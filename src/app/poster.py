"""Compose a poster collage out of the artwork of tracked media.

Every poster is placed whole -- never cropped -- at a size that follows the
chosen statistic, and the tiles are packed around a growing cluster. The result
is therefore not a rectangle: its silhouette is ragged and the background stays
transparent, and a seed decides the exact shape so the same selection can be
reshuffled into a different one.
"""

import hashlib
import io
import itertools
import logging
import random
from concurrent.futures import ThreadPoolExecutor

import requests
from django.apps import apps
from django.core.cache import cache
from django.db.models import Q, TextChoices
from django.utils import timezone
from PIL import Image, ImageChops, ImageDraw, ImageFont

from app import config
from app.models import MediaTypes, Status

logger = logging.getLogger(__name__)

# No tile cap by default: a poster covers everything in range. A limit can be
# asked for, and it is then split evenly between the selected media types.
# Every tile is one HTTP request for artwork, cached for a week, so only the
# first build of a large library is slow.
NO_LIMIT = 0
IMAGE_TIMEOUT = 10
IMAGE_MAX_BYTES = 8 * 1024 * 1024
IMAGE_CACHE_TIMEOUT = 60 * 60 * 24 * 7  # a week
DOWNLOAD_WORKERS = 12
# Some artwork hosts (Wikimedia among them) reject the default requests
# user agent outright, which would silently turn those tiles into placeholders.
USER_AGENT = "Yamtrack poster builder"

# Tile height the heaviest entry is drawn at. Artwork is never scaled above its
# own resolution, so this is a ceiling rather than a target: a 500px-tall
# poster stays 500px tall and simply ends up smaller in the collage.
BASE_TILE_HEIGHT = 900
# The lightest entry still gets this share of the tallest tile.
MIN_TILE_SHARE = 0.35
# How much a tile may drift from its grade's size, as a fraction of the gap
# between grades. Below 0.5 so grades can never cross over into each other.
SIZE_JITTER = 0.35
# How far artwork may be enlarged past its own resolution to reach the size its
# statistic earned it. Beyond this the interpolation shows.
MAX_UPSCALE = 2.0
# Artwork that could not be downloaded is drawn at this aspect ratio.
PLACEHOLDER_RATIO = 2 / 3

MAX_GAP = 64

# Not a user-facing limit: a runaway selection (150 large covers) would
# otherwise allocate gigabytes of RGBA. Everything is scaled down together when
# the canvas would exceed this, which keeps relative sizes intact.
MAX_CANVAS_PIXELS = 120_000_000

# Longest edge of the preview rendered on the page. The download is always the
# full-size render.
PREVIEW_MAX_EDGE = 1400

# TV and season rows have no date columns: theirs come from their episodes.
EPISODE_DATE_PATHS = {
    MediaTypes.TV.value: "seasons__episodes__end_date",
    MediaTypes.SEASON.value: "episodes__end_date",
}

# Placed tiles whose faces are offered to the next tile. Only the outermost
# ones can have a free side, so this bounds the work per tile.
FRONTIER_TILES = 60

# Outward nudge applied to a chosen spot, in pixels, so neighbours stack
# unevenly rather than snapping flush.
JITTER = 10

# How far inside the radius already reached a tile may still settle. Below 1.0
# so tiles can tuck into the outer fringe, high enough that they cannot drop
# into the middle of the poster.
INTERIOR_FLOOR = 0.82

# How many of the best candidate spots a tile picks between, and how often each
# is taken. Never taking the tightest one would scatter the collage; always
# taking it would close the silhouette back into a rectangle.
ROUGHNESS_CHOICES = 4
ROUGHNESS_WEIGHTS = (0.55, 0.25, 0.13, 0.07)


class PosterStatistics(TextChoices):
    """Statistics a poster can size its tiles by."""

    SCORE = "score", "Score"
    PROGRESS = "progress", "Progress"
    RECENCY = "recency", "Recently tracked"
    EQUAL = "equal", "Equal size"


class GapStyles(TextChoices):
    """How the space between tiles is drawn."""

    NONE = "none", "Touching"
    SOLID = "solid", "Spaced"
    GRADIENT = "gradient", "Feathered"


def collect_tiles(user_media, statistic, limit=NO_LIMIT):
    """Return ``(media, weight)`` pairs to draw, heaviest first.

    ``user_media`` is the ``{media_type: queryset}`` mapping that the
    statistics page already builds for a date range, so a poster always covers
    exactly what the statistics for that range cover.

    ``limit`` of 0 means everything. When one is given it is shared between
    media types rather than handed to whichever type happens to weigh most: a
    library with a thousand anime would otherwise fill every tile and push the
    games and manga out entirely.
    """
    per_type = {}

    for media_type, queryset in user_media.items():
        # Progress is counted in different units per media type (episodes,
        # minutes, pages), so it is only comparable within a type.
        media_list = list(queryset)
        type_max = None
        if statistic == PosterStatistics.PROGRESS:
            type_max = _max_progress(media_list)

        weighted = []
        for media in media_list:
            if media.item.image in (None, ""):
                continue

            weighted.append((media, _weight(media, statistic, type_max)))

        if weighted:
            weighted.sort(key=lambda pair: pair[1], reverse=True)
            per_type[media_type] = weighted

    per_type = _drop_covered_entries(per_type)

    if limit:
        selected = _take_round_robin(per_type, limit)
    else:
        selected = [entry for entries in per_type.values() for entry in entries]

    selected.sort(key=lambda pair: pair[1], reverse=True)
    return selected


def _drop_covered_entries(per_type):
    """Collapse entries that would draw the same title more than once.

    Tracking a show also tracks its seasons, so a poster of both would show
    one tile for the show and another seven for its seasons. When the show
    itself is in the selection its seasons stand down; picking seasons alone
    still gives a tile each.

    The second rule catches manual entries: anything sharing a title's id and
    its exact artwork is the same picture twice over.
    """
    tracked_shows = {media.id for media, _ in per_type.get(MediaTypes.TV.value, ())}

    seen = set()
    kept = {media_type: [] for media_type in per_type}

    # Heaviest first, so the survivor of a collapse is the biggest tile.
    ranked = sorted(
        (
            (weight, media_type, media)
            for media_type, entries in per_type.items()
            for media, weight in entries
        ),
        key=lambda entry: entry[0],
        reverse=True,
    )

    for weight, media_type, media in ranked:
        if (
            media_type == MediaTypes.SEASON.value
            and getattr(media, "related_tv_id", None) in tracked_shows
        ):
            continue

        key = (media.item.source, media.item.media_id, media.item.image)
        if key in seen:
            continue

        seen.add(key)
        kept[media_type].append((media, weight))

    return {media_type: entries for media_type, entries in kept.items() if entries}


def _take_round_robin(per_type, limit):
    """Take entries one media type at a time until the budget runs out.

    Types that run out early hand their remaining share back to the others, so
    a small games list does not cost anime its tiles and vice versa.
    """
    taken = []
    cursors = dict.fromkeys(per_type, 0)

    while len(taken) < limit:
        progressed = False
        for media_type, entries in per_type.items():
            index = cursors[media_type]
            if index >= len(entries):
                continue

            taken.append(entries[index])
            cursors[media_type] = index + 1
            progressed = True

            if len(taken) >= limit:
                break

        if not progressed:
            break

    return taken


def _max_progress(media_list):
    """Return the largest progress value in a list, or None when all are zero."""
    largest = max((media.progress or 0 for media in media_list), default=0)
    return largest or None


def _weight(media, statistic, type_max):
    """Return the raw weight of one media row.

    Nothing is ever dropped for lacking the statistic: a missing score, no
    progress or no timestamp weighs zero, which scale_weights turns into the
    smallest tile. Dropping them instead silently emptied whole media types
    out of a poster -- people tend to score their anime and not their games.
    """
    if statistic == PosterStatistics.EQUAL:
        return 1.0

    if statistic == PosterStatistics.SCORE:
        return float(media.score) if media.score is not None else 0.0

    if statistic == PosterStatistics.PROGRESS:
        if not type_max:
            return 0.0
        return (media.progress or 0) / type_max

    # Recency: newer activity weighs more. progressed_at is maintained by
    # MonitorField on every progress change, so it is the closest thing to a
    # "last touched" timestamp that every media type has.
    touched = media.progressed_at or media.created_at
    if touched is None:
        return 0.0
    age_days = max((timezone.now() - touched).days, 0)
    return 1.0 / (1.0 + age_days / 30)


def scale_weights(weights):
    """Rescale weights into tile-height shares between MIN_TILE_SHARE and 1.

    Ranked by distinct value rather than scaled linearly: a library scored 7 to
    10 with some unscored entries would otherwise squeeze four grades into the
    top sixth of the range, leaving a 7 and a 10 all but the same size. Each
    distinct score becomes its own visible step instead.
    """
    if not weights:
        return []

    levels = sorted(set(weights))
    if len(levels) == 1:
        return [1.0] * len(weights)

    step = (1 - MIN_TILE_SHARE) / (len(levels) - 1)
    share_of = {
        value: MIN_TILE_SHARE + rank * step for rank, value in enumerate(levels)
    }
    return [share_of[weight] for weight in weights]


def build_poster(tiles, gap=0, gap_style=GapStyles.NONE, seed=0):
    """Render tiles into a transparent, ragged-edged poster and return it.

    The canvas is whatever the packing needs: there is no target size, and no
    artwork is cropped or enlarged.
    """
    if not tiles:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    rng = random.Random(seed)  # noqa: S311 - layout variety, not security
    shares = jitter_shares(scale_weights([weight for _, weight in tiles]), rng)
    raw = _fetch_images([media.item.image for media, _ in tiles])

    sources = []
    for (media, _), share in zip(tiles, shares, strict=False):
        sources.append(_prepare_tile(raw.get(media.item.image), media, share))

    sizes = _limit_total_pixels([image.size for image in sources])
    spacing = gap if gap_style == GapStyles.SOLID else 0

    # Pack biggest first: the packer keeps each tile outside the radius already
    # reached, so whatever goes in first ends up in the middle. Ordering by the
    # final pixel size -- after jitter and after any artwork came back smaller
    # than asked for -- is what actually puts the big ones at the centre.
    order = sorted(
        range(len(sizes)),
        key=lambda index: sizes[index][0] * sizes[index][1],
        reverse=True,
    )
    positions, canvas_size = pack([sizes[index] for index in order], spacing, seed)

    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    feather = gap if gap_style == GapStyles.GRADIENT else 0

    for slot, index in enumerate(order):
        image = sources[index]
        size = sizes[index]
        tile = image if image.size == size else image.resize(size, Image.LANCZOS)
        mask = _feather_mask(tile, feather)
        canvas.paste(tile, positions[slot], mask)

    return canvas


def jitter_shares(shares, rng):
    """Vary tile sizes within their grade so no two are stamped alike.

    The nudge is a fraction of the gap between grades, so a 9 can end up a
    little larger or smaller than another 9 while still never reaching the
    size of a 10.
    """
    levels = sorted(set(shares))
    if len(levels) < 2:  # noqa: PLR2004 - a single grade has no gap to work in
        spread = (1 - MIN_TILE_SHARE) * SIZE_JITTER
    else:
        step = min(b - a for a, b in itertools.pairwise(levels))
        spread = step * SIZE_JITTER

    return [
        min(1.0, max(MIN_TILE_SHARE / 2, share + rng.uniform(-spread, spread)))
        for share in shares
    ]


def preview(image, max_edge=PREVIEW_MAX_EDGE):
    """Return a downscaled copy for on-page display, or the image if small."""
    if max(image.size) <= max_edge:
        return image

    thumbnail = image.copy()
    thumbnail.thumbnail((max_edge, max_edge), Image.LANCZOS)
    return thumbnail


def pack(sizes, spacing, seed):
    """Place rectangles around a centre without overlapping.

    Each tile is tried against the outward faces of the tiles already placed
    and takes a position close to the cluster's centre of mass, so the collage
    grows outwards in every direction and ends up roughly round rather than
    squared off against an origin. Tiles stack unevenly: the choice is made
    between the best few spots and then nudged outwards, so edges stay rough
    and neighbours do not all line up flush.

    ``sizes`` arrives heaviest-first and placement keeps it that way: a tile
    may not settle back inside the radius already reached, so size falls off
    from the middle outwards and the unscored, smallest entries end up on the
    rim instead of dotted through the centre.
    """
    if not sizes:
        return [], (1, 1)

    rng = random.Random(seed)  # noqa: S311 - layout variety, not security
    placed = []
    positions = []
    bounds = None
    centre = (0.0, 0.0)
    mass = 0.0
    reach = 0.0

    for width, height in sizes:
        if not placed:
            # Centre the first tile on the origin so growth is symmetric.
            spot = (-width // 2, -height // 2)
        else:
            spot = _best_spot((placed, centre, reach), (width, height), spacing, rng)

        x, y = spot
        positions.append(spot)
        placed.append((x, y, width, height))
        bounds = _grow(bounds, x, y, width, height)

        # Area-weighted centre of mass: big tiles anchor the middle.
        area = width * height
        centre = (
            (centre[0] * mass + (x + width / 2) * area) / (mass + area),
            (centre[1] * mass + (y + height / 2) * area) / (mass + area),
        )
        mass += area
        reach = max(
            reach,
            _distance(x + width / 2, y + height / 2, centre),
        )

    origin_x, origin_y, right, bottom = bounds
    positions = [(x - origin_x, y - origin_y) for x, y in positions]
    return positions, (right - origin_x, bottom - origin_y)


def _best_spot(cluster, size, spacing, rng):
    """Return a free position for one tile, on the outside of the cluster.

    ``cluster`` is the ``(placed, centre, reach)`` state the packer carries.
    """
    placed, centre, reach = cluster
    width, height = size

    # Only the outermost tiles can offer a free face, and checking all of them
    # on a large poster is wasted work.
    frontier = sorted(
        placed,
        key=lambda rect: (
            -_distance(rect[0] + rect[2] / 2, rect[1] + rect[3] / 2, centre)
        ),
    )[:FRONTIER_TILES]

    scored = []
    for candidate in _candidate_spots(frontier, size, spacing):
        x, y = candidate
        if _collides(placed, x, y, width, height, spacing):
            continue

        # Radial distance keeps the blob round; the seeded jitter keeps two
        # posters of the same library from resolving ties identically.
        distance = _distance(x + width / 2, y + height / 2, centre)
        scored.append((distance * rng.uniform(0.9, 1.1), distance, candidate))

    if not scored:
        # Nothing free anywhere near: drop the tile below the cluster.
        return (round(centre[0] - width / 2), round(centre[1] + height))

    # Later tiles are lighter, so keep them out of the middle: anything that
    # would settle well inside the radius already reached is backfilling a hole
    # left by a bigger tile, which is what put unscored entries in the centre.
    outside = [entry for entry in scored if entry[1] >= reach * INTERIOR_FLOOR]
    if outside:
        scored = outside

    # Always taking the closest spot packs the cluster into a tight lattice.
    # Picking among the best few leaves tiles jutting out, which is what gives
    # the poster its rough edge -- and another axis of variety between seeds.
    scored.sort(key=lambda entry: entry[0])
    shortlist = scored[:ROUGHNESS_CHOICES]
    weights = ROUGHNESS_WEIGHTS[: len(shortlist)]
    x, y = rng.choices(shortlist, weights=weights)[0][2]

    # A small outward nudge so neighbours stack unevenly instead of snapping
    # into flush rows. Outward only, so it can never create an overlap.
    if JITTER:
        dx = rng.randint(0, JITTER)
        dy = rng.randint(0, JITTER)
        x += dx if x + width / 2 >= centre[0] else -dx
        y += dy if y + height / 2 >= centre[1] else -dy

    return (x, y)


def _candidate_spots(frontier, size, spacing):
    """Yield positions where a tile would sit flush against a placed one."""
    width, height = size

    for px, py, pw, ph in frontier:
        right = px + pw + spacing
        left = px - width - spacing
        below = py + ph + spacing
        above = py - height - spacing

        yield (right, py)
        yield (right, py + ph - height)
        yield (left, py)
        yield (left, py + ph - height)
        yield (px, below)
        yield (px + pw - width, below)
        yield (px, above)
        yield (px + pw - width, above)


def _distance(x, y, centre):
    """Return the distance from a point to the cluster centre."""
    return ((x - centre[0]) ** 2 + (y - centre[1]) ** 2) ** 0.5


def _collides(placed, x, y, width, height, spacing):
    """Return whether a candidate rectangle touches any placed one."""
    left = x - spacing
    top = y - spacing
    right = x + width + spacing
    bottom = y + height + spacing

    for px, py, pw, ph in placed:
        if left < px + pw and right > px and top < py + ph and bottom > py:
            return True
    return False


def _grow(bounds, x, y, width, height):
    """Return the bounding box extended to include a rectangle."""
    if bounds is None:
        return (x, y, x + width, y + height)

    left, top, right, bottom = bounds
    return (
        min(left, x),
        min(top, y),
        max(right, x + width),
        max(bottom, y + height),
    )


def _prepare_tile(raw, media, share):
    """Return the whole artwork scaled to its share, or a titled placeholder.

    Size follows the statistic alone. Scaling to each source's own resolution
    instead made a show whose artwork happens to be 1000px tall tower over a
    500px one of the same score, which read as the statistic being broken.
    """
    height = max(1, round(BASE_TILE_HEIGHT * share))

    if raw is not None:
        try:
            with Image.open(io.BytesIO(raw)) as source:
                image = source.convert("RGBA")
                # Small artwork is allowed to grow a little so it keeps its
                # place in the ranking, but not so far that it turns to mush.
                target_height = min(height, round(image.height * MAX_UPSCALE))
                ratio = target_height / image.height
                target_width = max(1, round(image.width * ratio))
                if (target_width, target_height) == image.size:
                    return image.copy()
                return image.resize((target_width, target_height), Image.LANCZOS)
        except (OSError, ValueError):
            logger.warning("Unreadable poster artwork for %s", media.item.title)

    return _placeholder_tile(media, max(1, round(height * PLACEHOLDER_RATIO)), height)


def _placeholder_tile(media, width, height):
    """Return a flat tile carrying the title, for artwork that failed to load."""
    colour = config.get_stats_color(media.item.media_type)
    tile = Image.new("RGBA", (width, height), (*_dim(colour), 255))

    title = media.item.title
    if not title or width < 60 or height < 24:  # noqa: PLR2004 - unreadable below this
        return tile

    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    text = title if len(title) <= 24 else f"{title[:21]}..."  # noqa: PLR2004
    draw.text(
        (width / 2, height / 2),
        text,
        fill=(235, 235, 235, 255),
        font=font,
        anchor="mm",
    )
    return tile


def _dim(hex_colour, factor=0.35):
    """Return a darkened RGB tuple for a ``#rrggbb`` string."""
    value = hex_colour.lstrip("#")
    rgb = tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    return tuple(int(channel * factor) for channel in rgb)


def _limit_total_pixels(sizes, budget=MAX_CANVAS_PIXELS):
    """Shrink every tile by the same factor if the collage would be enormous."""
    total = sum(width * height for width, height in sizes)
    if total <= budget or total == 0:
        return sizes

    factor = (budget / total) ** 0.5
    logger.info("Poster scaled to %.2f of full size to stay within memory", factor)
    # Floor, not round: rounding up would push the total back over the budget.
    return [
        (max(1, int(width * factor)), max(1, int(height * factor)))
        for width, height in sizes
    ]


def _feather_mask(tile, feather):
    """Return the paste mask for a tile, fading its edges when asked.

    The tile's own alpha is the base, so transparent artwork stays transparent;
    feathering multiplies a border ramp on top of it.
    """
    mask = tile.getchannel("A")
    width, height = tile.size
    feather = min(feather, width // 2, height // 2)
    if feather <= 0:
        return mask

    ramp = Image.new("L", tile.size, 255)
    draw = ImageDraw.Draw(ramp)
    for step in range(feather):
        alpha = int(255 * (step + 1) / (feather + 1))
        draw.rectangle((step, step, width - 1 - step, height - 1 - step), outline=alpha)

    return ImageChops.multiply(mask, ramp)


def _fetch_images(urls):
    """Download artwork concurrently and return ``{url: bytes}``."""
    unique = {url for url in urls if url}
    if not unique:
        return {}

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        results = pool.map(_fetch_image, unique)

    return {url: data for url, data in zip(unique, results, strict=True) if data}


def _fetch_image(url):
    """Return the bytes of one image, from cache when possible."""
    key = f"poster_art_{hashlib.sha256(url.encode()).hexdigest()}"
    cached = cache.get(key)
    if cached is not None:
        return cached

    try:
        response = requests.get(
            url,
            timeout=IMAGE_TIMEOUT,
            stream=True,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()

        length = response.headers.get("Content-Length")
        if length and int(length) > IMAGE_MAX_BYTES:
            logger.warning("Skipping oversized poster artwork: %s", url)
            return None

        data = response.content[:IMAGE_MAX_BYTES]
    except (requests.RequestException, ValueError):
        logger.warning("Could not download poster artwork: %s", url)
        return None

    cache.set(key, data, IMAGE_CACHE_TIMEOUT)
    return data


def parse_limit(value):
    """Return how many tiles to draw, 0 (the default) meaning all of them."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return NO_LIMIT
    return max(NO_LIMIT, parsed)


def parse_gap(value):
    """Return a gap width clamped to the supported range."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(parsed, MAX_GAP))


def parse_statistic(value):
    """Return a valid statistic choice, falling back to score."""
    if value in PosterStatistics.values:
        return value
    return PosterStatistics.SCORE.value


def parse_gap_style(value):
    """Return a valid gap style, falling back to none."""
    if value in GapStyles.values:
        return value
    return GapStyles.NONE.value


def parse_seed(value):
    """Return the layout seed, or a fresh random one when absent.

    The seed travels in the URL so the preview on the page and the full-size
    download render the identical shape, and reshuffling is just a new seed.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return random.randrange(1, 1_000_000)  # noqa: S311 - layout variety


def gather_media(user, media_types, statuses, start_date=None, end_date=None):
    """Return ``{media_type: queryset}`` of everything the poster should draw.

    Deliberately not the statistics page's selector: that one derives TV and
    seasons from tracked *episodes*, so a show marked complete without
    per-episode rows -- which is most of them after an import -- never appears
    at all. A poster asks each media model for the user's own rows instead, so
    what you track is what you get.
    """
    gathered = {}

    for media_type in media_types:
        if media_type not in MediaTypes.values:
            continue
        # Episodes are children of a season, tracked through it, and carry no
        # status or user of their own.
        if media_type == MediaTypes.EPISODE.value:
            continue

        model = apps.get_model(app_label="app", model_name=media_type)
        queryset = model.objects.filter(user=user).select_related("item")

        if statuses:
            queryset = queryset.filter(status__in=statuses)

        if start_date and end_date:
            queryset = _apply_range(queryset, media_type, start_date, end_date)

        gathered[media_type] = queryset

    return gathered


def _apply_range(queryset, media_type, start_date, end_date):
    """Narrow a queryset to a date range.

    TV shows and seasons carry no dates of their own -- theirs are computed
    from the episodes underneath them -- so they are dated through those
    instead. A show with no episode history cannot be placed in time at all
    and only appears when no range is asked for.
    """
    episode_path = EPISODE_DATE_PATHS.get(media_type)
    if episode_path:
        return queryset.filter(
            **{f"{episode_path}__range": (start_date, end_date)},
        ).distinct()

    return queryset.filter(_within(start_date, end_date))


def _within(start_date, end_date):
    """Return the filter for entries that fall inside a date range.

    Matches the statistics page: an entry overlaps the range through whichever
    of its dates it has. Entries carrying neither date cannot be placed in time
    and only show up when no range is asked for.
    """
    return (
        (
            Q(start_date__isnull=False)
            & Q(end_date__isnull=False)
            & ~(Q(end_date__lt=start_date) | Q(start_date__gt=end_date))
        )
        | (
            Q(start_date__isnull=False)
            & Q(end_date__isnull=True)
            & Q(start_date__gte=start_date)
            & Q(start_date__lte=end_date)
        )
        | (
            Q(start_date__isnull=True)
            & Q(end_date__isnull=False)
            & Q(end_date__gte=start_date)
            & Q(end_date__lte=end_date)
        )
    )


def parse_statuses(values):
    """Return the requested tracking statuses, or all of them when none are."""
    wanted = [value for value in values if value in Status.values]
    return wanted or list(Status.values)


def parse_media_types(values, fallback):
    """Return the requested media types, or the user's enabled ones."""
    wanted = [value for value in values if value in MediaTypes.values]
    return wanted or list(fallback)
