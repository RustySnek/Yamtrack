import datetime
import io
import itertools
import random
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from PIL import Image

from app import poster
from app.models import (
    TV,
    Anime,
    Episode,
    Game,
    Item,
    MediaTypes,
    Movie,
    Season,
    Sources,
    Status,
)


def artwork_bytes(colour=(120, 30, 30), size=(300, 450)):
    """Return PNG bytes standing in for a downloaded poster."""
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


class PackTests(TestCase):
    """Test the collage packing."""

    def test_tiles_do_not_overlap(self):
        """Packed rectangles never cover each other."""
        sizes = [(300, 450), (200, 300), (150, 220), (400, 260), (180, 180)]

        positions, _ = poster.pack(sizes, spacing=0, seed=1)

        placed = [
            (x, y, width, height)
            for (x, y), (width, height) in zip(positions, sizes, strict=True)
        ]
        for index, (x1, y1, w1, h1) in enumerate(placed):
            for x2, y2, w2, h2 in placed[index + 1 :]:
                overlaps = (
                    x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2
                )
                self.assertFalse(overlaps)

    def test_growth_is_radial_not_from_a_corner(self):
        """Tiles spread around the centre, so no edge comes out flush."""
        sizes = [(300, 450)] * 9

        positions, canvas = poster.pack(sizes, spacing=0, seed=11)

        # A corner-anchored packer puts several tiles at x=0 and y=0; a radial
        # one leaves the extremes to single tiles on each side.
        flush_left = sum(1 for x, _ in positions if x == 0)
        flush_top = sum(1 for _, y in positions if y == 0)
        self.assertLessEqual(flush_left, 2)
        self.assertLessEqual(flush_top, 2)

        # And the result is not a tall column or a wide strip.
        ratio = canvas[0] / canvas[1]
        self.assertGreater(ratio, 0.4)
        self.assertLess(ratio, 2.5)

    def test_small_tiles_stay_out_of_the_middle(self):
        """Later, lighter tiles land on the rim, not in interior holes."""
        big = [(400, 600)] * 8
        small = [(120, 180)] * 8

        positions, canvas = poster.pack(big + small, spacing=0, seed=17)
        sizes = big + small

        centre_x, centre_y = canvas[0] / 2, canvas[1] / 2
        radii = [
            (
                ((x + width / 2 - centre_x) ** 2 + (y + height / 2 - centre_y) ** 2)
                ** 0.5
            )
            for (x, y), (width, height) in zip(positions, sizes, strict=True)
        ]

        big_radius = sum(radii[:8]) / 8
        small_radius = sum(radii[8:]) / 8
        self.assertGreater(small_radius, big_radius)

    def test_canvas_is_the_bounding_box(self):
        """The canvas is exactly what the packed tiles occupy."""
        sizes = [(300, 450), (200, 300), (150, 220)]

        positions, canvas = poster.pack(sizes, spacing=0, seed=7)

        right = max(
            x + width for (x, _), (width, _) in zip(positions, sizes, strict=True)
        )
        bottom = max(
            y + height for (_, y), (_, height) in zip(positions, sizes, strict=True)
        )
        self.assertEqual(canvas, (right, bottom))
        self.assertGreaterEqual(min(x for x, _ in positions), 0)
        self.assertGreaterEqual(min(y for _, y in positions), 0)

    def test_the_same_seed_packs_the_same_way(self):
        """A seed makes the shape reproducible, so preview and download match."""
        sizes = [(300, 450), (200, 300), (150, 220), (400, 260)]

        first, first_canvas = poster.pack(sizes, spacing=0, seed=42)
        second, second_canvas = poster.pack(sizes, spacing=0, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(first_canvas, second_canvas)

    def test_different_seeds_pack_differently(self):
        """Reshuffling gives a different silhouette."""
        sizes = [(300, 450), (200, 300), (150, 220), (400, 260), (260, 380)]

        layouts = {
            tuple(poster.pack(sizes, spacing=0, seed=seed)[0]) for seed in range(12)
        }

        self.assertGreater(len(layouts), 1)

    def test_spacing_keeps_tiles_apart(self):
        """Spacing separates neighbours instead of letting them touch."""
        sizes = [(200, 300), (200, 300), (200, 300)]

        positions, _ = poster.pack(sizes, spacing=20, seed=3)

        placed = [
            (x, y, width, height)
            for (x, y), (width, height) in zip(positions, sizes, strict=True)
        ]
        for index, (x1, y1, w1, h1) in enumerate(placed):
            for x2, y2, w2, h2 in placed[index + 1 :]:
                gap_x = max(x2 - (x1 + w1), x1 - (x2 + w2))
                gap_y = max(y2 - (y1 + h1), y1 - (y2 + h2))
                self.assertGreaterEqual(max(gap_x, gap_y), 20)

    def test_empty_input(self):
        """No tiles means no positions, not a crash."""
        self.assertEqual(poster.pack([], spacing=0, seed=1), ([], (1, 1)))

    def test_scale_weights_keeps_the_smallest_visible(self):
        """The smallest tile keeps a usable share of the tallest."""
        scaled = poster.scale_weights([10, 1])

        self.assertEqual(max(scaled), 1.0)
        self.assertEqual(min(scaled), poster.MIN_TILE_SHARE)

    def test_scale_weights_gives_every_grade_its_own_step(self):
        """Clustered scores still differ visibly in size."""
        # A typical library: grades 7-10 plus unscored entries at zero.
        scaled = poster.scale_weights([0, 7, 8, 9, 10])

        self.assertEqual(scaled[0], poster.MIN_TILE_SHARE)
        self.assertEqual(scaled[-1], 1.0)

        steps = [round(b - a, 6) for a, b in itertools.pairwise(scaled)]
        # Even steps, and each one big enough to see.
        self.assertEqual(len(set(steps)), 1)
        self.assertGreater(steps[0], 0.1)

    def test_jitter_varies_sizes_within_a_grade(self):
        """Same-grade tiles differ a little, and grades never cross over."""
        shares = poster.scale_weights([10, 10, 10, 9, 9, 9, 7, 7])
        rng = random.Random(4)  # noqa: S311 - layout variety
        jittered = poster.jitter_shares(shares, rng)

        tens = jittered[:3]
        nines = jittered[3:6]

        # Varied inside the grade...
        self.assertGreater(len(set(tens)), 1)
        # ...but a nine never grows into a ten.
        self.assertGreater(min(tens), max(nines))

    def test_jitter_is_seeded(self):
        """The same seed jitters identically, so a poster is reproducible."""
        shares = poster.scale_weights([10, 9, 8, 7])

        first = poster.jitter_shares(shares, random.Random(12))  # noqa: S311
        second = poster.jitter_shares(shares, random.Random(12))  # noqa: S311
        other = poster.jitter_shares(shares, random.Random(13))  # noqa: S311

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_jitter_stays_within_bounds(self):
        """No tile jitters past full size or down to nothing."""
        shares = poster.scale_weights([10, 9, 8, 7, 0])

        for seed in range(20):
            jittered = poster.jitter_shares(shares, random.Random(seed))  # noqa: S311
            self.assertLessEqual(max(jittered), 1.0)
            self.assertGreater(min(jittered), 0)

    def test_scale_weights_ties_share_a_size(self):
        """Two entries with the same score come out the same size."""
        scaled = poster.scale_weights([9, 7, 9, 10])

        self.assertEqual(scaled[0], scaled[2])
        self.assertLess(scaled[1], scaled[0])
        self.assertEqual(scaled[3], 1.0)

    def test_total_pixels_are_capped(self):
        """An enormous selection is scaled down as a whole, keeping proportions."""
        sizes = [(20000, 20000), (10000, 10000)]

        limited = poster._limit_total_pixels(sizes)

        total = sum(width * height for width, height in limited)
        self.assertLessEqual(total, poster.MAX_CANVAS_PIXELS)
        # 2:1 in each dimension before, still 2:1 after.
        self.assertAlmostEqual(limited[0][0] / limited[1][0], 2, places=1)


class BuildPosterTests(TestCase):
    """Test poster rendering."""

    def setUp(self):
        """Create a user with a few tracked movies."""
        self.user = get_user_model().objects.create_user(
            username="poster",
            password="12345",  # noqa: S106
        )
        self.media = []
        for index, score in enumerate((9, 5, 2), start=1):
            item = Item.objects.create(
                media_id=str(index),
                source=Sources.MANUAL.value,
                media_type=MediaTypes.MOVIE.value,
                title=f"Movie {index}",
                image=f"https://example.com/{index}.jpg",
            )
            self.media.append(
                Movie.objects.create(
                    item=item,
                    user=self.user,
                    status=Status.COMPLETED.value,
                    score=score,
                    progress=index,
                    end_date=timezone.now(),
                ),
            )

    def test_collect_tiles_orders_by_statistic(self):
        """Tiles come back heaviest first, with unscored entries last."""
        unscored = Item.objects.create(
            media_id="99",
            source=Sources.MANUAL.value,
            media_type=MediaTypes.MOVIE.value,
            title="Unscored",
            image="https://example.com/99.jpg",
        )
        Movie.objects.create(
            item=unscored,
            user=self.user,
            status=Status.COMPLETED.value,
        )

        tiles = poster.collect_tiles(
            {MediaTypes.MOVIE.value: Movie.objects.filter(user=self.user)},
            poster.PosterStatistics.SCORE,
        )

        self.assertEqual(
            [media.item.title for media, _ in tiles],
            ["Movie 1", "Movie 2", "Movie 3", "Unscored"],
        )

    def test_collect_tiles_equal_keeps_unscored(self):
        """The equal-size statistic does not need a score."""
        tiles = poster.collect_tiles(
            {MediaTypes.MOVIE.value: Movie.objects.filter(user=self.user)},
            poster.PosterStatistics.EQUAL,
        )

        self.assertEqual(len(tiles), 3)
        self.assertEqual({weight for _, weight in tiles}, {1.0})

    def test_no_limit_draws_everything(self):
        """The default limit of 0 means every entry in range is drawn."""
        for index in range(12):
            item = Item.objects.create(
                media_id=f"bulk-{index}",
                source=Sources.MANUAL.value,
                media_type=MediaTypes.ANIME.value,
                title=f"Anime {index}",
                image=f"https://example.com/b{index}.jpg",
            )
            Anime.objects.create(
                item=item,
                user=self.user,
                status=Status.COMPLETED.value,
                score=6,
            )

        tiles = poster.collect_tiles(
            {
                MediaTypes.ANIME.value: Anime.objects.filter(user=self.user),
                MediaTypes.MOVIE.value: Movie.objects.filter(user=self.user),
            },
            poster.PosterStatistics.SCORE,
        )

        # 12 anime + the 3 scored movies from setUp, nothing dropped.
        self.assertEqual(len(tiles), 15)

    def test_budget_is_shared_between_media_types(self):
        """A crowded media type cannot push the others out of the poster."""
        many = []
        for index in range(20):
            item = Item.objects.create(
                media_id=f"crowd-{index}",
                source=Sources.MANUAL.value,
                media_type=MediaTypes.ANIME.value,
                title=f"Anime {index}",
                image=f"https://example.com/a{index}.jpg",
            )
            many.append(
                Anime.objects.create(
                    item=item,
                    user=self.user,
                    status=Status.COMPLETED.value,
                    score=10,
                ),
            )

        tiles = poster.collect_tiles(
            {
                MediaTypes.ANIME.value: Anime.objects.filter(user=self.user),
                MediaTypes.MOVIE.value: Movie.objects.filter(user=self.user),
            },
            poster.PosterStatistics.SCORE,
            limit=6,
        )

        types = {media.item.media_type for media, _ in tiles}
        self.assertEqual(len(tiles), 6)
        self.assertIn(MediaTypes.ANIME.value, types)
        # Every anime scores 10 and every movie less, yet movies still appear.
        self.assertIn(MediaTypes.MOVIE.value, types)

    def test_budget_reallocates_from_short_types(self):
        """A type with few entries gives its unused share back."""
        for index in range(5):
            item = Item.objects.create(
                media_id=f"extra-{index}",
                source=Sources.MANUAL.value,
                media_type=MediaTypes.ANIME.value,
                title=f"Anime {index}",
                image=f"https://example.com/a{index}.jpg",
            )
            Anime.objects.create(
                item=item,
                user=self.user,
                status=Status.COMPLETED.value,
                score=7,
            )

        tiles = poster.collect_tiles(
            {
                MediaTypes.ANIME.value: Anime.objects.filter(user=self.user),
                MediaTypes.MOVIE.value: Movie.objects.filter(user=self.user),
            },
            poster.PosterStatistics.SCORE,
            limit=8,
        )

        # 5 anime + 3 movies is everything there is, and all of it fits.
        self.assertEqual(len(tiles), 8)

    def test_entries_without_the_statistic_are_kept(self):
        """An unscored game still reaches a score poster, drawn smallest."""
        item = Item.objects.create(
            media_id="unscored-game",
            source=Sources.MANUAL.value,
            media_type=MediaTypes.GAME.value,
            title="Unscored Game",
            image="https://example.com/g.jpg",
        )
        Game.objects.create(item=item, user=self.user, status=Status.COMPLETED.value)

        tiles = poster.collect_tiles(
            {
                MediaTypes.MOVIE.value: Movie.objects.filter(user=self.user),
                MediaTypes.GAME.value: Game.objects.filter(user=self.user),
            },
            poster.PosterStatistics.SCORE,
        )

        titles = [media.item.title for media, _ in tiles]
        self.assertIn("Unscored Game", titles)
        # Weighs nothing, so it sorts last and scales to the smallest tile.
        self.assertEqual(titles[-1], "Unscored Game")
        shares = poster.scale_weights([weight for _, weight in tiles])
        self.assertEqual(shares[-1], poster.MIN_TILE_SHARE)

    def test_show_and_its_seasons_are_not_drawn_twice(self):
        """A season reusing its show's poster collapses into the show."""
        show_item = Item.objects.create(
            media_id="1668",
            source=Sources.MANUAL.value,
            media_type=MediaTypes.TV.value,
            title="Show",
            image="https://example.com/show.jpg",
        )
        # Planning: a completed show would have the provider fill in seasons.
        show = TV.objects.create(
            item=show_item,
            user=self.user,
            status=Status.PLANNING.value,
            score=9,
        )
        for number in (1, 2):
            season_item = Item.objects.create(
                media_id="1668",
                source=Sources.MANUAL.value,
                media_type=MediaTypes.SEASON.value,
                title="Show",
                # Seasons without their own art carry the show's.
                image="https://example.com/show.jpg",
                season_number=number,
            )
            Season.objects.create(
                item=season_item,
                related_tv=show,
                user=self.user,
                status=Status.PLANNING.value,
                score=8,
            )

        tiles = poster.collect_tiles(
            poster.gather_media(
                self.user,
                [MediaTypes.TV.value, MediaTypes.SEASON.value],
                list(Status.values),
            ),
            poster.PosterStatistics.SCORE,
        )

        # One tile for the show, and it is the heaviest of the three.
        drawn = [media for media, _ in tiles]
        self.assertEqual(len(drawn), 1)
        self.assertEqual(drawn[0].item.media_type, MediaTypes.TV.value)

        # Seasons alone: these two carry the same picture, so one tile.
        seasons_only = poster.collect_tiles(
            poster.gather_media(
                self.user,
                [MediaTypes.SEASON.value],
                list(Status.values),
            ),
            poster.PosterStatistics.SCORE,
        )
        self.assertEqual(len(seasons_only), 1)

    def test_tracked_show_covers_its_seasons(self):
        """A season is not drawn separately when its show is in the poster."""
        show_item = Item.objects.create(
            media_id="1669",
            source=Sources.MANUAL.value,
            media_type=MediaTypes.TV.value,
            title="Other Show",
            image="https://example.com/other-show.jpg",
        )
        show = TV.objects.create(
            item=show_item,
            user=self.user,
            status=Status.PLANNING.value,
            score=9,
        )
        season_item = Item.objects.create(
            media_id="1669",
            source=Sources.MANUAL.value,
            media_type=MediaTypes.SEASON.value,
            title="Other Show",
            image="https://example.com/other-season-2.jpg",
            season_number=2,
        )
        Season.objects.create(
            item=season_item,
            related_tv=show,
            user=self.user,
            status=Status.PLANNING.value,
            score=8,
        )

        tiles = poster.collect_tiles(
            poster.gather_media(
                self.user,
                [MediaTypes.TV.value, MediaTypes.SEASON.value],
                list(Status.values),
            ),
            poster.PosterStatistics.SCORE,
        )

        # The show covers it: distinct artwork or not, it is the same title.
        self.assertEqual(
            [media.item.media_type for media, _ in tiles],
            [MediaTypes.TV.value],
        )

    def test_seasons_alone_keep_their_own_tiles(self):
        """Seasons picked without their show each get a tile of their own."""
        show_item = Item.objects.create(
            media_id="1670",
            source=Sources.MANUAL.value,
            media_type=MediaTypes.TV.value,
            title="Show",
            image="https://example.com/s.jpg",
        )
        show = TV.objects.create(
            item=show_item,
            user=self.user,
            status=Status.PLANNING.value,
            score=9,
        )
        for number in (1, 2):
            season_item = Item.objects.create(
                media_id="1670",
                source=Sources.MANUAL.value,
                media_type=MediaTypes.SEASON.value,
                title="Show",
                # Distinct artwork per season, as TMDB actually supplies.
                image=f"https://example.com/s{number}.jpg",
                season_number=number,
            )
            Season.objects.create(
                item=season_item,
                related_tv=show,
                user=self.user,
                status=Status.PLANNING.value,
                score=8,
            )

        tiles = poster.collect_tiles(
            poster.gather_media(
                self.user,
                [MediaTypes.SEASON.value],
                list(Status.values),
            ),
            poster.PosterStatistics.SCORE,
        )

        self.assertEqual(len(tiles), 2)

    def test_status_filter(self):
        """Only the requested tracking statuses reach the poster."""
        planned = Item.objects.create(
            media_id="planned",
            source=Sources.MANUAL.value,
            media_type=MediaTypes.MOVIE.value,
            title="Planned",
            image="https://example.com/p.jpg",
        )
        Movie.objects.create(
            item=planned,
            user=self.user,
            status=Status.PLANNING.value,
            score=5,
        )

        completed = poster.gather_media(
            self.user,
            [MediaTypes.MOVIE.value],
            [Status.COMPLETED.value],
        )
        titles = [
            media.item.title
            for media, _ in poster.collect_tiles(
                completed,
                poster.PosterStatistics.SCORE,
            )
        ]
        self.assertNotIn("Planned", titles)

        both = poster.gather_media(
            self.user,
            [MediaTypes.MOVIE.value],
            [Status.COMPLETED.value, Status.PLANNING.value],
        )
        titles = [
            media.item.title
            for media, _ in poster.collect_tiles(both, poster.PosterStatistics.SCORE)
        ]
        self.assertIn("Planned", titles)

    def test_parse_statuses(self):
        """No status asked for means every status."""
        self.assertEqual(poster.parse_statuses([]), list(Status.values))
        self.assertEqual(poster.parse_statuses(["nonsense"]), list(Status.values))
        self.assertEqual(
            poster.parse_statuses([Status.DROPPED.value]),
            [Status.DROPPED.value],
        )

    def test_collect_tiles_skips_items_without_artwork(self):
        """An entry with no image cannot contribute a tile."""
        Item.objects.filter(media_id="1").update(image="")

        tiles = poster.collect_tiles(
            {MediaTypes.MOVIE.value: Movie.objects.filter(user=self.user)},
            poster.PosterStatistics.EQUAL,
        )

        self.assertEqual(len(tiles), 2)

    @patch("app.poster._fetch_image")
    def test_artwork_is_never_cropped(self, mock_fetch):
        """Tiles keep the whole artwork, so their aspect ratio survives."""
        mock_fetch.return_value = artwork_bytes(size=(400, 600))

        tiles = poster.collect_tiles(
            {MediaTypes.MOVIE.value: Movie.objects.filter(user=self.user)},
            poster.PosterStatistics.SCORE,
        )
        prepared = poster._prepare_tile(
            artwork_bytes(size=(400, 600)),
            tiles[0][0],
            1.0,
        )

        self.assertAlmostEqual(prepared.width / prepared.height, 400 / 600, places=2)

    @patch("app.poster._fetch_image")
    def test_size_follows_the_statistic_not_the_source_resolution(self, mock_fetch):
        """Two entries of equal weight match in size, however big their art is."""
        mock_fetch.return_value = artwork_bytes()

        tiles = poster.collect_tiles(
            {MediaTypes.MOVIE.value: Movie.objects.filter(user=self.user)},
            poster.PosterStatistics.SCORE,
        )
        small_source = poster._prepare_tile(
            artwork_bytes(size=(500, 750)),
            tiles[0][0],
            1.0,
        )
        big_source = poster._prepare_tile(
            artwork_bytes(size=(1000, 1500)),
            tiles[0][0],
            1.0,
        )

        self.assertEqual(small_source.height, big_source.height)
        self.assertEqual(small_source.height, poster.BASE_TILE_HEIGHT)

    @patch("app.poster._fetch_image")
    def test_tiny_artwork_is_not_blown_up(self, mock_fetch):
        """Enlarging is capped so a thumbnail does not turn into mush."""
        mock_fetch.return_value = artwork_bytes(size=(100, 150))

        tiles = poster.collect_tiles(
            {MediaTypes.MOVIE.value: Movie.objects.filter(user=self.user)},
            poster.PosterStatistics.SCORE,
        )
        prepared = poster._prepare_tile(
            artwork_bytes(size=(100, 150)),
            tiles[0][0],
            1.0,
        )

        self.assertEqual(prepared.height, int(150 * poster.MAX_UPSCALE))

    @patch("app.poster._fetch_image")
    def test_poster_is_transparent_and_ragged(self, mock_fetch):
        """The background stays transparent and the outline is not a rectangle."""
        mock_fetch.return_value = artwork_bytes()

        tiles = poster.collect_tiles(
            {MediaTypes.MOVIE.value: Movie.objects.filter(user=self.user)},
            poster.PosterStatistics.SCORE,
        )
        image = poster.build_poster(tiles, seed=5)

        self.assertEqual(image.mode, "RGBA")
        alpha = image.getchannel("A")
        # Differently sized tiles cannot tile a rectangle, so some of the
        # bounding box is left fully transparent.
        self.assertIn(0, alpha.getdata())
        self.assertIn(255, alpha.getdata())

    @patch("app.poster._fetch_image")
    def test_poster_has_no_size_ceiling(self, mock_fetch):
        """Nothing clamps the canvas: it is as large as the artwork needs."""
        mock_fetch.return_value = artwork_bytes(size=(1000, 1500))

        tiles = poster.collect_tiles(
            {MediaTypes.MOVIE.value: Movie.objects.filter(user=self.user)},
            poster.PosterStatistics.EQUAL,
        )
        image = poster.build_poster(tiles, seed=1)

        # Three 600x900 tiles (capped by BASE_TILE_HEIGHT) cannot fit in a
        # canvas smaller than one of them.
        self.assertGreater(max(image.size), poster.BASE_TILE_HEIGHT)

    @patch("app.poster._fetch_image")
    def test_biggest_tiles_land_nearest_the_centre(self, mock_fetch):
        """Packing follows the final tile size, so the big ones are central."""
        mock_fetch.return_value = artwork_bytes()

        for index in range(9):
            item = Item.objects.create(
                media_id=f"spread-{index}",
                source=Sources.MANUAL.value,
                media_type=MediaTypes.ANIME.value,
                title=f"Anime {index}",
                image=f"https://example.com/s{index}.jpg",
            )
            Anime.objects.create(
                item=item,
                user=self.user,
                status=Status.COMPLETED.value,
                # three each of 10, 6 and unscored
                score=[10, 6, None][index % 3],
            )

        tiles = poster.collect_tiles(
            {MediaTypes.ANIME.value: Anime.objects.filter(user=self.user)},
            poster.PosterStatistics.SCORE,
        )
        rng = random.Random(3)  # noqa: S311 - layout variety
        shares = poster.jitter_shares(
            poster.scale_weights([weight for _, weight in tiles]),
            rng,
        )
        sizes = [
            (
                round(poster.BASE_TILE_HEIGHT * share * 2 / 3),
                round(poster.BASE_TILE_HEIGHT * share),
            )
            for share in shares
        ]
        order = sorted(
            range(len(sizes)),
            key=lambda index: sizes[index][0] * sizes[index][1],
            reverse=True,
        )
        positions, canvas = poster.pack([sizes[i] for i in order], 0, 3)

        centre_x, centre_y = canvas[0] / 2, canvas[1] / 2
        radii = {}
        for slot, index in enumerate(order):
            width, height = sizes[index]
            x, y = positions[slot]
            distance = (
                (x + width / 2 - centre_x) ** 2 + (y + height / 2 - centre_y) ** 2
            ) ** 0.5
            radii[tiles[index][0].score] = radii.get(tiles[index][0].score, [])
            radii[tiles[index][0].score].append(distance)

        top = sum(radii[10]) / len(radii[10])
        unscored = sum(radii[None]) / len(radii[None])
        self.assertLess(top, unscored)

    @patch("app.poster._fetch_image")
    def test_the_same_seed_renders_the_same_poster(self, mock_fetch):
        """Preview and download share a seed, so they share a layout."""
        mock_fetch.return_value = artwork_bytes()

        tiles = poster.collect_tiles(
            {MediaTypes.MOVIE.value: Movie.objects.filter(user=self.user)},
            poster.PosterStatistics.SCORE,
        )
        first = poster.build_poster(tiles, seed=99)
        second = poster.build_poster(tiles, seed=99)

        self.assertEqual(first.size, second.size)
        self.assertEqual(list(first.getdata()), list(second.getdata()))

    @patch("app.poster._fetch_image")
    def test_preview_is_smaller_than_the_full_render(self, mock_fetch):
        """The page gets a thumbnail; the download keeps the full pixels."""
        mock_fetch.return_value = artwork_bytes(size=(1200, 1800))

        tiles = poster.collect_tiles(
            {MediaTypes.MOVIE.value: Movie.objects.filter(user=self.user)},
            poster.PosterStatistics.EQUAL,
        )
        full = poster.build_poster(tiles, seed=2)
        small = poster.preview(full, max_edge=200)

        self.assertLessEqual(max(small.size), 200)
        self.assertGreater(max(full.size), max(small.size))
        self.assertEqual(small.mode, "RGBA")

    @patch("app.poster._fetch_image")
    def test_unreachable_artwork_falls_back_to_a_placeholder(self, mock_fetch):
        """A failed download still produces a tile instead of a hole."""
        mock_fetch.return_value = None

        tiles = poster.collect_tiles(
            {MediaTypes.MOVIE.value: Movie.objects.filter(user=self.user)},
            poster.PosterStatistics.SCORE,
        )
        image = poster.build_poster(tiles, seed=3)

        self.assertEqual(image.mode, "RGBA")
        self.assertGreater(max(image.size), 1)

    def test_build_poster_without_tiles(self):
        """An empty selection renders an empty canvas rather than failing."""
        image = poster.build_poster([])

        self.assertEqual(image.mode, "RGBA")
        self.assertEqual(image.size, (1, 1))

    def test_gather_media_keeps_tv_without_episode_rows(self):
        """A show tracked at show level still reaches the poster.

        The statistics selector derives TV from tracked episodes, so a show
        marked complete without per-episode rows is invisible to it.
        """
        item = Item.objects.create(
            media_id="tv-1",
            source=Sources.MANUAL.value,
            media_type=MediaTypes.TV.value,
            title="Show",
            image="https://example.com/tv.jpg",
        )
        TV.objects.create(item=item, user=self.user, status=Status.COMPLETED.value)

        gathered = poster.gather_media(
            self.user,
            [MediaTypes.TV.value],
            list(Status.values),
        )

        self.assertEqual(
            [media.item.title for media in gathered[MediaTypes.TV.value]],
            ["Show"],
        )

    def test_gather_media_dates_tv_through_its_episodes(self):
        """TV has no dates of its own, so a range is applied to its episodes."""
        show_item = Item.objects.create(
            media_id="dated-show",
            source=Sources.MANUAL.value,
            media_type=MediaTypes.TV.value,
            title="Dated Show",
            image="https://example.com/s.jpg",
        )
        TV.objects.create(item=show_item, user=self.user, status=Status.COMPLETED.value)
        season_item = Item.objects.create(
            media_id="dated-show",
            source=Sources.MANUAL.value,
            media_type=MediaTypes.SEASON.value,
            title="Dated Show",
            image="https://example.com/s.jpg",
            season_number=1,
        )
        season = Season.objects.create(
            item=season_item,
            user=self.user,
            status=Status.COMPLETED.value,
        )
        episode_item = Item.objects.create(
            media_id="dated-show",
            source=Sources.MANUAL.value,
            media_type=MediaTypes.EPISODE.value,
            title="Dated Show",
            image="https://example.com/s.jpg",
            season_number=1,
            episode_number=1,
        )
        Episode.objects.create(
            item=episode_item,
            related_season=season,
            end_date=timezone.now(),
        )

        # A show whose episodes fall in the range is dated by them.
        start = timezone.now() - datetime.timedelta(days=7)
        end = timezone.now() + datetime.timedelta(days=7)
        in_range = poster.gather_media(
            self.user,
            [MediaTypes.TV.value],
            list(Status.values),
            start,
            end,
        )
        self.assertEqual(
            [media.item.title for media in in_range[MediaTypes.TV.value]],
            ["Dated Show"],
        )

        # Outside it, it drops out rather than raising on a missing column.
        old_start = timezone.now() - datetime.timedelta(days=700)
        old_end = timezone.now() - datetime.timedelta(days=600)
        out_of_range = poster.gather_media(
            self.user,
            [MediaTypes.TV.value],
            list(Status.values),
            old_start,
            old_end,
        )
        self.assertEqual(list(out_of_range[MediaTypes.TV.value]), [])

    def test_gather_media_selects_requested_types(self):
        """Only the ticked media types are queried, episodes never."""
        gathered = poster.gather_media(
            self.user,
            [MediaTypes.MOVIE.value, MediaTypes.EPISODE.value, "nonsense"],
            list(Status.values),
        )

        self.assertEqual(set(gathered), {MediaTypes.MOVIE.value})

    def test_gather_media_range_excludes_undated(self):
        """A range covers dated entries; undated ones need all time."""
        undated = Item.objects.create(
            media_id="undated",
            source=Sources.MANUAL.value,
            media_type=MediaTypes.MOVIE.value,
            title="Undated",
            image="https://example.com/u.jpg",
        )
        Movie.objects.create(
            item=undated,
            user=self.user,
            status=Status.COMPLETED.value,
            score=5,
        )

        start = timezone.now() - datetime.timedelta(days=7)
        end = timezone.now() + datetime.timedelta(days=7)
        ranged = poster.gather_media(
            self.user,
            [MediaTypes.MOVIE.value],
            list(Status.values),
            start,
            end,
        )
        all_time = poster.gather_media(
            self.user,
            [MediaTypes.MOVIE.value],
            list(Status.values),
        )

        ranged_titles = [media.item.title for media in ranged[MediaTypes.MOVIE.value]]
        all_titles = [media.item.title for media in all_time[MediaTypes.MOVIE.value]]
        self.assertNotIn("Undated", ranged_titles)
        self.assertIn("Undated", all_titles)

    def test_parse_media_types(self):
        """Nothing ticked means the user's enabled types."""
        self.assertEqual(
            poster.parse_media_types([], [MediaTypes.MOVIE.value]),
            [MediaTypes.MOVIE.value],
        )
        self.assertEqual(
            poster.parse_media_types(["nonsense"], [MediaTypes.MOVIE.value]),
            [MediaTypes.MOVIE.value],
        )
        self.assertEqual(
            poster.parse_media_types([MediaTypes.GAME.value], [MediaTypes.MOVIE.value]),
            [MediaTypes.GAME.value],
        )

    def test_parsers_clamp_input(self):
        """Query parameters are clamped instead of trusted."""
        self.assertEqual(poster.parse_gap("999"), poster.MAX_GAP)
        # A limit is opt-in: anything unparseable or negative means "all".
        self.assertEqual(poster.parse_limit("25"), 25)
        self.assertEqual(poster.parse_limit(None), poster.NO_LIMIT)
        self.assertEqual(poster.parse_limit("-5"), poster.NO_LIMIT)
        self.assertEqual(poster.parse_gap(None), 0)
        self.assertEqual(
            poster.parse_statistic("nonsense"),
            poster.PosterStatistics.SCORE.value,
        )
        self.assertEqual(
            poster.parse_gap_style("nonsense"),
            poster.GapStyles.NONE.value,
        )
        # A missing seed is generated, an explicit one is kept.
        self.assertEqual(poster.parse_seed("77"), 77)
        self.assertIsInstance(poster.parse_seed(None), int)


class PosterViewTests(TestCase):
    """Test the poster page and image endpoint."""

    def setUp(self):
        """Create a user with one tracked movie and log in."""
        self.credentials = {"username": "test", "password": "12345"}
        self.user = get_user_model().objects.create_user(**self.credentials)
        self.client.login(**self.credentials)

        item = Item.objects.create(
            media_id="1",
            source=Sources.MANUAL.value,
            media_type=MediaTypes.MOVIE.value,
            title="Movie",
            image="https://example.com/1.jpg",
        )
        Movie.objects.create(
            item=item,
            user=self.user,
            status=Status.COMPLETED.value,
            score=8,
            end_date=timezone.now(),
        )

    def test_poster_page(self):
        """The builder page renders with the form defaults."""
        response = self.client.get(reverse("poster"))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "app/poster.html")
        self.assertEqual(
            response.context["selected_statistic"],
            poster.PosterStatistics.SCORE.value,
        )

    @patch("app.poster._fetch_image")
    def test_poster_image(self, mock_fetch):
        """The image endpoint returns a transparent PNG at full size."""
        mock_fetch.return_value = artwork_bytes()

        response = self.client.get(
            reverse("poster_image"),
            {
                "statistic": "score",
                "seed": "5",
                "start-date": "all",
                "end-date": "all",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "image/png")

        with Image.open(io.BytesIO(response.content)) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(
                response["X-Poster-Size"],
                f"{image.width}x{image.height}",
            )

    @patch("app.poster._fetch_image")
    def test_preview_is_smaller_but_reports_the_full_size(self, mock_fetch):
        """The preview is downscaled; the header still names the real size."""
        mock_fetch.return_value = artwork_bytes(size=(1200, 1800))

        params = {
            "statistic": "equal",
            "seed": "5",
            "start-date": "all",
            "end-date": "all",
        }
        full = self.client.get(reverse("poster_image"), params)
        small = self.client.get(reverse("poster_image"), {**params, "preview": "1"})

        self.assertEqual(full["X-Poster-Size"], small["X-Poster-Size"])

        with (
            Image.open(io.BytesIO(full.content)) as full_image,
            Image.open(io.BytesIO(small.content)) as preview_image,
        ):
            self.assertGreaterEqual(max(full_image.size), max(preview_image.size))
            self.assertLessEqual(max(preview_image.size), poster.PREVIEW_MAX_EDGE)

    @patch("app.poster._fetch_image")
    def test_poster_image_download(self, mock_fetch):
        """The download flag turns the response into an attachment."""
        mock_fetch.return_value = artwork_bytes()

        response = self.client.get(reverse("poster_image"), {"download": "1"})

        self.assertIn("attachment", response["Content-Disposition"])
        self.assertIn("yamtrack-poster-", response["Content-Disposition"])

    @patch("app.poster._fetch_image")
    def test_default_range_is_all_time(self, mock_fetch):
        """A poster nobody narrowed covers undated entries too."""
        mock_fetch.return_value = artwork_bytes()

        undated = Item.objects.create(
            media_id="undated",
            source=Sources.MANUAL.value,
            media_type=MediaTypes.MOVIE.value,
            title="Undated",
            image="https://example.com/undated.jpg",
        )
        Movie.objects.create(
            item=undated,
            user=self.user,
            status=Status.COMPLETED.value,
            score=7,
        )

        response = self.client.get(reverse("poster"))

        self.assertTrue(response.context["all_time"])

        image_response = self.client.get(
            reverse("poster_image"), {"statistic": "score"}
        )
        width, height = (
            int(value) for value in image_response["X-Poster-Size"].split("x")
        )
        # Both the dated and the undated entry are in the render.
        self.assertGreater(width * height, 0)
        self.assertEqual(image_response.status_code, 200)

    @patch("app.poster._fetch_image")
    def test_blank_dates_are_not_a_range(self, mock_fetch):
        """The form leaves its date inputs in place; blank means all time."""
        mock_fetch.return_value = artwork_bytes()

        undated = Item.objects.create(
            media_id="undated-blank",
            source=Sources.MANUAL.value,
            media_type=MediaTypes.MOVIE.value,
            title="Undated",
            image="https://example.com/u.jpg",
        )
        Movie.objects.create(
            item=undated,
            user=self.user,
            status=Status.COMPLETED.value,
            score=6,
        )

        response = self.client.get(
            reverse("poster"),
            {"start-date": "", "end-date": ""},
        )

        self.assertTrue(response.context["all_time"])

    def test_poster_requires_login(self):
        """Posters are private: anonymous visitors get bounced to login."""
        self.client.logout()

        response = self.client.get(reverse("poster"))

        self.assertEqual(response.status_code, 302)
