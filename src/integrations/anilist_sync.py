"""Push anime and manga changes from Yamtrack to AniList.

Users connect their AniList account from the integrations settings page, which
stores an AniListConnection; their changes are then mirrored to that account.
Only MyAnimeList-sourced items can be synced, since AniList is queried by MAL
ID. A push that fails is recorded as an AniListSyncFailure and retried hourly.
"""

import logging

import requests
from cryptography.fernet import InvalidToken
from django.apps import apps
from django.utils import timezone

import app
from app.models import Sources, Status, UserMessage, UserMessageLevel
from integrations.imports import helpers
from integrations.models import AniListConnection, AniListSyncFailure

logger = logging.getLogger(__name__)

URL = "https://graphql.anilist.co"

LOOKUP_QUERY = """
query ($idMal: Int, $type: MediaType) {
    Media(idMal: $idMal, type: $type) {
        id
        mediaListEntry {
            id
        }
    }
}
"""

SAVE_MUTATION = """
mutation (
    $mediaId: Int,
    $status: MediaListStatus,
    $scoreRaw: Int,
    $progress: Int,
    $repeat: Int,
    $notes: String,
    $startedAt: FuzzyDateInput,
    $completedAt: FuzzyDateInput
) {
    SaveMediaListEntry(
        mediaId: $mediaId,
        status: $status,
        scoreRaw: $scoreRaw,
        progress: $progress,
        repeat: $repeat,
        notes: $notes,
        startedAt: $startedAt,
        completedAt: $completedAt
    ) {
        id
    }
}
"""

DELETE_MUTATION = """
mutation ($id: Int) {
    DeleteMediaListEntry(id: $id) {
        deleted
    }
}
"""

STATUS = {
    Status.COMPLETED.value: "COMPLETED",
    Status.IN_PROGRESS.value: "CURRENT",
    Status.PLANNING.value: "PLANNING",
    Status.PAUSED.value: "PAUSED",
    Status.DROPPED.value: "DROPPED",
}


def is_enabled_for(media):
    """Return whether a change to this media entry should be pushed."""
    return (
        media.item.source == Sources.MAL.value
        and AniListConnection.objects.filter(user_id=media.user_id).exists()
    )


def connect(user, access_token, username):
    """Store the user's AniList token, replacing any previous connection."""
    AniListConnection.objects.update_or_create(
        user=user,
        defaults={"token": helpers.encrypt(access_token), "username": username},
    )


def push(user_id, media_type, media_id):
    """Sync one item, recording a failure for the hourly retry or clearing it."""
    try:
        result = sync(user_id, media_type, media_id)
    except Exception as error:
        failure, _ = AniListSyncFailure.objects.get_or_create(
            user_id=user_id,
            media_type=media_type,
            media_id=media_id,
        )
        failure.attempts += 1
        failure.error = f"{type(error).__name__}: {error}"
        failure.save()
        raise

    AniListSyncFailure.objects.filter(
        user_id=user_id,
        media_type=media_type,
        media_id=media_id,
    ).delete()
    return result


def retry_failed(user_id=None):
    """Push every item whose last sync failed again, optionally for one user."""
    failures = AniListSyncFailure.objects.order_by("failed_at")
    if user_id is not None:
        failures = failures.filter(user_id=user_id)
    failures = list(failures.values_list("user_id", "media_type", "media_id"))

    still_failing = 0
    for failed_user_id, media_type, media_id in failures:
        try:
            push(failed_user_id, media_type, media_id)
        except Exception:
            logger.exception(
                "Retrying AniList sync of %s %s for user %s failed",
                media_type,
                media_id,
                failed_user_id,
            )
            still_failing += 1

    return (
        f"Retried {len(failures)} failed AniList syncs, {still_failing} still failing."
    )


def sync(user_id, media_type, media_id):
    """Make the AniList entry match the user's current entries for a MAL item."""
    connection = (
        AniListConnection.objects.select_related("user").filter(user_id=user_id).first()
    )
    if connection is None:
        # Disconnected, or the user was deleted: a cascade of deletions must
        # not wipe the AniList list.
        return "User is not connected to AniList."

    model = apps.get_model(app_label="app", model_name=media_type)
    entries = list(
        model.objects.filter(
            user_id=user_id,
            item__media_id=media_id,
            item__source=Sources.MAL.value,
        ).order_by("created_at", "id"),
    )

    try:
        media = _request(
            connection,
            LOOKUP_QUERY,
            {"idMal": int(media_id), "type": media_type.upper()},
        )["Media"]
    except requests.exceptions.HTTPError as error:
        if error.response.status_code != requests.codes.not_found:
            raise
        title = entries[-1].item.title if entries else f"MAL ID {media_id}"
        _notify(
            connection.user,
            UserMessageLevel.WARNING,
            f"{title} has no AniList entry, it was not synced.",
        )
        return f"No AniList {media_type} for MAL ID {media_id}."

    if not entries:
        if media["mediaListEntry"] is None:
            return "Not on AniList list, nothing to delete."
        _request(connection, DELETE_MUTATION, {"id": media["mediaListEntry"]["id"]})
        return f"Deleted AniList entry for media {media['id']}."

    _request(connection, SAVE_MUTATION, _entry_variables(media["id"], entries))
    return f"Saved AniList entry for media {media['id']}."


def _entry_variables(anilist_id, entries):
    """Build SaveMediaListEntry variables from a user's entries for one item.

    Mirrors the AniList importer: earlier entries are past repeats and the
    latest one holds the current state, so an in-progress latest entry after
    earlier ones is a rewatch/reread.
    """
    current = entries[-1]
    repeat = len(entries) - 1

    status = STATUS[current.status]
    if status == "CURRENT" and repeat:
        status = "REPEATING"

    return {
        "mediaId": anilist_id,
        "status": status,
        "scoreRaw": int(current.score * 10) if current.score is not None else 0,
        "progress": current.progress,
        "repeat": repeat,
        "notes": current.notes,
        "startedAt": _fuzzy_date(current.start_date),
        "completedAt": _fuzzy_date(current.end_date),
    }


def _fuzzy_date(value):
    """Return an AniList FuzzyDateInput; all nulls clear the date."""
    if value is None:
        return {"year": None, "month": None, "day": None}
    local = timezone.localtime(value)
    return {"year": local.year, "month": local.month, "day": local.day}


def _request(connection, query, variables):
    """Send a GraphQL request as the connected AniList user and return its data."""
    rejected_message = (
        f"AniList rejected the connection to {connection.username}, anime and manga "
        "changes are not being synced. Reconnect AniList in the integrations settings."
    )
    try:
        # Undecryptable after SECRET changes; the token is useless then too.
        token = helpers.decrypt(connection.token)
    except InvalidToken:
        _notify(connection.user, UserMessageLevel.ERROR, rejected_message)
        raise

    try:
        response = app.providers.services.api_request(
            "ANILIST",
            "POST",
            URL,
            params={"query": query, "variables": variables},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
    except requests.exceptions.HTTPError as error:
        # A bad or expired token is a 400 "Invalid token", a missing one a 401.
        if _error_messages(error.response) & {"Invalid token", "Unauthorized."}:
            _notify(connection.user, UserMessageLevel.ERROR, rejected_message)
        raise
    return response["data"]


def _error_messages(response):
    """Return the GraphQL error messages of a failed AniList response."""
    try:
        return {error.get("message") for error in response.json().get("errors", [])}
    except ValueError:
        return set()


def _notify(user, level, message):
    """Show the user a message, unless the same one is still waiting unseen.

    Hourly retries hit the same error again; one toast per error is enough.
    """
    if not UserMessage.objects.filter(
        user=user,
        message=message,
        shown_at__isnull=True,
    ).exists():
        UserMessage.objects.create(user=user, level=level, message=message)
