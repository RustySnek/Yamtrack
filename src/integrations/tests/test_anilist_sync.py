from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import requests
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from app.mixins import disable_fetch_releases
from app.models import Anime, Item, Manga, MediaTypes, Sources, Status, UserMessage
from integrations import anilist_sync
from integrations.imports import helpers
from integrations.models import AniListConnection, AniListSyncFailure
from integrations.tasks import retry_anilist_syncs, sync_anilist


def http_error(status_code, message):
    """Build the HTTPError api_request raises for a failed AniList response."""
    response = MagicMock(status_code=status_code)
    response.json.return_value = {"errors": [{"message": message}]}
    return requests.exceptions.HTTPError(response=response)


def lookup(anilist_id=500, list_entry_id=None):
    """Return an AniList Media lookup response."""
    entry = {"id": list_entry_id} if list_entry_id else None
    return {"data": {"Media": {"id": anilist_id, "mediaListEntry": entry}}}


class AniListSyncTests(TestCase):
    """Test pushing anime and manga state to AniList."""

    def setUp(self):
        """Create a connected user and a MAL anime item."""
        self.user = get_user_model().objects.create_user(username="test")
        anilist_sync.connect(self.user, "token", "aniuser")
        self.item = Item.objects.create(
            media_id="10271",
            source=Sources.MAL.value,
            media_type=MediaTypes.ANIME.value,
            title="Kaiji",
            image="http://example.com/image.jpg",
        )

    def add_entries(self, *entries):
        """Create anime entries without running save() side effects or signals."""
        Anime.objects.bulk_create(
            Anime(item=self.item, user=self.user, **entry) for entry in entries
        )

    def sync(self):
        """Run the sync task for the test item."""
        return sync_anilist(self.user.id, MediaTypes.ANIME.value, "10271")

    def test_connect_stores_encrypted_token(self):
        """The token is only kept encrypted, and reconnecting replaces it."""
        anilist_sync.connect(self.user, "new-token", "renamed")

        connection = AniListConnection.objects.get()
        self.assertEqual(connection.username, "renamed")
        self.assertNotIn("new-token", connection.token)
        self.assertEqual(helpers.decrypt(connection.token), "new-token")

    @patch("app.providers.services.api_request")
    def test_save_maps_latest_entry(self, api_request):
        """Push the latest entry's state, with score on AniList's 100 scale."""
        self.add_entries(
            {
                "status": Status.COMPLETED.value,
                "score": Decimal("8.5"),
                "progress": 26,
                # 22:00 UTC is the next day in Warsaw.
                "start_date": datetime(2016, 8, 17, 22, 0, tzinfo=UTC),
                "end_date": None,
                "notes": "good",
            },
        )
        api_request.side_effect = [lookup(), {"data": {"SaveMediaListEntry": {}}}]

        with self.settings(TIME_ZONE="Europe/Warsaw"):
            self.sync()

        variables = api_request.call_args_list[1].kwargs["params"]["variables"]
        self.assertEqual(
            variables,
            {
                "mediaId": 500,
                "status": "COMPLETED",
                "scoreRaw": 85,
                "progress": 26,
                "repeat": 0,
                "notes": "good",
                "startedAt": {"year": 2016, "month": 8, "day": 18},
                "completedAt": {"year": None, "month": None, "day": None},
            },
        )
        self.assertEqual(
            api_request.call_args.kwargs["headers"]["Authorization"],
            "Bearer token",
        )

    @patch("app.providers.services.api_request")
    def test_in_progress_after_earlier_entries_is_repeating(self, api_request):
        """Earlier entries count as repeats, an in-progress latest one as a rewatch."""
        self.add_entries(
            {"status": Status.COMPLETED.value, "progress": 26},
            {"status": Status.IN_PROGRESS.value, "progress": 3, "score": None},
        )
        api_request.side_effect = [lookup(), {"data": {"SaveMediaListEntry": {}}}]

        self.sync()

        variables = api_request.call_args.kwargs["params"]["variables"]
        self.assertEqual(variables["status"], "REPEATING")
        self.assertEqual(variables["repeat"], 1)
        self.assertEqual(variables["progress"], 3)
        self.assertEqual(variables["scoreRaw"], 0)

    @patch("app.providers.services.api_request")
    def test_no_entries_deletes_from_anilist(self, api_request):
        """Delete the AniList list entry once no Yamtrack entry is left."""
        api_request.side_effect = [
            lookup(list_entry_id=77),
            {"data": {"DeleteMediaListEntry": {"deleted": True}}},
        ]

        self.sync()

        self.assertEqual(
            api_request.call_args.kwargs["params"]["variables"],
            {"id": 77},
        )

    @patch("app.providers.services.api_request")
    def test_no_entries_and_not_on_anilist_does_nothing(self, api_request):
        """Skip the delete when AniList has no list entry either."""
        api_request.return_value = lookup()

        self.sync()

        self.assertEqual(api_request.call_count, 1)

    @patch("app.providers.services.api_request")
    def test_disconnected_user_does_not_touch_anilist(self, api_request):
        """Without a connection, including after the user is deleted, do nothing."""
        AniListConnection.objects.all().delete()

        self.sync()

        api_request.assert_not_called()

    @patch("app.providers.services.api_request")
    def test_unknown_mal_id_warns_without_failing(self, api_request):
        """A title AniList lacks is reported once, not retried."""
        self.add_entries({"status": Status.PLANNING.value})
        api_request.side_effect = http_error(404, "Not Found.")

        self.sync()
        self.sync()

        self.assertFalse(AniListSyncFailure.objects.exists())
        self.assertEqual(
            list(UserMessage.objects.values_list("message", flat=True)),
            ["Kaiji has no AniList entry, it was not synced."],
        )

    @patch("app.providers.services.api_request")
    def test_failure_is_recorded_then_cleared_by_retry(self, api_request):
        """A failed push is stored, and the hourly retry clears it on success."""
        self.add_entries({"status": Status.PLANNING.value})
        api_request.side_effect = requests.exceptions.ConnectionError("down")

        with self.assertRaises(requests.exceptions.ConnectionError):
            self.sync()
        retry_anilist_syncs()

        failure = AniListSyncFailure.objects.get()
        self.assertEqual(failure.user, self.user)
        self.assertEqual(failure.attempts, 2)
        self.assertEqual(failure.error, "ConnectionError: down")

        api_request.side_effect = [lookup(), {"data": {"SaveMediaListEntry": {}}}]
        result = retry_anilist_syncs()

        self.assertFalse(AniListSyncFailure.objects.exists())
        self.assertEqual(result, "Retried 1 failed AniList syncs, 0 still failing.")

    @patch("integrations.anilist_sync.push")
    def test_retry_for_one_user(self, push):
        """A user's retry only touches their own failures."""
        other = get_user_model().objects.create_user(username="other")
        AniListSyncFailure.objects.create(
            user=self.user, media_type="anime", media_id="1"
        )
        AniListSyncFailure.objects.create(user=other, media_type="anime", media_id="2")

        retry_anilist_syncs(user_id=self.user.id)

        push.assert_called_once_with(self.user.id, "anime", "1")

    @patch("app.providers.services.api_request")
    def test_invalid_token_notifies_once(self, api_request):
        """Repeated token failures leave a single unseen error message."""
        self.add_entries({"status": Status.PLANNING.value})
        api_request.side_effect = http_error(400, "Invalid token")

        with self.assertRaises(requests.exceptions.HTTPError):
            self.sync()
        retry_anilist_syncs()

        self.assertEqual(
            list(UserMessage.objects.values_list("message", flat=True)),
            [
                "AniList rejected the connection to aniuser, anime and manga changes "
                "are not being synced. Reconnect AniList in the integrations settings.",
            ],
        )
        self.assertEqual(AniListSyncFailure.objects.get().attempts, 2)


@patch("integrations.tasks.sync_anilist.delay")
@patch("app.models.Item.fetch_releases")
class AniListSyncSignalTests(TestCase):
    """Test which anime and manga changes queue an AniList sync."""

    def setUp(self):
        """Create a connected user and a MAL manga item."""
        self.user = get_user_model().objects.create_user(username="test")
        anilist_sync.connect(self.user, "token", "aniuser")
        self.item = Item.objects.create(
            media_id="2",
            source=Sources.MAL.value,
            media_type=MediaTypes.MANGA.value,
            title="Berserk",
            image="http://example.com/image.jpg",
        )

    def create_manga(self, user=None, item=None):
        """Create a manga entry, running any on-commit callbacks."""
        with self.captureOnCommitCallbacks(execute=True):
            return Manga.objects.create(
                item=item or self.item,
                user=user or self.user,
                status=Status.PLANNING.value,
            )

    def test_save_and_delete_queue_sync(self, _fetch_releases, delay):
        """Creating, editing and deleting an entry each queue a push."""
        manga = self.create_manga()
        with self.captureOnCommitCallbacks(execute=True):
            manga.status = Status.DROPPED.value
            manga.save()
        with self.captureOnCommitCallbacks(execute=True):
            manga.delete()

        self.assertEqual(delay.call_count, 3)
        delay.assert_called_with(self.user.id, MediaTypes.MANGA.value, "2")

    def test_unconnected_user_is_not_synced(self, _fetch_releases, delay):
        """Users without an AniList connection are left alone."""
        other = get_user_model().objects.create_user(username="other")

        self.create_manga(user=other)

        delay.assert_not_called()

    def test_non_mal_item_is_not_synced(self, _fetch_releases, delay):
        """Items AniList cannot look up by MAL ID are skipped."""
        item = Item.objects.create(
            media_id="123",
            source=Sources.MANGAUPDATES.value,
            media_type=MediaTypes.MANGA.value,
            title="Other",
            image="http://example.com/image.jpg",
        )

        self.create_manga(item=item)

        delay.assert_not_called()

    def test_imports_are_not_synced(self, _fetch_releases, delay):
        """Changes made while importing are not pushed back."""
        with disable_fetch_releases():
            self.create_manga()

        delay.assert_not_called()


@override_settings(
    URLS=["https://yamtrack.example.com"],
    ANILIST_ID="client",
    ANILIST_SECRET="secret",  # noqa: S106
)
class AniListSyncViewTests(TestCase):
    """Test connecting and managing AniList sync from the settings page."""

    def setUp(self):
        """Create and log in a user."""
        credentials = {"username": "testuser", "password": "testpass123"}
        self.user = get_user_model().objects.create_user(**credentials)
        self.client.login(**credentials)

    def start_connect(self):
        """Begin the OAuth flow and return the state token it stored."""
        response = self.client.post(reverse("anilist_sync_connect"))
        return parse_qs(urlparse(response["Location"]).query)["state"][0]

    def test_connect_redirects_to_anilist(self):
        """Authorization goes through the import's registered callback URL."""
        response = self.client.post(reverse("anilist_sync_connect"))

        redirect = urlparse(response["Location"])
        query = parse_qs(redirect.query)
        self.assertEqual(redirect.netloc, "anilist.co")
        self.assertEqual(query["client_id"], ["client"])
        self.assertEqual(
            query["redirect_uri"],
            ["https://yamtrack.example.com/import/anilist/private"],
        )
        self.assertEqual(self.client.session[query["state"][0]], {"anilist_sync": True})

    @override_settings(ANILIST_ID="")
    def test_connect_without_client_configured(self):
        """Explain the missing configuration instead of redirecting to AniList."""
        response = self.client.post(reverse("anilist_sync_connect"), follow=True)

        self.assertRedirects(response, reverse("integrations"))
        self.assertContains(response, "ANILIST_ID and ANILIST_SECRET")

    @patch("integrations.views.tasks.retry_anilist_syncs.delay")
    @patch("integrations.views.tasks.import_anilist.delay")
    @patch("integrations.views.anilist.get_token")
    def test_callback_stores_connection(self, get_token, import_anilist, retry):
        """The shared callback connects sync instead of importing."""
        get_token.return_value = {"access_token": "token", "username": "aniuser"}
        state = self.start_connect()

        response = self.client.get(
            reverse("import_anilist_private"),
            {"code": "abc", "state": state},
        )

        self.assertRedirects(response, reverse("integrations"))
        connection = AniListConnection.objects.get(user=self.user)
        self.assertEqual(connection.username, "aniuser")
        self.assertEqual(helpers.decrypt(connection.token), "token")
        import_anilist.assert_not_called()
        retry.assert_called_once_with(user_id=self.user.id)
        self.assertNotIn(state, self.client.session)

    @patch("integrations.views.anilist.get_token")
    def test_callback_after_denied_authorization(self, get_token):
        """Denying access on AniList leaves the user disconnected."""
        state = self.start_connect()

        response = self.client.get(
            reverse("import_anilist_private"),
            {"error": "access_denied", "state": state},
            follow=True,
        )

        self.assertContains(response, "AniList authorization was cancelled.")
        get_token.assert_not_called()
        self.assertFalse(AniListConnection.objects.exists())

    def test_disconnect_forgets_token_and_failures(self):
        """Disconnecting removes the token and pending retries."""
        anilist_sync.connect(self.user, "token", "aniuser")
        AniListSyncFailure.objects.create(
            user=self.user, media_type="anime", media_id="1"
        )

        self.client.post(reverse("anilist_sync_disconnect"))

        self.assertFalse(AniListConnection.objects.exists())
        self.assertFalse(AniListSyncFailure.objects.exists())

    @patch("integrations.views.tasks.retry_anilist_syncs.delay")
    def test_retry_now(self, retry):
        """The retry button queues the user's failures immediately."""
        self.client.post(reverse("anilist_sync_retry"))

        retry.assert_called_once_with(user_id=self.user.id)

    def test_integrations_page_shows_state(self):
        """The settings page shows connect, then connection and failures."""
        response = self.client.get(reverse("integrations"))
        self.assertContains(response, "Connect AniList")

        anilist_sync.connect(self.user, "token", "aniuser")
        AniListSyncFailure.objects.create(
            user=self.user,
            media_type="anime",
            media_id="1",
            error="ConnectionError: down",
        )
        response = self.client.get(reverse("integrations"))

        self.assertContains(response, "aniuser")
        self.assertContains(response, "1 change failed to sync")
        self.assertContains(response, "ConnectionError: down")
        self.assertNotContains(response, "Connect AniList")
