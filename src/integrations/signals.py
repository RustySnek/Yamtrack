from django.db import transaction
from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from app.models import Anime, Item, Manga
from integrations import anilist_sync, tasks


@receiver(post_save, sender=Anime)
@receiver(post_save, sender=Manga)
@receiver(post_delete, sender=Anime)
@receiver(post_delete, sender=Manga)
def queue_anilist_sync(sender, instance, **kwargs):  # noqa: ARG001
    """Push the item's new state to AniList once the change is committed."""
    # Imports run under disable_fetch_releases(); their changes come from
    # elsewhere (often AniList itself) and are not pushed back.
    if Item._disable_calendar_triggers or not anilist_sync.is_enabled_for(instance):
        return

    user_id = instance.user_id
    media_type = instance.item.media_type
    media_id = instance.item.media_id
    transaction.on_commit(
        lambda: tasks.sync_anilist.delay(user_id, media_type, media_id),
    )
