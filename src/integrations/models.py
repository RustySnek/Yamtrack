from django.conf import settings
from django.db import models


class AniListConnection(models.Model):
    """An AniList account a user's anime and manga changes are pushed to."""

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="anilist_connection",
    )
    username = models.CharField(max_length=255)
    token = models.TextField()  # encrypted with integrations.imports.helpers
    connected_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        """Return the user and the AniList account they are connected to."""
        return f"{self.user} -> {self.username}"


class AniListSyncFailure(models.Model):
    """An anime or manga whose last push to AniList failed, pending a retry."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="anilist_sync_failures",
    )
    media_type = models.CharField(max_length=10)
    media_id = models.CharField(max_length=36)
    error = models.TextField()
    attempts = models.PositiveIntegerField(default=0)
    failed_at = models.DateTimeField(auto_now=True)

    class Meta:
        """Meta options for the model."""

        ordering = ["-failed_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["user", "media_type", "media_id"],
                name="unique_anilist_sync_failure",
            ),
        ]

    def __str__(self):
        """Return the failed item and its last error."""
        return f"{self.media_type} {self.media_id}: {self.error}"
