"""Email notification channel (stub for future implementation)."""

from __future__ import annotations

import structlog

from alfred.notifications.models import Notification, NotificationResult

log = structlog.get_logger()


class EmailClient:
    """Sends email via SendGrid. Stub — will be implemented in Phase 4."""

    channel_type = "email"

    def __init__(self, api_key: str, from_email: str = "alfred@example.com") -> None:
        self._api_key = api_key
        self._from_email = from_email

    async def send(self, notification: Notification) -> NotificationResult:
        """Send an email notification."""
        log.warning("email_not_implemented", recipient=notification.recipient)
        return NotificationResult(
            success=False,
            channel="email",
            error="Email channel not yet implemented",
        )

    async def health_check(self) -> bool:
        return False

    async def close(self) -> None:
        pass
