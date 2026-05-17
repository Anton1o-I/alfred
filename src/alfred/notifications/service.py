"""Notification service — routes to the appropriate channel and logs delivery."""

from __future__ import annotations

from typing import Any

import structlog

from alfred.audit.logger import AuditLogger
from alfred.core.config import NotificationConfig, RecipientConfig
from alfred.core.constants import AuditEventType
from alfred.notifications.models import Notification, NotificationResult
from alfred.storage.database import Database

log = structlog.get_logger()


class NotificationService:
    """Routes notifications to the appropriate channel and logs delivery."""

    def __init__(
        self,
        channels: dict[str, Any],  # channel_type -> client
        config: NotificationConfig,
        audit_logger: AuditLogger,
        db: Database,
    ) -> None:
        self._channels = channels
        self.channels = channels  # Public read-only view for callers that need a specific channel
        self._config = config
        self._audit = audit_logger
        self._db = db
        self._recipients = {r.user_id: r for r in config.recipients}

    async def send(
        self, notification: Notification, request_id: str = ""
    ) -> NotificationResult:
        """Route to channel, send, log to notification_log table, audit."""
        channel = self._channels.get(notification.channel)
        if channel is None:
            return NotificationResult(
                success=False,
                channel=notification.channel,
                error=f"Channel '{notification.channel}' not configured",
            )

        result = await channel.send(notification)

        # Log to notification_log table
        await self._db.execute(
            "INSERT INTO notification_log "
            "(request_id, channel, recipient, subject, body, status, error, sent_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (
                request_id,
                notification.channel,
                notification.recipient,
                notification.subject,
                notification.body,
                "sent" if result.success else "failed",
                result.error,
            ),
        )

        await self._audit.log_event(
            event_type=AuditEventType.NOTIFICATION,
            request_id=request_id,
            details={
                "channel": notification.channel,
                "recipient": notification.recipient,
                "success": result.success,
                "error": result.error,
            },
        )

        return result

    async def send_to_family(
        self,
        body: str,
        subject: str | None = None,
        html_body: str | None = None,
        request_id: str = "",
        from_name: str | None = None,
    ) -> list[NotificationResult]:
        """Send a notification to all family members via their preferred channel."""
        metadata = {"from_name": from_name} if from_name else {}
        results = []
        for recipient in self._config.recipients:
            notification = Notification(
                recipient=recipient.phone or recipient.email or recipient.user_id,
                subject=subject,
                body=body,
                html_body=html_body,
                channel=recipient.preferred_channel,
                metadata=metadata,
            )
            result = await self.send(notification, request_id)
            results.append(result)
        return results

    async def send_to_user(
        self,
        user_id: str,
        body: str,
        subject: str | None = None,
        html_body: str | None = None,
        request_id: str = "",
    ) -> NotificationResult:
        """Send a notification to a specific user via their preferred channel."""
        recipient = self._recipients.get(user_id)
        if recipient is None:
            return NotificationResult(
                success=False,
                channel="unknown",
                error=f"Unknown user: {user_id}",
            )

        notification = Notification(
            recipient=recipient.phone or recipient.email or user_id,
            subject=subject,
            body=body,
            html_body=html_body,
            channel=recipient.preferred_channel,
        )
        return await self.send(notification, request_id)

    def get_recipient(self, user_id: str) -> RecipientConfig | None:
        """Look up a recipient by user_id."""
        return self._recipients.get(user_id)
