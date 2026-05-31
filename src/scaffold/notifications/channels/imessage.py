"""BlueBubbles iMessage client."""

from __future__ import annotations

import httpx
import structlog

from scaffold.notifications.models import Notification, NotificationResult

log = structlog.get_logger()


class BlueBubblesClient:
    """Sends iMessages via the BlueBubbles REST API on the Mac Mini."""

    channel_type = "imessage"

    def __init__(self, base_url: str, password: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._password = password
        self._client = httpx.AsyncClient(timeout=30)

    async def send(self, notification: Notification) -> NotificationResult:
        """Send an iMessage via BlueBubbles."""
        try:
            resp = await self._client.post(
                f"{self.base_url}/api/v1/message/text",
                json={
                    "chatGuid": f"iMessage;-;{notification.recipient}",
                    "tempGuid": "",
                    "message": self._format_message(notification),
                    "method": "private-api",
                },
                params={"password": self._password},
            )
            resp.raise_for_status()

            return NotificationResult(success=True, channel="imessage")
        except Exception as e:
            log.error("imessage_send_failed", error=str(e), recipient=notification.recipient)
            return NotificationResult(
                success=False, channel="imessage", error=str(e)
            )

    async def health_check(self) -> bool:
        """Check if BlueBubbles server is reachable."""
        try:
            resp = await self._client.get(
                f"{self.base_url}/api/v1/server/info",
                params={"password": self._password},
            )
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def _format_message(self, notification: Notification) -> str:
        """Format notification into an iMessage-friendly text."""
        if notification.subject:
            return f"{notification.subject}\n\n{notification.body}"
        return notification.body

    async def close(self) -> None:
        await self._client.aclose()
