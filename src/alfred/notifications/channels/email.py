"""Gmail-API-backed email notification channel."""

from __future__ import annotations

import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path
from typing import Any

import structlog
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from alfred.integrations.google_auth import (
    GMAIL_SEND_SCOPE,
    load_credentials,
)
from alfred.notifications.models import Notification, NotificationResult

log = structlog.get_logger()


class EmailClient:
    """Sends mail via the Gmail API using the shared Google OAuth token.

    The From header is whatever Google account authorized the OAuth flow —
    we send as `userId="me"`. No SMTP, no app passwords.
    """

    channel_type = "email"

    def __init__(
        self,
        credentials_path: Path,
        token_path: Path,
        from_name: str = "Alfred",
        from_address: str = "",
    ) -> None:
        self._credentials_path = credentials_path
        self._token_path = token_path
        self._from_name = from_name
        self._from_address = from_address
        self._service = None  # lazy-built; refresh handled by load_credentials

    def _get_service(self):  # type: ignore[no-untyped-def]
        if self._service is not None:
            return self._service
        creds = load_credentials(
            token_path=self._token_path,
            credentials_path=self._credentials_path,
            scopes=[GMAIL_SEND_SCOPE],
        )
        if creds is None:
            raise RuntimeError(
                "No usable Google token. Run `alfred google-auth` to grant Gmail scope."
            )
        self._service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        return self._service

    def _get_from_header(self) -> str:
        """Return the formatted From header: `Display Name <user@gmail.com>`."""
        if not self._from_address:
            # Address must match the OAuth-authorized account; gmail.send can't
            # introspect the address, so it has to come from config.
            return self._from_name
        return formataddr((self._from_name, self._from_address))

    async def send(self, notification: Notification) -> NotificationResult:
        subject = notification.subject or "(no subject)"
        try:
            service = self._get_service()
            if notification.html_body:
                msg: MIMEText | MIMEMultipart = MIMEMultipart("alternative")
                msg.attach(MIMEText(notification.body, "plain", "utf-8"))
                msg.attach(MIMEText(notification.html_body, "html", "utf-8"))
            else:
                msg = MIMEText(notification.body, "plain", "utf-8")
            msg["from"] = self._get_from_header()
            msg["to"] = notification.recipient
            msg["subject"] = subject
            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
            sent = (
                service.users()
                .messages()
                .send(userId="me", body={"raw": raw})
                .execute()
            )
            log.info(
                "email_sent",
                recipient=notification.recipient,
                subject=subject,
                message_id=sent.get("id"),
            )
            return NotificationResult(success=True, channel="email")
        except HttpError as e:
            log.error("email_http_error", error=str(e), recipient=notification.recipient)
            return NotificationResult(success=False, channel="email", error=str(e))
        except Exception as e:
            log.error("email_send_failed", error=str(e), recipient=notification.recipient)
            return NotificationResult(success=False, channel="email", error=str(e))

    async def send_reply(
        self,
        recipient: str,
        subject: str,
        body: str,
        thread_id: str = "",
        in_reply_to: str = "",
        references: str = "",
        html_body: str | None = None,
    ) -> NotificationResult:
        """Send a reply that threads with the original message in Gmail."""
        try:
            service = self._get_service()
            if html_body:
                msg: MIMEText | MIMEMultipart = MIMEMultipart("alternative")
                msg.attach(MIMEText(body, "plain", "utf-8"))
                msg.attach(MIMEText(html_body, "html", "utf-8"))
            else:
                msg = MIMEText(body, "plain", "utf-8")
            msg["from"] = self._get_from_header()
            msg["to"] = recipient
            msg["subject"] = subject
            if in_reply_to:
                msg["In-Reply-To"] = in_reply_to
                # Gmail wants References to be a chain; appending the parent is sufficient.
                msg["References"] = (references + " " + in_reply_to).strip()

            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
            payload: dict[str, Any] = {"raw": raw}
            if thread_id:
                payload["threadId"] = thread_id
            sent = (
                service.users()
                .messages()
                .send(userId="me", body=payload)
                .execute()
            )
            log.info(
                "email_reply_sent",
                recipient=recipient,
                subject=subject,
                message_id=sent.get("id"),
                thread_id=thread_id,
            )
            return NotificationResult(success=True, channel="email")
        except HttpError as e:
            log.error("email_reply_http_error", error=str(e), recipient=recipient)
            return NotificationResult(success=False, channel="email", error=str(e))
        except Exception as e:
            log.error("email_reply_failed", error=str(e), recipient=recipient)
            return NotificationResult(success=False, channel="email", error=str(e))

    async def health_check(self) -> bool:
        try:
            self._get_service()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        pass
