"""iCloud-SMTP email notification channel.

App-password auth against smtp.mail.me.com:587 (STARTTLS). No OAuth, no
expiring refresh tokens. The "From" address must be a verified alias on
the authenticated Apple ID.
"""

from __future__ import annotations

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, make_msgid

import aiosmtplib
import structlog

from alfred.notifications.models import Notification, NotificationResult

log = structlog.get_logger()

SMTP_HOST = "smtp.mail.me.com"
SMTP_PORT = 587


class IcloudEmailClient:
    """Sends mail via Apple's iCloud SMTP relay.

    `smtp_username` defaults to `from_address`. Apple usually accepts auth
    as the alias address with the parent Apple ID's app-specific password;
    if your account requires the primary Apple ID for auth, set
    `smtp_username` explicitly.
    """

    channel_type = "email"

    def __init__(
        self,
        from_address: str,
        app_password: str,
        from_name: str = "Alfred",
        smtp_username: str | None = None,
    ) -> None:
        if not from_address:
            raise ValueError("from_address is required for IcloudEmailClient")
        if not app_password:
            raise ValueError("app_password is required for IcloudEmailClient")
        self._from_address = from_address
        self._from_name = from_name
        self._username = smtp_username or from_address
        self._password = app_password

    def _build_message(
        self,
        recipient: str,
        subject: str,
        body: str,
        html_body: str | None,
        extra_headers: dict[str, str] | None = None,
        from_name: str | None = None,
    ) -> MIMEText | MIMEMultipart:
        if html_body:
            msg: MIMEText | MIMEMultipart = MIMEMultipart("alternative")
            msg.attach(MIMEText(body, "plain", "utf-8"))
            msg.attach(MIMEText(html_body, "html", "utf-8"))
        else:
            msg = MIMEText(body, "plain", "utf-8")
        msg["From"] = formataddr((from_name or self._from_name, self._from_address))
        msg["To"] = recipient
        msg["Subject"] = subject
        msg["Message-Id"] = make_msgid(domain="icloud.com")
        if extra_headers:
            for k, v in extra_headers.items():
                if v:
                    msg[k] = v
        return msg

    async def _deliver(
        self, msg: MIMEText | MIMEMultipart, recipient: str
    ) -> NotificationResult:
        try:
            await aiosmtplib.send(
                msg,
                hostname=SMTP_HOST,
                port=SMTP_PORT,
                start_tls=True,
                username=self._username,
                password=self._password,
            )
            msg_id = msg.get("Message-Id", "")
            log.info(
                "email_sent",
                recipient=recipient,
                subject=msg["Subject"],
                channel="icloud",
                message_id=msg_id,
            )
            return NotificationResult(
                success=True, channel="email", message_id=msg_id
            )
        except aiosmtplib.SMTPException as e:
            log.error("email_smtp_error", error=str(e), recipient=recipient)
            return NotificationResult(success=False, channel="email", error=str(e))
        except Exception as e:
            log.error("email_send_failed", error=str(e), recipient=recipient)
            return NotificationResult(success=False, channel="email", error=str(e))

    async def send(self, notification: Notification) -> NotificationResult:
        subject = notification.subject or "(no subject)"
        msg = self._build_message(
            recipient=notification.recipient,
            subject=subject,
            body=notification.body,
            html_body=notification.html_body,
            from_name=notification.metadata.get("from_name") if notification.metadata else None,
        )
        return await self._deliver(msg, notification.recipient)

    async def send_reply(
        self,
        recipient: str,
        subject: str,
        body: str,
        thread_id: str = "",  # unused for SMTP — kept for interface parity
        in_reply_to: str = "",
        references: str = "",
        html_body: str | None = None,
        from_name: str | None = None,
    ) -> NotificationResult:
        """Send a threaded reply. SMTP threads via In-Reply-To / References headers."""
        extra: dict[str, str] = {}
        if in_reply_to:
            extra["In-Reply-To"] = in_reply_to
            extra["References"] = (references + " " + in_reply_to).strip()
        msg = self._build_message(
            recipient=recipient,
            subject=subject,
            body=body,
            html_body=html_body,
            extra_headers=extra,
            from_name=from_name,
        )
        return await self._deliver(msg, recipient)

    async def health_check(self) -> bool:
        try:
            client = aiosmtplib.SMTP(hostname=SMTP_HOST, port=SMTP_PORT, start_tls=True)
            await client.connect()
            await client.login(self._username, self._password)
            await client.quit()
            return True
        except Exception as e:
            log.warning("email_health_check_failed", error=str(e))
            return False

    async def close(self) -> None:
        pass
