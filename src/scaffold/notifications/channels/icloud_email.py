"""iCloud-SMTP email notification channel.

App-password auth against smtp.mail.me.com:587 (STARTTLS). No OAuth, no
expiring refresh tokens. The "From" address must be a verified alias on
the authenticated Apple ID.

After each successful SMTP send we also APPEND a copy of the message to
the iCloud "Sent Messages" mailbox over IMAPS — SMTP only transmits to
the recipient and does not populate the sender's Sent folder. The IMAP
APPEND is best-effort: a failure is logged but does not flip the send
result, since the recipient has already been delivered to.
"""

from __future__ import annotations

import asyncio
import imaplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, make_msgid

import aiosmtplib
import structlog

from scaffold.notifications.models import Notification, NotificationResult

log = structlog.get_logger()

SMTP_HOST = "smtp.mail.me.com"
SMTP_PORT = 587
IMAP_HOST = "imap.mail.me.com"
IMAP_PORT = 993
# iCloud's Sent mailbox is named "Sent Messages" — the same string Mail.app
# and the iCloud web UI use. Don't change without checking your account.
SENT_MAILBOX = "Sent Messages"


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

    def _imap_append_sent(self, msg: MIMEText | MIMEMultipart) -> None:
        """Blocking IMAP APPEND to iCloud's Sent Messages mailbox.

        Runs in a thread (see `_save_to_sent`). Stdlib `imaplib` is
        synchronous; the per-send overhead is ~200-400 ms and we don't
        block the event loop because we wrap the call in `asyncio.to_thread`.

        Two quirks worth knowing if you debug this:
          - The message bytes MUST use CRLF line endings; `as_bytes` produces
            LF by default and iCloud answers `BAD Parse Error` to LF input.
          - The mailbox name `Sent Messages` contains a space; we pass it
            inside double quotes so `imaplib` doesn't split on whitespace.
        """
        raw = msg.as_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT) as imap:
            imap.login(self._username, self._password)
            # `\Seen` so the message doesn't show as unread in Sent.
            status, data = imap.append(
                f'"{SENT_MAILBOX}"',
                "(\\Seen)",
                imaplib.Time2Internaldate(time.time()),
                raw,
            )
            if status != "OK":
                raise RuntimeError(f"IMAP APPEND status={status} data={data!r}")

    async def _save_to_sent(self, msg: MIMEText | MIMEMultipart) -> None:
        """Best-effort copy of `msg` into iCloud's Sent Messages mailbox.

        Never raises — IMAP issues are logged but don't propagate, because
        the SMTP send has already succeeded by the time we get here.
        """
        try:
            await asyncio.to_thread(self._imap_append_sent, msg)
            log.info(
                "email_saved_to_sent",
                message_id=msg.get("Message-Id", ""),
                mailbox=SENT_MAILBOX,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "email_save_to_sent_failed",
                error=str(e),
                message_id=msg.get("Message-Id", ""),
            )

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
            await self._save_to_sent(msg)
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
