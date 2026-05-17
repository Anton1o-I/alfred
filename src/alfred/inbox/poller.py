"""IMAP-based inbox poller for iCloud Mail.

Fetches unread messages from imap.mail.me.com, filters to the authorized
senders allowlist, hands the body to the orchestrator, sends a threaded
reply, and marks the message read. App-specific password auth — no OAuth.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from email.utils import parseaddr
from typing import TYPE_CHECKING

import structlog
from imap_tools import AND, MailBox

from alfred.core.constants import RequestSource
from alfred.core.models import AgentRequest
from alfred.inbox.email_parse import format_thread_for_agent, parse_thread
from alfred.notifications.signature import (
    append_to_body,
    append_to_html,
    render_signature,
)

if TYPE_CHECKING:
    from alfred.app import App

log = structlog.get_logger()

IMAP_HOST = "imap.mail.me.com"
IMAP_PORT = 993

# Folders to scan. Junk is included because iCloud's spam filter often misroutes
# legitimate messages from authorized senders, especially for newer accounts.
# Our allowlist is the real security boundary.
POLLED_FOLDERS = ("INBOX", "Junk")


@dataclass
class _Fetched:
    uid: str
    from_addr: str
    subject: str
    body: str
    message_id: str
    references: str
    folder: str


def _header(headers: dict, key: str) -> str:
    """imap_tools returns header values as tuples; coerce to first scalar string."""
    val = headers.get(key, "")
    if isinstance(val, tuple):
        return val[0] if val else ""
    return val or ""


class InboxPoller:
    """One-shot IMAP poller. Call process_once() from a scheduled task."""

    def __init__(
        self,
        username: str,
        app_password: str,
        authorized_senders: list[str],
    ) -> None:
        if not username or not app_password:
            raise ValueError("username and app_password are required")
        self._username = username
        self._password = app_password
        self._allowlist = {s.lower() for s in authorized_senders}

    def _is_authorized(self, from_addr: str) -> bool:
        _, addr = parseaddr(from_addr)
        return addr.lower() in self._allowlist

    def _fetch_blocking(self) -> tuple[list[_Fetched], dict[str, list[str]]]:
        """Fetch unread mail across POLLED_FOLDERS.

        Returns (authorized_fetched, unauthorized_uids_by_folder).
        Each _Fetched carries its source folder so we can mark seen in the
        same folder it came from.
        """
        authorized: list[_Fetched] = []
        unauthorized_uids: dict[str, list[str]] = {}
        with MailBox(IMAP_HOST, IMAP_PORT).login(
            self._username, self._password
        ) as mailbox:
            for folder in POLLED_FOLDERS:
                try:
                    mailbox.folder.set(folder)
                except Exception as e:  # noqa: BLE001
                    log.warning("inbox_folder_skip", folder=folder, error=str(e))
                    continue
                for msg in mailbox.fetch(AND(seen=False), mark_seen=False, limit=20):
                    if not self._is_authorized(msg.from_):
                        log.info(
                            "inbox_skip_unauthorized",
                            uid=msg.uid,
                            from_=msg.from_,
                            folder=folder,
                        )
                        unauthorized_uids.setdefault(folder, []).append(msg.uid)
                        continue
                    # Keep the full email body — quote chain included.
                    # The parser/preamble logic in _dispatch turns the quote
                    # chain into structured prior turns for the agent.
                    if msg.text:
                        body = msg.text
                    elif msg.html:
                        body = re.sub(r"<[^>]+>", "", msg.html)
                    else:
                        body = ""
                    authorized.append(
                        _Fetched(
                            uid=msg.uid,
                            from_addr=msg.from_,
                            subject=msg.subject or "",
                            body=body,
                            message_id=_header(msg.headers, "message-id"),
                            references=_header(msg.headers, "references"),
                            folder=folder,
                        )
                    )
        return authorized, unauthorized_uids

    def _mark_seen_blocking(self, by_folder: dict[str, list[str]]) -> None:
        if not by_folder:
            return
        with MailBox(IMAP_HOST, IMAP_PORT).login(
            self._username, self._password
        ) as mailbox:
            for folder, uids in by_folder.items():
                if not uids:
                    continue
                try:
                    mailbox.folder.set(folder)
                    mailbox.flag(uids, "\\Seen", True)
                except Exception as e:  # noqa: BLE001
                    log.warning("inbox_mark_seen_failed", folder=folder, error=str(e))

    async def process_once(self, app: App) -> dict[str, int]:
        counts = {"scanned": 0, "dispatched": 0, "skipped": 0, "errors": 0}

        try:
            fetched, unauth_by_folder = await asyncio.to_thread(self._fetch_blocking)
        except Exception as e:  # noqa: BLE001
            log.error("inbox_connect_failed", error=str(e))
            counts["errors"] += 1
            return counts

        unauth_count = sum(len(v) for v in unauth_by_folder.values())
        counts["scanned"] = len(fetched) + unauth_count
        counts["skipped"] = unauth_count
        await asyncio.to_thread(self._mark_seen_blocking, unauth_by_folder)

        successful_by_folder: dict[str, list[str]] = {}
        for item in fetched:
            try:
                _, sender_addr = parseaddr(item.from_addr)
                log.info(
                    "inbox_dispatching",
                    uid=item.uid,
                    folder=item.folder,
                    sender=sender_addr,
                    subject=item.subject[:80],
                    body_chars=len(item.body),
                )
                await _dispatch(
                    app,
                    body=item.body,
                    subject=item.subject,
                    sender=sender_addr,
                    message_id=item.message_id,
                    in_reply_to=item.message_id,  # threading: reply to THIS message
                    references=item.references,
                )
                successful_by_folder.setdefault(item.folder, []).append(item.uid)
                counts["dispatched"] += 1
            except Exception as e:  # noqa: BLE001
                log.error("inbox_process_failed", uid=item.uid, error=str(e))
                counts["errors"] += 1

        await asyncio.to_thread(self._mark_seen_blocking, successful_by_folder)

        if counts["scanned"]:
            log.info("inbox_poll_summary", **counts)
        return counts


async def _dispatch(
    app: App,
    body: str,
    subject: str,
    sender: str,
    message_id: str,
    in_reply_to: str,
    references: str,
) -> None:
    """Route the email body through the orchestrator and reply with the result.

    The full email body (quote chain included) is parsed into turns so the
    agent sees the prior conversation natively — no separate history store.
    """
    turns = parse_thread(body)
    enriched_body = format_thread_for_agent(turns)

    request = AgentRequest(
        user_message=enriched_body,
        source=RequestSource.EMAIL,
        user_id=None,
    )
    response = await app.orchestrator.handle(request)
    reply_subject = subject if subject.lower().startswith("re:") else f"Re: {subject}"

    email_channel = app.notification_service.channels.get("email")
    if email_channel is None or not hasattr(email_channel, "send_reply"):
        log.warning("inbox_reply_skipped_no_email_channel")
        return

    # Use the per-agent From-name if configured, so a calendar reply looks
    # like it came from "Alfred · Scheduler" rather than the default.
    cfg = app.settings.notifications
    agent_key = response.agent_name or ""
    from_name = cfg.email_from_names_by_agent.get(agent_key)
    tagline = cfg.email_taglines_by_agent.get(agent_key, "")
    sig_plain, sig_html = render_signature(from_name or "", tagline)

    # The agent is responsible for producing fully rendered plain + html.
    # We just append the signature.
    plain_body: str = response.message
    html_body: str | None = (response.data or {}).get("reply_html")

    plain_body = append_to_body(plain_body, sig_plain)
    if html_body:
        html_body = append_to_html(html_body, sig_html)

    await email_channel.send_reply(
        recipient=sender,
        subject=reply_subject,
        body=plain_body,
        html_body=html_body,
        in_reply_to=in_reply_to,
        references=references,
        from_name=from_name,
    )
