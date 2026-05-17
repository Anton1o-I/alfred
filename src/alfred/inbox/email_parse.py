"""Split an email body into the current message + prior thread turns.

Email clients quote prior replies in a few common formats:

- Gmail / Apple Mail:
    On Sun, May 17, 2026 at 12:37 PM Alfred · Scheduler <a@x.com> wrote:
    > prior body line 1
    > prior body line 2

- Outlook (inserted as a header block, no `> ` prefix):
    From: Alfred · Scheduler <a@x.com>
    Sent: Sunday, May 17, 2026 12:37 PM
    To: ...
    Subject: ...

We find all marker positions in the body, slice the text between them into
chunks, strip `> ` prefixes, and label each chunk with sender/timestamp
metadata pulled from the preceding marker. When no markers are found, the
whole body is the current message — the agent still sees raw text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class EmailTurn:
    """One message in an email thread.

    `sender` and `sent_at` are best-effort, pulled from the quote header
    when available. None when the marker style didn't expose them.
    """

    body: str
    sender: str | None = None
    sent_at: str | None = None


# Gmail / Apple Mail: "On <date>[, ]<sender> wrote:" (possibly wrapped across
# two lines, possibly preceded by `> ` quote prefixes when nested).
_GMAIL_MARKER = re.compile(
    r"\n[> \t]*On\b(?P<when>[^\n]*?)(?:\n[> \t]*(?P<cont>[^\n]*?))?\bwrote:\s*\n",
    re.IGNORECASE,
)

# Outlook reply block. Captures: 1=sender, 2=sent_at.
_OUTLOOK_MARKER = re.compile(
    r"\n[> \t]*(?:-+\s*Original Message\s*-+\s*\n)?"
    r"From:[ \t]+(?P<sender>[^\n]+)\n"
    r"(?:Sent|Date):[ \t]+(?P<sent>[^\n]+)\n"
    r"(?:To:[^\n]*\n)?"
    r"(?:Cc:[^\n]*\n)?"
    r"(?:Subject:[^\n]*\n)",
    re.IGNORECASE,
)

# After splitting on a marker, the chunk may have `> ` prefixes from quoting.
# Strip them line-by-line (and tolerate multiple levels: `> > `).
_QUOTE_PREFIX = re.compile(r"^[ \t]*(?:>[ \t]?)+", re.MULTILINE)


def _strip_quote_prefix(text: str) -> str:
    return _QUOTE_PREFIX.sub("", text).strip()


def _parse_gmail_marker(match: re.Match) -> tuple[str | None, str | None]:
    """Pull (sender, sent_at) out of an 'On <date>, <sender> wrote:' match."""
    when_part = (match.group("when") or "").strip()
    cont = (match.group("cont") or "").strip()
    combined = (when_part + " " + cont).strip() if cont else when_part
    # The boundary between the date phrase and the sender is reliably the
    # AM/PM token (Gmail/Apple Mail) or the year (some clients). Try those
    # in order before falling back to broader heuristics.
    m_boundary = re.search(r"\b(?:AM|PM)\b,?\s+(.+?<[^>]+>)\s*$", combined, re.IGNORECASE)
    if m_boundary:
        sender = m_boundary.group(1).strip()
        sent_at = combined[: m_boundary.start()].strip().rstrip(",")
        return sender, sent_at or None
    m_email = re.search(r"([^\s,][^,\n]*?<[^>]+>)\s*$", combined)
    if m_email:
        sender = m_email.group(1).strip()
        sent_at = combined[: m_email.start()].strip().rstrip(",")
        return sender, sent_at or None
    if "," in combined:
        idx = combined.rfind(",")
        return combined[idx + 1 :].strip() or None, combined[:idx].strip() or None
    return combined or None, None


def parse_thread(body: str) -> list[EmailTurn]:
    """Split a raw email body into turns, newest first.

    The first element is the current message. Subsequent elements are
    prior quoted turns. Returns a single-element list if no markers found.
    """
    if not body:
        return [EmailTurn(body="")]

    text = "\n" + body  # leading newline simplifies anchors

    # Collect all marker matches across both patterns, sorted by position.
    matches: list[tuple[int, int, str | None, str | None]] = []
    for m in _GMAIL_MARKER.finditer(text):
        sender, sent_at = _parse_gmail_marker(m)
        matches.append((m.start(), m.end(), sender, sent_at))
    for m in _OUTLOOK_MARKER.finditer(text):
        matches.append((m.start(), m.end(), m.group("sender").strip(), m.group("sent").strip()))
    matches.sort(key=lambda t: t[0])

    if not matches:
        return [EmailTurn(body=text.lstrip("\n").rstrip())]

    turns: list[EmailTurn] = []
    # First chunk: from start to the first marker — this is the current message.
    first_marker_start = matches[0][0]
    current_body = text[:first_marker_start].lstrip("\n").rstrip()
    turns.append(EmailTurn(body=current_body))

    # Subsequent chunks: between markers. Each chunk's metadata comes from
    # the marker that precedes it.
    for i, (_start, end, sender, sent_at) in enumerate(matches):
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        chunk_raw = text[end:next_start]
        chunk = _strip_quote_prefix(chunk_raw)
        if not chunk:
            continue
        turns.append(EmailTurn(body=chunk, sender=sender, sent_at=sent_at))

    return turns


def format_thread_for_agent(turns: list[EmailTurn]) -> str:
    """Render a parsed thread as a labeled preamble for the LLM.

    Returns just the current message when there are no prior turns.
    """
    if len(turns) <= 1:
        return turns[0].body if turns else ""

    current = turns[0]
    prior = turns[1:]

    parts = [
        "=== CURRENT MESSAGE ===",
        current.body,
        "",
        "=== EARLIER IN THIS THREAD ===",
        "(Newest first. Each block shows who sent it.)",
        "",
    ]
    for t in prior:
        header_bits: list[str] = []
        if t.sender:
            header_bits.append(t.sender)
        if t.sent_at:
            header_bits.append(t.sent_at)
        header = " · ".join(header_bits) if header_bits else "(unknown sender)"
        parts.append(f"[{header}]")
        parts.append(t.body)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"
