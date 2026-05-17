"""Render calendar agent replies as styled definition-list emails.

Produces (plain, html) pairs so we can send multipart/alternative. HTML
is inline-styled to survive Gmail / iCloud / Apple Mail rendering.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any

_FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _format_date(dt: datetime) -> str:
    # "Tuesday, May 19, 2026"
    return dt.strftime("%A, %B %-d, %Y") if hasattr(dt, "strftime") else str(dt)


def _format_time_range(start: datetime, end: datetime) -> str:
    # "3:00 PM – 4:00 PM" — collapse leading zero on hour
    def _fmt(d: datetime) -> str:
        return d.strftime("%-I:%M %p")
    tz = start.strftime("%Z") or ""
    base = f"{_fmt(start)} – {_fmt(end)}"
    return f"{base} {tz}".rstrip() if tz else base


def _rows(event: dict[str, Any]) -> list[tuple[str, str]]:
    """Build label/value rows from an event dict. Omits empty optional fields."""
    rows: list[tuple[str, str]] = []
    if event.get("summary"):
        rows.append(("Event", event["summary"]))
    start = _parse_iso(event.get("start_iso"))
    end = _parse_iso(event.get("end_iso"))
    if start:
        rows.append(("Date", _format_date(start)))
    if start and end:
        rows.append(("Time", _format_time_range(start, end)))
    if event.get("location"):
        rows.append(("Location", event["location"]))
    if event.get("description"):
        rows.append(("Notes", event["description"]))
    cal = event.get("calendar_name")
    if cal:
        rows.append(("Calendar", cal))
    return rows


# ── Plain text ──────────────────────────────────────────────────────────────


def render_event_plain(event: dict[str, Any]) -> str:
    rows = _rows(event)
    if not rows:
        return "Event added to your calendar."
    label_width = max(len(label) for label, _ in rows) + 2
    lines = ["✓ Event added", ""]
    for label, value in rows:
        lines.append(f"{label:<{label_width}}{value}")
    return "\n".join(lines)


def render_clarification_plain(message: str) -> str:
    """Pass-through for now — the agent's clarification text is already labeled."""
    return message.strip()


# ── HTML ────────────────────────────────────────────────────────────────────

_BODY_STYLE = _font = (
    "margin: 0; padding: 0; background: #f5f5f7; "
    f"font-family: {_FONT_STACK}; color: #1d1d1f;"
)
_CONTAINER_STYLE = (
    "max-width: 560px; margin: 0 auto; padding: 32px 24px; background: #ffffff;"
)
_HEADER_STYLE = (
    "margin: 0 0 24px; font-size: 14px; font-weight: 600; "
    "letter-spacing: 0.04em; text-transform: uppercase; color: #34a853;"
)
_HEADER_STYLE_NEUTRAL = _HEADER_STYLE.replace("#34a853", "#0040a0")
_TITLE_STYLE = (
    "margin: 0 0 20px; font-size: 22px; font-weight: 600; "
    "letter-spacing: -0.01em; color: #111;"
)
_TABLE_STYLE = "width: 100%; border-collapse: collapse; margin: 0 0 16px;"
_LABEL_STYLE = (
    "padding: 8px 16px 8px 0; vertical-align: top; font-size: 13px; "
    "color: #86868b; font-weight: 500; white-space: nowrap;"
)
_VALUE_STYLE = (
    "padding: 8px 0; vertical-align: top; font-size: 15px; "
    "line-height: 1.45; color: #1d1d1f;"
)


def render_event_html(event: dict[str, Any]) -> str:
    rows = _rows(event)
    summary = escape(event.get("summary", "Event"))
    table_rows = "".join(
        f'<tr><td style="{_LABEL_STYLE}">{escape(label)}</td>'
        f'<td style="{_VALUE_STYLE}">{escape(value)}</td></tr>'
        for label, value in rows
        if label != "Event"
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8"></head>'
        f'<body style="{_BODY_STYLE}">'
        f'<div style="{_CONTAINER_STYLE}">'
        f'<p style="{_HEADER_STYLE}">✓ Event added</p>'
        f'<h1 style="{_TITLE_STYLE}">{summary}</h1>'
        f'<table style="{_TABLE_STYLE}">{table_rows}</table>'
        "</div></body></html>"
    )


def render_clarification_html(message: str) -> str:
    """A simple bordered card for clarification responses."""
    safe = escape(message).replace("\n", "<br>")
    return (
        '<!doctype html><html><head><meta charset="utf-8"></head>'
        f'<body style="{_BODY_STYLE}">'
        f'<div style="{_CONTAINER_STYLE}">'
        f'<p style="{_HEADER_STYLE_NEUTRAL}">Need a bit more info</p>'
        f'<div style="font-size: 15px; line-height: 1.55; color: #1d1d1f;">{safe}</div>'
        "</div></body></html>"
    )
