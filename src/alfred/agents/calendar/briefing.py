"""Calendar event-shape analysis + rendering helpers.

These are the calendar-agent-owned primitives the routines/daily_briefing
routine composes with chore data and shame routing. Keep this file
narrowly scoped to event-shaped concerns: fetch, flatten, analyze, render.

Public surface used by routines/daily_briefing:
- ``analyze_day``, ``analyze_week`` — pure stats over events
- ``_fetch_events_for_window``, ``_get_calendar_client`` — pull events
- ``_render_events_plain`` / ``_render_events_html`` — daily list rows
- ``_render_weekly_events_plain`` / ``_render_weekly_events_html`` — weekly rows
- ``_flatten_event``, ``_fmt_time``, ``_fmt_short_date`` — formatting primitives
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import structlog

if TYPE_CHECKING:
    from alfred.app import App

log = structlog.get_logger()


# ── Event helpers ─────────────────────────────────────────────────────────


def _flatten_event(ev: dict) -> dict[str, Any] | None:
    """Pull (start_iso, end_iso, summary) out of a CalDAV-shaped event dict."""
    s = (ev.get("start") or {}).get("dateTime")
    e = (ev.get("end") or {}).get("dateTime")
    if not s or not e:
        return None
    return {
        "start_iso": s,
        "end_iso": e,
        "summary": ev.get("summary", "Untitled"),
        "location": ev.get("location"),
    }


def _fmt_time(iso: str, tz: ZoneInfo) -> str:
    dt = datetime.fromisoformat(iso).astimezone(tz)
    return dt.strftime("%-I:%M %p").lower()


def _fmt_short_date(d: date) -> str:
    return d.strftime("%a, %b %-d")


def _fetch_events_for_window(app: App, start: datetime, end: datetime) -> list[dict]:
    """Pull events from the configured calendar and flatten to a flat list."""
    client = _get_calendar_client(app)
    raw = client.list_events(time_min=start, time_max=end)
    out = []
    for ev in raw:
        flat = _flatten_event(ev)
        if flat:
            out.append(flat)
    out.sort(key=lambda e: e["start_iso"])
    return out


def _get_calendar_client(app: App):  # type: ignore[no-untyped-def]
    """Reach the calendar client. The CalendarAgent has it as ._client."""
    agent = app.agent_registry.get("calendar")
    if agent is None:
        raise RuntimeError("calendar agent is not registered")
    return agent._client  # noqa: SLF001


# ── Analysis ──────────────────────────────────────────────────────────────


def analyze_day(events: list[dict], tz: ZoneInfo) -> dict:
    """Compute the day's shape: gaps, totals, tight transitions."""
    if not events:
        return {
            "event_count": 0,
            "total_minutes": 0,
            "first_start": None,
            "last_end": None,
            "open_blocks": [],
            "tight_transitions": [],
        }
    total_minutes = 0.0
    for ev in events:
        start = datetime.fromisoformat(ev["start_iso"])
        end = datetime.fromisoformat(ev["end_iso"])
        total_minutes += (end - start).total_seconds() / 60

    open_blocks: list[dict] = []
    tight: list[dict] = []
    for i in range(len(events) - 1):
        end_a = datetime.fromisoformat(events[i]["end_iso"])
        start_b = datetime.fromisoformat(events[i + 1]["start_iso"])
        gap_min = (start_b - end_a).total_seconds() / 60
        if gap_min >= 60:
            open_blocks.append(
                {
                    "after": events[i]["summary"],
                    "duration_minutes": int(gap_min),
                    "from_local": _fmt_time(events[i]["end_iso"], tz),
                    "to_local": _fmt_time(events[i + 1]["start_iso"], tz),
                }
            )
        elif 0 <= gap_min < 15:
            tight.append(
                {
                    "before": events[i]["summary"],
                    "after": events[i + 1]["summary"],
                    "gap_minutes": int(gap_min),
                }
            )

    return {
        "event_count": len(events),
        "total_minutes": int(total_minutes),
        "first_start": _fmt_time(events[0]["start_iso"], tz),
        "last_end": _fmt_time(events[-1]["end_iso"], tz),
        "open_blocks": open_blocks,
        "tight_transitions": tight,
    }


def analyze_week(by_day: dict[str, list[dict]]) -> dict:
    """Aggregate counts + busiest/lightest from per-day buckets."""
    totals: list[tuple[str, int, int]] = []  # (day_iso, count, minutes)
    for day, evs in by_day.items():
        minutes = 0.0
        for ev in evs:
            start = datetime.fromisoformat(ev["start_iso"])
            end = datetime.fromisoformat(ev["end_iso"])
            minutes += (end - start).total_seconds() / 60
        totals.append((day, len(evs), int(minutes)))
    days_with_events = [(d, c, m) for d, c, m in totals if c > 0]
    busiest = max(days_with_events, key=lambda t: t[2], default=None)
    lightest = min(days_with_events, key=lambda t: t[2], default=None)
    return {
        "event_count": sum(c for _, c, _ in totals),
        "total_minutes": sum(m for _, _, m in totals),
        "busiest_day": busiest[0] if busiest else None,
        "lightest_day": lightest[0] if lightest else None,
        "by_day_totals": [
            {"date": d, "count": c, "minutes": m} for d, c, m in totals
        ],
    }


# ── Rendering ─────────────────────────────────────────────────────────────


def _escape(s: str) -> str:
    from html import escape

    return escape(s or "")


def _render_events_plain(events: list[dict], tz: ZoneInfo) -> str:
    if not events:
        return "(nothing on the calendar)"
    lines = []
    for ev in events:
        t = _fmt_time(ev["start_iso"], tz)
        line = f"  {t:<10} {ev['summary']}"
        if ev.get("location"):
            line += f"  ·  {ev['location']}"
        lines.append(line)
    return "\n".join(lines)


def _render_events_html(events: list[dict], tz: ZoneInfo) -> str:
    if not events:
        return '<p style="font-style: italic; color: #86868b;">(nothing on the calendar)</p>'
    rows = []
    row_style = (
        "padding: 6px 0; border-bottom: 1px solid #f0f0f3; font-size: 15px; line-height: 1.4;"
    )
    time_style = (
        "display: inline-block; min-width: 80px; color: #86868b; "
        "font-variant-numeric: tabular-nums;"
    )
    title_style = "color: #111;"
    loc_style = "color: #86868b; font-size: 13px;"
    for ev in events:
        t = _fmt_time(ev["start_iso"], tz)
        line = (
            f'<div style="{row_style}">'
            f'<span style="{time_style}">{_escape(t)}</span>'
            f'<span style="{title_style}">{_escape(ev["summary"])}</span>'
        )
        if ev.get("location"):
            line += f' <span style="{loc_style}">· {_escape(ev["location"])}</span>'
        line += "</div>"
        rows.append(line)
    return "".join(rows)


def _render_weekly_events_plain(by_day: dict[str, list[dict]], tz: ZoneInfo) -> str:
    lines = []
    for day_iso in sorted(by_day.keys()):
        day = date.fromisoformat(day_iso)
        evs = by_day[day_iso]
        if not evs:
            lines.append(f"  {_fmt_short_date(day):<14}  (clear)")
        else:
            times = ", ".join(_fmt_time(e["start_iso"], tz) for e in evs)
            label = f"{len(evs)} event{'s' if len(evs) > 1 else ''}"
            lines.append(f"  {_fmt_short_date(day):<14}  {label:<10}  {times}")
    return "\n".join(lines)


def _render_weekly_events_html(by_day: dict[str, list[dict]], tz: ZoneInfo) -> str:
    rows = []
    row_style = (
        "padding: 6px 0; border-bottom: 1px solid #f0f0f3; font-size: 15px; line-height: 1.4;"
    )
    day_style = (
        "display: inline-block; min-width: 120px; color: #111; font-weight: 500;"
    )
    detail_style = "color: #86868b;"
    clear_style = "color: #86868b; font-style: italic;"
    for day_iso in sorted(by_day.keys()):
        day = date.fromisoformat(day_iso)
        evs = by_day[day_iso]
        cell = f'<span style="{day_style}">{_escape(_fmt_short_date(day))}</span>'
        if not evs:
            cell += f'<span style="{clear_style}">clear</span>'
        else:
            times = ", ".join(_fmt_time(e["start_iso"], tz) for e in evs)
            label = f"{len(evs)} event{'s' if len(evs) > 1 else ''}"
            cell += (
                f'<span style="{detail_style}">{_escape(label)} · {_escape(times)}</span>'
            )
        rows.append(f'<div style="{row_style}">{cell}</div>')
    return "".join(rows)
