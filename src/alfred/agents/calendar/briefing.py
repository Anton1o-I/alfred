"""Daily next-day briefing + Sunday weekly preview.

Linear pipeline (no LangGraph — no branching):

  fetch_events (CalDAV)
       │
       ▼
  analyze_shape (pure code — gaps, totals, lunch check, tight transitions)
       │
       ▼
  generate_narrative (LLM specialist, typed DailyNarrative / WeeklyNarrative)
       │
       ▼
  render (definition-list HTML + plain text with greeting + summary + observations)
       │
       ▼
  send (notification_service.send_to_family)

The LLM only generates flavor — greeting, summary sentence, observation bullets.
All facts (event list, times, day stats) come from code. The LLM can't invent.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import structlog
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    from alfred.app import App

log = structlog.get_logger()


# ── LLM output schemas ────────────────────────────────────────────────────


class DailyNarrative(BaseModel):
    greeting: str = Field(
        description=(
            "Short opener, 5-10 words, friendly but understated. "
            "Examples: 'Heads up for tomorrow', 'Quick rundown', 'Light Monday ahead'."
        )
    )
    summary: str = Field(
        description=(
            "2-3 sentences narrating the shape of the day — when it's busy, when it's "
            "open, anything worth flagging. Stick to what's in the events + analysis. "
            "Don't invent meetings or people."
        )
    )
    observations: list[str] = Field(
        default_factory=list,
        description=(
            "0-3 short bullets, each ≤80 chars. Useful things like '3-hour open block 12–3pm' "
            "or 'No lunch on the calendar'. Skip if nothing notable."
        ),
    )


class WeeklyNarrative(BaseModel):
    greeting: str = Field(description="Short opener, e.g. 'Week ahead'.")
    summary: str = Field(
        description=(
            "2-3 sentences on the week's shape — front-loaded/back-loaded, "
            "busiest and lightest days, total density. Facts only."
        )
    )
    observations: list[str] = Field(
        default_factory=list,
        description="0-3 short bullets on patterns worth noting.",
    )


# ── Prompts ───────────────────────────────────────────────────────────────


_DAILY_PROMPT = (
    "You are writing a brief, friendly heads-up about tomorrow's calendar for one person.\n"
    "\n"
    "Tomorrow: {label}\n"
    "Timezone: {tz}\n"
    "\n"
    "Events ({event_count}):\n"
    "{events_block}\n"
    "\n"
    "Analysis (computed):\n"
    "- Total scheduled time: {total_minutes} minutes\n"
    "- First event starts at: {first_start}\n"
    "- Last event ends at: {last_end}\n"
    "- Open blocks (>= 60 min between events): {open_blocks}\n"
    "- Tight transitions (< 15 min gap): {tight_transitions}\n"
    "- Lunch window covered (11:30am - 1:30pm): {has_lunch}\n"
    "\n"
    "Write a short, friendly briefing. Keep it grounded in the data — do not invent "
    "events, people, or context. Return a typed DailyNarrative.\n"
    "\n"
    "Tone notes:\n"
    "- Friendly but understated — like a thoughtful assistant, not a marketer.\n"
    "- No forced cheer ('Have a great day!').\n"
    "- If the day is very light, say so plainly.\n"
    "- If lunch isn't covered, you may mention it as an observation.\n"
    "- Times in greetings/summary should use 12-hour with am/pm (lowercase ok)."
)


_WEEKLY_PROMPT = (
    "You are writing a brief, friendly preview of the upcoming week's calendar.\n"
    "\n"
    "Week: {start_label} to {end_label}\n"
    "Timezone: {tz}\n"
    "\n"
    "Events by day:\n"
    "{events_block}\n"
    "\n"
    "Analysis (computed):\n"
    "- Total events: {event_count}\n"
    "- Total scheduled time: {total_minutes} minutes\n"
    "- Busiest day: {busiest_day}\n"
    "- Lightest day: {lightest_day}\n"
    "\n"
    "Write a short, friendly week-ahead summary. Keep facts grounded in the data. "
    "Return a typed WeeklyNarrative."
)


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
    """Compute the day's shape: gaps, totals, lunch coverage, tight transitions."""
    if not events:
        return {
            "event_count": 0,
            "total_minutes": 0,
            "first_start": None,
            "last_end": None,
            "open_blocks": [],
            "tight_transitions": [],
            "has_lunch_window": False,
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

    # Lunch coverage: any event whose start is between 11:30 and 13:30 local.
    has_lunch = False
    for ev in events:
        dt = datetime.fromisoformat(ev["start_iso"]).astimezone(tz)
        if dt.hour + dt.minute / 60 < 13.5 and dt.hour + dt.minute / 60 >= 11.5:
            has_lunch = True
            break

    return {
        "event_count": len(events),
        "total_minutes": int(total_minutes),
        "first_start": _fmt_time(events[0]["start_iso"], tz),
        "last_end": _fmt_time(events[-1]["end_iso"], tz),
        "open_blocks": open_blocks,
        "tight_transitions": tight,
        "has_lunch_window": has_lunch,
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


# ── LLM specialists ───────────────────────────────────────────────────────


def _make_narrative_agent(litellm_client, output_type: type, model_name: str) -> Agent:
    model = OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(
            base_url=f"{litellm_client.base_url}/v1",
            api_key=litellm_client._api_key,  # noqa: SLF001
        ),
    )
    return Agent(model=model, output_type=output_type)


# ── Render ────────────────────────────────────────────────────────────────


_FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)


def _render_email(
    *,
    headline: str,
    narrative: DailyNarrative | WeeklyNarrative,
    events_section_plain: str,
    events_section_html: str,
    observations: list[str],
    chores_section_plain: str | None = None,
    chores_section_html: str | None = None,
) -> tuple[str, str]:
    """Shared render shape for daily + weekly."""
    # Plain
    plain_parts = [
        narrative.greeting.strip(),
        "",
        narrative.summary.strip(),
        "",
        "Events",
        "─" * 30,
        events_section_plain,
    ]
    if chores_section_plain:
        plain_parts.append("")
        plain_parts.append("Chores")
        plain_parts.append("─" * 30)
        plain_parts.append(chores_section_plain)
    if observations:
        plain_parts.append("")
        plain_parts.append("Observations")
        plain_parts.append("─" * 30)
        for obs in observations:
            plain_parts.append(f"• {obs}")
    plain = "\n".join(plain_parts).rstrip() + "\n"

    # HTML
    body_style = (
        "margin: 0; padding: 0; background: #f5f5f7; "
        f"font-family: {_FONT_STACK}; color: #1d1d1f;"
    )
    container_style = (
        "max-width: 560px; margin: 0 auto; padding: 32px 24px; background: #ffffff;"
    )
    header_style = (
        "margin: 0 0 6px; font-size: 14px; font-weight: 600; "
        "letter-spacing: 0.04em; text-transform: uppercase; color: #6e6e73;"
    )
    greeting_style = (
        "margin: 0 0 16px; font-size: 22px; font-weight: 600; "
        "letter-spacing: -0.01em; color: #111;"
    )
    summary_style = (
        "margin: 0 0 24px; font-size: 15px; line-height: 1.6; color: #1d1d1f;"
    )
    section_h_style = (
        "margin: 24px 0 12px; padding-top: 16px; "
        "border-top: 1px solid #e5e5ea; font-size: 11px; font-weight: 600; "
        "letter-spacing: 0.08em; text-transform: uppercase; color: #6e6e73;"
    )
    obs_list_style = "margin: 0; padding: 0 0 0 18px; font-size: 14px; line-height: 1.55;"

    parts = [
        '<!doctype html><html><head><meta charset="utf-8"></head>',
        f'<body style="{body_style}">',
        f'<div style="{container_style}">',
        f'<p style="{header_style}">{headline}</p>',
        f'<h1 style="{greeting_style}">{_escape(narrative.greeting)}</h1>',
        f'<p style="{summary_style}">{_escape(narrative.summary)}</p>',
        f'<h2 style="{section_h_style}">Events</h2>',
        events_section_html,
    ]
    if chores_section_html:
        parts.append(f'<h2 style="{section_h_style}">Chores</h2>')
        parts.append(chores_section_html)
    if observations:
        parts.append(f'<h2 style="{section_h_style}">Observations</h2>')
        parts.append(f'<ul style="{obs_list_style}">')
        for obs in observations:
            parts.append(f"<li>{_escape(obs)}</li>")
        parts.append("</ul>")
    parts.append("</div></body></html>")
    html = "".join(parts)
    return plain, html


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


# ── Entry points ──────────────────────────────────────────────────────────


async def run_daily_briefing(app: App) -> dict:
    """Fetch tomorrow's events + chore status and email a unified briefing."""
    timezone_name = _calendar_timezone(app)
    tz = ZoneInfo(timezone_name)
    now_local = datetime.now(tz)
    tomorrow = now_local.date() + timedelta(days=1)
    start = datetime.combine(tomorrow, datetime.min.time(), tzinfo=tz)
    end = start + timedelta(days=1)

    events = _fetch_events_for_window(app, start, end)
    chore_data = await _fetch_chore_data(app, now_local, tz)
    log.info(
        "daily_briefing_fetch",
        events=len(events),
        chores_overdue=len(chore_data["categorized"]["overdue"]),
        chores_due_today=len(chore_data["categorized"]["due_today"]),
        chores_due_tomorrow=len(chore_data["categorized"]["due_tomorrow"]),
        date=tomorrow.isoformat(),
    )

    has_chores_to_show = bool(
        chore_data["categorized"]["overdue"]
        or chore_data["categorized"]["due_today"]
        or chore_data["categorized"]["due_tomorrow"]
    )
    if not events and not has_chores_to_show:
        log.info("daily_briefing_skip_empty", date=tomorrow.isoformat())
        return {"events": 0, "chores": 0, "emailed": False}

    analysis = analyze_day(events, tz)
    narrative = await _generate_daily_narrative(
        app, tomorrow, events, analysis, tz, chore_data["categorized"]
    )

    events_plain = _render_events_plain(events, tz)
    events_html = _render_events_html(events, tz)
    headline = f"TOMORROW · {tomorrow.strftime('%a, %B %-d')}"

    chores_plain: str | None = None
    chores_html: str | None = None
    if has_chores_to_show:
        from alfred.notifications.tasks_render import (
            render_chores_html,
            render_chores_plain,
        )

        chores_plain = render_chores_plain(chore_data["categorized"])
        chores_html = render_chores_html(chore_data["categorized"])

    plain, html = _render_email(
        headline=headline,
        narrative=narrative,
        events_section_plain=events_plain,
        events_section_html=events_html,
        observations=narrative.observations,
        chores_section_plain=chores_plain,
        chores_section_html=chores_html,
    )
    subject = f"Tomorrow's schedule — {_fmt_short_date(tomorrow)}"
    await _send(app, subject, plain, html, force_user_ids=chore_data["shame_user_ids"])
    log.info(
        "daily_briefing_sent",
        date=tomorrow.isoformat(),
        events=len(events),
        chores_shown=sum(
            len(chore_data["categorized"][k])
            for k in ("overdue", "due_today", "due_tomorrow")
        ),
        shame_user_ids=chore_data["shame_user_ids"],
    )
    return {
        "events": len(events),
        "chores": sum(
            len(chore_data["categorized"][k])
            for k in ("overdue", "due_today", "due_tomorrow")
        ),
        "emailed": True,
    }


async def _fetch_chore_data(app: App, now_local: datetime, tz: ZoneInfo) -> dict:
    """Pull chore statuses from the tasks agent's store (if registered).

    Returns categorized status buckets + the list of user_ids that should be
    force-CC'd because at least one of their chores is past `shame_after_days`.
    Returns empty data when the tasks agent isn't enabled.
    """
    from alfred.notifications.tasks_render import categorize_statuses, shame_assignees

    empty = {
        "categorized": {"overdue": [], "due_today": [], "due_tomorrow": [], "later": []},
        "shame_user_ids": [],
    }
    tasks_agent = app.agent_registry.get("tasks")
    if tasks_agent is None:
        return empty
    try:
        statuses = await tasks_agent.store.status_for_all_active(now=now_local, tz=tz)
    except Exception as e:  # noqa: BLE001
        log.warning("daily_briefing_chores_fetch_failed", error=str(e))
        return empty
    categorized = categorize_statuses(statuses)
    shame_ids = sorted(shame_assignees(statuses))
    return {"categorized": categorized, "shame_user_ids": shame_ids}


async def run_weekly_preview(app: App) -> dict:
    """Sunday evening: email next-week preview to the family."""
    timezone_name = _calendar_timezone(app)
    tz = ZoneInfo(timezone_name)
    now_local = datetime.now(tz)
    # Next Monday → following Sunday inclusive
    days_until_monday = (0 - now_local.weekday()) % 7 or 7
    start_date = now_local.date() + timedelta(days=days_until_monday)
    end_date = start_date + timedelta(days=7)
    start = datetime.combine(start_date, datetime.min.time(), tzinfo=tz)
    end = datetime.combine(end_date, datetime.min.time(), tzinfo=tz)

    events = _fetch_events_for_window(app, start, end)
    log.info(
        "weekly_preview_fetch",
        events=len(events),
        start=start_date.isoformat(),
        end=(end_date - timedelta(days=1)).isoformat(),
    )

    by_day: dict[str, list[dict]] = {
        (start_date + timedelta(days=i)).isoformat(): [] for i in range(7)
    }
    for ev in events:
        d = datetime.fromisoformat(ev["start_iso"]).astimezone(tz).date().isoformat()
        if d in by_day:
            by_day[d].append(ev)

    if not events:
        log.info("weekly_preview_skip_empty")
        return {"events": 0, "emailed": False}

    analysis = analyze_week(by_day)
    narrative = await _generate_weekly_narrative(
        app, start_date, end_date - timedelta(days=1), by_day, analysis
    )

    events_plain = _render_weekly_events_plain(by_day, tz)
    events_html = _render_weekly_events_html(by_day, tz)
    headline = (
        f"WEEK AHEAD · {_fmt_short_date(start_date)} – "
        f"{_fmt_short_date(end_date - timedelta(days=1))}"
    )
    plain, html = _render_email(
        headline=headline,
        narrative=narrative,
        events_section_plain=events_plain,
        events_section_html=events_html,
        observations=narrative.observations,
    )
    subject = f"Week ahead — {_fmt_short_date(start_date)}"
    await _send(app, subject, plain, html)
    log.info("weekly_preview_sent", events=len(events))
    return {"events": len(events), "emailed": True}


def _calendar_timezone(app: App) -> str:
    agent = app.agent_registry.get("calendar")
    if agent is None:
        return "UTC"
    return agent._config.timezone  # noqa: SLF001


async def _generate_daily_narrative(
    app: App,
    day: date,
    events: list[dict],
    analysis: dict,
    tz: ZoneInfo,
    categorized_chores: dict | None = None,
) -> DailyNarrative:
    agent = _make_narrative_agent(app.litellm_client, DailyNarrative, "local-default")
    events_block = _render_events_plain(events, tz)
    chores_summary = "(none tracked)"
    if categorized_chores:
        bits = []
        if categorized_chores["overdue"]:
            worst = max(s.overdue_days for s in categorized_chores["overdue"])
            bits.append(f"{len(categorized_chores['overdue'])} overdue (worst {worst}d)")
        if categorized_chores["due_today"]:
            bits.append(f"{len(categorized_chores['due_today'])} due today")
        if categorized_chores["due_tomorrow"]:
            bits.append(f"{len(categorized_chores['due_tomorrow'])} due tomorrow")
        if bits:
            chores_summary = ", ".join(bits)
    prompt = (
        _DAILY_PROMPT.format(
            label=day.strftime("%A, %B %-d, %Y"),
            tz=tz.key,
            event_count=analysis["event_count"],
            events_block=events_block,
            total_minutes=analysis["total_minutes"],
            first_start=analysis["first_start"] or "—",
            last_end=analysis["last_end"] or "—",
            open_blocks=analysis["open_blocks"] or "(none)",
            tight_transitions=analysis["tight_transitions"] or "(none)",
            has_lunch=analysis["has_lunch_window"],
        )
        + f"\n\nChore burden: {chores_summary}.\n"
        "If chores are overdue or due tomorrow, briefly acknowledge them in the "
        "summary or as an observation — do not invent any."
    )
    result = await agent.run(prompt)
    return result.output


async def _generate_weekly_narrative(
    app: App,
    start_date: date,
    end_date: date,
    by_day: dict[str, list[dict]],
    analysis: dict,
) -> WeeklyNarrative:
    agent = _make_narrative_agent(app.litellm_client, WeeklyNarrative, "local-default")
    tz = ZoneInfo(_calendar_timezone(app))
    events_block = _render_weekly_events_plain(by_day, tz)
    prompt = _WEEKLY_PROMPT.format(
        start_label=start_date.strftime("%A %b %-d"),
        end_label=end_date.strftime("%A %b %-d"),
        tz=tz.key,
        events_block=events_block,
        event_count=analysis["event_count"],
        total_minutes=analysis["total_minutes"],
        busiest_day=analysis["busiest_day"] or "—",
        lightest_day=analysis["lightest_day"] or "—",
    )
    result = await agent.run(prompt)
    return result.output


async def _send(
    app: App,
    subject: str,
    plain: str,
    html: str,
    force_user_ids: list[str] | None = None,
) -> None:
    from alfred.notifications.signature import append_to_body, append_to_html, render_signature

    cfg = app.settings.notifications
    from_name = cfg.email_from_names_by_agent.get("calendar", cfg.email_from_name)
    tagline = cfg.email_taglines_by_agent.get("calendar", "")
    sig_plain, sig_html = render_signature(from_name or "", tagline)
    plain_final = append_to_body(plain, sig_plain)
    html_final = append_to_html(html, sig_html) if html else None
    await app.notification_service.send_to_family(
        body=plain_final,
        subject=subject,
        html_body=html_final,
        from_name=from_name,
        agent_name="calendar",
        force_user_ids=force_user_ids,
    )
