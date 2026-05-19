"""Sunday evening: email a next-week preview to the family.

Pipeline:

  fetch_events (Mon-Sun of next week)
       │
       ▼
  analyze_week (pure code — counts, busiest/lightest, totals)
       │
       ▼
  generate_weekly_narrative (LLM specialist, typed WeeklyNarrative)
       │
       ▼
  render (shared email frame from routines/_common)
       │
       ▼
  send (notification_service.send_to_family)

Separate from `routines/daily_briefing` because the trigger, prompt,
output schema, and reader experience are all distinct. Shared helpers
(email frame, dispatcher, LLM agent factory) live in `routines/_common`.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import structlog
from pydantic import BaseModel, Field

from alfred.agents.calendar.briefing import (
    _fetch_events_for_window,
    _fmt_short_date,
    _render_weekly_events_html,
    _render_weekly_events_plain,
    analyze_week,
)
from alfred.routines._common import (
    _calendar_timezone,
    _make_narrative_agent,
    _render_email,
    _send,
)

if TYPE_CHECKING:
    from alfred.app import App

log = structlog.get_logger()


# ── LLM output schema ─────────────────────────────────────────────────────


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


# ── Prompt ────────────────────────────────────────────────────────────────


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


# ── Routine ───────────────────────────────────────────────────────────────


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
        greeting=narrative.greeting,
        summary=narrative.summary,
        events_section_plain=events_plain,
        events_section_html=events_html,
        observations=narrative.observations,
    )
    subject = f"Week ahead — {_fmt_short_date(start_date)}"
    await _send(app, subject, plain, html)
    log.info("weekly_preview_sent", events=len(events))
    return {"events": len(events), "emailed": True}


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
