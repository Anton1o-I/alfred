"""Morning briefing routine: today's events + tasks + week-ahead glance.

Fires early (5 AM by default) to set up the day. Differs from the evening
``daily_briefing`` in three ways:

  1. Window is TODAY (not tomorrow).
  2. Chore section shows overdue + due-today (no due-tomorrow).
  3. Adds a hybrid "week ahead" block: a terse one-line-per-day bulleted
     glance, framed by a single LLM-written headline sentence.

Pipeline mirrors the evening briefing: fetch → analyze → narrate → render
→ send. Shared frame, dispatcher, and chore helpers live in
``routines/_common``.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal
from zoneinfo import ZoneInfo

import structlog
from pydantic import BaseModel, Field

from alfred.agents.calendar.briefing import (
    _fetch_events_for_window,
    _fmt_short_date,
    _render_events_html,
    _render_events_plain,
    analyze_day,
    analyze_week,
)
from alfred.routines._common import (
    _build_assignee_names,
    _calendar_timezone,
    _make_narrative_agent,
    _render_chores_for_prompt,
    _render_email,
    _send,
    fetch_chore_data,
    generate_chore_roasts,
    render_week_ahead_bullets_html,
    render_week_ahead_bullets_plain,
)

if TYPE_CHECKING:
    from alfred.app import App

log = structlog.get_logger()


# ── LLM output schemas ────────────────────────────────────────────────────


class MorningNarrative(BaseModel):
    lead_section: Literal["events", "chores"] = Field(
        description=(
            "Which section appears FIRST in the email. Use 'chores' when "
            "(a) today has no events, OR (b) there is at least one overdue "
            "chore, OR (c) events are light while multiple chores are due "
            "today. Otherwise 'events'."
        )
    )
    greeting: str = Field(
        description="Short opener, 5-10 words. e.g. 'Today's plan', 'Light day ahead'."
    )
    summary: str = Field(
        description=(
            "1-2 sentences narrating today. Mention the lead section's "
            "content concretely. Facts only."
        )
    )
    week_headline: str = Field(
        description=(
            "ONE sentence framing the next 7 days — e.g. 'Front-loaded "
            "week with three meetings Monday' or 'Quiet stretch through "
            "Wednesday, busier Thursday-Friday.' Grounded in the by-day "
            "counts shown. No invented events."
        )
    )
    observations: list[str] = Field(
        default_factory=list,
        description="0-3 short bullets on the shape of TODAY. Skip if nothing notable.",
    )


# ── Prompt ────────────────────────────────────────────────────────────────


_MORNING_PROMPT = (
    "You are writing a brief, friendly morning briefing for one person.\n"
    "Three sections may appear: today's events, household chores (overdue\n"
    "and due-today only), and a week-ahead glance. Decide which section\n"
    "(events vs. chores) LEADS the email, then write a short narrative\n"
    "and ONE sentence framing the rest of the week.\n"
    "\n"
    "Today: {today_label}\n"
    "Timezone: {tz}\n"
    "\n"
    "Today's events ({event_count}):\n"
    "{events_block}\n"
    "\n"
    "Event analysis (computed):\n"
    "- Total scheduled time today: {total_minutes} minutes\n"
    "- First event starts at: {first_start}\n"
    "- Last event ends at: {last_end}\n"
    "- Open blocks (>= 60 min): {open_blocks}\n"
    "- Tight transitions (< 15 min): {tight_transitions}\n"
    "\n"
    "Chores (overdue + due today only — not tomorrow):\n"
    "{chores_block}\n"
    "\n"
    "Week ahead — by-day counts (next 7 days starting tomorrow):\n"
    "{week_block}\n"
    "Week totals: {week_event_count} events, busiest {busiest_day}, "
    "lightest {lightest_day}.\n"
    "\n"
    "Tone & content rules:\n"
    "- Friendly but understated. No forced cheer, no emoji.\n"
    "- Name today's events specifically (title + time) when they exist.\n"
    "- Never call a 'due today' chore 'overdue'.\n"
    "- The week_headline must reflect the by-day pattern above. ONE sentence.\n"
    "- Observations only for structural notes about TODAY (open blocks,\n"
    "  tight transitions). Empty list if nothing notable.\n"
    "\n"
    "Never invent events, chores, or names not in the data above.\n"
    "\n"
    "Return a typed MorningNarrative."
)


# ── Routine ───────────────────────────────────────────────────────────────


async def run_morning_briefing(app: App) -> dict:
    """Fetch today's events + chores + week-ahead and email a unified briefing."""
    timezone_name = _calendar_timezone(app)
    tz = ZoneInfo(timezone_name)
    now_local = datetime.now(tz)
    today = now_local.date()

    # Today window
    today_start = datetime.combine(today, datetime.min.time(), tzinfo=tz)
    today_end = today_start + timedelta(days=1)
    today_events = _fetch_events_for_window(app, today_start, today_end)

    # Next 7 days starting tomorrow (week-ahead glance)
    week_start_date = today + timedelta(days=1)
    week_end_date = week_start_date + timedelta(days=7)
    week_start = datetime.combine(week_start_date, datetime.min.time(), tzinfo=tz)
    week_end = datetime.combine(week_end_date, datetime.min.time(), tzinfo=tz)
    week_events = _fetch_events_for_window(app, week_start, week_end)

    week_by_day: dict[str, list[dict]] = {
        (week_start_date + timedelta(days=i)).isoformat(): [] for i in range(7)
    }
    for ev in week_events:
        d = datetime.fromisoformat(ev["start_iso"]).astimezone(tz).date().isoformat()
        if d in week_by_day:
            week_by_day[d].append(ev)

    chore_data = await fetch_chore_data(app, now_local, tz)

    log.info(
        "morning_briefing_fetch",
        events_today=len(today_events),
        week_events=len(week_events),
        chores_overdue=len(chore_data["categorized"]["overdue"]),
        chores_due_today=len(chore_data["categorized"]["due_today"]),
        date=today.isoformat(),
    )

    # Morning briefing scopes chores to overdue + due-today only.
    morning_categorized = {
        "overdue": chore_data["categorized"]["overdue"],
        "due_today": chore_data["categorized"]["due_today"],
        "due_tomorrow": [],
        "later": [],
    }
    has_chores_to_show = bool(
        morning_categorized["overdue"] or morning_categorized["due_today"]
    )
    if not today_events and not has_chores_to_show and not week_events:
        log.info("morning_briefing_skip_empty", date=today.isoformat())
        return {"events": 0, "chores": 0, "emailed": False}

    today_analysis = analyze_day(today_events, tz)
    week_analysis = analyze_week(week_by_day)
    assignee_names = _build_assignee_names(app)
    narrative = await _generate_morning_narrative(
        app,
        today,
        today_events,
        today_analysis,
        week_by_day,
        week_analysis,
        tz,
        morning_categorized,
        assignee_names,
    )

    # Render today's events
    events_plain = _render_events_plain(today_events, tz)
    events_html = _render_events_html(today_events, tz)
    headline = f"TODAY · {today.strftime('%a, %B %-d')}"

    from alfred.agents.tasks.renderers import (
        ShameTierTable,
        render_chores_html,
        render_chores_plain,
    )

    tier_table = ShameTierTable(app.settings.notifications.shame_tiers)
    roast_lines = await generate_chore_roasts(
        app, morning_categorized["overdue"], tier_table
    )

    chores_plain: str | None = None
    chores_html: str | None = None
    if has_chores_to_show:
        chores_plain = render_chores_plain(
            morning_categorized,
            assignee_names,
            shame_tiers=tier_table,
            roast_lines=roast_lines,
        )
        chores_html = render_chores_html(
            morning_categorized,
            assignee_names,
            shame_tiers=tier_table,
            roast_lines=roast_lines,
        )

    # Week-ahead block: hybrid headline + bulleted glance.
    week_bullets_plain = render_week_ahead_bullets_plain(week_by_day, tz)
    week_bullets_html = render_week_ahead_bullets_html(week_by_day, tz)

    # Pack the week section into the events column so we get the same
    # styled section header treatment from the shared frame.
    events_plain_with_week = (
        f"{events_plain}\n\nTHE WEEK AHEAD\n\n"
        f"{narrative.week_headline}\n\n{week_bullets_plain}"
    )
    events_html_with_week = (
        f"{events_html}"
        f'<h3 style="margin: 24px 0 8px; font-size: 11px; font-weight: 600; '
        f'letter-spacing: 0.08em; text-transform: uppercase; color: #6e6e73;">'
        f"The week ahead</h3>"
        f'<p style="margin: 0 0 12px; font-size: 15px; line-height: 1.55; '
        f'color: #1d1d1f;">{narrative.week_headline}</p>'
        f"{week_bullets_html}"
    )

    plain, html = _render_email(
        headline=headline,
        greeting=narrative.greeting,
        summary=narrative.summary,
        events_section_plain=events_plain_with_week,
        events_section_html=events_html_with_week,
        observations=narrative.observations,
        chores_section_plain=chores_plain,
        chores_section_html=chores_html,
        lead_section=narrative.lead_section,
    )
    subject = f"Today + week ahead — {_fmt_short_date(today)}"
    await _send(app, subject, plain, html, force_user_ids=chore_data["shame_user_ids"])

    log.info(
        "morning_briefing_sent",
        date=today.isoformat(),
        events_today=len(today_events),
        chores_shown=len(morning_categorized["overdue"])
        + len(morning_categorized["due_today"]),
    )
    return {
        "events": len(today_events),
        "chores": len(morning_categorized["overdue"])
        + len(morning_categorized["due_today"]),
        "emailed": True,
    }


async def _generate_morning_narrative(
    app: App,
    today: date,
    today_events: list[dict],
    today_analysis: dict,
    week_by_day: dict[str, list[dict]],
    week_analysis: dict,
    tz: ZoneInfo,
    categorized_chores: dict,
    assignee_names: dict[str, str],
) -> MorningNarrative:
    agent = _make_narrative_agent(
        app.litellm_client, MorningNarrative, "local-default"
    )
    events_block = _render_events_plain(today_events, tz)
    chores_block = _render_chores_for_prompt(categorized_chores, assignee_names)
    week_block = render_week_ahead_bullets_plain(week_by_day, tz)
    prompt = _MORNING_PROMPT.format(
        today_label=today.strftime("%A, %B %-d, %Y"),
        tz=tz.key,
        event_count=today_analysis["event_count"],
        events_block=events_block,
        total_minutes=today_analysis["total_minutes"],
        first_start=today_analysis["first_start"] or "—",
        last_end=today_analysis["last_end"] or "—",
        open_blocks=today_analysis["open_blocks"] or "(none)",
        tight_transitions=today_analysis["tight_transitions"] or "(none)",
        chores_block=chores_block,
        week_block=week_block,
        week_event_count=week_analysis["event_count"],
        busiest_day=week_analysis["busiest_day"] or "—",
        lightest_day=week_analysis["lightest_day"] or "—",
    )
    result = await agent.run(prompt)
    return result.output


# Reference unused-by-design imports so they're not flagged.
_ = Any
