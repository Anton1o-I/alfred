"""Daily next-day briefing routine.

Linear pipeline (no LangGraph — no branching):

  fetch_events (CalDAV) + fetch_chore_data
       │
       ▼
  analyze_day (pure code — gaps, totals, tight transitions)
       │
       ▼
  generate_daily_narrative (LLM specialist, typed DailyNarrative)
       │
       ▼
  render (shared email frame from routines/_common)
       │
       ▼
  send + optional split "Alfred · Disappointed" shame email

Composes calendar event data with tasks chore data + shame tier routing.
Lives in routines/ because it spans multiple agents; the calendar agent
only provides the event-shape analysis helpers it imports. Shared frame
+ dispatcher live in routines/_common; weekly preview lives in
routines/weekly_preview.
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
)
from alfred.routines._common import (
    _FONT_STACK,
    _build_assignee_names,
    _calendar_timezone,
    _escape,
    _make_narrative_agent,
    _render_chores_for_prompt,
    _render_email,
    _send,
    fetch_chore_data,
    generate_chore_roasts,
)

if TYPE_CHECKING:
    from alfred.app import App

log = structlog.get_logger()


# ── LLM output schema ─────────────────────────────────────────────────────


class DailyNarrative(BaseModel):
    lead_section: Literal["events", "chores"] = Field(
        description=(
            "Which section should appear FIRST in the email. Use 'chores' "
            "when: (a) the calendar has no events, OR (b) there is at "
            "least one overdue chore, OR (c) the events are light/routine "
            "while there are multiple chores due today or tomorrow. "
            "Use 'events' otherwise. This is a judgment call — pick the "
            "section the reader most needs to act on."
        )
    )
    greeting: str = Field(
        description=(
            "Short opener, 5-10 words, friendly but understated. "
            "Examples: 'Heads up for tomorrow', 'Quick rundown', 'Light Monday ahead', "
            "'Chores to tackle tomorrow'. Match the lead_section in tone — if chores "
            "lead, the greeting should acknowledge that, not pretend it's about events."
        )
    )
    summary: str = Field(
        description=(
            "1-2 sentences narrating what tomorrow looks like. Mention the lead "
            "section's content concretely (e.g. 'two overdue chores' or 'three meetings, "
            "open afternoon'). Facts only — never invent events, names, or chores not in "
            "the data. If the day is genuinely uneventful, say so plainly in ONE sentence."
        )
    )
    observations: list[str] = Field(
        default_factory=list,
        description=(
            "0-3 short bullets, each ≤80 chars, ONLY for things genuinely worth "
            "flagging: a 3+ hour open block, a tight transition, no lunch, an "
            "overdue chore worth a nudge. Return an EMPTY LIST if nothing is "
            "notable — do not pad. Never restate what's already in the summary."
        ),
    )


# ── Prompt ────────────────────────────────────────────────────────────────


_DAILY_PROMPT = (
    "You are writing a brief, friendly heads-up about tomorrow for one person.\n"
    "Two sections may appear: calendar events and household chores. Your job is to\n"
    "(a) decide which section LEADS the email, and (b) write a short narrative.\n"
    "\n"
    "Tomorrow: {label}\n"
    "Timezone: {tz}\n"
    "\n"
    "Events ({event_count}):\n"
    "{events_block}\n"
    "\n"
    "Event analysis (computed):\n"
    "- Total scheduled time: {total_minutes} minutes\n"
    "- First event starts at: {first_start}\n"
    "- Last event ends at: {last_end}\n"
    "- Open blocks (>= 60 min between events): {open_blocks}\n"
    "- Tight transitions (< 15 min gap): {tight_transitions}\n"
    "\n"
    "Chores:\n"
    "{chores_block}\n"
    "\n"
    "How to pick lead_section:\n"
    "- If there are no events, lead with 'chores'.\n"
    "- If any chore is overdue, lead with 'chores' (the user needs to act).\n"
    "- If events are sparse (e.g. 1 short event) and multiple chores are due today\n"
    "  or tomorrow, lead with 'chores'.\n"
    "- Otherwise lead with 'events'.\n"
    "\n"
    "Tone & content rules:\n"
    "- Friendly but understated. No forced cheer ('Have a great day!'). No emoji.\n"
    "- Comment on what IS on the calendar. Do not invent things to worry about\n"
    "  (no 'remember to eat', no 'don't forget lunch', no filler advice).\n"
    "- When the day is light, ONE short sentence is enough — e.g. 'Tomorrow looks\n"
    "  clear, nothing on the calendar.' Do not pad with observations.\n"
    "- When events exist, name them specifically (title + time) in the summary,\n"
    "  don't just say 'you have a few meetings'.\n"
    "- Times use 12-hour with am/pm (lowercase ok).\n"
    "\n"
    "Chore status precision — read carefully:\n"
    "- Items under OVERDUE are overdue. Items under DUE TODAY are NOT overdue;\n"
    "  they are due today. Items under DUE TOMORROW are not yet due.\n"
    "- Never call a 'due today' or 'due tomorrow' chore 'overdue'.\n"
    "\n"
    "Observations rules — read carefully:\n"
    "- Observations must add NEW information not already in the Events or\n"
    "  Chores sections. Do NOT restate or re-list chores or events.\n"
    "- Valid observations are about the SHAPE of the day: tight transitions\n"
    "  between specific events, an unusually long open block, the day starting\n"
    "  unusually early or running unusually late.\n"
    "- Listing a chore that already appears in the Chores section is NOT a\n"
    "  valid observation. Do not do it.\n"
    "- If nothing structurally notable, return an EMPTY observations list.\n"
    "  An empty list is the correct answer most of the time.\n"
    "\n"
    "Never invent events, chores, names, or context not in the data above.\n"
    "\n"
    "Return a typed DailyNarrative."
)


# ── Routine ───────────────────────────────────────────────────────────────


async def run_daily_briefing(app: App) -> dict:
    """Fetch tomorrow's events + chore status and email a unified briefing."""
    timezone_name = _calendar_timezone(app)
    tz = ZoneInfo(timezone_name)
    now_local = datetime.now(tz)
    tomorrow = now_local.date() + timedelta(days=1)
    start = datetime.combine(tomorrow, datetime.min.time(), tzinfo=tz)
    end = start + timedelta(days=1)

    events = _fetch_events_for_window(app, start, end)
    chore_data = await fetch_chore_data(app, now_local, tz)
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
    assignee_names = _build_assignee_names(app)
    narrative = await _generate_daily_narrative(
        app,
        tomorrow,
        events,
        analysis,
        tz,
        chore_data["categorized"],
        assignee_names,
    )

    events_plain = _render_events_plain(events, tz)
    events_html = _render_events_html(events, tz)
    headline = f"TOMORROW · {tomorrow.strftime('%a, %B %-d')}"

    from alfred.agents.tasks.renderers import (
        ShameTierTable,
        render_chores_html,
        render_chores_plain,
    )

    tier_table = ShameTierTable(app.settings.notifications.shame_tiers)
    split_for_shame: bool = bool(chore_data.get("split_for_shame"))
    roast_lines = await generate_chore_roasts(
        app, chore_data["categorized"]["overdue"], tier_table
    )

    # Categorized chores for the *regular* briefing: when splitting, peel
    # tier-2+ overdue rows out so they only appear in the Disappointed mail.
    if split_for_shame:
        briefing_categorized = {
            **chore_data["categorized"],
            "overdue": chore_data["non_shame_overdue"],
        }
    else:
        briefing_categorized = chore_data["categorized"]

    briefing_has_chores = bool(
        briefing_categorized["overdue"]
        or briefing_categorized["due_today"]
        or briefing_categorized["due_tomorrow"]
    )

    chores_plain: str | None = None
    chores_html: str | None = None
    if briefing_has_chores:
        chores_plain = render_chores_plain(
            briefing_categorized,
            assignee_names,
            shame_tiers=tier_table,
            roast_lines=roast_lines,
        )
        chores_html = render_chores_html(
            briefing_categorized,
            assignee_names,
            shame_tiers=tier_table,
            roast_lines=roast_lines,
        )

    plain, html = _render_email(
        headline=headline,
        greeting=narrative.greeting,
        summary=narrative.summary,
        events_section_plain=events_plain,
        events_section_html=events_html,
        observations=narrative.observations,
        chores_section_plain=chores_plain,
        chores_section_html=chores_html,
        lead_section=narrative.lead_section,
    )
    subject = f"Tomorrow's schedule — {_fmt_short_date(tomorrow)}"
    await _send(app, subject, plain, html, force_user_ids=chore_data["shame_user_ids"])

    if split_for_shame:
        await _send_shame_email(
            app,
            tomorrow=tomorrow,
            shame_overdue=chore_data["shame_overdue"],
            assignee_names=assignee_names,
            tier_table=tier_table,
            force_user_ids=chore_data["shame_user_ids"],
            roast_lines=roast_lines,
        )
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


async def _generate_daily_narrative(
    app: App,
    day: date,
    events: list[dict],
    analysis: dict,
    tz: ZoneInfo,
    categorized_chores: dict | None = None,
    assignee_names: dict[str, str] | None = None,
) -> DailyNarrative:
    agent = _make_narrative_agent(app.litellm_client, DailyNarrative, "local-default")
    events_block = _render_events_plain(events, tz)
    chores_block = _render_chores_for_prompt(categorized_chores, assignee_names)
    prompt = _DAILY_PROMPT.format(
        label=day.strftime("%A, %B %-d, %Y"),
        tz=tz.key,
        event_count=analysis["event_count"],
        events_block=events_block,
        total_minutes=analysis["total_minutes"],
        first_start=analysis["first_start"] or "—",
        last_end=analysis["last_end"] or "—",
        open_blocks=analysis["open_blocks"] or "(none)",
        tight_transitions=analysis["tight_transitions"] or "(none)",
        chores_block=chores_block,
    )
    result = await agent.run(prompt)
    return result.output


async def _send_shame_email(
    app: App,
    *,
    tomorrow: date,
    shame_overdue: list[Any],
    assignee_names: dict[str, str],
    tier_table: Any,
    force_user_ids: list[str] | None,
    roast_lines: dict[str, str] | None = None,
) -> None:
    """Send the tier-2+ overdue chores as a separate "Disappointed" email.

    Composed and sent in addition to the regular briefing — never as a
    replacement. Reuses the chore-row renderers with the tier table so
    the visual treatment matches what users see inside the briefing's
    chore section, just with a different envelope and persona.
    """
    from alfred.agents.tasks.renderers import (
        render_chores_html,
        render_chores_plain,
    )
    from alfred.core.persona import Persona, resolve_persona
    from alfred.notifications.signature import (
        append_to_body,
        append_to_html,
        render_signature,
    )

    cfg = app.settings.notifications
    persona_name, persona_tagline = resolve_persona(
        Persona.TASKS_SHAME, cfg.personas
    )

    categorized = {
        "overdue": shame_overdue,
        "due_today": [],
        "due_tomorrow": [],
        "later": [],
    }
    chores_plain = render_chores_plain(
        categorized,
        assignee_names,
        shame_tiers=tier_table,
        roast_lines=roast_lines,
    )
    chores_html = render_chores_html(
        categorized,
        assignee_names,
        shame_tiers=tier_table,
        roast_lines=roast_lines,
    )

    headline = f"OVERDUE CHORES · {_fmt_short_date(tomorrow)}"
    intro_plain = (
        "Several chores have been pending long enough that they need attention "
        "today. Listed below with how long they've been waiting."
    )
    intro_html = _escape(intro_plain)

    body_style = (
        "margin: 0; padding: 0; background: #f5f5f7; "
        f"font-family: {_FONT_STACK}; color: #1d1d1f;"
    )
    container_style = (
        "max-width: 560px; margin: 0 auto; padding: 32px 24px; background: #ffffff;"
    )
    header_style = (
        "margin: 0 0 6px; font-size: 14px; font-weight: 600; "
        "letter-spacing: 0.04em; text-transform: uppercase; color: #b91c1c;"
    )
    intro_style = (
        "margin: 0 0 20px; font-size: 15px; line-height: 1.55; color: #1d1d1f;"
    )
    html_parts = [
        '<!doctype html><html><head><meta charset="utf-8"></head>',
        f'<body style="{body_style}">',
        f'<div style="{container_style}">',
        f'<p style="{header_style}">{headline}</p>',
        f'<p style="{intro_style}">{intro_html}</p>',
        chores_html,
        "</div></body></html>",
    ]
    html = "".join(html_parts)
    plain_parts = [headline, "", intro_plain, "", chores_plain]
    plain = "\n".join(plain_parts).rstrip() + "\n"

    sig_plain, sig_html = render_signature(persona_name, persona_tagline)
    plain_final = append_to_body(plain, sig_plain)
    html_final = append_to_html(html, sig_html)

    subject = f"Overdue chores — {_fmt_short_date(tomorrow)}"
    await app.notification_service.send_to_family(
        body=plain_final,
        subject=subject,
        html_body=html_final,
        agent_name="calendar",
        force_user_ids=force_user_ids,
        persona_override=Persona.TASKS_SHAME,
    )
