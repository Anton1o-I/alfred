"""Daily next-day briefing + Sunday weekly preview routine.

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

Composes calendar event data with tasks chore data + shame tier routing. Lives in
routines/ because it spans multiple agents; the calendar agent only provides the
event-shape analysis helpers it imports.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal
from zoneinfo import ZoneInfo

import structlog
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from alfred.agents.calendar.briefing import (
    _fetch_events_for_window,
    _fmt_short_date,
    _render_events_html,
    _render_events_plain,
    _render_weekly_events_html,
    _render_weekly_events_plain,
    analyze_day,
    analyze_week,
)

if TYPE_CHECKING:
    from alfred.app import App

log = structlog.get_logger()


# ── LLM output schemas ────────────────────────────────────────────────────


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
    lead_section: Literal["events", "chores"] = "events",
) -> tuple[str, str]:
    """Shared render shape for daily + weekly.

    `lead_section` controls whether Events or Chores appears first. Weekly
    previews don't have chores; they always pass lead_section='events'.
    """
    has_chores = bool(chores_section_plain or chores_section_html)
    chores_lead = lead_section == "chores" and has_chores

    # Plain — section order driven by lead_section. No horizontal-rule
    # dividers under headers: Gmail mobile's "Show trimmed content"
    # heuristic detects any line of repeated dash-like characters in the
    # plain part as a signature/quote boundary and collapses everything
    # below. We use uppercase-styled headers and blank-line spacing to
    # carry the visual hierarchy in plain text instead.
    plain_parts = [
        narrative.greeting.strip(),
        "",
        narrative.summary.strip(),
    ]
    sections_plain: list[tuple[str, str]] = []
    if chores_lead:
        if chores_section_plain:
            sections_plain.append(("Chores", chores_section_plain))
        sections_plain.append(("Events", events_section_plain))
    else:
        sections_plain.append(("Events", events_section_plain))
        if chores_section_plain:
            sections_plain.append(("Chores", chores_section_plain))
    for title, body in sections_plain:
        plain_parts.append("")
        plain_parts.append(title.upper())
        plain_parts.append("")
        plain_parts.append(body)
    if observations:
        plain_parts.append("")
        plain_parts.append("OBSERVATIONS")
        plain_parts.append("")
        for obs in observations:
            plain_parts.append(f"• {obs}")
    plain = "\n".join(plain_parts).rstrip() + "\n"

    # HTML — same section ordering.
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
    # NOTE: no border-top divider. Gmail mobile's "Show trimmed content"
    # heuristic treats horizontal-rule patterns as a signature/quote
    # boundary and collapses everything below — losing the Events / Chores
    # / Observations sections entirely. Use generous margin + small-caps
    # styling for visual separation instead.
    section_h_style = (
        "margin: 32px 0 12px; font-size: 11px; font-weight: 600; "
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
    ]
    sections_html: list[tuple[str, str]] = []
    if chores_lead:
        if chores_section_html:
            sections_html.append(("Chores", chores_section_html))
        sections_html.append(("Events", events_section_html))
    else:
        sections_html.append(("Events", events_section_html))
        if chores_section_html:
            sections_html.append(("Chores", chores_section_html))
    for title, body in sections_html:
        parts.append(f'<h2 style="{section_h_style}">{title}</h2>')
        parts.append(body)
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
    roast_lines = await _generate_chore_roasts(
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
        narrative=narrative,
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


async def _generate_chore_roasts(
    app: App,
    overdue: list[Any],
    tier_table: Any,
) -> dict[str, str]:
    """Build the specialist inputs and call the roaster for shame-tier chores.

    Returns `{chore_id: roast}` for every chore the LLM successfully
    roasted. Returns `{}` on any failure (the renderers fall back per
    chore to the static `fallback_label`). Skipped entirely when no
    overdue chore qualifies for a shame tier (tier ≥ 1).
    """
    from alfred.agents.tasks.renderers import shame_tier
    from alfred.specialists.shame.specialist import (
        ChoreRoastInput,
        generate_roasts,
    )

    inputs: list[ChoreRoastInput] = []
    for s in overdue:
        if s.chore.assignee == "household":
            # Household chores skip shame; static label is correct here.
            continue
        tier = shame_tier(
            s.overdue_days,
            shame_after_days=s.chore.shame_after_days,
            table=tier_table,
        )
        if tier < 1:
            continue
        chore_id = getattr(s.chore, "id", None)
        if not chore_id:
            continue
        inputs.append(
            ChoreRoastInput(
                chore_id=str(chore_id),
                title=s.chore.title,
                assignee=s.chore.assignee,
                tier=tier,
                days=s.overdue_days,
            )
        )
    if not inputs:
        return {}
    try:
        return await generate_roasts(inputs, app.litellm_client)
    except Exception as e:  # noqa: BLE001
        log.warning("shame_roast_generation_failed", error=str(e))
        return {}


async def _fetch_chore_data(app: App, now_local: datetime, tz: ZoneInfo) -> dict:
    """Pull chore statuses from the tasks agent's store (if registered).

    Returns categorized status buckets + the list of user_ids that should be
    force-CC'd because at least one of their chores is past `shame_after_days`.
    Returns empty data when the tasks agent isn't enabled.
    """
    from alfred.agents.tasks.renderers import (
        ShameTierTable,
        _should_split_for_shame,
        categorize_statuses,
        shame_assignees,
        shame_tier,
    )

    empty = {
        "categorized": {"overdue": [], "due_today": [], "due_tomorrow": [], "later": []},
        "shame_user_ids": [],
        "statuses": [],
        "split_for_shame": False,
        "shame_overdue": [],
        "non_shame_overdue": [],
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

    table = ShameTierTable(app.settings.notifications.shame_tiers)
    split = _should_split_for_shame(statuses, table)
    shame_overdue: list[Any] = []
    non_shame_overdue: list[Any] = []
    if split:
        for s in categorized["overdue"]:
            if s.chore.assignee == "household":
                non_shame_overdue.append(s)
                continue
            tier = shame_tier(
                s.overdue_days,
                shame_after_days=s.chore.shame_after_days,
                table=table,
            )
            if tier >= 2:
                shame_overdue.append(s)
            else:
                non_shame_overdue.append(s)

    return {
        "categorized": categorized,
        "shame_user_ids": shame_ids,
        "statuses": statuses,
        "split_for_shame": split,
        "shame_overdue": shame_overdue,
        "non_shame_overdue": non_shame_overdue,
    }


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


def _build_assignee_names(app: App) -> dict[str, str]:
    """user_id -> display name map sourced from env vars via recipient configs.

    Mirrors the lookup in `app.py` for the tasks agent — keeps personal
    names out of tracked config and lets us show "Alex" instead of
    "primary" in chore lines.
    """
    import os

    out: dict[str, str] = {}
    for r in app.settings.notifications.recipients:
        if r.name_env:
            name = os.environ.get(r.name_env, "").strip()
            if name:
                out[r.user_id] = name
    return out


def _render_chores_for_prompt(
    categorized: dict | None,
    assignee_names: dict[str, str] | None = None,
) -> str:
    """Plain-text chore listing fed to the LLM so it can reason about specifics.

    Format mirrors the rendered email so the LLM and the user see the same facts.
    Resolving assignees here means the LLM-generated summary references real
    names rather than the internal user_id tokens.
    """
    if not categorized:
        return "(no chores tracked)"
    buckets = (
        ("OVERDUE", categorized.get("overdue", [])),
        ("DUE TODAY", categorized.get("due_today", [])),
        ("DUE TOMORROW", categorized.get("due_tomorrow", [])),
    )
    lines: list[str] = []
    for label, statuses in buckets:
        if not statuses:
            continue
        lines.append(label)
        for s in statuses:
            who = (
                assignee_names.get(s.chore.assignee, s.chore.assignee)
                if assignee_names
                else s.chore.assignee
            )
            suffix = f" — {s.overdue_days}d overdue" if s.overdue_days > 0 else ""
            lines.append(f"  - {s.chore.title} ({who}){suffix}")
    return "\n".join(lines) if lines else "(no chores due in the next day)"


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
