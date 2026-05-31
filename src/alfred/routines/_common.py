"""Shared helpers for the briefing routines.

Used by `routines/daily_briefing.py` and `routines/weekly_preview.py`:
the email frame (HTML + plain composer), the dispatcher (`_send`), the
LLM agent factory, and small app-level lookups (timezone, assignee
display names, chore prompt rendering).

The email frame intentionally takes `greeting` and `summary` as plain
strings rather than a narrative object so this module stays free of
narrative-type imports — daily and weekly each own their narrative
schemas locally.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from zoneinfo import ZoneInfo

import structlog
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    from alfred.app import App

log = structlog.get_logger()


_FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)


def _make_narrative_agent(litellm_client, output_type: type, model_name: str) -> Agent:
    model = OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(
            base_url=f"{litellm_client.base_url}/v1",
            api_key=litellm_client._api_key,  # noqa: SLF001
        ),
    )
    return Agent(model=model, output_type=output_type)


def _escape(s: str) -> str:
    from html import escape

    return escape(s or "")


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


def _render_email(
    *,
    headline: str,
    greeting: str,
    summary: str,
    events_section_plain: str,
    events_section_html: str,
    observations: list[str],
    chores_section_plain: str | None = None,
    chores_section_html: str | None = None,
    lead_section: Literal["events", "chores"] = "events",
) -> tuple[str, str]:
    """Shared render shape for daily + weekly briefings.

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
        greeting.strip(),
        "",
        summary.strip(),
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
        f'<h1 style="{greeting_style}">{_escape(greeting)}</h1>',
        f'<p style="{summary_style}">{_escape(summary)}</p>',
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


async def generate_chore_roasts(
    app: App,
    overdue: list[Any],
    tier_table: Any,
) -> dict[str, str]:
    """Build the specialist inputs and call the roaster for shame-tier chores.

    Returns ``{chore_id: roast}`` for every chore the LLM successfully
    roasted. Returns ``{}`` on any failure (the renderers fall back per
    chore to the static ``fallback_label``). Skipped entirely when no
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
            continue
        tier = shame_tier(s.overdue_days, table=tier_table)
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


async def fetch_chore_data(app: App, now_local: datetime, tz: ZoneInfo) -> dict:
    """Pull chore statuses from the tasks agent's store (if registered).

    Returns categorized status buckets + the list of user_ids that should
    be force-CC'd because at least one of their chores is overdue, plus
    the tier-2+ split for the optional "Disappointed" envelope.
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
        log.warning("briefing_chores_fetch_failed", error=str(e))
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
            tier = shame_tier(s.overdue_days, table=table)
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


def render_week_ahead_bullets_plain(
    by_day: dict[str, list[dict]], tz: ZoneInfo
) -> str:
    """One-line-per-day terse glance of the next 7 days (plain text).

    Each line: ``Mon Jun 2 — 3 events, first 9:00 am``. Empty days show
    ``— clear``. Used by the morning briefing's hybrid week-ahead block.
    """
    from datetime import date as date_cls

    lines: list[str] = []
    for day_iso in sorted(by_day.keys()):
        evs = by_day[day_iso]
        d = date_cls.fromisoformat(day_iso)
        label = d.strftime("%a %b %-d")
        if not evs:
            lines.append(f"  {label} — clear")
            continue
        evs_sorted = sorted(evs, key=lambda e: e["start_iso"])
        first = datetime.fromisoformat(evs_sorted[0]["start_iso"]).astimezone(tz)
        first_str = first.strftime("%-I:%M %p").lower()
        n = len(evs)
        word = "event" if n == 1 else "events"
        lines.append(f"  {label} — {n} {word}, first {first_str}")
    return "\n".join(lines) if lines else "  (no events on the calendar)"


def render_week_ahead_bullets_html(
    by_day: dict[str, list[dict]], tz: ZoneInfo
) -> str:
    """HTML version of the week-ahead glance — inline-styled list rows."""
    from datetime import date as date_cls
    from html import escape

    row_style = (
        "padding: 6px 0; border-bottom: 1px solid #f0f0f3; "
        "font-size: 14px; line-height: 1.4;"
    )
    day_style = (
        "display: inline-block; min-width: 110px; color: #6e6e73; "
        "font-variant-numeric: tabular-nums;"
    )
    detail_style = "color: #1d1d1f;"
    clear_style = "color: #86868b; font-style: italic;"

    parts: list[str] = []
    for day_iso in sorted(by_day.keys()):
        evs = by_day[day_iso]
        d = date_cls.fromisoformat(day_iso)
        label = d.strftime("%a %b %-d")
        if not evs:
            parts.append(
                f'<div style="{row_style}">'
                f'<span style="{day_style}">{escape(label)}</span>'
                f'<span style="{clear_style}">clear</span></div>'
            )
            continue
        evs_sorted = sorted(evs, key=lambda e: e["start_iso"])
        first = datetime.fromisoformat(evs_sorted[0]["start_iso"]).astimezone(tz)
        first_str = first.strftime("%-I:%M %p").lower()
        n = len(evs)
        word = "event" if n == 1 else "events"
        detail = f"{n} {word}, first {first_str}"
        parts.append(
            f'<div style="{row_style}">'
            f'<span style="{day_style}">{escape(label)}</span>'
            f'<span style="{detail_style}">{escape(detail)}</span></div>'
        )
    return "".join(parts) or '<p style="color:#86868b;">(no events on the calendar)</p>'


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
