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

from typing import TYPE_CHECKING, Literal

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    from alfred.app import App


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
