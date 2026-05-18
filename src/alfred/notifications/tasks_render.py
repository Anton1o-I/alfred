"""Rendering primitives for the tasks agent's outbound mail.

Two surfaces live here:

1. Daily-briefing chore section (`categorize_statuses`, `render_chores_*`,
   `shame_assignees`) — composed alongside the calendar event sections.
2. Per-reply card (`ReplyPayload` + `render_reply_*`) — the response that
   goes back to the user after a create/complete/delete/update/list/etc.
   Uses a definition-table layout for actions with structured details
   and a plain prose card for clarifications and confirmations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from typing import Any

_FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)


def categorize_statuses(statuses: list[Any]) -> dict[str, list[Any]]:
    """Split chore statuses into overdue / due-today / due-tomorrow / later.

    Inputs are ChoreStatus dataclass instances from ChoreStore.status_for_all_active.
    """
    out: dict[str, list[Any]] = {
        "overdue": [],
        "due_today": [],
        "due_tomorrow": [],
        "later": [],
    }
    for s in statuses:
        if s.overdue_days > 0:
            out["overdue"].append(s)
        elif s.overdue_days == 0:
            out["due_today"].append(s)
        elif s.overdue_days == -1:
            out["due_tomorrow"].append(s)
        else:
            out["later"].append(s)
    out["overdue"].sort(key=lambda s: -s.overdue_days)
    return out


def shame_assignees(statuses: list[Any]) -> set[str]:
    """Return the set of assignees whose chores are past shame_after_days.

    'household' is intentionally excluded — shame only applies to chores
    explicitly assigned to a person.
    """
    return {
        s.chore.assignee
        for s in statuses
        if s.overdue_days >= s.chore.shame_after_days
        and s.chore.assignee != "household"
    }


def _display_assignee(
    assignee: str, assignee_names: dict[str, str] | None
) -> str:
    """Resolve a stored user_id token to a display name when one is configured."""
    if not assignee_names:
        return assignee
    return assignee_names.get(assignee, assignee)


def render_chores_plain(
    categorized: dict[str, list[Any]],
    assignee_names: dict[str, str] | None = None,
) -> str:
    """Plain-text section showing overdue / today / tomorrow chores.

    `assignee_names` maps stored user_id tokens ("primary", "secondary",
    "household") to display names sourced from env vars. Falls back to
    the raw token if no mapping exists.
    """
    if not (categorized["overdue"] or categorized["due_today"] or categorized["due_tomorrow"]):
        return "(no chores due in the next day)"
    lines: list[str] = []
    if categorized["overdue"]:
        lines.append("OVERDUE")
        for s in categorized["overdue"]:
            who = _display_assignee(s.chore.assignee, assignee_names)
            lines.append(
                f"  ⚠  {s.chore.title} ({who}) — {s.overdue_days}d overdue"
            )
    if categorized["due_today"]:
        if lines:
            lines.append("")
        lines.append("DUE TODAY")
        for s in categorized["due_today"]:
            who = _display_assignee(s.chore.assignee, assignee_names)
            lines.append(f"  •  {s.chore.title} ({who})")
    if categorized["due_tomorrow"]:
        if lines:
            lines.append("")
        lines.append("DUE TOMORROW")
        for s in categorized["due_tomorrow"]:
            who = _display_assignee(s.chore.assignee, assignee_names)
            lines.append(f"  •  {s.chore.title} ({who})")
    return "\n".join(lines)


def render_chores_html(
    categorized: dict[str, list[Any]],
    assignee_names: dict[str, str] | None = None,
) -> str:
    """Inline-styled HTML chore section for the briefing email."""
    if not (categorized["overdue"] or categorized["due_today"] or categorized["due_tomorrow"]):
        return (
            '<p style="font-style: italic; color: #86868b;">'
            "(no chores due in the next day)</p>"
        )

    row_style = (
        "padding: 6px 0; border-bottom: 1px solid #f0f0f3; "
        "font-size: 15px; line-height: 1.4;"
    )
    overdue_marker = (
        "display: inline-block; min-width: 90px; color: #b91c1c; "
        "font-weight: 600; font-variant-numeric: tabular-nums;"
    )
    due_marker = (
        "display: inline-block; min-width: 90px; color: #86868b; "
        "font-variant-numeric: tabular-nums;"
    )
    title_overdue = "color: #111; font-weight: 600;"
    title_normal = "color: #111;"
    assignee_style = "color: #86868b; font-size: 13px;"
    group_h_style = (
        "margin: 14px 0 8px; font-size: 11px; font-weight: 600; "
        "letter-spacing: 0.08em; text-transform: uppercase; color: #6e6e73;"
    )
    overdue_h_style = group_h_style + " color: #b91c1c;"

    parts: list[str] = []
    if categorized["overdue"]:
        parts.append(f'<h3 style="{overdue_h_style}">Overdue</h3>')
        for s in categorized["overdue"]:
            who = _display_assignee(s.chore.assignee, assignee_names)
            parts.append(
                f'<div style="{row_style}">'
                f'<span style="{overdue_marker}">{s.overdue_days}d late</span>'
                f'<span style="{title_overdue}">{escape(s.chore.title)}</span>'
                f' <span style="{assignee_style}">· {escape(who)}</span>'
                "</div>"
            )
    if categorized["due_today"]:
        parts.append(f'<h3 style="{group_h_style}">Due today</h3>')
        for s in categorized["due_today"]:
            who = _display_assignee(s.chore.assignee, assignee_names)
            parts.append(
                f'<div style="{row_style}">'
                f'<span style="{due_marker}">today</span>'
                f'<span style="{title_normal}">{escape(s.chore.title)}</span>'
                f' <span style="{assignee_style}">· {escape(who)}</span>'
                "</div>"
            )
    if categorized["due_tomorrow"]:
        parts.append(f'<h3 style="{group_h_style}">Due tomorrow</h3>')
        for s in categorized["due_tomorrow"]:
            who = _display_assignee(s.chore.assignee, assignee_names)
            parts.append(
                f'<div style="{row_style}">'
                f'<span style="{due_marker}">tomorrow</span>'
                f'<span style="{title_normal}">{escape(s.chore.title)}</span>'
                f' <span style="{assignee_style}">· {escape(who)}</span>'
                "</div>"
            )
    return "".join(parts)


# ── Per-reply payload + renderer ────────────────────────────────────────────


@dataclass
class ReplyPayload:
    """Structured content for a single agent reply.

    Renderers (`render_reply_plain` / `render_reply_html`) consume this and
    produce the strings the inbox poller actually sends. Keeping the
    structure separate from the rendering means the workflow handlers stay
    rendering-agnostic and the layout can evolve without touching them.

    Fields:
      header: short banner (HTML title + plain-text heading). Required.
      title:  primary subject of the action — typically the chore title.
              Rendered as a prominent line above the fields table. Optional.
      fields: ordered (label, value) pairs rendered as a definition table.
              Empty list = no table.
      body:   free-form prose below the table (clarifications, asks).
              Used when the reply isn't structured enough for fields.
      footer: optional small-text footnote (e.g. "id: trash · reply 'undo'").
    """

    header: str
    title: str | None = None
    fields: list[tuple[str, str]] = field(default_factory=list)
    body: str | None = None
    footer: str | None = None


def render_reply_plain(payload: ReplyPayload) -> str:
    """Render a ReplyPayload as plain text suitable for the email body."""
    lines: list[str] = []
    lines.append(payload.header)
    lines.append("─" * max(len(payload.header), 14))
    lines.append("")

    if payload.title:
        lines.append(payload.title)
        lines.append("")

    if payload.fields:
        label_width = max(len(label) for label, _ in payload.fields)
        for label, value in payload.fields:
            lines.append(f"  {label:<{label_width}}    {value}")
        lines.append("")

    if payload.body:
        lines.append(payload.body.strip())
        lines.append("")

    if payload.footer:
        lines.append(payload.footer)

    return "\n".join(lines).rstrip() + "\n"


_REPLY_BODY_STYLE = (
    "margin: 0; padding: 0; background: #f5f5f7; "
    f"font-family: {_FONT_STACK}; color: #1d1d1f;"
)
_REPLY_CONTAINER_STYLE = (
    "max-width: 560px; margin: 0 auto; padding: 32px 24px; background: #ffffff;"
)
_REPLY_HEADER_STYLE = (
    "margin: 0 0 20px; font-size: 14px; font-weight: 600; "
    "letter-spacing: 0.04em; text-transform: uppercase; color: #34a853;"
)
_REPLY_TITLE_STYLE = (
    "margin: 0 0 20px; font-size: 22px; font-weight: 600; "
    "letter-spacing: -0.01em; color: #111;"
)
_REPLY_TABLE_STYLE = "width: 100%; border-collapse: collapse; margin: 0 0 16px;"
_REPLY_LABEL_STYLE = (
    "padding: 8px 16px 8px 0; vertical-align: top; font-size: 13px; "
    "color: #86868b; font-weight: 500; white-space: nowrap;"
)
_REPLY_VALUE_STYLE = (
    "padding: 8px 0; vertical-align: top; font-size: 15px; color: #1d1d1f;"
)
_REPLY_BODY_TEXT_STYLE = (
    "font-size: 15px; line-height: 1.55; color: #1d1d1f; margin: 0 0 16px;"
)
_REPLY_FOOTER_STYLE = (
    "font-size: 12px; color: #86868b; margin: 12px 0 0;"
)


def render_reply_html(payload: ReplyPayload) -> str:
    """Render a ReplyPayload as inline-styled HTML for the email body."""
    parts: list[str] = [
        '<!doctype html><html><head><meta charset="utf-8"></head>',
        f'<body style="{_REPLY_BODY_STYLE}">',
        f'<div style="{_REPLY_CONTAINER_STYLE}">',
        f'<p style="{_REPLY_HEADER_STYLE}">{escape(payload.header)}</p>',
    ]
    if payload.title:
        parts.append(f'<h1 style="{_REPLY_TITLE_STYLE}">{escape(payload.title)}</h1>')
    if payload.fields:
        parts.append(f'<table style="{_REPLY_TABLE_STYLE}">')
        for label, value in payload.fields:
            parts.append(
                "<tr>"
                f'<td style="{_REPLY_LABEL_STYLE}">{escape(label)}</td>'
                f'<td style="{_REPLY_VALUE_STYLE}">{escape(value)}</td>'
                "</tr>"
            )
        parts.append("</table>")
    if payload.body:
        body_html = escape(payload.body.strip()).replace("\n", "<br>")
        parts.append(f'<p style="{_REPLY_BODY_TEXT_STYLE}">{body_html}</p>')
    if payload.footer:
        parts.append(f'<p style="{_REPLY_FOOTER_STYLE}">{escape(payload.footer)}</p>')
    parts.append("</div></body></html>")
    return "".join(parts)


# Reference _FONT_STACK so module-level constants are discoverable for
# future composer refactor; intentionally unused otherwise.
_ = _FONT_STACK
