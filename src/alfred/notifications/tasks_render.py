"""Render chore status sections for the unified daily briefing.

Produces plain-text and HTML blocks. The briefing composes these alongside
the calendar event sections so users get one morning view of both.
"""

from __future__ import annotations

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


def render_chores_plain(categorized: dict[str, list[Any]]) -> str:
    """Plain-text section showing overdue / today / tomorrow chores."""
    if not (categorized["overdue"] or categorized["due_today"] or categorized["due_tomorrow"]):
        return "(no chores due in the next day)"
    lines: list[str] = []
    if categorized["overdue"]:
        lines.append("OVERDUE")
        for s in categorized["overdue"]:
            lines.append(
                f"  ⚠  {s.chore.title} ({s.chore.assignee}) — "
                f"{s.overdue_days}d overdue"
            )
    if categorized["due_today"]:
        if lines:
            lines.append("")
        lines.append("DUE TODAY")
        for s in categorized["due_today"]:
            lines.append(f"  •  {s.chore.title} ({s.chore.assignee})")
    if categorized["due_tomorrow"]:
        if lines:
            lines.append("")
        lines.append("DUE TOMORROW")
        for s in categorized["due_tomorrow"]:
            lines.append(f"  •  {s.chore.title} ({s.chore.assignee})")
    return "\n".join(lines)


def render_chores_html(categorized: dict[str, list[Any]]) -> str:
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
            parts.append(
                f'<div style="{row_style}">'
                f'<span style="{overdue_marker}">{s.overdue_days}d late</span>'
                f'<span style="{title_overdue}">{escape(s.chore.title)}</span>'
                f' <span style="{assignee_style}">· {escape(s.chore.assignee)}</span>'
                "</div>"
            )
    if categorized["due_today"]:
        parts.append(f'<h3 style="{group_h_style}">Due today</h3>')
        for s in categorized["due_today"]:
            parts.append(
                f'<div style="{row_style}">'
                f'<span style="{due_marker}">today</span>'
                f'<span style="{title_normal}">{escape(s.chore.title)}</span>'
                f' <span style="{assignee_style}">· {escape(s.chore.assignee)}</span>'
                "</div>"
            )
    if categorized["due_tomorrow"]:
        parts.append(f'<h3 style="{group_h_style}">Due tomorrow</h3>')
        for s in categorized["due_tomorrow"]:
            parts.append(
                f'<div style="{row_style}">'
                f'<span style="{due_marker}">tomorrow</span>'
                f'<span style="{title_normal}">{escape(s.chore.title)}</span>'
                f' <span style="{assignee_style}">· {escape(s.chore.assignee)}</span>'
                "</div>"
            )
    return "".join(parts)


# Reference _FONT_STACK so module-level constants are discoverable for
# future composer refactor; intentionally unused otherwise.
_ = _FONT_STACK
