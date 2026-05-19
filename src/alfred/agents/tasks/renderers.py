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

# Sentinel tier value for chores that have not yet crossed their personal
# shame_after_days threshold. Used by `shame_tier` and `_should_split_for_shame`.
TIER_NONE: int = 0


@dataclass(frozen=True)
class ShameTierCopy:
    """Resolved copy for one shame tier — derived from notifications.yaml.

    `fallback_label` is the static "{days}d overdue" string used when the
    roast specialist has nothing to say for a given chore (LLM error,
    missing chore_id in response, empty roast). When a roast IS available
    the renderer uses it instead and the fallback never appears.
    """

    tier: int
    min_days: int
    max_days: int | None  # None = open-ended top tier
    fallback_label: str  # may contain "{days}" placeholder
    row_prefix: str

    def format_fallback(self, overdue_days: int) -> str:
        return self.fallback_label.format(days=overdue_days)


class ShameTierTable:
    """Tier-lookup table built from `NotificationConfig.shame_tiers`.

    Resolution rule: `tier_for(overdue_days)` returns the first tier whose
    `[min_days, max_days]` range contains `overdue_days`. Tiers are sorted
    by `min_days` at construction so callers can rely on stable ordering.
    """

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        rows: list[ShameTierCopy] = []
        for e in entries:
            rows.append(
                ShameTierCopy(
                    tier=int(e["tier"]),
                    min_days=int(e["min_days"]),
                    max_days=None if e.get("max_days") is None else int(e["max_days"]),
                    fallback_label=str(e.get("fallback_label", "{days}d overdue")),
                    row_prefix=str(e.get("row_prefix", "")),
                )
            )
        rows.sort(key=lambda r: r.min_days)
        self._rows = rows

    def tier_for(self, overdue_days: int) -> int:
        """Return the tier (1..N) for an overdue_days value, or TIER_NONE."""
        for row in self._rows:
            if overdue_days < row.min_days:
                continue
            if row.max_days is not None and overdue_days > row.max_days:
                continue
            return row.tier
        return TIER_NONE

    def copy_for_tier(self, tier: int) -> ShameTierCopy | None:
        for row in self._rows:
            if row.tier == tier:
                return row
        return None

    def __bool__(self) -> bool:
        return bool(self._rows)


def shame_tier(
    overdue_days: int,
    *,
    shame_after_days: int,
    table: ShameTierTable | None,
) -> int:
    """Return the shame tier (1..N) for a chore, or `TIER_NONE`.

    A chore qualifies for tiering only once it has crossed its own
    `shame_after_days` threshold. The tier itself is selected purely
    from `overdue_days` via the configured `ShameTierTable`. Returns
    `TIER_NONE` if no table is configured or no tier range matches.
    """
    if table is None:
        return TIER_NONE
    if overdue_days < shame_after_days:
        return TIER_NONE
    return table.tier_for(overdue_days)


def _should_split_for_shame(
    statuses: list[Any], table: ShameTierTable | None
) -> bool:
    """Return True iff at least one non-household chore is at tier 2 or higher.

    Used by the daily briefing to decide whether to peel the shame chores
    out into a separate "Alfred · Disappointed" email. Isolated as a
    single predicate so the splitting decision stays one branch, not an
    if-tree across multiple call sites.
    """
    if table is None:
        return False
    for s in statuses:
        if s.chore.assignee == "household":
            continue
        tier = shame_tier(
            s.overdue_days,
            shame_after_days=s.chore.shame_after_days,
            table=table,
        )
        if tier >= 2:
            return True
    return False

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


def _bucket_overdue_by_tier(
    overdue: list[Any], table: ShameTierTable
) -> dict[int, list[Any]]:
    """Group overdue statuses by their resolved shame tier.

    Statuses that don't qualify for a tier (sub-`shame_after_days`) land
    under `TIER_NONE`. Within each bucket, ordering follows the input
    (which `categorize_statuses` already sorts by descending overdue_days).
    """
    out: dict[int, list[Any]] = {}
    for s in overdue:
        if s.chore.assignee == "household":
            tier = TIER_NONE
        else:
            tier = shame_tier(
                s.overdue_days,
                shame_after_days=s.chore.shame_after_days,
                table=table,
            )
        out.setdefault(tier, []).append(s)
    return out


def _label_for(
    status: Any,
    copy: ShameTierCopy | None,
    roast_lines: dict[str, str] | None,
) -> str:
    """Resolve the label for one overdue row.

    Priority:
      1. Roast line for this chore_id (if provided).
      2. Tier's `fallback_label` formatted with overdue_days.
      3. Hardcoded "{days}d overdue" if no tier copy.
    """
    chore_id = getattr(status.chore, "id", None)
    if roast_lines and chore_id and chore_id in roast_lines:
        return roast_lines[chore_id]
    if copy is not None:
        return copy.format_fallback(status.overdue_days)
    return f"{status.overdue_days}d overdue"


def _sorted_overdue_with_tier(
    overdue: list[Any], table: ShameTierTable
) -> list[tuple[int, Any]]:
    """Return [(tier, status), …] sorted tier-desc, then days-desc.

    Section headers were removed in favor of letting the row prefix
    (`·` / `!` / `!!`) plus the per-chore roast carry severity. Sorting
    most-severe-first keeps the visual hierarchy intact.
    """
    buckets = _bucket_overdue_by_tier(overdue, table)
    flat: list[tuple[int, Any]] = []
    for tier in sorted(buckets.keys(), reverse=True):
        rows = sorted(buckets[tier], key=lambda s: -s.overdue_days)
        for s in rows:
            flat.append((tier, s))
    return flat


def render_chores_plain(
    categorized: dict[str, list[Any]],
    assignee_names: dict[str, str] | None = None,
    shame_tiers: ShameTierTable | None = None,
    roast_lines: dict[str, str] | None = None,
) -> str:
    """Plain-text section showing overdue / today / tomorrow chores.

    `assignee_names` maps stored user_id tokens ("primary", "secondary",
    "household") to display names sourced from env vars. Falls back to
    the raw token if no mapping exists.

    `roast_lines` maps `chore.id` → LLM-generated roast string. When a
    chore_id is present its roast replaces the static fallback label;
    when absent the renderer uses the tier's `fallback_label`.
    """
    if not (categorized["overdue"] or categorized["due_today"] or categorized["due_tomorrow"]):
        return "(no chores due in the next day)"
    lines: list[str] = []
    if categorized["overdue"]:
        if shame_tiers:
            for tier, s in _sorted_overdue_with_tier(
                categorized["overdue"], shame_tiers
            ):
                copy = shame_tiers.copy_for_tier(tier)
                prefix = copy.row_prefix if copy is not None else "⚠"
                who = _display_assignee(s.chore.assignee, assignee_names)
                label = _label_for(s, copy, roast_lines)
                lines.append(
                    f"  {prefix}  {s.chore.title} ({who}) — {label}"
                )
        else:
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
    shame_tiers: ShameTierTable | None = None,
    roast_lines: dict[str, str] | None = None,
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

    # Per-tier accent colors for the marker. Lower tiers stay neutral;
    # tier 2/3 get progressively redder to match the section header.
    tier_marker_colors = {
        1: "#86868b",
        2: "#b91c1c",
        3: "#7f1d1d",
    }

    parts: list[str] = []
    if categorized["overdue"]:
        if shame_tiers:
            # No section headers. Tier color on the row marker + the
            # row prefix in the label carry severity. Tier 1 stays
            # neutral; tier 2/3 progress redder to match the persona.
            for tier, s in _sorted_overdue_with_tier(
                categorized["overdue"], shame_tiers
            ):
                copy = shame_tiers.copy_for_tier(tier)
                row_title_style = title_overdue if tier >= 2 else title_normal
                marker_color = tier_marker_colors.get(tier, "#b91c1c")
                # Wider marker column so the roast line (≤ 100 chars) has
                # room without wrapping awkwardly.
                marker_style = (
                    f"display: inline-block; color: {marker_color}; "
                    "font-weight: 600; padding-right: 12px;"
                )
                who = _display_assignee(s.chore.assignee, assignee_names)
                label = _label_for(s, copy, roast_lines)
                prefix = copy.row_prefix if copy is not None else "⚠"
                parts.append(
                    f'<div style="{row_style}">'
                    f'<span style="{marker_style}">{escape(prefix)} {escape(label)}</span>'
                    f'<span style="{row_title_style}">{escape(s.chore.title)}</span>'
                    f' <span style="{assignee_style}">· {escape(who)}</span>'
                    "</div>"
                )
        else:
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
