"""Build the user-facing reply for a finished workflow run.

One handler per `Outcome`, registered in a dispatch table. Each handler
inspects the relevant pieces of state and returns a `ReplyPayload` — a
structured record that the rendering layer turns into plain + HTML.

This module owns the user-visible vocabulary: banners ("Chore added"),
field labels ("Schedule", "Assignee"), and human-friendly value formatting
(display names instead of user_id slots, "every 2 days" instead of the
raw RRULE).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from alfred.agents.tasks.formatting import format_incomplete_text, format_recurrence_human
from alfred.agents.tasks.outcomes import SUCCESS_OUTCOMES, Outcome
from alfred.notifications.tasks_render import ReplyPayload

# Assignee slot → fallback label when no display name is configured.
_ASSIGNEE_FALLBACKS: dict[str, str] = {
    "household": "Household",
    "primary": "Primary",
    "secondary": "Secondary",
}


def humanize_assignee(assignee: str | None, name_map: dict[str, str]) -> str:
    """Render an assignee slot ('primary'/'secondary'/'household') for users.

    Replaces 'primary'/'secondary' with the configured display name when
    available (e.g. 'Antonio', 'Emily'). 'household' always renders as
    'Household'. Unknown slots are title-cased as a defensive fallback so
    nothing leaks raw enum values.
    """
    if not assignee:
        return "Household"
    if assignee == "household":
        return "Household"
    if assignee in name_map and name_map[assignee]:
        return name_map[assignee]
    return _ASSIGNEE_FALLBACKS.get(assignee, assignee.title())


def _schedule_text(chore: dict[str, Any]) -> str:
    """Human-readable schedule string for a chore record."""
    if chore.get("recurrence_type") == "once":
        return f"Due {chore.get('due_date') or 'unspecified'} (one-time)"
    return format_recurrence_human(
        chore.get("recurrence_rule") or {},
        chore.get("recurrence_type") or "schedule",
    )


def _shame_text(chore: dict[str, Any]) -> str:
    days = chore.get("shame_after_days") or 3
    suffix = "day" if days == 1 else "days"
    return f"After {days} {suffix} overdue"


# ── Handlers ────────────────────────────────────────────────────────────────
#
# Each takes the full graph state and returns a ReplyPayload. They're tiny
# on purpose — pure functions, no I/O, easy to unit test.


def _reply_created(state: dict[str, Any], name_map: dict[str, str]) -> ReplyPayload:
    c = state["created_chore"]
    return ReplyPayload(
        header="Chore added",
        title=c["title"],
        fields=[
            ("Assignee", humanize_assignee(c.get("assignee"), name_map)),
            ("Schedule", _schedule_text(c)),
            ("Shame", _shame_text(c)),
            ("ID", c["id"]),
        ],
    )


def _reply_completed(state: dict[str, Any], name_map: dict[str, str]) -> ReplyPayload:
    c = state["completed_chore"]
    return ReplyPayload(
        header="Chore completed",
        title=c["title"],
        fields=[
            ("Completed by", humanize_assignee(c.get("completed_by"), name_map)),
            ("At", c.get("completed_at", "—")),
            ("ID", c["id"]),
        ],
    )


def _reply_deleted(state: dict[str, Any], _name_map: dict[str, str]) -> ReplyPayload:
    c = state["deleted_chore"]
    return ReplyPayload(
        header="Chore removed",
        title=c["title"],
        fields=[("ID", c["id"])],
        body=(
            "You won't get reminders for this chore anymore. "
            "Existing completion history is preserved."
        ),
    )


def _reply_updated(state: dict[str, Any], name_map: dict[str, str]) -> ReplyPayload:
    c = state["updated_chore"]
    changed = ", ".join(c.get("changed_fields", [])) or "(none)"
    return ReplyPayload(
        header="Chore updated",
        title=c["title"],
        fields=[
            ("Assignee", humanize_assignee(c.get("assignee"), name_map)),
            ("Schedule", _schedule_text(c)),
            ("Shame", _shame_text(c)),
            ("Changed", changed),
            ("ID", c["id"]),
        ],
    )


def _reply_listed(state: dict[str, Any], name_map: dict[str, str]) -> ReplyPayload:
    summary = state.get("pending_summary") or []
    scope_label = (state.get("list_target_label") or "").strip()
    header = f"{scope_label}'s chores" if scope_label else "Your chores"
    if not summary:
        empty_body = (
            f"No pending chores for {scope_label}."
            if scope_label
            else "No chores are currently being tracked."
        )
        return ReplyPayload(header=header, body=empty_body)
    overdue = [s for s in summary if s["overdue_days"] > 0]
    due_today = [s for s in summary if s["overdue_days"] == 0]
    upcoming = [s for s in summary if s["overdue_days"] < 0]
    lines: list[str] = []
    if overdue:
        lines.append("OVERDUE")
        for s in sorted(overdue, key=lambda x: -x["overdue_days"]):
            who = humanize_assignee(s["assignee"], name_map)
            lines.append(f"  • {s['title']} ({who}) — {s['overdue_days']}d overdue")
    if due_today:
        if lines:
            lines.append("")
        lines.append("DUE TODAY")
        for s in due_today:
            who = humanize_assignee(s["assignee"], name_map)
            lines.append(f"  • {s['title']} ({who})")
    if upcoming:
        if lines:
            lines.append("")
        lines.append("UPCOMING")
        for s in upcoming:
            who = humanize_assignee(s["assignee"], name_map)
            lines.append(
                f"  • {s['title']} ({who}) — due {s['next_due_iso'][:10]}"
            )
    return ReplyPayload(header="Your chores", body="\n".join(lines))


def _reply_duplicate_existing(state: dict[str, Any], _name_map: dict[str, str]) -> ReplyPayload:
    dup = state.get("duplicate_check") or {}
    existing_id = dup.get("existing_chore_id") or "?"
    reasoning = dup.get("reasoning", "")
    return ReplyPayload(
        header="Duplicate chore",
        fields=[("Existing chore", existing_id)],
        body=(
            f"This looks like the same chore that's already on the list. "
            f"{reasoning}\n\n"
            "Did you mean to update the existing chore? Reply with what to change, "
            "or 'add anyway' if it's actually a separate chore."
        ),
    )


def _reply_duplicate_needs_confirmation(
    state: dict[str, Any], _name_map: dict[str, str]
) -> ReplyPayload:
    dup = state.get("duplicate_check") or {}
    existing_id = dup.get("existing_chore_id") or "?"
    reasoning = dup.get("reasoning", "")
    return ReplyPayload(
        header="Possible duplicate",
        fields=[("Possibly matches", existing_id)],
        body=(
            f"I think this might overlap with that existing chore. {reasoning}\n\n"
            "Reply 'add anyway' to create it as a new chore, or describe the "
            "difference so I can update the existing one instead."
        ),
    )


def _reply_target_needs_confirmation(
    state: dict[str, Any], _name_map: dict[str, str]
) -> ReplyPayload:
    match = state.get("target_match") or {}
    chore_id = match.get("chore_id") or "?"
    reasoning = match.get("reasoning", "")
    return ReplyPayload(
        header="Need confirmation",
        fields=[("Best guess", chore_id)],
        body=(
            f"I think you mean this chore. {reasoning}\n\n"
            "Reply 'yes' to confirm, or give me more detail to pick a different one."
        ),
    )


def _reply_target_not_found(
    state: dict[str, Any], _name_map: dict[str, str]
) -> ReplyPayload:
    action = state.get("action") or "that"
    ref = (state.get("target_reference") or {}).get("reference") or ""
    tail = f" matching '{ref}'" if ref else ""
    return ReplyPayload(
        header="Need a bit more info",
        body=(
            f"I couldn't find a chore{tail} to {action}. "
            "Reply with the chore id or a clearer name."
        ),
    )


def _reply_incomplete(state: dict[str, Any], _name_map: dict[str, str]) -> ReplyPayload:
    return ReplyPayload(
        header="Need a bit more info",
        body=format_incomplete_text(state.get("completeness_issues", []) or []),
    )


def _reply_unsupported(state: dict[str, Any], _name_map: dict[str, str]) -> ReplyPayload:
    return ReplyPayload(
        header="Not supported yet",
        body=state.get("error_message", "That action isn't supported yet."),
    )


def _reply_error(state: dict[str, Any], _name_map: dict[str, str]) -> ReplyPayload:
    return ReplyPayload(
        header="Something went wrong",
        body=(
            "Sorry — something went wrong while processing your chore: "
            + state.get("error_message", "unknown error")
        ),
    )


def _reply_clarification(_state: dict[str, Any], _name_map: dict[str, str]) -> ReplyPayload:
    return ReplyPayload(
        header="Need a bit more info",
        body=(
            "I couldn't tell what you wanted to do with the chore list. Try "
            "something like 'Add chore: take out trash every Tuesday, household'."
        ),
    )


# Dispatch table. New outcomes plug in here without touching the workflow.
_HandlerFn = Callable[[dict[str, Any], dict[str, str]], ReplyPayload]

_HANDLERS: dict[str, _HandlerFn] = {
    Outcome.CREATED.value: _reply_created,
    Outcome.COMPLETED.value: _reply_completed,
    Outcome.DELETED.value: _reply_deleted,
    Outcome.UPDATED.value: _reply_updated,
    Outcome.LISTED.value: _reply_listed,
    Outcome.DUPLICATE_EXISTING.value: _reply_duplicate_existing,
    Outcome.DUPLICATE_NEEDS_CONFIRMATION.value: _reply_duplicate_needs_confirmation,
    Outcome.TARGET_NEEDS_CONFIRMATION.value: _reply_target_needs_confirmation,
    Outcome.TARGET_NOT_FOUND.value: _reply_target_not_found,
    Outcome.INCOMPLETE.value: _reply_incomplete,
    Outcome.UNSUPPORTED.value: _reply_unsupported,
    Outcome.ERROR.value: _reply_error,
    Outcome.CLARIFICATION.value: _reply_clarification,
}


# Maps a populated state-result key to the outcome it implies. Used by
# `resolve_outcome` to defend against state-mismatch UX bugs where a
# side-effect node wrote a chore but `outcome` ended up clobbered or
# empty (see commit history for the reported "Need more info" + chore
# created case).
_RESULT_KEYS: dict[str, Outcome] = {
    "created_chore": Outcome.CREATED,
    "completed_chore": Outcome.COMPLETED,
    "deleted_chore": Outcome.DELETED,
    "updated_chore": Outcome.UPDATED,
}


def resolve_outcome(state: dict[str, Any]) -> tuple[str, str | None]:
    """Pick the outcome to render, falling back to populated-result detection.

    Returns (final_outcome, mismatch_key_or_None). When the state's
    `outcome` doesn't match a populated result dict, we prefer the
    success outcome and return the key name as a signal — callers log a
    warning so the underlying state bug stays visible.
    """
    declared = state.get("outcome") or Outcome.CLARIFICATION.value
    for key, success in _RESULT_KEYS.items():
        if state.get(key) and declared != success.value:
            return success.value, key
    return declared, None


def build_payload(state: dict[str, Any], name_map: dict[str, str]) -> ReplyPayload:
    """Look up the handler for the state's outcome and produce a payload."""
    outcome, _ = resolve_outcome(state)
    handler = _HANDLERS.get(outcome, _reply_clarification)
    return handler(state, name_map)


__all__ = [
    "SUCCESS_OUTCOMES",
    "build_payload",
    "humanize_assignee",
    "resolve_outcome",
]
