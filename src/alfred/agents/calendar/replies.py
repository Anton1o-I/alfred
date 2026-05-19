"""Build the user-facing reply for a finished calendar workflow run.

One handler per `Outcome`, registered in a dispatch table. Mirrors the
tasks `replies.py` pattern. Each handler inspects the relevant pieces
of state and returns a (plain, html) tuple — the workflow's `build_reply`
node just delegates here.

Keeping the user-visible vocabulary (banners, confirmation copy, "Heads
up — recurring" warnings) in one file makes adding new outcomes a
one-handler change instead of an if/elif edit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from alfred.agents.calendar.outcomes import Outcome
from alfred.agents.calendar.renderers import (
    render_clarification_html,
    render_clarification_plain,
    render_event_deleted_html,
    render_event_deleted_plain,
    render_event_html,
    render_event_plain,
)

ReplyPair = tuple[str, str]
_HandlerFn = Callable[[dict[str, Any]], ReplyPair]


# ── Small helpers ───────────────────────────────────────────────────────────


def _format_incomplete_text(issues: list[str]) -> str:
    return (
        "I couldn't fully parse your request.\n"
        "Issues: " + "; ".join(issues) + ".\n\n"
        "Reply with the full details, e.g. 'Meeting with Eric next Wednesday at 10am'."
    )


def _wrap_clarification(text: str) -> ReplyPair:
    return render_clarification_plain(text), render_clarification_html(text)


def _candidate_summary_lines(target: dict[str, Any]) -> str:
    start = target.get("start") or ""
    out = f"  • {target.get('summary', 'Untitled')}\n    {start}\n"
    if target.get("location"):
        out += f"    Location: {target['location']}\n"
    return out


# ── Handlers ────────────────────────────────────────────────────────────────


def _reply_created(state: dict[str, Any]) -> ReplyPair:
    event = state["created_event"]
    return render_event_plain(event), render_event_html(event)


def _reply_deleted(state: dict[str, Any]) -> ReplyPair:
    event = state["deleted_event"]
    return render_event_deleted_plain(event), render_event_deleted_html(event)


def _reply_delete_not_found(state: dict[str, Any]) -> ReplyPair:
    target = state.get("delete_target") or {}
    intent_summary = target.get("intent_summary") or "(no description)"
    when = (
        f" on {target['date_hint_iso']}"
        if target.get("date_hint_iso")
        else " in the next 30 days"
    )
    txt = (
        f"I couldn't find an event matching '{intent_summary}'{when}.\n\n"
        "Could you reply with the event title and date so I can find it?"
    )
    return _wrap_clarification(txt)


def _reply_delete_needs_confirmation(state: dict[str, Any]) -> ReplyPair:
    decision = state.get("match_decision") or {}
    candidates = state.get("matching_events") or []
    target = next(
        (c for c in candidates if c.get("id") == decision.get("event_id")), None
    )
    if target is None:
        txt = (
            f"I think you mean an event matching: {decision.get('reasoning', '')}. "
            "Could you reply with more details?"
        )
    else:
        reasoning = decision.get("reasoning", "")
        why = f"\nWhy I think this: {reasoning}\n" if reasoning else "\n"
        txt = (
            "I think you want to delete:\n\n"
            + _candidate_summary_lines(target)
            + why
            + "\nReply 'yes' to confirm, or give me more details to pick a different one."
        )
    return _wrap_clarification(txt)


# ── Update branch ───────────────────────────────────────────────────────────


def _format_update_changes(patch: dict[str, Any]) -> list[str]:
    """Human-readable bullet list of what we changed (or plan to)."""
    bits: list[str] = []
    if patch.get("new_title"):
        bits.append(f"title → {patch['new_title']}")
    if patch.get("new_start"):
        bits.append(f"start → {patch['new_start']}")
    if patch.get("new_end"):
        bits.append(f"end → {patch['new_end']}")
    if patch.get("new_location"):
        bits.append(f"location → {patch['new_location']}")
    if patch.get("new_notes"):
        bits.append(f"notes → {patch['new_notes']}")
    return bits


def _reply_updated(state: dict[str, Any]) -> ReplyPair:
    event = state["updated_event"]
    changes = _format_update_changes(state.get("update_target") or {})
    rows_event = {
        "summary": event.get("summary"),
        "start_iso": event.get("start_iso"),
        "end_iso": event.get("end_iso"),
        "location": event.get("location"),
        "description": event.get("description"),
        "calendar_name": event.get("calendar_name"),
    }
    # Reuse the "created" card with a swapped header so users see what
    # the event now looks like, plus a small changes block + (optional)
    # recurring-series warning.
    plain_lines = ["✓ Event updated", ""]
    for label, value in _event_rows(rows_event):
        plain_lines.append(f"{label}: {value}")
    if changes:
        plain_lines.append("")
        plain_lines.append("Changed:")
        for b in changes:
            plain_lines.append(f"  • {b}")
    if event.get("is_recurring"):
        plain_lines.append("")
        plain_lines.append(
            "Heads up — this is a recurring event, so the update applies "
            "to the whole series."
        )
    txt = "\n".join(plain_lines)
    html = render_clarification_html(txt, header="Event updated")
    return render_clarification_plain(txt), html


def _event_rows(event: dict[str, Any]) -> list[tuple[str, str]]:
    """Minimal label/value rows for an updated event (mirrors render._rows)."""
    rows: list[tuple[str, str]] = []
    if event.get("summary"):
        rows.append(("Event", event["summary"]))
    if event.get("start_iso"):
        rows.append(("Start", event["start_iso"]))
    if event.get("end_iso"):
        rows.append(("End", event["end_iso"]))
    if event.get("location"):
        rows.append(("Location", event["location"]))
    if event.get("description"):
        rows.append(("Notes", event["description"]))
    if event.get("calendar_name"):
        rows.append(("Calendar", event["calendar_name"]))
    return rows


def _reply_update_not_found(state: dict[str, Any]) -> ReplyPair:
    target = state.get("update_target") or {}
    ref = target.get("target") or {}
    intent_summary = ref.get("intent_summary") or "(no description)"
    when = (
        f" on {ref['date_hint_iso']}"
        if ref.get("date_hint_iso")
        else " in the next 30 days"
    )
    txt = (
        f"I couldn't find an event matching '{intent_summary}'{when} to update.\n\n"
        "Could you reply with the event title and date so I can find it?"
    )
    return _wrap_clarification(txt)


def _reply_update_needs_confirmation(state: dict[str, Any]) -> ReplyPair:
    decision = state.get("match_decision") or {}
    candidates = state.get("matching_events") or []
    target = next(
        (c for c in candidates if c.get("id") == decision.get("event_id")), None
    )
    changes = _format_update_changes(state.get("update_target") or {})
    change_block = ""
    if changes:
        change_block = "\nProposed changes:\n" + "\n".join(f"  • {b}" for b in changes) + "\n"
    if target is None:
        txt = (
            f"I think you mean an event matching: {decision.get('reasoning', '')}. "
            "Could you reply with more details?"
        )
    else:
        reasoning = decision.get("reasoning", "")
        why = f"\nWhy I think this: {reasoning}\n" if reasoning else "\n"
        txt = (
            "I think you want to update:\n\n"
            + _candidate_summary_lines(target)
            + change_block
            + why
            + "\nReply 'yes' to confirm, or give me more details to pick a different one."
        )
    return _wrap_clarification(txt)


def _reply_update_no_change(_state: dict[str, Any]) -> ReplyPair:
    txt = (
        "I couldn't tell what you wanted to change about the event. "
        "Try something like 'Move my 3pm tomorrow to 4pm' or "
        "'Rename the dentist appointment to checkup'."
    )
    return _wrap_clarification(txt)


# ── Generic fallbacks ───────────────────────────────────────────────────────


def _reply_incomplete(state: dict[str, Any]) -> ReplyPair:
    txt = _format_incomplete_text(state.get("completeness_issues", []))
    return _wrap_clarification(txt)


def _reply_unsupported(state: dict[str, Any]) -> ReplyPair:
    txt = state.get("error_message", "That action isn't supported yet.")
    return _wrap_clarification(txt)


def _reply_error(state: dict[str, Any]) -> ReplyPair:
    txt = (
        "Sorry — something went wrong while processing your request: "
        + state.get("error_message", "unknown error")
    )
    return _wrap_clarification(txt)


def _reply_clarification(_state: dict[str, Any]) -> ReplyPair:
    txt = (
        "I couldn't tell what calendar action you wanted. I can create, "
        "update, and cancel events — try something like 'Lunch with Sarah "
        "next Wednesday at 12:30pm'."
    )
    return _wrap_clarification(txt)


# ── Dispatch table ──────────────────────────────────────────────────────────


_HANDLERS: dict[str, _HandlerFn] = {
    Outcome.CREATED.value: _reply_created,
    Outcome.DELETED.value: _reply_deleted,
    Outcome.DELETE_NOT_FOUND.value: _reply_delete_not_found,
    Outcome.DELETE_NEEDS_CONFIRMATION.value: _reply_delete_needs_confirmation,
    Outcome.UPDATED.value: _reply_updated,
    Outcome.UPDATE_NOT_FOUND.value: _reply_update_not_found,
    Outcome.UPDATE_NEEDS_CONFIRMATION.value: _reply_update_needs_confirmation,
    Outcome.UPDATE_NO_CHANGE.value: _reply_update_no_change,
    Outcome.INCOMPLETE.value: _reply_incomplete,
    Outcome.UNSUPPORTED.value: _reply_unsupported,
    Outcome.ERROR.value: _reply_error,
    Outcome.CLARIFICATION.value: _reply_clarification,
}


def build_reply_pair(state: dict[str, Any]) -> ReplyPair:
    """Look up the handler for the state's outcome and produce (plain, html)."""
    outcome = state.get("outcome") or Outcome.CLARIFICATION.value
    handler = _HANDLERS.get(outcome, _reply_clarification)
    return handler(state)


__all__ = ["build_reply_pair"]
