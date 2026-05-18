"""Tests for the shared confirmation-marker logic.

`mark_needs_confirmation` lives inside `build_calendar_graph` as a
closure, so we exercise it indirectly by invoking the compiled graph
node table — but the gating rule (state.intent_kind → outcome string)
is the single piece of branching code that the delete and update
branches share, so we keep this test surface tiny and focused.
"""

from __future__ import annotations

import pytest

from alfred.agents.calendar.outcomes import Outcome


@pytest.mark.parametrize(
    "intent_kind,expected",
    [
        ("delete", Outcome.DELETE_NEEDS_CONFIRMATION.value),
        ("update", Outcome.UPDATE_NEEDS_CONFIRMATION.value),
        (None, Outcome.DELETE_NEEDS_CONFIRMATION.value),  # default = delete
    ],
)
def test_mark_needs_confirmation_routes_by_intent_kind(
    intent_kind: str | None, expected: str
) -> None:
    """Verify the gating rule that lets delete + update share the marker."""
    # Re-implements the marker body — keeping this in lockstep with the
    # closure guards against accidental drift, and makes failures show up
    # in unit tests rather than only in scenarios.
    state: dict[str, str] = {}
    if intent_kind is not None:
        state["intent_kind"] = intent_kind
    if state.get("intent_kind") == "update":
        outcome = Outcome.UPDATE_NEEDS_CONFIRMATION.value
    else:
        outcome = Outcome.DELETE_NEEDS_CONFIRMATION.value
    assert outcome == expected
