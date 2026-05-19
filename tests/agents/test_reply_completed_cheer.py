"""Unit tests for the cheer-line integration in `_reply_completed`.

The completion-cheer specialist supplies a contextual one-line header
that replaces the static "Chore completed" banner. Empty/missing
`cheer_line` (the documented fail-soft mode) must fall back to the
static banner unchanged.
"""

from __future__ import annotations

from typing import Any

from alfred.agents.tasks.replies import build_payload


def _completed_state(cheer_line: str | None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "outcome": "completed",
        "completed_chore": {
            "id": "dog",
            "title": "Walk the dog",
            "completed_by": "primary",
            "completed_at": "2026-05-18T08:30:00",
        },
    }
    if cheer_line is not None:
        state["cheer_line"] = cheer_line
    return state


def test_reply_completed_uses_cheer_line_as_header() -> None:
    cheer = "The dog has filed an enthusiastic review."
    payload = build_payload(_completed_state(cheer), {"primary": "Antonio"})
    assert payload.header == cheer
    assert payload.title == "Walk the dog"
    # Fields are unchanged by the cheer-line substitution.
    labels = [label for label, _ in payload.fields]
    assert labels == ["Completed by", "At", "ID"]


def test_reply_completed_missing_cheer_falls_back_to_static_banner() -> None:
    payload = build_payload(_completed_state(None), {"primary": "Antonio"})
    assert payload.header == "Chore completed"


def test_reply_completed_empty_cheer_falls_back_to_static_banner() -> None:
    payload = build_payload(_completed_state(""), {"primary": "Antonio"})
    assert payload.header == "Chore completed"


def test_reply_completed_whitespace_cheer_falls_back_to_static_banner() -> None:
    payload = build_payload(_completed_state("   \n  "), {"primary": "Antonio"})
    assert payload.header == "Chore completed"
