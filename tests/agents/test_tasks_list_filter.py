"""Unit tests for the list-by-assignee filter helpers in tasks workflow.

Covers:
- `_resolve_assignee_filter`: phrase → assignee slot resolution incl.
  household aliases and case-insensitive display names.
- `_filter_pending_by_assignee`: personal slots do NOT include household
  chores; household filter only matches household chores.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from alfred.agents.tasks.workflow import (
    _filter_pending_by_assignee,
    _resolve_assignee_filter,
)

NAME_MAP = {"primary": "Alex", "secondary": "Sam"}


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        # Slot literals
        ("primary", "primary"),
        ("secondary", "secondary"),
        ("household", "household"),
        # Case-insensitive display names
        ("Alex", "primary"),
        ("alex", "primary"),
        ("SAM", "secondary"),
        # Household aliases
        ("us", "household"),
        ("the family", "household"),
        ("everyone", "household"),
        # Loose phrasings the model might echo verbatim
        ("what does Alex owe", "primary"),
        ("Sam's chores", "secondary"),
        ("our household", "household"),
        # No match
        ("nobody", None),
        ("", None),
        ("   ", None),
    ],
)
def test_resolve_assignee_filter(reference: str, expected: str | None) -> None:
    assert _resolve_assignee_filter(reference, NAME_MAP) == expected


def _fake_status(assignee: str, chore_id: str) -> SimpleNamespace:
    """Build a fake object shaped enough like ChoreStatus for the filter."""
    chore = SimpleNamespace(id=chore_id, assignee=assignee, title=chore_id)
    return SimpleNamespace(
        chore=chore,
        next_due=datetime(2026, 5, 17, tzinfo=UTC),
        overdue_days=0,
        last_completed_at=None,
    )


def test_filter_personal_excludes_household() -> None:
    """A name filter must not include shared household chores — the user
    asked specifically what THIS PERSON owes."""
    statuses = [
        _fake_status("primary", "mow-lawn"),
        _fake_status("household", "take-out-trash"),
        _fake_status("secondary", "clean-bathrooms"),
    ]
    result = _filter_pending_by_assignee(statuses, "primary")
    assert [s.chore.id for s in result] == ["mow-lawn"]


def test_filter_household_excludes_personal() -> None:
    """Household filter matches only household-assigned chores."""
    statuses = [
        _fake_status("primary", "mow-lawn"),
        _fake_status("household", "take-out-trash"),
        _fake_status("household", "clean-kitchen"),
        _fake_status("secondary", "clean-bathrooms"),
    ]
    result = _filter_pending_by_assignee(statuses, "household")
    assert sorted(s.chore.id for s in result) == ["clean-kitchen", "take-out-trash"]


def test_filter_secondary_only() -> None:
    statuses = [
        _fake_status("primary", "mow-lawn"),
        _fake_status("household", "take-out-trash"),
        _fake_status("secondary", "clean-bathrooms"),
    ]
    result = _filter_pending_by_assignee(statuses, "secondary")
    assert [s.chore.id for s in result] == ["clean-bathrooms"]


def test_filter_returns_empty_when_no_match() -> None:
    statuses = [_fake_status("household", "take-out-trash")]
    result = _filter_pending_by_assignee(statuses, "primary")
    assert result == []
