"""Unit tests for the calendar update schemas + EventPatch."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from alfred.agents.calendar.icloud_client import EventPatch
from alfred.agents.calendar.workflow import EventReference, UpdateEventDraft

_UTC = UTC


def _ref(intent: str = "my 3pm") -> EventReference:
    return EventReference(intent_summary=intent, date_hint_iso=None, time_hint=None)


class TestUpdateEventDraftHasAnyChange:
    def test_all_none_returns_false(self) -> None:
        draft = UpdateEventDraft(target=_ref())
        assert draft.has_any_change() is False

    @pytest.mark.parametrize(
        "field,value",
        [
            ("new_start_iso", "2026-06-01T14:00:00+00:00"),
            ("new_end_iso", "2026-06-01T15:00:00+00:00"),
            ("new_title", "Renamed"),
            ("new_location", "Cafe Luna"),
            ("new_notes", "bring laptop"),
        ],
    )
    def test_any_single_field_returns_true(self, field: str, value: str) -> None:
        draft = UpdateEventDraft(target=_ref(), **{field: value})
        assert draft.has_any_change() is True


class TestEventPatch:
    def test_empty_patch_is_empty(self) -> None:
        assert EventPatch().is_empty() is True

    def test_patch_with_summary_is_not_empty(self) -> None:
        assert EventPatch(summary="new").is_empty() is False

    def test_end_before_start_raises(self) -> None:
        start = datetime(2026, 6, 1, 14, tzinfo=_UTC)
        end = start - timedelta(hours=1)
        with pytest.raises(ValueError, match="end must be after start"):
            EventPatch(start=start, end=end)

    def test_end_after_start_ok(self) -> None:
        start = datetime(2026, 6, 1, 14, tzinfo=_UTC)
        end = start + timedelta(hours=1)
        patch = EventPatch(start=start, end=end)
        assert patch.is_empty() is False
