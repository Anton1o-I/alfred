"""Tests for InMemoryCalendarClient.update_event (sim stand-in for CalDAV).

The icloud client's `update_event` shares the same EventPatch surface and
the same lookup-failure-via-LookupError contract; testing the in-memory
analog gives us coverage of the patch semantics without bringing
caldav.py into the unit test path.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sim.in_memory_calendar import InMemoryCalendarClient

from alfred.agents.calendar.icloud_client import EventPatch

_UTC = UTC


def _seed_one(client: InMemoryCalendarClient) -> str:
    """Drop one event onto the calendar, return its uid."""
    ev = client.create_event(
        summary="Standup",
        start=datetime(2026, 6, 1, 14, tzinfo=_UTC),
        end=datetime(2026, 6, 1, 15, tzinfo=_UTC),
        location="Zoom",
    )
    return ev["id"]


def test_update_event_changes_summary() -> None:
    client = InMemoryCalendarClient()
    uid = _seed_one(client)
    patch = EventPatch(summary="Retro")
    result = client.update_event(uid, patch=patch)
    assert result["summary"] == "Retro"


def test_update_event_changes_start_and_end() -> None:
    client = InMemoryCalendarClient()
    uid = _seed_one(client)
    new_start = datetime(2026, 6, 1, 16, tzinfo=_UTC)
    new_end = datetime(2026, 6, 1, 17, tzinfo=_UTC)
    result = client.update_event(uid, patch=EventPatch(start=new_start, end=new_end))
    assert result["start"]["dateTime"] == new_start.isoformat()
    assert result["end"]["dateTime"] == new_end.isoformat()


def test_update_event_preserves_unspecified_fields() -> None:
    client = InMemoryCalendarClient()
    uid = _seed_one(client)
    client.update_event(uid, patch=EventPatch(summary="Retro"))
    assert client.events[0]["location"] == "Zoom"


def test_update_event_unknown_uid_raises_lookup_error() -> None:
    client = InMemoryCalendarClient()
    _seed_one(client)
    with pytest.raises(LookupError):
        client.update_event("does-not-exist", patch=EventPatch(summary="Retro"))


def test_update_event_empty_patch_raises_value_error() -> None:
    client = InMemoryCalendarClient()
    uid = _seed_one(client)
    with pytest.raises(ValueError):
        client.update_event(uid, patch=EventPatch())
