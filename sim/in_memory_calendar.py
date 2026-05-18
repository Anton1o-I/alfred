"""In-memory calendar client for tests + simulation.

Mirrors the surface of IcloudCalendarClient (list_events / create_event /
delete_event / list_calendars) but keeps everything in a Python list. No
network, no auth, no eventual-consistency lag — every write is visible
to subsequent reads immediately.

Pre-seed by calling `seed(...)` with raw event dicts or by repeated
`create_event(...)` calls before the test runs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any


class InMemoryCalendarClient:
    """Drop-in stand-in for IcloudCalendarClient in tests + simulation."""

    def __init__(
        self,
        calendar_name: str = "Alfred",
        timezone: str = "UTC",
    ) -> None:
        self._calendar_name = calendar_name
        self._timezone = timezone
        self._events: list[dict[str, Any]] = []

    # ── seeding ───────────────────────────────────────────────────────────

    def seed(self, events: list[dict[str, Any]]) -> None:
        """Pre-populate the calendar. Each entry should look like
        {title, start, end, location?, description?} with ISO datetimes.
        """
        for ev in events:
            start = datetime.fromisoformat(ev["start"])
            end = datetime.fromisoformat(ev["end"])
            self.create_event(
                summary=ev["title"],
                start=start,
                end=end,
                location=ev.get("location"),
                description=ev.get("description"),
            )

    # ── read ──────────────────────────────────────────────────────────────

    def list_events(
        self,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        calendar_id: str = "primary",  # noqa: ARG002 — interface parity
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        if time_min is None:
            time_min = datetime.now(UTC)
        if time_max is None:
            time_max = time_min + timedelta(days=7)

        out: list[dict[str, Any]] = []
        for ev in self._events:
            try:
                ev_start = datetime.fromisoformat(ev["start"]["dateTime"])
                ev_end = datetime.fromisoformat(ev["end"]["dateTime"])
            except (KeyError, ValueError):
                continue
            if ev_start < time_max and ev_end > time_min:
                out.append(ev)
            if len(out) >= max_results:
                break
        return out

    def list_calendars(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "in-memory",
                "summary": self._calendar_name,
                "primary": True,
                "access_role": "owner",
            }
        ]

    def list_all_family_events(
        self,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        return {"primary": self.list_events(time_min, time_max)}

    def find_conflicts(
        self,
        time_min: datetime | None = None,  # noqa: ARG002
        time_max: datetime | None = None,  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        return []

    def get_calendar_id(self, family_member: str | None = None) -> str:  # noqa: ARG002
        return "primary"

    # ── write ─────────────────────────────────────────────────────────────

    def create_event(
        self,
        summary: str,
        start: datetime,
        end: datetime | None = None,
        description: str | None = None,
        location: str | None = None,
        calendar_id: str = "primary",  # noqa: ARG002
        attendees: list[str] | None = None,  # noqa: ARG002
        recurrence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if end is None:
            end = start + timedelta(hours=1)
        uid = f"sim-{int(start.timestamp())}-{abs(hash(summary)) % 10_000_000}"
        ev = {
            "id": uid,
            "summary": summary,
            "start": {"dateTime": start.isoformat()},
            "end": {"dateTime": end.isoformat()},
            "location": location,
            "description": description,
            "recurrence": recurrence,
            "htmlLink": None,
        }
        self._events.append(ev)
        return ev

    def update_event(self, event_uid: str, *, patch: Any) -> dict[str, Any]:
        """In-memory analog of IcloudCalendarClient.update_event.

        Accepts the same `EventPatch` shape (duck-typed to avoid the
        circular import between the sim package and the icloud client).
        Raises LookupError when the UID is unknown.
        """
        if hasattr(patch, "is_empty") and patch.is_empty():
            raise ValueError("EventPatch has no fields set")
        target = next((e for e in self._events if e.get("id") == event_uid), None)
        if target is None:
            raise LookupError(f"event uid not found: {event_uid}")
        if getattr(patch, "summary", None) is not None:
            target["summary"] = patch.summary
        if getattr(patch, "start", None) is not None:
            target["start"] = {"dateTime": patch.start.isoformat()}
        if getattr(patch, "end", None) is not None:
            target["end"] = {"dateTime": patch.end.isoformat()}
        if getattr(patch, "location", None) is not None:
            target["location"] = patch.location
        if getattr(patch, "description", None) is not None:
            target["description"] = patch.description
        return target

    def delete_event(self, event_uid: str) -> bool:
        before = len(self._events)
        self._events = [e for e in self._events if e.get("id") != event_uid]
        return len(self._events) < before

    # ── inspection helpers (for assertions) ───────────────────────────────

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)
