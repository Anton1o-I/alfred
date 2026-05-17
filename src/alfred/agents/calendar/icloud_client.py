"""iCloud Calendar client via CalDAV.

App-password auth, no OAuth. Mirrors GoogleCalendarClient's surface so the
calendar agent doesn't need to know which provider is behind it.

caldav.py is sync — these methods block the calling thread/event loop
briefly while the HTTP request completes. That matches GoogleCalendarClient's
shape, so the calendar agent's tools (which are async wrappers over sync
client calls) work either way.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from typing import Any

import caldav
import structlog
from icalendar import Calendar as ICal
from icalendar import Event as IEvent

log = structlog.get_logger()

CALDAV_URL = "https://caldav.icloud.com"


def _build_rrule(recurrence: dict[str, Any]) -> dict[str, Any] | None:
    """Map a RecurrenceRule dict (frequency/interval/byday/until_iso/count)
    to the dict form icalendar expects for an RRULE property."""
    freq = recurrence.get("frequency")
    if not freq:
        return None
    out: dict[str, Any] = {"FREQ": freq}
    interval = recurrence.get("interval") or 1
    if interval and interval != 1:
        out["INTERVAL"] = interval
    byday = recurrence.get("byday") or []
    if byday:
        out["BYDAY"] = byday
    until = recurrence.get("until_iso")
    if until:
        with contextlib.suppress(ValueError):
            out["UNTIL"] = datetime.fromisoformat(until)
    count = recurrence.get("count")
    if count:
        out["COUNT"] = count
    return out


def _ical_to_dict(ical_event: IEvent) -> dict[str, Any]:
    """Flatten an icalendar Event to the shape Alfred's agent expects."""
    def _dt(name: str) -> str | None:
        val = ical_event.get(name)
        if val is None:
            return None
        try:
            return val.dt.isoformat()
        except AttributeError:
            return str(val)

    return {
        "id": str(ical_event.get("uid", "")),
        "summary": str(ical_event.get("summary", "Untitled")),
        "description": str(ical_event.get("description", "") or "") or None,
        "location": str(ical_event.get("location", "") or "") or None,
        "start": {"dateTime": _dt("dtstart")},
        "end": {"dateTime": _dt("dtend")},
        "htmlLink": None,
    }


class IcloudCalendarClient:
    """Sync wrapper over caldav.py against iCloud."""

    # In-memory cache of events written by THIS process — bridges iCloud's
    # CalDAV read-after-write lag (writes don't appear in subsequent reads
    # for several seconds). Entries expire after RECENT_WRITES_TTL_SECONDS;
    # by then CalDAV will have caught up, and querying real iCloud is truth.
    RECENT_WRITES_TTL_SECONDS = 300

    def __init__(
        self,
        username: str,
        app_password: str,
        calendar_name: str = "",
        timezone: str = "UTC",
    ) -> None:
        if not username or not app_password:
            raise ValueError("username and app_password are required")
        self._username = username
        self._password = app_password
        self._calendar_name = calendar_name  # "" → principal's default calendar
        self._timezone = timezone
        self._client: caldav.DAVClient | None = None
        self._calendar: caldav.Calendar | None = None
        # recent_writes: list of dicts shaped like list_events() output
        self._recent_writes: list[dict[str, Any]] = []
        # Track expiry per entry as a parallel list of (uid, expires_at_epoch)
        self._recent_writes_meta: list[tuple[str, float]] = []

    def _get_client(self) -> caldav.DAVClient:
        if self._client is None:
            self._client = caldav.DAVClient(
                url=CALDAV_URL, username=self._username, password=self._password
            )
        return self._client

    def _get_calendar(self) -> caldav.Calendar:
        if self._calendar is not None:
            return self._calendar
        principal = self._get_client().principal()
        calendars = principal.calendars()
        if not calendars:
            raise RuntimeError("No CalDAV calendars found for this iCloud account.")
        if self._calendar_name:
            for cal in calendars:
                if cal.name == self._calendar_name:
                    self._calendar = cal
                    break
            if self._calendar is None:
                raise RuntimeError(
                    f"Calendar named '{self._calendar_name}' not found. "
                    f"Available: {[c.name for c in calendars]}"
                )
        else:
            self._calendar = calendars[0]
        return self._calendar

    # ── Read ──────────────────────────────────────────────────────────────

    def _prune_expired_writes(self) -> None:
        import time

        now = time.time()
        keep_indices = [i for i, (_, exp) in enumerate(self._recent_writes_meta) if exp > now]
        self._recent_writes = [self._recent_writes[i] for i in keep_indices]
        self._recent_writes_meta = [self._recent_writes_meta[i] for i in keep_indices]

    def _merge_with_recent_writes(
        self,
        caldav_results: list[dict[str, Any]],
        time_min: datetime,
        time_max: datetime,
    ) -> list[dict[str, Any]]:
        """Add cached recent writes that overlap the window, deduped by UID."""
        self._prune_expired_writes()
        if not self._recent_writes:
            return caldav_results
        seen_uids = {e.get("id") for e in caldav_results if e.get("id")}
        merged = list(caldav_results)
        for cached in self._recent_writes:
            uid = cached.get("id")
            if uid in seen_uids:
                continue
            # Check window overlap
            try:
                c_start = datetime.fromisoformat(cached["start"]["dateTime"])
                c_end = datetime.fromisoformat(cached["end"]["dateTime"])
            except (KeyError, ValueError, TypeError):
                continue
            if c_start < time_max and c_end > time_min:
                merged.append(cached)
        return merged

    def list_events(
        self,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        calendar_id: str = "primary",  # unused — single configured calendar
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        if time_min is None:
            time_min = datetime.now(UTC)
        if time_max is None:
            time_max = time_min + timedelta(days=7)

        cal = self._get_calendar()
        results = cal.search(start=time_min, end=time_max, event=True, expand=True)
        out: list[dict[str, Any]] = []
        for item in results[:max_results]:
            try:
                ical = ICal.from_ical(item.data)
                for component in ical.walk("VEVENT"):
                    out.append(_ical_to_dict(component))
            except Exception as e:  # noqa: BLE001
                log.warning("caldav_parse_failed", error=str(e))
        # Bridge iCloud's read-after-write lag with our local cache.
        return self._merge_with_recent_writes(out, time_min, time_max)

    def list_all_family_events(
        self,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Match GoogleCalendarClient interface — currently single-calendar."""
        return {"primary": self.list_events(time_min, time_max)}

    def list_calendars(self) -> list[dict[str, Any]]:
        principal = self._get_client().principal()
        return [
            {
                "id": str(c.url),
                "summary": c.name,
                "primary": False,
                "access_role": "owner",
            }
            for c in principal.calendars()
        ]

    # ── Write ─────────────────────────────────────────────────────────────

    def create_event(
        self,
        summary: str,
        start: datetime,
        end: datetime | None = None,
        description: str | None = None,
        location: str | None = None,
        calendar_id: str = "primary",  # unused — single configured calendar
        attendees: list[str] | None = None,  # noqa: ARG002 (CalDAV attendees TBD)
        recurrence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if end is None:
            end = start + timedelta(hours=1)

        cal = self._get_calendar()
        ical = ICal()
        ical.add("prodid", "-//Alfred//CalendarAgent//EN")
        ical.add("version", "2.0")
        event = IEvent()
        event.add("summary", summary)
        event.add("dtstart", start)
        event.add("dtend", end)
        if description:
            event.add("description", description)
        if location:
            event.add("location", location)
        if recurrence:
            rrule = _build_rrule(recurrence)
            if rrule:
                event.add("rrule", rrule)
        uid = f"alfred-{int(start.timestamp())}-{abs(hash(summary)) % 10_000_000}"
        event.add("uid", uid)
        ical.add_component(event)
        cal.save_event(ical.to_ical().decode("utf-8"))
        log.info(
            "calendar_event_created",
            summary=summary,
            start=start.isoformat(),
            event_id=uid,
            provider="icloud",
        )
        out = {
            "id": uid,
            "summary": summary,
            "start": {"dateTime": start.isoformat()},
            "end": {"dateTime": end.isoformat()},
            "htmlLink": None,
        }
        # Cache so future list_events queries see it even if iCloud's
        # CalDAV view hasn't caught up to the write yet.
        import time

        self._recent_writes.append(out)
        self._recent_writes_meta.append(
            (uid, time.time() + self.RECENT_WRITES_TTL_SECONDS)
        )
        return out

    def delete_event(self, event_uid: str) -> bool:
        """Delete an event by UID. Returns True if found and deleted, False otherwise.

        Also drops the entry from the recent-writes cache so subsequent
        list_events queries reflect the deletion.
        """
        cal = self._get_calendar()
        deleted = False
        for evt in cal.events():
            if event_uid in evt.data:
                evt.delete()
                deleted = True
                log.info("calendar_event_deleted", event_uid=event_uid, provider="icloud")
                break
        # Drop from cache too
        self._recent_writes = [e for e in self._recent_writes if e.get("id") != event_uid]
        self._recent_writes_meta = [
            m for m in self._recent_writes_meta if m[0] != event_uid
        ]
        if not deleted:
            log.info("calendar_event_delete_miss", event_uid=event_uid, provider="icloud")
        return deleted

    def find_conflicts(
        self,
        time_min: datetime | None = None,  # noqa: ARG002
        time_max: datetime | None = None,  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        """Single-calendar mode has no cross-member conflicts. Returns []."""
        return []

    def get_calendar_id(self, family_member: str | None = None) -> str:  # noqa: ARG002
        return "primary"
