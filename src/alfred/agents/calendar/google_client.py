"""Google Calendar API client — read/write operations on the shared OAuth token."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
from googleapiclient.discovery import build

from alfred.integrations.google_auth import load_credentials

log = structlog.get_logger()

CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar"

DEFAULT_CREDENTIALS_PATH = Path("config/google_credentials.json")
DEFAULT_TOKEN_PATH = Path("config/google_token.json")


class GoogleCalendarClient:
    """Async-compatible wrapper around the Google Calendar API.

    Setup: Run `alfred google-auth` once to complete the OAuth2 flow.
    This stores a refresh token at config/google_token.json. Subsequent
    calls use the refresh token automatically.

    Calendar sharing: The wife's calendar should be shared with the husband's
    Google account (read+write). Then this single client can access both.
    """

    def __init__(
        self,
        credentials_path: Path | None = None,
        token_path: Path | None = None,
        calendar_ids: dict[str, str] | None = None,
        timezone: str = "UTC",
    ) -> None:
        self._credentials_path = credentials_path or DEFAULT_CREDENTIALS_PATH
        self._token_path = token_path or DEFAULT_TOKEN_PATH
        self._calendar_ids = calendar_ids or {}
        self._timezone = timezone
        self._service = None

    def _get_service(self):  # type: ignore[no-untyped-def]
        """Get or create the Calendar API service."""
        if self._service is None:
            creds = load_credentials(
                token_path=self._token_path,
                credentials_path=self._credentials_path,
                scopes=[CALENDAR_SCOPE],
            )
            if creds is None:
                raise RuntimeError(
                    "No valid Google token. Run `alfred google-auth` to authenticate."
                )
            self._service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        return self._service

    def get_calendar_id(self, family_member: str | None = None) -> str:
        """Resolve a family member name to a calendar ID."""
        if family_member and family_member in self._calendar_ids:
            return self._calendar_ids[family_member]
        return "primary"

    def list_events(
        self,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        calendar_id: str = "primary",
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch events from a calendar within a time range."""
        service = self._get_service()

        if time_min is None:
            time_min = datetime.now(UTC)
        if time_max is None:
            time_max = time_min + timedelta(days=7)

        result = (
            service.events()
            .list(
                calendarId=calendar_id,
                timeMin=time_min.isoformat(),
                timeMax=time_max.isoformat(),
                maxResults=max_results,
                singleEvents=True,  # Expands recurring events
                orderBy="startTime",
            )
            .execute()
        )

        return result.get("items", [])

    def list_all_family_events(
        self,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch events from all family member calendars."""
        all_events = {}
        for member, cal_id in self._calendar_ids.items():
            events = self.list_events(
                time_min=time_min,
                time_max=time_max,
                calendar_id=cal_id,
            )
            all_events[member] = events

        # If no calendar_ids configured, at least get primary
        if not self._calendar_ids:
            all_events["primary"] = self.list_events(
                time_min=time_min,
                time_max=time_max,
                calendar_id="primary",
            )

        return all_events

    def create_event(
        self,
        summary: str,
        start: datetime,
        end: datetime | None = None,
        description: str | None = None,
        location: str | None = None,
        calendar_id: str = "primary",
        attendees: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new calendar event."""
        service = self._get_service()

        if end is None:
            end = start + timedelta(hours=1)

        event_body: dict[str, Any] = {
            "summary": summary,
            "start": {"dateTime": start.isoformat(), "timeZone": self._timezone},
            "end": {"dateTime": end.isoformat(), "timeZone": self._timezone},
        }

        if description:
            event_body["description"] = description
        if location:
            event_body["location"] = location
        if attendees:
            event_body["attendees"] = [{"email": a} for a in attendees]

        result = (
            service.events()
            .insert(calendarId=calendar_id, body=event_body)
            .execute()
        )

        log.info(
            "calendar_event_created",
            summary=summary,
            start=start.isoformat(),
            calendar=calendar_id,
            event_id=result.get("id"),
        )

        return result

    def find_conflicts(
        self,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Detect overlapping events across all family calendars."""
        all_events = self.list_all_family_events(time_min, time_max)
        members = list(all_events.keys())
        conflicts = []

        for i, member_a in enumerate(members):
            for member_b in members[i + 1 :]:
                for event_a in all_events[member_a]:
                    for event_b in all_events[member_b]:
                        a_start = _parse_event_time(event_a, "start")
                        a_end = _parse_event_time(event_a, "end")
                        b_start = _parse_event_time(event_b, "start")
                        b_end = _parse_event_time(event_b, "end")

                        if (
                            a_start and a_end and b_start and b_end
                            and a_start < b_end and b_start < a_end
                        ):
                            conflicts.append({
                                "member_a": member_a,
                                "event_a": event_a.get("summary", "Untitled"),
                                "member_b": member_b,
                                "event_b": event_b.get("summary", "Untitled"),
                                "overlap_start": max(a_start, b_start).isoformat(),
                                "overlap_end": min(a_end, b_end).isoformat(),
                            })

        return conflicts

    def list_calendars(self) -> list[dict[str, Any]]:
        """List all calendars accessible to the authenticated account."""
        service = self._get_service()
        result = service.calendarList().list().execute()
        return [
            {
                "id": cal["id"],
                "summary": cal.get("summary", ""),
                "primary": cal.get("primary", False),
                "access_role": cal.get("accessRole", ""),
            }
            for cal in result.get("items", [])
        ]


def _parse_event_time(event: dict[str, Any], key: str) -> datetime | None:
    """Parse a start or end time from a Google Calendar event."""
    time_data = event.get(key, {})
    dt_str = time_data.get("dateTime") or time_data.get("date")
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
        return None
