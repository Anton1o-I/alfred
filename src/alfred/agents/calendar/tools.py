"""Google Calendar tools for the calendar agent."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from alfred.agents.calendar.google_client import GoogleCalendarClient
from alfred.tools.base import Tool


class CalendarReadTool(Tool):
    """Fetch events from family calendars."""

    name = "google_calendar_read"
    description = (
        "Read events from Google Calendar. "
        "Can fetch events for a specific date range and family member. "
        "Use this to check schedules, find free time, or detect conflicts."
    )
    requires_confirmation = False

    def __init__(self, client: GoogleCalendarClient) -> None:
        self._client = client

    async def execute(self, arguments: dict[str, Any]) -> Any:
        action = arguments.get("action", "list_events")

        if action == "list_events":
            return self._list_events(arguments)
        elif action == "list_all_family":
            return self._list_all_family(arguments)
        elif action == "find_conflicts":
            return self._find_conflicts(arguments)
        elif action == "list_calendars":
            return self._client.list_calendars()
        else:
            return {"error": f"Unknown action: {action}"}

    def _list_events(self, arguments: dict[str, Any]) -> Any:
        time_min, time_max = _parse_time_range(arguments)
        family_member = arguments.get("family_member")
        calendar_id = self._client.get_calendar_id(family_member)

        events = self._client.list_events(
            time_min=time_min,
            time_max=time_max,
            calendar_id=calendar_id,
        )
        return _format_events(events)

    def _list_all_family(self, arguments: dict[str, Any]) -> Any:
        time_min, time_max = _parse_time_range(arguments)
        all_events = self._client.list_all_family_events(time_min, time_max)
        return {
            member: _format_events(events)
            for member, events in all_events.items()
        }

    def _find_conflicts(self, arguments: dict[str, Any]) -> Any:
        time_min, time_max = _parse_time_range(arguments)
        return self._client.find_conflicts(time_min, time_max)

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list_events",
                        "list_all_family",
                        "find_conflicts",
                        "list_calendars",
                    ],
                    "description": "The calendar read action to perform",
                },
                "family_member": {
                    "type": "string",
                    "description": (
                        "Which family member's calendar to read. "
                        "Omit to read the primary calendar."
                    ),
                },
                "days_ahead": {
                    "type": "integer",
                    "description": "Number of days ahead to fetch events for. Default 7.",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format. Defaults to today.",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format.",
                },
            },
            "required": ["action"],
        }


class CalendarWriteTool(Tool):
    """Create events on family calendars."""

    name = "google_calendar_write"
    description = (
        "Create a new event on Google Calendar. "
        "Provide a summary/title, start time, and optionally an end time, "
        "location, and description. Can target a specific family member's calendar."
    )
    requires_confirmation = False  # User explicitly said no confirmation needed

    def __init__(self, client: GoogleCalendarClient) -> None:
        self._client = client

    async def execute(self, arguments: dict[str, Any]) -> Any:
        summary = arguments.get("summary", "")
        if not summary:
            return {"error": "Event summary/title is required"}

        start_str = arguments.get("start")
        if not start_str:
            return {"error": "Event start time is required"}

        try:
            start = datetime.fromisoformat(start_str)
        except ValueError:
            return {"error": f"Invalid start time format: {start_str}"}

        end = None
        if arguments.get("end"):
            try:
                end = datetime.fromisoformat(arguments["end"])
            except ValueError:
                return {"error": f"Invalid end time format: {arguments['end']}"}

        duration_minutes = arguments.get("duration_minutes")
        if end is None and duration_minutes:
            end = start + timedelta(minutes=duration_minutes)

        family_member = arguments.get("family_member")
        calendar_id = self._client.get_calendar_id(family_member)

        result = self._client.create_event(
            summary=summary,
            start=start,
            end=end,
            description=arguments.get("description"),
            location=arguments.get("location"),
            calendar_id=calendar_id,
        )

        return {
            "status": "created",
            "event_id": result.get("id"),
            "summary": result.get("summary"),
            "start": result.get("start", {}).get("dateTime"),
            "end": result.get("end", {}).get("dateTime"),
            "link": result.get("htmlLink"),
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Event title/summary",
                },
                "start": {
                    "type": "string",
                    "description": (
                        "Event start time in ISO 8601 format "
                        "(e.g., 2026-04-20T14:00:00-07:00)"
                    ),
                },
                "end": {
                    "type": "string",
                    "description": "Event end time in ISO 8601 format. Optional.",
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": (
                        "Duration in minutes. Used if end time is not provided. "
                        "Defaults to 60 minutes."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Event description/notes",
                },
                "location": {
                    "type": "string",
                    "description": "Event location or address",
                },
                "family_member": {
                    "type": "string",
                    "description": (
                        "Which family member's calendar to add the event to. "
                        "Omit to use the primary calendar."
                    ),
                },
            },
            "required": ["summary", "start"],
        }


class NotifyTool(Tool):
    """Send a notification to family members."""

    name = "notify"
    description = (
        "Send a message notification to family members via iMessage. "
        "Use this to deliver calendar digests, conflict alerts, or confirmations."
    )
    requires_confirmation = False

    def __init__(self, notification_service: Any) -> None:
        self._notification_service = notification_service

    async def execute(self, arguments: dict[str, Any]) -> Any:
        body = arguments.get("message", "")
        if not body:
            return {"error": "Message body is required"}

        target = arguments.get("target", "family")
        subject = arguments.get("subject")
        request_id = arguments.get("request_id", "")

        if target == "family":
            results = await self._notification_service.send_to_family(
                body=body, subject=subject, request_id=request_id
            )
            return {
                "sent_to": "family",
                "results": [
                    {"success": r.success, "error": r.error} for r in results
                ],
            }
        else:
            result = await self._notification_service.send_to_user(
                user_id=target, body=body, subject=subject, request_id=request_id
            )
            return {
                "sent_to": target,
                "success": result.success,
                "error": result.error,
            }

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send",
                },
                "subject": {
                    "type": "string",
                    "description": "Optional subject line",
                },
                "target": {
                    "type": "string",
                    "description": (
                        "Who to notify: 'family' for all members, "
                        "or a specific user_id like 'primary'"
                    ),
                    "default": "family",
                },
            },
            "required": ["message"],
        }


def _parse_time_range(
    arguments: dict[str, Any],
) -> tuple[datetime, datetime]:
    """Parse time range from tool arguments."""
    now = datetime.now(UTC)

    if arguments.get("start_date"):
        time_min = datetime.fromisoformat(arguments["start_date"]).replace(
            tzinfo=UTC
        )
    else:
        time_min = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if arguments.get("end_date"):
        time_max = datetime.fromisoformat(arguments["end_date"]).replace(
            tzinfo=UTC
        )
    else:
        days = arguments.get("days_ahead", 7)
        time_max = time_min + timedelta(days=days)

    return time_min, time_max


def _format_events(events: list[dict]) -> list[dict]:
    """Format raw Google Calendar events into a cleaner structure."""
    formatted = []
    for event in events:
        start = event.get("start", {})
        end = event.get("end", {})
        formatted.append({
            "id": event.get("id"),
            "summary": event.get("summary", "Untitled"),
            "start": start.get("dateTime") or start.get("date"),
            "end": end.get("dateTime") or end.get("date"),
            "location": event.get("location"),
            "description": event.get("description"),
            "status": event.get("status"),
            "all_day": "date" in start and "dateTime" not in start,
        })
    return formatted
