"""Calendar agent — family calendar management, scheduling, and digests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from alfred.agents.base import AgentBase, AgentContext, AgentResult
from alfred.agents.calendar.google_client import GoogleCalendarClient
from alfred.core.models import TokenUsage
from alfred.routing.clients import LiteLLMClient

log = structlog.get_logger()


SYSTEM_PROMPT = """\
You are a family calendar assistant managing schedules for a household.

Family members: {family_members}
Today's date: {today}
Timezone: {timezone}

Your capabilities:
- Read events from all family members' Google Calendars (list_events, list_all_family_events)
- Create new events on any family member's calendar (create_event)
- Detect scheduling conflicts between family members (find_conflicts)
- Generate daily and weekly schedule digests
- Send notifications to family members via iMessage (notify_family)

Rules for creating events:
- Use ISO 8601 format with timezone {timezone} for all dates/times
- If no duration is specified, default to 1 hour
- If the sender doesn't specify whose calendar, use their own
- If they refer to another family member by name, use that person's calendar

Rules for digests and schedule queries:
- Always fetch events from ALL family calendars for the relevant time range
- Group events by day
- Include start times, titles, and locations when available
- Highlight conflicts between family members
- Keep formatting clean and scannable for iMessage delivery

SCOPE — STRICT:
You handle ONLY calendar, scheduling, and event-coordination requests.
If the user asks about anything outside that scope — math, general
knowledge, news, recipes, research, coding, personal advice, etc. —
respond with exactly:

  "I'm the calendar assistant — I can only help with scheduling,
  events, and calendar coordination. That request is outside my scope."

Do not attempt to answer out-of-scope questions even if you know the
answer. Do not invoke any tools for out-of-scope requests.
"""


class CalendarConfig:
    """Calendar-specific configuration loaded from calendar.yaml."""

    def __init__(self, config_dir: Path = Path("config")) -> None:
        config_path = config_dir / "calendar.yaml"
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        self.timezone = data.get("timezone", "UTC")
        self.family: list[dict[str, str]] = data.get("family", [])
        self.calendar_ids: dict[str, str] = {
            m["name"]: m["calendar_id"]
            for m in self.family
            if m.get("name") and m.get("calendar_id")
        }

    @property
    def family_names(self) -> list[str]:
        return [m["name"] for m in self.family if m.get("name")]


@dataclass
class CalendarDeps:
    """Dependencies passed into every tool invocation by Pydantic AI."""

    google: GoogleCalendarClient
    config: CalendarConfig
    notify: Any | None
    request_id: str = ""


def _format_events(events: list[dict]) -> list[dict]:
    """Flatten raw Google Calendar events to a cleaner, model-friendly shape."""
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


def _today_utc_midnight() -> datetime:
    return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def _build_agent(
    model_name: str, base_url: str, api_key: str
) -> Agent[CalendarDeps, str]:
    """Construct a Pydantic AI agent bound to a given LiteLLM model alias."""
    model = OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(base_url=f"{base_url}/v1", api_key=api_key),
    )
    agent: Agent[CalendarDeps, str] = Agent(model=model, deps_type=CalendarDeps)

    @agent.instructions
    def _instructions(ctx: RunContext[CalendarDeps]) -> str:
        members = (
            ", ".join(ctx.deps.config.family_names)
            if ctx.deps.config.family_names
            else "Not yet configured"
        )
        return SYSTEM_PROMPT.format(
            family_members=members,
            today=datetime.now(UTC).strftime("%A, %B %d, %Y"),
            timezone=ctx.deps.config.timezone,
        )

    @agent.tool
    async def list_events(
        ctx: RunContext[CalendarDeps],
        days_ahead: int = 7,
        family_member: str | None = None,
    ) -> list[dict]:
        """Get upcoming events for one family member (defaults to primary calendar)."""
        time_min = _today_utc_midnight()
        time_max = time_min + timedelta(days=days_ahead)
        cal_id = ctx.deps.google.get_calendar_id(family_member)
        events = ctx.deps.google.list_events(
            time_min=time_min, time_max=time_max, calendar_id=cal_id
        )
        return _format_events(events)

    @agent.tool
    async def list_all_family_events(
        ctx: RunContext[CalendarDeps], days_ahead: int = 7
    ) -> dict[str, list[dict]]:
        """Get events for ALL family members in the next N days. Use this for digests."""
        time_min = _today_utc_midnight()
        time_max = time_min + timedelta(days=days_ahead)
        all_events = ctx.deps.google.list_all_family_events(time_min, time_max)
        return {m: _format_events(e) for m, e in all_events.items()}

    @agent.tool
    async def find_conflicts(
        ctx: RunContext[CalendarDeps], days_ahead: int = 7
    ) -> Any:
        """Detect overlapping events across family members in the next N days."""
        time_min = _today_utc_midnight()
        time_max = time_min + timedelta(days=days_ahead)
        return ctx.deps.google.find_conflicts(time_min, time_max)

    @agent.tool
    async def create_event(
        ctx: RunContext[CalendarDeps],
        summary: str,
        start: str,
        end: str | None = None,
        duration_minutes: int | None = None,
        description: str | None = None,
        location: str | None = None,
        family_member: str | None = None,
    ) -> dict:
        """Create a calendar event. Times must be ISO 8601 (e.g. 2026-04-20T14:00:00-07:00)."""
        try:
            start_dt = datetime.fromisoformat(start)
        except ValueError:
            return {"error": f"Invalid start time format: {start}"}

        if end:
            try:
                end_dt = datetime.fromisoformat(end)
            except ValueError:
                return {"error": f"Invalid end time format: {end}"}
        else:
            end_dt = start_dt + timedelta(minutes=duration_minutes or 60)

        cal_id = ctx.deps.google.get_calendar_id(family_member)
        result = ctx.deps.google.create_event(
            summary=summary,
            start=start_dt,
            end=end_dt,
            description=description,
            location=location,
            calendar_id=cal_id,
        )
        return {
            "status": "created",
            "event_id": result.get("id"),
            "summary": result.get("summary"),
            "start": result.get("start", {}).get("dateTime"),
            "end": result.get("end", {}).get("dateTime"),
            "link": result.get("htmlLink"),
        }

    @agent.tool
    async def notify_family(
        ctx: RunContext[CalendarDeps],
        message: str,
        subject: str | None = None,
        target: str = "family",
    ) -> dict:
        """Send a notification (iMessage) to the whole family or a specific user_id."""
        if ctx.deps.notify is None:
            return {"error": "Notification service not configured"}
        if target == "family":
            results = await ctx.deps.notify.send_to_family(
                body=message, subject=subject, request_id=ctx.deps.request_id
            )
            return {
                "sent_to": "family",
                "results": [
                    {"success": r.success, "error": r.error} for r in results
                ],
            }
        result = await ctx.deps.notify.send_to_user(
            user_id=target, body=message, subject=subject, request_id=ctx.deps.request_id
        )
        return {"sent_to": target, "success": result.success, "error": result.error}

    return agent


class CalendarAgent(AgentBase):
    """Family calendar agent — thin wrapper around a Pydantic AI agent."""

    name = "calendar"
    description = "Family calendar management, scheduling, and conflict detection"
    model = "local-default"

    def __init__(
        self,
        google_client: GoogleCalendarClient,
        litellm_client: LiteLLMClient,
        calendar_config: CalendarConfig,
        notification_service: Any = None,
    ) -> None:
        self._google = google_client
        self._config = calendar_config
        self._notify = notification_service
        base_url = litellm_client.base_url
        api_key = litellm_client._api_key
        self._local_agent = _build_agent("local-default", base_url, api_key)
        self._cloud_agent = _build_agent("cloud-default", base_url, api_key)

    async def run(self, message: str, context: AgentContext) -> AgentResult:
        # Heavier formatting tasks go to the cloud model
        use_cloud = any(
            kw in message.lower()
            for kw in ("digest", "week", "summary", "briefing", "upcoming")
        )
        agent = self._cloud_agent if use_cloud else self._local_agent
        model_name = "cloud-default" if use_cloud else "local-default"

        deps = CalendarDeps(
            google=self._google,
            config=self._config,
            notify=self._notify,
            request_id=context.request_id,
        )

        result = await agent.run(message, deps=deps)
        usage = result.usage()

        return AgentResult(
            message=str(result.output),
            data={"model": model_name},
            usage=TokenUsage(
                provider="litellm",
                model=model_name,
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
            ),
        )
