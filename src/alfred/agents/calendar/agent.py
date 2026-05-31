"""Calendar agent — thin wrapper around the LangGraph calendar workflow.

The agent does NOT run a tool-using Pydantic AI loop. Instead it constructs
graph state from the inbound message and invokes the compiled workflow,
which uses code-controlled nodes for everything except the two LLM steps
(intent classification + event parsing). See `workflow.py` for the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import yaml
from opentelemetry import trace

from alfred.agents.base import AgentBase, AgentContext, AgentResult
from alfred.agents.calendar.google_client import GoogleCalendarClient
from alfred.agents.calendar.icloud_client import IcloudCalendarClient
from alfred.agents.calendar.workflow import build_calendar_graph
from alfred.routing.clients import LiteLLMClient
from scaffold.core.models import TokenUsage

_tracer = trace.get_tracer("alfred.calendar.agent")

CalendarClient = GoogleCalendarClient | IcloudCalendarClient

log = structlog.get_logger()


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
        self.calendar_id: str = data.get("calendar_id", "primary")
        # "icloud" (CalDAV + app password) or "google" (OAuth)
        self.provider: str = data.get("provider", "icloud")
        # iCloud-only: which named calendar to write to ("" → principal default)
        self.icloud_calendar_name: str = data.get("icloud_calendar_name", "")
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
class CalendarAgentResult:
    """Internal aggregate of what the workflow produced."""

    reply_plain: str
    reply_html: str | None
    created_event: dict | None
    outcome: str
    input_tokens: int
    output_tokens: int


class CalendarAgent(AgentBase):
    """Family calendar agent — runs the LangGraph calendar workflow."""

    name = "calendar"
    description = "Family calendar management, scheduling, and conflict detection"
    model = "local-default"

    def __init__(
        self,
        google_client: CalendarClient,
        litellm_client: LiteLLMClient,
        calendar_config: CalendarConfig,
        notification_service: Any = None,  # noqa: ARG002 — kept for registry signature compat
        now_fn: Any = None,  # optional override for "today" in the workflow (simulation)
    ) -> None:
        self._client = google_client
        self._config = calendar_config
        self._litellm = litellm_client
        # Build the compiled graph once at construction time so subsequent
        # runs don't repeat the assembly work.
        self._graph = build_calendar_graph(
            calendar_client=google_client,
            timezone_name=calendar_config.timezone,
            litellm_client=litellm_client,
            model_name="local-default",
            now_fn=now_fn,
        )

    async def run(self, message: str, context: AgentContext) -> AgentResult:  # noqa: ARG002
        # Parent span for the whole workflow — every node-level span and
        # every LLM span ends up as a child of this one in Phoenix.
        with _tracer.start_as_current_span("calendar.workflow") as span:
            span.set_attribute("openinference.span.kind", "AGENT")
            span.set_attribute("alfred.agent", "calendar")
            span.set_attribute("alfred.email_body_chars", len(message))
            initial_state = {"email_body": message}
            final_state = await self._graph.ainvoke(initial_state)
            outcome = final_state.get("outcome", "unknown")
            span.set_attribute("alfred.outcome", outcome)
            complexity = final_state.get("complexity", "simple")
            span.set_attribute("alfred.complexity", complexity)

        reply_plain = final_state.get("reply_plain", "")
        reply_html = final_state.get("reply_html")
        outcome = final_state.get("outcome", "unknown")
        created_event = final_state.get("created_event")
        input_tokens = final_state.get("input_tokens", 0)
        output_tokens = final_state.get("output_tokens", 0)

        log.info(
            "calendar_workflow_done",
            outcome=outcome,
            event_created=bool(created_event),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # AgentResult.message is the plain-text reply the inbox poller will
        # render into the email body. AgentResult.data["event"] is checked by
        # the poller to decide which renderer (confirmation vs clarification)
        # to use. The workflow's own rendering is canonical now, but the
        # poller still gets the event for the structured-card path.
        return AgentResult(
            message=reply_plain,
            data={
                "model": "local-default",
                "event": created_event,
                "outcome": outcome,
                "reply_html": reply_html,
            },
            usage=TokenUsage(
                provider="litellm",
                model="local-default",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )
