"""Calendar workflow — LangGraph state machine with specialist nodes.

Mirrors the orchestrator's pattern (StateGraph + conditional edges + closures
for deps). Each node has one job; the LLM only runs for intent classification
and event parsing — everything else (date enrichment, validation, conflict
check, calendar write, reply rendering) is pure code that can't be skipped.

Graph:

    START
      │
      ▼
    enrich_with_date_context          (pure code)
      │
      ▼
    classify_intent                   (LLM: typed action enum)
      │
      ├─ action=create   ──▶ parse_event ─▶ validate ─┬─▶ build_reply (clarify)
      │                                                │
      │                                                ▼
      │                                       check_conflicts ─┬─▶ build_reply (conflict)
      │                                                        │
      │                                                        ▼
      │                                                   create_event
      │                                                        │
      │                                                        ▼
      │                                                   build_reply (confirmation)
      │
      ├─ action=delete   ──▶ mark_unsupported   ──▶ build_reply
      ├─ action=query    ──▶ mark_unsupported   ──▶ build_reply
      └─ action=clarify  ──▶ mark_clarification ──▶ build_reply
                                                        │
                                                        ▼
                                                       END
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from zoneinfo import ZoneInfo

import structlog
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    from alfred.agents.calendar.google_client import GoogleCalendarClient
    from alfred.agents.calendar.icloud_client import IcloudCalendarClient
    from alfred.routing.clients import LiteLLMClient

    CalendarClient = IcloudCalendarClient | GoogleCalendarClient

log = structlog.get_logger()


# ── Typed schemas the LLM specialists produce ─────────────────────────────


class IntentClassification(BaseModel):
    """Output of the intent-classifier node."""

    action: Literal["create", "delete", "query", "clarify"] = Field(
        description="What the user wants to do with the calendar."
    )
    reasoning: str = Field(default="", description="One short sentence.")


class EventDraft(BaseModel):
    """Output of the event-parser node."""

    title: str = Field(description="Concise human-readable event name.")
    start_iso: str = Field(description="ISO 8601 with timezone offset.")
    end_iso: str = Field(description="ISO 8601 with timezone offset.")
    location: str | None = None
    description: str | None = None


# ── Graph state ───────────────────────────────────────────────────────────


class CalendarState(TypedDict, total=False):
    # Inputs
    email_body: str

    # Date context (enriched up front)
    today_iso: str
    today_day_of_week: str
    timezone_name: str
    upcoming_days: list[dict]

    # Classification
    action: str

    # Parsed event (for create branch)
    parsed_event: dict
    completeness_issues: list[str]

    # Conflicts
    conflicts: list[dict]

    # Create result
    created_event: dict

    # Final outcome — drives build_reply
    outcome: str  # created | conflict | incomplete | clarification | unsupported | error
    error_message: str

    # Reply text/html (output of build_reply)
    reply_plain: str
    reply_html: str

    # Usage accumulators
    input_tokens: int
    output_tokens: int


# ── Per-node prompts (narrow scope) ──────────────────────────────────────


_INTENT_PROMPT = (
    "Classify the user's email into one of these calendar actions:\n"
    "- create: user wants to schedule a new event\n"
    "- delete: user wants to cancel/remove an existing event\n"
    "- query: user wants to see what's on the calendar\n"
    "- clarify: the request is unclear, ambiguous, or outside calendar scope\n"
    "\n"
    "Return a typed IntentClassification.\n"
    "\n"
    "Email body:\n"
    "{email_body}"
)


_PARSE_PROMPT = (
    "Extract the event the user wants to schedule.\n"
    "\n"
    "Today is {today_day_of_week}, {today_iso} ({timezone_name}).\n"
    "\n"
    "Upcoming 15 days (use this table to resolve relative dates like "
    "'next Monday', 'tomorrow', 'the next one'):\n"
    "{upcoming_table}\n"
    "\n"
    "Rules:\n"
    "- start_iso and end_iso must be ISO 8601 with the timezone offset for {timezone_name}.\n"
    "- If duration is not specified, default to 1 hour.\n"
    "- title should be a short human-readable event name.\n"
    "- location and description are optional; only fill them if the user mentions them.\n"
    "\n"
    "Email body:\n"
    "{email_body}"
)


# ── Node builders ─────────────────────────────────────────────────────────


def _format_upcoming_table(upcoming: list[dict]) -> str:
    return "\n".join(
        f"  +{d['offset_days']:>2}d  {d['iso_date']}  {d['day_of_week']}"
        for d in upcoming
    )


def _make_specialist(
    litellm_client: LiteLLMClient, model_name: str, output_type: type
) -> Agent:
    """Build a typed-output Pydantic AI agent for a single workflow step."""
    model = OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(
            base_url=f"{litellm_client.base_url}/v1",
            api_key=litellm_client._api_key,  # noqa: SLF001
        ),
    )
    return Agent(model=model, output_type=output_type)


def _format_conflict_text(event: dict, conflicts: list[dict]) -> str:
    lines = [
        "Before I add this, you already have an event during that time:",
        "",
    ]
    for c in conflicts:
        lines.append(f"  • {c['summary']} ({c['start']} – {c['end']})")
    lines.append("")
    lines.append(
        f"Want me to schedule '{event['title']}' anyway, or pick a different time?"
    )
    return "\n".join(lines)


def _format_incomplete_text(issues: list[str]) -> str:
    return (
        "I couldn't fully parse your request.\n"
        "Issues: " + "; ".join(issues) + ".\n\n"
        "Reply with the full details, e.g. 'Meeting with Eric next Wednesday at 10am'."
    )


def build_calendar_graph(
    calendar_client: CalendarClient,
    timezone_name: str,
    litellm_client: LiteLLMClient,
    model_name: str = "local-default",
) -> Any:
    """Compile the calendar workflow with deps closed over."""
    intent_agent = _make_specialist(litellm_client, model_name, IntentClassification)
    parse_agent = _make_specialist(litellm_client, model_name, EventDraft)

    # ── Nodes ────────────────────────────────────────────────────────────

    async def enrich_with_date_context(state: CalendarState) -> dict:
        tz = ZoneInfo(timezone_name)
        now = datetime.now(tz)
        today = now.date()
        upcoming = [
            {
                "offset_days": i,
                "iso_date": (today + timedelta(days=i)).isoformat(),
                "day_of_week": (today + timedelta(days=i)).strftime("%A"),
            }
            for i in range(15)
        ]
        return {
            "today_iso": today.isoformat(),
            "today_day_of_week": today.strftime("%A"),
            "timezone_name": timezone_name,
            "upcoming_days": upcoming,
        }

    async def classify_intent(state: CalendarState) -> dict:
        prompt = _INTENT_PROMPT.format(email_body=state["email_body"])
        result = await intent_agent.run(prompt)
        usage = result.usage()
        log.info(
            "calendar_intent_classified",
            action=result.output.action,
            reasoning=result.output.reasoning[:120],
        )
        return {
            "action": result.output.action,
            "input_tokens": state.get("input_tokens", 0) + (usage.input_tokens or 0),
            "output_tokens": state.get("output_tokens", 0) + (usage.output_tokens or 0),
        }

    async def parse_event(state: CalendarState) -> dict:
        prompt = _PARSE_PROMPT.format(
            today_day_of_week=state["today_day_of_week"],
            today_iso=state["today_iso"],
            timezone_name=state["timezone_name"],
            upcoming_table=_format_upcoming_table(state["upcoming_days"]),
            email_body=state["email_body"],
        )
        try:
            result = await parse_agent.run(prompt)
            usage = result.usage()
            ev = result.output
            log.info(
                "calendar_event_parsed",
                title=ev.title,
                start_iso=ev.start_iso,
                end_iso=ev.end_iso,
            )
            return {
                "parsed_event": ev.model_dump(),
                "input_tokens": state.get("input_tokens", 0) + (usage.input_tokens or 0),
                "output_tokens": state.get("output_tokens", 0) + (usage.output_tokens or 0),
            }
        except Exception as e:  # noqa: BLE001
            log.error("calendar_event_parse_failed", error=str(e))
            return {
                "outcome": "incomplete",
                "completeness_issues": [f"parse_error: {e}"],
            }

    async def validate_event(state: CalendarState) -> dict:
        ev = state.get("parsed_event")
        if not ev:
            return {"outcome": "incomplete", "completeness_issues": ["no event parsed"]}
        issues: list[str] = []
        for field in ("title", "start_iso", "end_iso"):
            if not ev.get(field):
                issues.append(f"missing {field}")
        if not issues:
            try:
                datetime.fromisoformat(ev["start_iso"])
                datetime.fromisoformat(ev["end_iso"])
            except ValueError as e:
                issues.append(f"invalid ISO datetime: {e}")
        if issues:
            return {"outcome": "incomplete", "completeness_issues": issues}
        return {}

    async def check_conflicts(state: CalendarState) -> dict:
        ev = state["parsed_event"]
        start_dt = datetime.fromisoformat(ev["start_iso"])
        end_dt = datetime.fromisoformat(ev["end_iso"])
        existing = calendar_client.list_events(
            time_min=start_dt - timedelta(minutes=1),
            time_max=end_dt + timedelta(minutes=1),
        )
        conflicts: list[dict] = []
        for e in existing:
            ev_start_raw = (e.get("start") or {}).get("dateTime")
            ev_end_raw = (e.get("end") or {}).get("dateTime")
            if not ev_start_raw or not ev_end_raw:
                continue
            try:
                e_start = datetime.fromisoformat(ev_start_raw)
                e_end = datetime.fromisoformat(ev_end_raw)
            except ValueError:
                continue
            if e_start < end_dt and e_end > start_dt:
                conflicts.append(
                    {
                        "summary": e.get("summary", "Untitled"),
                        "start": ev_start_raw,
                        "end": ev_end_raw,
                    }
                )
        log.info(
            "calendar_conflict_check",
            existing_in_window=len(existing),
            conflicts=len(conflicts),
        )
        if conflicts:
            return {"conflicts": conflicts, "outcome": "conflict"}
        return {"conflicts": []}

    async def create_event(state: CalendarState) -> dict:
        ev = state["parsed_event"]
        try:
            start_dt = datetime.fromisoformat(ev["start_iso"])
            end_dt = datetime.fromisoformat(ev["end_iso"])
            result = calendar_client.create_event(
                summary=ev["title"],
                start=start_dt,
                end=end_dt,
                description=ev.get("description"),
                location=ev.get("location"),
            )
            return {
                "created_event": {
                    "id": result.get("id"),
                    "summary": result.get("summary") or ev["title"],
                    "start_iso": ev["start_iso"],
                    "end_iso": ev["end_iso"],
                    "location": ev.get("location"),
                    "description": ev.get("description"),
                    "calendar_name": getattr(calendar_client, "_calendar_name", "")
                    or "primary",
                    "timezone": timezone_name,
                },
                "outcome": "created",
            }
        except Exception as e:  # noqa: BLE001
            log.error("calendar_create_failed", error=str(e))
            return {"outcome": "error", "error_message": str(e)}

    async def mark_unsupported(state: CalendarState) -> dict:
        action = state.get("action", "unknown")
        return {
            "outcome": "unsupported",
            "error_message": (
                f"The '{action}' action isn't supported yet — I can only "
                "schedule new events for now. Cancel/list/update support is coming."
            ),
        }

    async def mark_clarification(state: CalendarState) -> dict:
        return {"outcome": "clarification"}

    async def build_reply(state: CalendarState) -> dict:
        from alfred.notifications.calendar_render import (
            render_clarification_html,
            render_clarification_plain,
            render_event_html,
            render_event_plain,
        )

        outcome = state.get("outcome", "clarification")
        if outcome == "created":
            event = state["created_event"]
            return {
                "reply_plain": render_event_plain(event),
                "reply_html": render_event_html(event),
            }
        if outcome == "conflict":
            txt = _format_conflict_text(state["parsed_event"], state["conflicts"])
            return {
                "reply_plain": render_clarification_plain(txt),
                "reply_html": render_clarification_html(txt),
            }
        if outcome == "incomplete":
            txt = _format_incomplete_text(state.get("completeness_issues", []))
            return {
                "reply_plain": render_clarification_plain(txt),
                "reply_html": render_clarification_html(txt),
            }
        if outcome == "unsupported":
            txt = state.get("error_message", "That action isn't supported yet.")
            return {
                "reply_plain": render_clarification_plain(txt),
                "reply_html": render_clarification_html(txt),
            }
        if outcome == "error":
            txt = (
                "Sorry — something went wrong while processing your request: "
                + state.get("error_message", "unknown error")
            )
            return {
                "reply_plain": render_clarification_plain(txt),
                "reply_html": render_clarification_html(txt),
            }
        # default: clarification
        txt = (
            "I couldn't tell what calendar action you wanted. I can create new "
            "events — try something like 'Lunch with Sarah next Wednesday at 12:30pm'."
        )
        return {
            "reply_plain": render_clarification_plain(txt),
            "reply_html": render_clarification_html(txt),
        }

    # ── Edges ────────────────────────────────────────────────────────────

    def route_after_intent(state: CalendarState) -> str:
        action = state.get("action", "clarify")
        if action == "create":
            return "parse_event"
        if action in ("delete", "query"):
            return "mark_unsupported"
        return "mark_clarification"

    def route_after_validate(state: CalendarState) -> str:
        return "build_reply" if state.get("outcome") == "incomplete" else "check_conflicts"

    def route_after_conflicts(state: CalendarState) -> str:
        return "build_reply" if state.get("outcome") == "conflict" else "create_event"

    # ── Assemble ─────────────────────────────────────────────────────────

    graph = StateGraph(CalendarState)
    graph.add_node("enrich_with_date_context", enrich_with_date_context)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("parse_event", parse_event)
    graph.add_node("validate_event", validate_event)
    graph.add_node("check_conflicts", check_conflicts)
    graph.add_node("create_event", create_event)
    graph.add_node("mark_unsupported", mark_unsupported)
    graph.add_node("mark_clarification", mark_clarification)
    graph.add_node("build_reply", build_reply)

    graph.set_entry_point("enrich_with_date_context")
    graph.add_edge("enrich_with_date_context", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "parse_event": "parse_event",
            "mark_unsupported": "mark_unsupported",
            "mark_clarification": "mark_clarification",
        },
    )
    graph.add_edge("parse_event", "validate_event")
    graph.add_conditional_edges(
        "validate_event",
        route_after_validate,
        {
            "build_reply": "build_reply",
            "check_conflicts": "check_conflicts",
        },
    )
    graph.add_conditional_edges(
        "check_conflicts",
        route_after_conflicts,
        {
            "build_reply": "build_reply",
            "create_event": "create_event",
        },
    )
    graph.add_edge("create_event", "build_reply")
    graph.add_edge("mark_unsupported", "build_reply")
    graph.add_edge("mark_clarification", "build_reply")
    graph.add_edge("build_reply", END)

    return graph.compile()
