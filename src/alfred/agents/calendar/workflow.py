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
from opentelemetry import trace
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
tracer = trace.get_tracer("alfred.calendar.workflow")

# openinference's span-kind attribute. Phoenix uses this to label/color
# spans in its UI; without it spans show up as "unknown type".
_OI_SPAN_KIND = "openinference.span.kind"


def _traced_node(name: str, fn):
    """Wrap a graph node so each invocation emits a CHAIN child span.

    The parent span is set by CalendarAgent.run (kind=AGENT) so the whole
    workflow appears as one trace with the node spans as children.
    """

    async def wrapper(state):  # type: ignore[no-untyped-def]
        with tracer.start_as_current_span(f"calendar.{name}") as span:
            span.set_attribute(_OI_SPAN_KIND, "CHAIN")
            span.set_attribute("alfred.node", name)
            action = state.get("action") if isinstance(state, dict) else None
            if action:
                span.set_attribute("alfred.action", action)
            result = await fn(state)
            if isinstance(result, dict):
                outcome = result.get("outcome")
                if outcome:
                    span.set_attribute("alfred.outcome", outcome)
            return result

    wrapper.__name__ = getattr(fn, "__name__", name)
    return wrapper


# ── Typed schemas the LLM specialists produce ─────────────────────────────


class IntentClassification(BaseModel):
    """Output of the intent-classifier node."""

    action: Literal["create", "delete", "query", "clarify"] = Field(
        description="What the user wants to do with the calendar."
    )
    reasoning: str = Field(default="", description="One short sentence.")


class RecurrenceRule(BaseModel):
    """How an event repeats. Maps directly onto iCalendar RRULE."""

    frequency: Literal["DAILY", "WEEKLY", "MONTHLY", "YEARLY"] = Field(
        description="How often the event repeats."
    )
    interval: int = Field(default=1, description="Repeat every N (e.g., 2 = every other week).")
    byday: list[str] = Field(
        default_factory=list,
        description=(
            "Days of the week the event runs on for weekly patterns. "
            "Use two-letter codes: MO TU WE TH FR SA SU."
        ),
    )
    until_iso: str | None = Field(
        default=None,
        description="ISO date when the recurrence ends (inclusive). Mutually exclusive with count.",
    )
    count: int | None = Field(
        default=None,
        description="Total number of occurrences. Mutually exclusive with until_iso.",
    )


class EventDraft(BaseModel):
    """Output of the event-parser node."""

    title: str = Field(description="Concise human-readable event name.")
    start_iso: str = Field(description="ISO 8601 with timezone offset.")
    end_iso: str = Field(description="ISO 8601 with timezone offset.")
    location: str | None = None
    description: str | None = None
    recurrence: RecurrenceRule | None = Field(
        default=None,
        description=(
            "Set only if the user explicitly asks for a recurring event "
            "('weekly', 'every Tuesday', 'monthly on the 15th', etc.). "
            "Leave null for one-off events."
        ),
    )


class DeleteTarget(BaseModel):
    """Output of the delete-parser node — what event(s) to remove."""

    intent_summary: str = Field(
        description=(
            "Brief restatement of what the user wants to delete, in their own words. "
            "Examples: 'the meeting with Wayne', 'lunch with Sarah', 'the standup'."
        )
    )
    date_hint_iso: str | None = Field(
        default=None,
        description=(
            "ISO date the event is on, if the user specified one (resolve relative "
            "dates using the upcoming-days table). Null if no date hint."
        ),
    )
    time_hint: str | None = Field(
        default=None,
        description="Optional time like '3pm' or 'morning'. Null if unspecified.",
    )


class DeleteMatchDecision(BaseModel):
    """Output of the match-reasoning node — which event (if any) to delete."""

    confidence: Literal["high", "medium", "low", "none"] = Field(
        description=(
            "high: clearly the right event — proceed with delete. "
            "medium: likely match but worth confirming with the user first. "
            "low: weak signal across multiple candidates. "
            "none: no candidate plausibly matches."
        )
    )
    event_id: str | None = Field(
        default=None,
        description="The matched event's id, or null if confidence is 'none'.",
    )
    reasoning: str = Field(
        default="",
        description="One short sentence explaining the match (shown to the user).",
    )


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

    # Delete branch
    delete_target: dict
    matching_events: list[dict]
    match_decision: dict
    deleted_event: dict

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


_DELETE_PARSE_PROMPT = (
    "Extract what event the user wants to cancel/delete. You only extract the "
    "intent + date — a downstream step will look at the actual calendar events "
    "and decide which one matches.\n"
    "\n"
    "Today is {today_day_of_week}, {today_iso} ({timezone_name}).\n"
    "\n"
    "Upcoming 15 days (use this table to resolve relative dates):\n"
    "{upcoming_table}\n"
    "\n"
    "Rules:\n"
    "- intent_summary: a brief restatement of the target event in the user's words.\n"
    "  'Cancel my lunch with Sarah' -> 'lunch with Sarah'.\n"
    "  'Remove the event with Wayne this Friday' -> 'the meeting with Wayne'.\n"
    "  'Delete the standup' -> 'the standup'.\n"
    "- date_hint_iso: ISO date if the user gives one (resolve relative dates "
    "  using the upcoming-days table). Null if no date is given.\n"
    "- time_hint: optional, e.g. '3pm' or 'morning'.\n"
    "\n"
    "Email body:\n"
    "{email_body}"
)


_DELETE_MATCH_PROMPT = (
    "Decide which calendar event the user wants to delete.\n"
    "\n"
    "User's intent: {intent_summary}\n"
    "Date the user gave: {date_hint}\n"
    "User's original message:\n"
    "{email_body}\n"
    "\n"
    "Candidate events (within the searched window):\n"
    "{candidates_block}\n"
    "\n"
    "Pick the best match. The match can be against title, location, or "
    "description — people's names sometimes end up in any of these fields.\n"
    "\n"
    "Confidence rubric:\n"
    "- high: one clear match. Title, location, OR description directly "
    "  references something the user said. Proceed with deletion.\n"
    "- medium: a likely match exists but it's worth confirming with the "
    "  user before deleting (e.g., only a partial name match, or one of "
    "  two plausible candidates).\n"
    "- low: multiple weak matches with no clear winner.\n"
    "- none: no candidate plausibly matches.\n"
    "\n"
    "Reasoning should be one short sentence — it gets shown to the user."
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
    "- title: short human-readable event name. Include who it's WITH (names "
    "  of people) or what it's ABOUT directly in the title. Examples: "
    "  'Lunch with Sarah', 'Q3 budget review with finance team', "
    "  'Coffee with Eric'. Do NOT prefix titles with action verbs like "
    "  'Add', 'Create', or 'Schedule' — those are commands, not part of "
    "  the event name.\n"
    "- location: ONLY physical places ('Cafe Luna', '123 Main St') or "
    "  video links ('Zoom', 'meet.google.com/abc'). Do NOT put people's "
    "  names here — they belong in the title.\n"
    "- description: optional free-form notes the user explicitly provides.\n"
    "\n"
    "Recurrence:\n"
    "- Leave recurrence null for one-off events.\n"
    "- Set recurrence ONLY if the user explicitly says it repeats ('weekly',\n"
    "  'every Tuesday', 'every other week', 'monthly on the 15th', etc.).\n"
    "- frequency: DAILY, WEEKLY, MONTHLY, or YEARLY.\n"
    "- interval defaults to 1 (every cycle). Use 2 for 'every other'.\n"
    "- byday is for weekly patterns; two-letter codes (MO TU WE TH FR SA SU).\n"
    "  Example: 'every Tuesday and Thursday' -> byday=[TU, TH].\n"
    "- until_iso: only set if the user gives an end date.\n"
    "- count: only set if the user says 'for N weeks' or similar.\n"
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
    delete_parse_agent = _make_specialist(litellm_client, model_name, DeleteTarget)
    delete_match_agent = _make_specialist(litellm_client, model_name, DeleteMatchDecision)

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
        """Informational only — conflicts are surfaced in the reply but do
        NOT block creation. Recurring events skip the check entirely because
        they'll naturally overlap with other things over time."""
        ev = state["parsed_event"]
        if ev.get("recurrence"):
            log.info("calendar_conflict_check_skipped", reason="recurring")
            return {"conflicts": []}
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
        return {"conflicts": conflicts}

    async def create_event(state: CalendarState) -> dict:
        ev = state["parsed_event"]
        try:
            start_dt = datetime.fromisoformat(ev["start_iso"])
            end_dt = datetime.fromisoformat(ev["end_iso"])
            recurrence = ev.get("recurrence")
            result = calendar_client.create_event(
                summary=ev["title"],
                start=start_dt,
                end=end_dt,
                description=ev.get("description"),
                location=ev.get("location"),
                recurrence=recurrence,
            )
            return {
                "created_event": {
                    "id": result.get("id"),
                    "summary": result.get("summary") or ev["title"],
                    "start_iso": ev["start_iso"],
                    "end_iso": ev["end_iso"],
                    "location": ev.get("location"),
                    "description": ev.get("description"),
                    "recurrence": recurrence,
                    "calendar_name": getattr(calendar_client, "_calendar_name", "")
                    or "primary",
                    "timezone": timezone_name,
                    # Conflicts are informational, not blocking. Surfaced in
                    # the reply as a "Heads up" section if present.
                    "potential_conflicts": state.get("conflicts", []),
                },
                "outcome": "created",
            }
        except Exception as e:  # noqa: BLE001
            log.error("calendar_create_failed", error=str(e))
            return {"outcome": "error", "error_message": str(e)}

    async def parse_delete_target(state: CalendarState) -> dict:
        prompt = _DELETE_PARSE_PROMPT.format(
            today_day_of_week=state["today_day_of_week"],
            today_iso=state["today_iso"],
            timezone_name=state["timezone_name"],
            upcoming_table=_format_upcoming_table(state["upcoming_days"]),
            email_body=state["email_body"],
        )
        try:
            result = await delete_parse_agent.run(prompt)
            usage = result.usage()
            target = result.output
            log.info(
                "calendar_delete_target_parsed",
                intent_summary=target.intent_summary,
                date_hint_iso=target.date_hint_iso,
            )
            return {
                "delete_target": target.model_dump(),
                "input_tokens": state.get("input_tokens", 0) + (usage.input_tokens or 0),
                "output_tokens": state.get("output_tokens", 0) + (usage.output_tokens or 0),
            }
        except Exception as e:  # noqa: BLE001
            log.error("calendar_delete_parse_failed", error=str(e))
            return {
                "outcome": "delete_not_found",
                "error_message": f"parse_error: {e}",
            }

    async def fetch_candidates_for_delete(state: CalendarState) -> dict:
        """Pure code: fetch all events in the relevant date window, no filtering."""
        target = state.get("delete_target") or {}
        date_hint = target.get("date_hint_iso")
        tz = ZoneInfo(state["timezone_name"])

        if date_hint:
            try:
                day = datetime.fromisoformat(date_hint).date()
            except ValueError:
                day = datetime.now(tz).date()
            time_min = datetime.combine(day, datetime.min.time(), tzinfo=tz)
            time_max = time_min + timedelta(days=1)
        else:
            time_min = datetime.now(tz)
            time_max = time_min + timedelta(days=30)

        existing = calendar_client.list_events(time_min=time_min, time_max=time_max)
        candidates = [
            {
                "id": ev.get("id"),
                "summary": ev.get("summary", "Untitled"),
                "location": ev.get("location"),
                "description": ev.get("description"),
                "start": (ev.get("start") or {}).get("dateTime"),
                "end": (ev.get("end") or {}).get("dateTime"),
            }
            for ev in existing
        ]
        log.info(
            "calendar_delete_candidates",
            window_events=len(candidates),
            date_hint=date_hint,
        )
        return {"matching_events": candidates}

    async def reason_about_delete_match(state: CalendarState) -> dict:
        """LLM specialist: pick the best match with a confidence rating."""
        target = state.get("delete_target") or {}
        candidates = state.get("matching_events") or []
        if not candidates:
            log.info("calendar_delete_no_candidates")
            return {
                "match_decision": {
                    "confidence": "none",
                    "event_id": None,
                    "reasoning": "No events in the searched window.",
                },
            }

        # Build a compact, readable candidate block for the prompt.
        lines = []
        for c in candidates:
            loc = f" · location: {c['location']}" if c.get("location") else ""
            desc = f" · notes: {c['description']}" if c.get("description") else ""
            lines.append(
                f"  - id={c['id']}  '{c['summary']}'  ({c['start']} – {c['end']}){loc}{desc}"
            )
        prompt = _DELETE_MATCH_PROMPT.format(
            intent_summary=target.get("intent_summary") or "(no summary)",
            date_hint=target.get("date_hint_iso") or "(none given)",
            email_body=state["email_body"],
            candidates_block="\n".join(lines),
        )
        try:
            result = await delete_match_agent.run(prompt)
            usage = result.usage()
            decision = result.output
            log.info(
                "calendar_delete_decision",
                confidence=decision.confidence,
                event_id=(decision.event_id or "")[:40],
                reasoning=decision.reasoning[:120],
            )
            return {
                "match_decision": decision.model_dump(),
                "input_tokens": state.get("input_tokens", 0) + (usage.input_tokens or 0),
                "output_tokens": state.get("output_tokens", 0) + (usage.output_tokens or 0),
            }
        except Exception as e:  # noqa: BLE001
            log.error("calendar_delete_reason_failed", error=str(e))
            return {
                "match_decision": {
                    "confidence": "none",
                    "event_id": None,
                    "reasoning": f"reasoning failed: {e}",
                },
            }

    async def delete_event_action(state: CalendarState) -> dict:
        decision = state.get("match_decision") or {}
        uid = decision.get("event_id")
        candidates = state.get("matching_events") or []
        candidate_ids = [c.get("id") for c in candidates]
        log.info(
            "calendar_delete_action_start",
            target_uid=uid,
            candidate_ids=candidate_ids,
            uid_match=uid in candidate_ids,
        )
        if not uid:
            return {"outcome": "delete_not_found"}
        # Find the candidate row so we can show details in the reply
        target = next((c for c in candidates if c.get("id") == uid), None)
        if target is None:
            log.warning("calendar_delete_target_not_in_candidates", uid=uid)
            return {"outcome": "delete_not_found"}
        try:
            success = calendar_client.delete_event(uid)
        except Exception as e:  # noqa: BLE001
            log.error("calendar_delete_failed", error=str(e), uid=uid)
            return {"outcome": "error", "error_message": str(e)}
        if not success:
            log.warning("calendar_delete_returned_false", uid=uid)
            return {"outcome": "delete_not_found"}
        return {
            "deleted_event": {
                "summary": target.get("summary"),
                "start_iso": target.get("start"),
                "end_iso": target.get("end"),
                "location": target.get("location"),
                "calendar_name": getattr(calendar_client, "_calendar_name", "")
                or "primary",
                "timezone": timezone_name,
            },
            "outcome": "deleted",
        }

    async def mark_needs_confirmation(state: CalendarState) -> dict:
        return {"outcome": "delete_needs_confirmation"}

    async def mark_unsupported(state: CalendarState) -> dict:
        action = state.get("action", "unknown")
        return {
            "outcome": "unsupported",
            "error_message": (
                f"The '{action}' action isn't supported yet — I can only "
                "schedule and cancel events for now. Listing/queries are coming."
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
        if outcome == "deleted":
            from alfred.notifications.calendar_render import (
                render_event_deleted_html,
                render_event_deleted_plain,
            )

            event = state["deleted_event"]
            return {
                "reply_plain": render_event_deleted_plain(event),
                "reply_html": render_event_deleted_html(event),
            }
        if outcome == "delete_not_found":
            target = state.get("delete_target") or {}
            kw = ", ".join(target.get("title_keywords", []) or []) or "(no keywords)"
            txt = (
                f"I couldn't find an event matching {kw}"
                + (
                    f" on {target['date_hint_iso']}"
                    if target.get("date_hint_iso")
                    else " in the next 30 days"
                )
                + ".\n\nCould you reply with the event title and date so I can find it?"
            )
            return {
                "reply_plain": render_clarification_plain(txt),
                "reply_html": render_clarification_html(txt),
            }
        if outcome == "delete_needs_confirmation":
            decision = state.get("match_decision") or {}
            event_id = decision.get("event_id")
            candidates = state.get("matching_events") or []
            target = next((c for c in candidates if c.get("id") == event_id), None)
            if target is None:
                txt = (
                    f"I think you mean an event matching: {decision.get('reasoning', '')}. "
                    "Could you reply with more details?"
                )
            else:
                start = target.get("start") or ""
                txt = (
                    "I think you want to delete:\n\n"
                    f"  • {target.get('summary', 'Untitled')}\n"
                    f"    {start}\n"
                    + (
                        f"    Location: {target['location']}\n"
                        if target.get("location")
                        else ""
                    )
                    + (
                        f"\nWhy I think this: {decision.get('reasoning', '')}\n"
                        if decision.get("reasoning")
                        else "\n"
                    )
                    + "\nReply 'yes' to confirm, or give me more details to pick a different one."
                )
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
        if action == "delete":
            return "parse_delete_target"
        if action == "query":
            return "mark_unsupported"
        return "mark_clarification"

    def route_after_validate(state: CalendarState) -> str:
        return "build_reply" if state.get("outcome") == "incomplete" else "check_conflicts"

    def route_after_delete_reasoning(state: CalendarState) -> str:
        outcome = state.get("outcome")
        decision = state.get("match_decision") or {}
        confidence = decision.get("confidence", "none")
        log.info(
            "calendar_delete_route_decision",
            current_outcome=outcome,
            confidence=confidence,
            event_id=(decision.get("event_id") or "")[:40],
        )
        # Parse step may have short-circuited if it failed.
        if outcome == "delete_not_found":
            return "build_reply"
        if confidence == "high":
            return "delete_event_action"
        if confidence == "medium":
            return "mark_needs_confirmation"
        return "mark_delete_not_found"  # low / none → ask user to clarify

    async def mark_delete_not_found(state: CalendarState) -> dict:
        return {"outcome": "delete_not_found"}

    # No router after check_conflicts anymore — conflicts are informational,
    # we always proceed to create_event.

    # ── Assemble ─────────────────────────────────────────────────────────

    graph = StateGraph(CalendarState)
    # Every node is wrapped with _traced_node so Phoenix sees the full
    # workflow shape — one child span per node, with action/outcome
    # attributes — rather than just the outer ainvoke() span.
    for node_name, fn in [
        ("enrich_with_date_context", enrich_with_date_context),
        ("classify_intent", classify_intent),
        ("parse_event", parse_event),
        ("validate_event", validate_event),
        ("check_conflicts", check_conflicts),
        ("create_event", create_event),
        ("parse_delete_target", parse_delete_target),
        ("fetch_candidates_for_delete", fetch_candidates_for_delete),
        ("reason_about_delete_match", reason_about_delete_match),
        ("delete_event_action", delete_event_action),
        ("mark_needs_confirmation", mark_needs_confirmation),
        ("mark_delete_not_found", mark_delete_not_found),
        ("mark_unsupported", mark_unsupported),
        ("mark_clarification", mark_clarification),
        ("build_reply", build_reply),
    ]:
        graph.add_node(node_name, _traced_node(node_name, fn))

    graph.set_entry_point("enrich_with_date_context")
    graph.add_edge("enrich_with_date_context", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "parse_event": "parse_event",
            "parse_delete_target": "parse_delete_target",
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
    graph.add_edge("check_conflicts", "create_event")
    graph.add_edge("create_event", "build_reply")
    # Delete branch
    graph.add_edge("parse_delete_target", "fetch_candidates_for_delete")
    graph.add_edge("fetch_candidates_for_delete", "reason_about_delete_match")
    graph.add_conditional_edges(
        "reason_about_delete_match",
        route_after_delete_reasoning,
        {
            "delete_event_action": "delete_event_action",
            "mark_needs_confirmation": "mark_needs_confirmation",
            "mark_delete_not_found": "mark_delete_not_found",
            "build_reply": "build_reply",
        },
    )
    graph.add_edge("delete_event_action", "build_reply")
    graph.add_edge("mark_needs_confirmation", "build_reply")
    graph.add_edge("mark_delete_not_found", "build_reply")
    graph.add_edge("mark_unsupported", "build_reply")
    graph.add_edge("mark_clarification", "build_reply")
    graph.add_edge("build_reply", END)

    return graph.compile()
