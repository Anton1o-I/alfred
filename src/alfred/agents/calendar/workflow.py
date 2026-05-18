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

import re
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

from alfred.agents.calendar.icloud_client import EventPatch
from alfred.agents.calendar.outcomes import Outcome
from alfred.agents.calendar.replies import build_reply_pair

if TYPE_CHECKING:
    from alfred.agents.calendar.google_client import GoogleCalendarClient
    from alfred.agents.calendar.icloud_client import IcloudCalendarClient
    from alfred.routing.clients import LiteLLMClient

    CalendarClient = IcloudCalendarClient | GoogleCalendarClient

log = structlog.get_logger()
tracer = trace.get_tracer("alfred.calendar.workflow")


# Stopwords for the delete branch's keyword-match heuristic. Anything
# in this set is filtered out of the user's intent before we look for
# slam-dunk title/location/description matches.
_DELETE_STOPWORDS = {
    # Common filler words
    "the", "a", "an", "and", "or", "for", "with", "from", "this", "that",
    "my", "your", "on", "at", "in", "to", "of", "is", "i", "me", "all",
    # Action verbs (the user said them, but they don't help identify the event)
    "cancel", "delete", "remove", "drop", "scrap", "kill",
    # Generic event nouns — too common to be distinctive
    "event", "meeting", "appointment",
    # Day / time / period words — they describe WHEN, not what the event IS
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
    "today", "tomorrow", "tonight", "yesterday",
    "morning", "afternoon", "evening", "night", "noon", "midnight",
    "week", "weeks", "month", "months", "year", "years", "weekend",
    "next", "last", "every",
}


def _extract_meaningful_words(text: str) -> set[str]:
    """Pull distinctive lower-cased words from an intent string for keyword matching."""
    return {
        w
        for w in re.findall(r"\b[a-z]{3,}\b", (text or "").lower())
        if w not in _DELETE_STOPWORDS
    }

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

    action: Literal["create", "delete", "update", "query", "clarify"] = Field(
        description="What the user wants to do with the calendar."
    )
    complexity: Literal["simple", "complex"] = Field(
        default="simple",
        description=(
            "How hard the request is to reason about. "
            "simple: one event with clear, explicit date+time and a "
            "straightforward title (e.g. 'Schedule lunch with Sarah on May 20 at 12:30pm'). "
            "complex: needs careful reasoning. Set this to 'complex' if ANY of "
            "the following apply: "
            "(1) negation phrasing ('Monday, not tomorrow, the next one'); "
            "(2) multiple events or actions in one email; "
            "(3) relative-to-relative dates ('the Wednesday after next', "
            "'the Tuesday after my doctor appointment'); "
            "(4) ambiguous identification of an existing event for delete/update; "
            "(5) the user explicitly says it's complicated or asks you to think carefully. "
            "When in doubt, prefer simple — escalation has a cost."
        ),
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
    start_iso: str | None = Field(
        default=None,
        description=(
            "ISO 8601 with timezone offset. **Leave null if the user did NOT "
            "give a clear date AND time** — never guess. The downstream "
            "validator will fall through to asking the user for clarification "
            "when this is null, which is exactly the right behavior for "
            "vague requests like 'schedule a meeting next week' (no day) "
            "or 'coffee with Eric tomorrow' (no time)."
        ),
    )
    end_iso: str | None = Field(
        default=None,
        description=(
            "ISO 8601 with timezone offset. Leave null whenever start_iso is "
            "null. When start_iso is set, default to start + 1 hour if the "
            "user gave no explicit end or duration."
        ),
    )
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


class EventReference(BaseModel):
    """Reference to an existing calendar event — shared by delete and update."""

    intent_summary: str = Field(
        description=(
            "Brief restatement of the target event in the user's own words. "
            "Examples: 'the meeting with Wayne', 'lunch with Sarah', 'my 3pm'."
        )
    )
    date_hint_iso: str | None = Field(
        default=None,
        description=(
            "ISO date the event is on, if the user specified one. Null if no date hint."
        ),
    )
    time_hint: str | None = Field(
        default=None,
        description="Optional time like '3pm' or 'morning'. Null if unspecified.",
    )


class UpdateEventDraft(BaseModel):
    """Output of the update-parser node — what event to modify and how."""

    target: EventReference = Field(
        description="Which existing event the user wants to change."
    )
    new_start_iso: str | None = Field(
        default=None,
        description=(
            "New start time in ISO 8601 with timezone offset. Set only if the "
            "user explicitly moves/reschedules the event. Null otherwise."
        ),
    )
    new_end_iso: str | None = Field(
        default=None,
        description=(
            "New end time in ISO 8601 with timezone offset. If the user gives a "
            "new start but no new end, leave null — the action node will keep the "
            "existing duration."
        ),
    )
    new_title: str | None = Field(
        default=None,
        description="New event title, if the user renames it. Null otherwise.",
    )
    new_location: str | None = Field(
        default=None,
        description="New event location, if the user relocates it. Null otherwise.",
    )
    new_notes: str | None = Field(
        default=None,
        description="New free-form notes/description, if the user changes them.",
    )

    def has_any_change(self) -> bool:
        return any(
            v is not None
            for v in (
                self.new_start_iso,
                self.new_end_iso,
                self.new_title,
                self.new_location,
                self.new_notes,
            )
        )


class EventMatchDecision(BaseModel):
    """Output of the match-reasoning node — which event (if any) was matched.

    Shared by the delete and update branches via `_reason_about_event_match`.
    """

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
    complexity: str  # "simple" → local-default; "complex" → cloud-default

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

    # Update branch
    update_target: dict  # serialized UpdateEventDraft + flat patch fields for replies
    updated_event: dict

    # Which intent owns the current match-resolution flow ("delete" | "update").
    # Set by the parse step so mark_needs_confirmation / mark_*_not_found can
    # emit the correct outcome string without forking the marker nodes.
    intent_kind: str

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
    "- update: user wants to change/move/rename/relocate an existing event "
    "(e.g. 'move my 3pm to 4pm', 'change the dentist to Friday', "
    "'rename my standup to retro', 'move it to the cafe')\n"
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


_MATCH_PROMPT = (
    "Decide which calendar event the user wants to {verb}.\n"
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
    "  references something the user said. Proceed with the action.\n"
    "- medium: a likely match exists but it's worth confirming with the "
    "  user before acting (e.g., only a partial name match, or one of "
    "  two plausible candidates).\n"
    "- low: multiple weak matches with no clear winner.\n"
    "- none: no candidate plausibly matches.\n"
    "\n"
    "Reasoning should be one short sentence — it gets shown to the user."
)


_UPDATE_PARSE_PROMPT = (
    "Extract what existing event the user wants to change AND what changes "
    "they want to make. You only extract the intent + proposed changes — a "
    "downstream step will look at the actual calendar and decide which event "
    "matches.\n"
    "\n"
    "Today is {today_day_of_week}, {today_iso} ({timezone_name}).\n"
    "\n"
    "Upcoming 15 days (use this table to resolve relative dates):\n"
    "{upcoming_table}\n"
    "\n"
    "Rules:\n"
    "- target.intent_summary: brief restatement of the event in the user's words. "
    "'Move my 3pm tomorrow to 4pm' -> 'my 3pm'. "
    "'Rename the dentist appointment to checkup' -> 'the dentist appointment'.\n"
    "- target.date_hint_iso: ISO date the existing event is on, if mentioned.\n"
    "- target.time_hint: optional time hint about the existing event.\n"
    "\n"
    "New-value fields — set ONLY the ones the user explicitly changes. Leave the "
    "rest null.\n"
    "- new_start_iso / new_end_iso: ISO 8601 with timezone offset for {timezone_name}. "
    "If the user gives a new start but no new end, set new_end_iso=null and the "
    "system will keep the original duration. If the user moves only the date "
    "(not time), preserve the existing clock time — but if you can't, set "
    "new_start_iso to the new date at the user's stated/implied time.\n"
    "- new_title: only if the user renames the event.\n"
    "- new_location: only if the user moves it to a new place.\n"
    "- new_notes: only if the user adds/changes free-form notes.\n"
    "\n"
    "If you can't tell what change the user wanted, leave ALL new_* fields null. "
    "The system will reply asking for clarification.\n"
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
    "- title: short human-readable event name. Include who it's WITH (names "
    "  of people) or what it's ABOUT directly in the title. Examples: "
    "  'Lunch with Sarah', 'Q3 budget review with finance team', "
    "  'Coffee with Eric'. Do NOT prefix titles with action verbs like "
    "  'Add', 'Create', or 'Schedule' — those are commands, not part of "
    "  the event name.\n"
    "- start_iso and end_iso must be ISO 8601 with the timezone offset for {timezone_name}.\n"
    "- If a duration is not specified but start_iso IS set, default end to "
    "  start + 1 hour.\n"
    "- location: ONLY physical places ('Cafe Luna', '123 Main St') or "
    "  video links ('Zoom', 'meet.google.com/abc'). Do NOT put people's "
    "  names here — they belong in the title.\n"
    "- description: optional free-form notes the user explicitly provides.\n"
    "\n"
    "DO NOT GUESS DATES OR TIMES. If the user did not give a clear date AND\n"
    "time, leave start_iso and end_iso as null. The system will ask the user\n"
    "for clarification rather than booking a wrong event. Examples:\n"
    "- 'Schedule a meeting next week' → start_iso=null, end_iso=null "
    "(no specific day, no time)\n"
    "- 'Coffee with Eric tomorrow' → start_iso=null, end_iso=null "
    "(has a date but no time)\n"
    "- 'Meeting at 3pm to review the budget' → start_iso=null, end_iso=null "
    "(has a time but no date)\n"
    "- 'Lunch with Sarah on May 20 at 12:30pm' → both fields filled in\n"
    "Returning null is the correct, expected behavior for vague requests.\n"
    "Filling in invented values silently books wrong events.\n"
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
    """Build a typed-output Pydantic AI agent for a single workflow step.

    The `/no_think` directive disables Qwen 3's thinking mode for these
    nodes — they're narrow structured-extraction tasks where the model
    doesn't benefit from chain-of-thought emission, and the <think> blocks
    can confuse the typed-output parser. Other models (Qwen 2.5, Sonnet)
    ignore the directive harmlessly.
    """
    model = OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(
            base_url=f"{litellm_client.base_url}/v1",
            api_key=litellm_client._api_key,  # noqa: SLF001
        ),
    )
    return Agent(
        model=model,
        output_type=output_type,
        system_prompt=(
            "/no_think\n"
            "You produce strictly-typed structured output. Fill the schema "
            "directly without preamble, explanation, or chain-of-thought."
        ),
    )


def build_calendar_graph(
    calendar_client: CalendarClient,
    timezone_name: str,
    litellm_client: LiteLLMClient,
    model_name: str = "local-default",
    now_fn: Any = None,
) -> Any:
    """Compile the calendar workflow with deps closed over.

    `now_fn` overrides the source of "today" for the enrich node — pass a
    zero-arg callable returning a timezone-aware datetime when running
    simulations against fixed dates. Defaults to `datetime.now(tz)`.
    """
    # Intent classification ALWAYS runs locally — it's a small classification
    # task and routing decisions shouldn't themselves cost cloud tokens.
    intent_agent = _make_specialist(litellm_client, model_name, IntentClassification)
    # Build local + cloud variants of each downstream specialist. The graph
    # picks per-call based on the complexity tag the intent classifier set.
    parse_agent_local = _make_specialist(litellm_client, model_name, EventDraft)
    parse_agent_cloud = _make_specialist(litellm_client, "cloud-default", EventDraft)
    delete_parse_local = _make_specialist(litellm_client, model_name, DeleteTarget)
    delete_parse_cloud = _make_specialist(litellm_client, "cloud-default", DeleteTarget)
    # Match-reasoning is shared between the delete and update branches —
    # `_reason_about_event_match` runs the same prompt with just the verb
    # swapped so we don't fork two near-identical specialists.
    match_agent_local = _make_specialist(litellm_client, model_name, EventMatchDecision)
    match_agent_cloud = _make_specialist(litellm_client, "cloud-default", EventMatchDecision)
    update_parse_local = _make_specialist(litellm_client, model_name, UpdateEventDraft)
    update_parse_cloud = _make_specialist(litellm_client, "cloud-default", UpdateEventDraft)

    def _pick_model_label(state: CalendarState) -> tuple[str, str]:
        """Return (specialist_pool_label, model_name_for_logging)."""
        if state.get("complexity") == "complex":
            return ("cloud", "cloud-default")
        return ("local", model_name)

    # ── Nodes ────────────────────────────────────────────────────────────

    async def enrich_with_date_context(state: CalendarState) -> dict:
        tz = ZoneInfo(timezone_name)
        now = now_fn() if now_fn is not None else datetime.now(tz)
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
            complexity=result.output.complexity,
            reasoning=result.output.reasoning[:120],
        )
        return {
            "action": result.output.action,
            "complexity": result.output.complexity,
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
        pool, model_label = _pick_model_label(state)
        agent = parse_agent_cloud if pool == "cloud" else parse_agent_local
        try:
            result = await agent.run(prompt)
            usage = result.usage()
            ev = result.output
            log.info(
                "calendar_event_parsed",
                title=ev.title,
                start_iso=ev.start_iso,
                end_iso=ev.end_iso,
                model=model_label,
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
        """Structural validation only — required fields and ISO format.

        We deliberately do NOT check that the email body "mentions" a date
        or time. That regex-based guard was overfit to the scenario suite
        and produced false-negatives on natural phrasing ("June 17th" with
        an ordinal suffix, weekday names in quoted-reply headers, etc.).
        Qwen 3 with the typed-output schema is disciplined enough to be
        trusted on the parse; the reply will surface obvious mistakes for
        the user to correct.
        """
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
            log.info("calendar_validate_failed", issues=issues)
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
        pool, model_label = _pick_model_label(state)
        agent = delete_parse_cloud if pool == "cloud" else delete_parse_local
        try:
            result = await agent.run(prompt)
            usage = result.usage()
            target = result.output
            log.info(
                "calendar_delete_target_parsed",
                intent_summary=target.intent_summary,
                date_hint_iso=target.date_hint_iso,
                model=model_label,
            )
            return {
                "delete_target": target.model_dump(),
                "intent_kind": "delete",
                "input_tokens": state.get("input_tokens", 0) + (usage.input_tokens or 0),
                "output_tokens": state.get("output_tokens", 0) + (usage.output_tokens or 0),
            }
        except Exception as e:  # noqa: BLE001
            log.error("calendar_delete_parse_failed", error=str(e))
            return {
                "outcome": Outcome.DELETE_NOT_FOUND.value,
                "error_message": f"parse_error: {e}",
                "intent_kind": "delete",
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

    async def _reason_about_event_match(
        state: CalendarState, intent_kind: Literal["delete", "update"]
    ) -> dict:
        """Shared matcher used by both the delete and update branches.

        Same prompt, same keyword/slam-dunk heuristics, same overconfidence
        downgrade — only the verb in the prompt and the source of the
        target reference change between the two callers. Keeping it one
        function prevents the delete/update reasoning from drifting apart.

        Code-side calibration wraps the LLM:
        - If exactly one candidate's title/location/description contains
          all of the user's intent keywords, return high confidence and
          skip the LLM call entirely. Avoids Qwen flapping on clear cases.
        - After the LLM responds with 'high' confidence, downgrade to
          'medium' if more than one candidate equally matches the keywords —
          forces a confirmation when the LLM was overconfident.
        """
        if intent_kind == "delete":
            target = state.get("delete_target") or {}
            verb = "delete"
        else:
            ut = state.get("update_target") or {}
            target = ut.get("target") or {}
            verb = "update"
        candidates = state.get("matching_events") or []
        if not candidates:
            log.info("calendar_match_no_candidates", intent_kind=intent_kind)
            return {
                "match_decision": {
                    "confidence": "none",
                    "event_id": None,
                    "reasoning": "No events in the searched window.",
                },
            }

        # Heuristic keyword pass — every candidate whose searchable text
        # contains all of the user's intent words is a "keyword match".
        intent_text = target.get("intent_summary") or ""
        intent_words = _extract_meaningful_words(intent_text)
        keyword_matches: list[dict] = []
        if intent_words:
            for c in candidates:
                haystack = " ".join(
                    str(c.get(k) or "")
                    for k in ("summary", "location", "description")
                ).lower()
                if all(w in haystack for w in intent_words):
                    keyword_matches.append(c)

        # Slam-dunk: exactly one keyword match → high confidence, skip LLM.
        if len(keyword_matches) == 1 and intent_words:
            m = keyword_matches[0]
            log.info(
                "calendar_match_slam_dunk",
                intent_kind=intent_kind,
                uid=(m.get("id") or "")[:40],
                intent_words=sorted(intent_words),
            )
            return {
                "match_decision": {
                    "confidence": "high",
                    "event_id": m.get("id"),
                    "reasoning": (
                        f"Single match: all keywords from "
                        f"'{intent_text}' appear in '{m.get('summary')}'."
                    ),
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
        prompt = _MATCH_PROMPT.format(
            verb=verb,
            intent_summary=target.get("intent_summary") or "(no summary)",
            date_hint=target.get("date_hint_iso") or "(none given)",
            email_body=state["email_body"],
            candidates_block="\n".join(lines),
        )
        pool, model_label = _pick_model_label(state)
        match_agent = match_agent_cloud if pool == "cloud" else match_agent_local
        try:
            result = await match_agent.run(prompt)
            usage = result.usage()
            decision = result.output
            confidence = decision.confidence
            reasoning = decision.reasoning
            # Calibration: if LLM said "high" but multiple candidates equally
            # match the intent keywords, downgrade to "medium" so we ask the
            # user to confirm rather than deleting the wrong one.
            if confidence == "high" and len(keyword_matches) > 1:
                log.info(
                    "calendar_match_confidence_downgrade",
                    intent_kind=intent_kind,
                    reason="multiple_equal_keyword_matches",
                    keyword_match_count=len(keyword_matches),
                )
                confidence = "medium"
                reasoning = (
                    reasoning
                    + f" (Multiple events match — {len(keyword_matches)} candidates. "
                    f"Please confirm before I {verb}.)"
                )
            log.info(
                "calendar_match_decision",
                intent_kind=intent_kind,
                confidence=confidence,
                event_id=(decision.event_id or "")[:40],
                reasoning=reasoning[:120],
                model=model_label,
            )
            return {
                "match_decision": {
                    "confidence": confidence,
                    "event_id": decision.event_id,
                    "reasoning": reasoning,
                },
                "input_tokens": state.get("input_tokens", 0) + (usage.input_tokens or 0),
                "output_tokens": state.get("output_tokens", 0) + (usage.output_tokens or 0),
            }
        except Exception as e:  # noqa: BLE001
            log.error("calendar_match_reason_failed", intent_kind=intent_kind, error=str(e))
            return {
                "match_decision": {
                    "confidence": "none",
                    "event_id": None,
                    "reasoning": f"reasoning failed: {e}",
                },
            }

    async def reason_about_delete_match(state: CalendarState) -> dict:
        """Thin wrapper — delegates to the shared event-match helper."""
        return await _reason_about_event_match(state, "delete")

    async def reason_about_update_match(state: CalendarState) -> dict:
        """Thin wrapper — delegates to the shared event-match helper."""
        return await _reason_about_event_match(state, "update")

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
            return {"outcome": Outcome.DELETE_NOT_FOUND.value}
        # Find the candidate row so we can show details in the reply
        target = next((c for c in candidates if c.get("id") == uid), None)
        if target is None:
            log.warning("calendar_delete_target_not_in_candidates", uid=uid)
            return {"outcome": Outcome.DELETE_NOT_FOUND.value}
        try:
            success = calendar_client.delete_event(uid)
        except Exception as e:  # noqa: BLE001
            log.error("calendar_delete_failed", error=str(e), uid=uid)
            return {"outcome": Outcome.ERROR.value, "error_message": str(e)}
        if not success:
            log.warning("calendar_delete_returned_false", uid=uid)
            return {"outcome": Outcome.DELETE_NOT_FOUND.value}
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
            "outcome": Outcome.DELETED.value,
        }

    # ── Update branch nodes ───────────────────────────────────────────────

    async def parse_update_target(state: CalendarState) -> dict:
        prompt = _UPDATE_PARSE_PROMPT.format(
            today_day_of_week=state["today_day_of_week"],
            today_iso=state["today_iso"],
            timezone_name=state["timezone_name"],
            upcoming_table=_format_upcoming_table(state["upcoming_days"]),
            email_body=state["email_body"],
        )
        pool, model_label = _pick_model_label(state)
        agent = update_parse_cloud if pool == "cloud" else update_parse_local
        try:
            result = await agent.run(prompt)
            usage = result.usage()
            draft: UpdateEventDraft = result.output
            log.info(
                "calendar_update_target_parsed",
                intent_summary=draft.target.intent_summary,
                has_change=draft.has_any_change(),
                model=model_label,
            )
            # Flatten the draft so replies.py and the matcher both have a
            # uniform dict shape ({target: {...}, new_start, new_end, ...}).
            update_target_dict: dict[str, Any] = {
                "target": draft.target.model_dump(),
                "new_start": draft.new_start_iso,
                "new_end": draft.new_end_iso,
                "new_title": draft.new_title,
                "new_location": draft.new_location,
                "new_notes": draft.new_notes,
            }
            base: dict[str, Any] = {
                "update_target": update_target_dict,
                "intent_kind": "update",
                "input_tokens": state.get("input_tokens", 0) + (usage.input_tokens or 0),
                "output_tokens": state.get("output_tokens", 0) + (usage.output_tokens or 0),
            }
            # No-op early-exit: if the user didn't ask for any change we
            # don't need to fetch candidates or call the matcher.
            if not draft.has_any_change():
                base["outcome"] = Outcome.UPDATE_NO_CHANGE.value
            return base
        except Exception as e:  # noqa: BLE001
            log.error("calendar_update_parse_failed", error=str(e))
            return {
                "outcome": Outcome.UPDATE_NOT_FOUND.value,
                "error_message": f"parse_error: {e}",
                "intent_kind": "update",
            }

    async def fetch_candidates_for_update(state: CalendarState) -> dict:
        """Pure code: fetch all events in the relevant date window.

        Mirrors `fetch_candidates_for_delete` — same window heuristic
        (date hint → that day; no hint → next 30 days). Kept as a separate
        node from the delete fetcher because the targets live in different
        state keys; the body is intentionally small.
        """
        target = (state.get("update_target") or {}).get("target") or {}
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
                "recurrence": ev.get("recurrence"),
            }
            for ev in existing
        ]
        log.info(
            "calendar_update_candidates",
            window_events=len(candidates),
            date_hint=date_hint,
        )
        return {"matching_events": candidates}

    async def update_event_action(state: CalendarState) -> dict:
        decision = state.get("match_decision") or {}
        uid = decision.get("event_id")
        candidates = state.get("matching_events") or []
        update_target = state.get("update_target") or {}
        if not uid:
            return {"outcome": Outcome.UPDATE_NOT_FOUND.value}
        target = next((c for c in candidates if c.get("id") == uid), None)
        if target is None:
            log.warning("calendar_update_target_not_in_candidates", uid=uid)
            return {"outcome": Outcome.UPDATE_NOT_FOUND.value}

        # Build the EventPatch. If only new_start is given and no new_end,
        # preserve the original duration so a "move my 3pm to 4pm" doesn't
        # accidentally collapse the meeting to 0min.
        def _opt_dt(value: str | None) -> datetime | None:
            if not value:
                return None
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None

        new_start = _opt_dt(update_target.get("new_start"))
        new_end = _opt_dt(update_target.get("new_end"))
        if new_start and not new_end:
            old_start = _opt_dt(target.get("start"))
            old_end = _opt_dt(target.get("end"))
            if old_start and old_end:
                new_end = new_start + (old_end - old_start)

        try:
            patch = EventPatch(
                start=new_start,
                end=new_end,
                summary=update_target.get("new_title"),
                location=update_target.get("new_location"),
                description=update_target.get("new_notes"),
            )
        except Exception as e:  # noqa: BLE001 — validation error
            log.error("calendar_update_patch_invalid", error=str(e), uid=uid)
            return {"outcome": Outcome.ERROR.value, "error_message": str(e)}

        if patch.is_empty():
            return {"outcome": Outcome.UPDATE_NO_CHANGE.value}

        if not hasattr(calendar_client, "update_event"):
            log.error(
                "calendar_update_not_supported_by_client",
                client_type=type(calendar_client).__name__,
            )
            return {
                "outcome": Outcome.ERROR.value,
                "error_message": "This calendar provider does not support updates yet.",
            }
        try:
            updated = calendar_client.update_event(uid, patch=patch)
        except LookupError:
            log.warning("calendar_update_event_uid_missing", uid=uid)
            return {"outcome": Outcome.UPDATE_NOT_FOUND.value}
        except Exception as e:  # noqa: BLE001
            log.error("calendar_update_failed", error=str(e), uid=uid)
            return {"outcome": Outcome.ERROR.value, "error_message": str(e)}

        is_recurring = bool(target.get("recurrence"))
        return {
            "updated_event": {
                "summary": updated.get("summary") or target.get("summary"),
                "start_iso": (updated.get("start") or {}).get("dateTime")
                or target.get("start"),
                "end_iso": (updated.get("end") or {}).get("dateTime")
                or target.get("end"),
                "location": updated.get("location") or target.get("location"),
                "description": updated.get("description"),
                "calendar_name": getattr(calendar_client, "_calendar_name", "")
                or "primary",
                "timezone": timezone_name,
                "is_recurring": is_recurring,
            },
            "outcome": Outcome.UPDATED.value,
        }

    async def mark_update_not_found(state: CalendarState) -> dict:
        return {"outcome": Outcome.UPDATE_NOT_FOUND.value}

    async def mark_needs_confirmation(state: CalendarState) -> dict:
        """Emit the right *_needs_confirmation outcome for whichever branch
        we're in. The parse step sets `intent_kind` so this single marker
        node handles both delete and update without forking."""
        if state.get("intent_kind") == "update":
            return {"outcome": Outcome.UPDATE_NEEDS_CONFIRMATION.value}
        return {"outcome": Outcome.DELETE_NEEDS_CONFIRMATION.value}

    async def mark_unsupported(state: CalendarState) -> dict:
        action = state.get("action", "unknown")
        return {
            "outcome": Outcome.UNSUPPORTED.value,
            "error_message": (
                f"The '{action}' action isn't supported yet — I can only "
                "schedule, update, and cancel events for now. Listing/queries are coming."
            ),
        }

    async def mark_clarification(state: CalendarState) -> dict:
        return {"outcome": Outcome.CLARIFICATION.value}

    async def build_reply(state: CalendarState) -> dict:
        """Render the user-facing reply via the dispatch-table in replies.py.

        Adding a new outcome is a one-handler change in replies.py rather
        than an if/elif edit here.
        """
        plain, html = build_reply_pair(dict(state))
        return {"reply_plain": plain, "reply_html": html}


    # ── Edges ────────────────────────────────────────────────────────────

    def route_after_intent(state: CalendarState) -> str:
        action = state.get("action", "clarify")
        if action == "create":
            return "parse_event"
        if action == "delete":
            return "parse_delete_target"
        if action == "update":
            return "parse_update_target"
        if action == "query":
            return "mark_unsupported"
        return "mark_clarification"

    def route_after_parse_update(state: CalendarState) -> str:
        # If parse short-circuited (no change at all, or parse error), go
        # straight to build_reply — no point fetching candidates.
        outcome = state.get("outcome")
        if outcome in (
            Outcome.UPDATE_NO_CHANGE.value,
            Outcome.UPDATE_NOT_FOUND.value,
        ):
            return "build_reply"
        return "fetch_candidates_for_update"

    def route_after_update_reasoning(state: CalendarState) -> str:
        outcome = state.get("outcome")
        decision = state.get("match_decision") or {}
        confidence = decision.get("confidence", "none")
        log.info(
            "calendar_update_route_decision",
            current_outcome=outcome,
            confidence=confidence,
            event_id=(decision.get("event_id") or "")[:40],
        )
        if outcome == Outcome.UPDATE_NOT_FOUND.value:
            return "build_reply"
        if confidence == "high":
            return "update_event_action"
        if confidence == "medium":
            return "mark_needs_confirmation"
        return "mark_update_not_found"  # low / none → ask user to clarify

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
        if outcome == Outcome.DELETE_NOT_FOUND.value:
            return "build_reply"
        if confidence == "high":
            return "delete_event_action"
        if confidence == "medium":
            return "mark_needs_confirmation"
        return "mark_delete_not_found"  # low / none → ask user to clarify

    async def mark_delete_not_found(state: CalendarState) -> dict:
        return {"outcome": Outcome.DELETE_NOT_FOUND.value}

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
        ("parse_update_target", parse_update_target),
        ("fetch_candidates_for_update", fetch_candidates_for_update),
        ("reason_about_update_match", reason_about_update_match),
        ("update_event_action", update_event_action),
        ("mark_needs_confirmation", mark_needs_confirmation),
        ("mark_delete_not_found", mark_delete_not_found),
        ("mark_update_not_found", mark_update_not_found),
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
            "parse_update_target": "parse_update_target",
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
    # Update branch
    graph.add_conditional_edges(
        "parse_update_target",
        route_after_parse_update,
        {
            "fetch_candidates_for_update": "fetch_candidates_for_update",
            "build_reply": "build_reply",
        },
    )
    graph.add_edge("fetch_candidates_for_update", "reason_about_update_match")
    graph.add_conditional_edges(
        "reason_about_update_match",
        route_after_update_reasoning,
        {
            "update_event_action": "update_event_action",
            "mark_needs_confirmation": "mark_needs_confirmation",
            "mark_update_not_found": "mark_update_not_found",
            "build_reply": "build_reply",
        },
    )
    graph.add_edge("update_event_action", "build_reply")
    graph.add_edge("mark_update_not_found", "build_reply")
    graph.add_edge("mark_unsupported", "build_reply")
    graph.add_edge("mark_clarification", "build_reply")
    graph.add_edge("build_reply", END)

    return graph.compile()
