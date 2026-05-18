"""Tasks workflow — LangGraph state machine with specialist nodes.

Mirrors the calendar workflow shape: pure-code nodes for orchestration,
typed Pydantic AI specialists for the few steps that need an LLM
(intent classification, chore parsing, semantic duplicate detection).

Phase 2 scope: classify_intent + create branch (with duplicate detection).
Phase 3 will add complete / list / delete / update branches; for now those
short-circuit to "unsupported".
"""

from __future__ import annotations

import re
from datetime import datetime
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
    from alfred.agents.tasks.store import ChoreStore
    from alfred.routing.clients import LiteLLMClient

log = structlog.get_logger()
tracer = trace.get_tracer("alfred.tasks.workflow")

_OI_SPAN_KIND = "openinference.span.kind"

def _norm_shape(s: str | None) -> str | None:
    """Lowercase + strip shape fields for deterministic comparison."""
    if not s:
        return None
    out = s.strip().lower()
    return out or None


def _shape_verdict(
    new_obj: str | None,
    new_qual: str | None,
    ex_obj: str | None,
    ex_qual: str | None,
) -> str:
    """Pure-code dedup comparison. Returns 'high' | 'medium' | 'none'.

    Truth table (after _norm_shape):
      different object               → none
      same object, both qualifiers, same → high
      same object, both qualifiers, diff → none
      same object, one qualified, one not → medium
      same object, neither qualified → high
    """
    if not new_obj or not ex_obj:
        return "none"
    if new_obj != ex_obj:
        return "none"
    if new_qual and ex_qual:
        return "high" if new_qual == ex_qual else "none"
    if new_qual or ex_qual:
        return "medium"
    return "high"


def _shape_reason(
    new_title: str,
    ex_title: str,
    new_obj: str | None,
    new_qual: str | None,
    ex_obj: str | None,
    ex_qual: str | None,
    verdict: str,
) -> str:
    """One-sentence explanation for the user/log."""
    if verdict == "high":
        return f"'{new_title}' is the same chore as '{ex_title}' (same target)."
    if verdict == "medium":
        unq = new_title if not new_qual else ex_title
        q = ex_title if not new_qual else new_title
        return (
            f"'{unq}' might be the same chore as '{q}' — one of them is more "
            "specific. Reply to confirm or split them apart."
        )
    return f"'{new_title}' and '{ex_title}' target different things."


def _slugify(s: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return slug or "chore"


def _normalize_title(s: str) -> str:
    """Canonicalize a chore title so dedup compares apples to apples.

    Rules:
    - Strip leading/trailing whitespace and trailing punctuation.
    - Drop articles ('the', 'a', 'an') that appear AFTER the first word,
      so 'Take out the trash' → 'Take out trash' and 'Clean the kitchen' →
      'Clean kitchen'. Articles in the leading position are left alone
      (rare in imperatives, but conservative).
    - Capitalize the first letter; leave the rest of the casing intact so
      proper nouns ('Amazon', 'UPS') and the LLM's chosen casing survive.
    """
    if not s:
        return s
    cleaned = s.strip().rstrip(".!?,;:")
    parts = cleaned.split()
    if len(parts) > 1:
        filtered = [parts[0]] + [w for w in parts[1:] if w.lower() not in {"the", "a", "an"}]
        cleaned = " ".join(filtered)
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _traced_node(name: str, fn):
    async def wrapper(state):  # type: ignore[no-untyped-def]
        with tracer.start_as_current_span(f"tasks.{name}") as span:
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
    """What the user wants to do with their chore list."""

    action: Literal["create", "complete", "list", "delete", "update", "clarify"] = (
        Field(description="What the user wants to do.")
    )
    complexity: Literal["simple", "complex"] = Field(
        default="simple",
        description=(
            "How hard the request is to reason about. simple: one action, "
            "explicit chore name or single recurrence rule. complex: multiple "
            "chores in one email, ambiguous identification of an existing "
            "chore, or unusual recurrence phrasing. When in doubt, prefer simple."
        ),
    )
    reasoning: str = Field(default="", description="One short sentence.")


class ChoreRecurrence(BaseModel):
    """How the chore repeats. Parallels the calendar RecurrenceRule shape."""

    frequency: Literal["DAILY", "WEEKLY", "MONTHLY", "YEARLY"] = Field(
        description="How often the chore repeats."
    )
    interval: int = Field(default=1, description="Repeat every N (e.g., 2 = every other).")
    byday: list[str] = Field(
        default_factory=list,
        description=(
            "Days of the week for weekly patterns. Two-letter codes: "
            "MO TU WE TH FR SA SU."
        ),
    )


class ChoreDraft(BaseModel):
    """Parsed chore from the user's message."""

    id: str | None = Field(
        default=None,
        description=(
            "Optional short stable id the user gave (e.g. 'trash', 'bathrooms'). "
            "Leave null if the user didn't name one — the system will slugify the title."
        ),
    )
    title: str | None = Field(
        default=None,
        description=(
            "Concise chore name. Examples: 'Take out trash', 'Clean bathrooms'. "
            "Leave null if the user did not actually describe a chore."
        ),
    )
    description: str | None = None
    assignee: Literal["household", "primary", "secondary"] | None = Field(
        default=None,
        description=(
            "Who is responsible. 'household' if either spouse can do it; "
            "'primary' for the message owner; 'secondary' for the other spouse. "
            "Leave null if the user didn't say — the system will default to household."
        ),
    )
    recurrence_type: Literal["schedule", "completion", "once"] | None = Field(
        default=None,
        description=(
            "'schedule' = anchored to calendar dates ('every Tuesday'). "
            "'completion' = anchored to last completion ('every 7 days after I do it'). "
            "'once' = a single one-time task with a specific due date. "
            "Leave null if the user didn't specify and the system will infer "
            "(once if due_date_iso is set, else schedule)."
        ),
    )
    recurrence: ChoreRecurrence | None = Field(
        default=None,
        description=(
            "Required for recurring chores. Leave null for one-time tasks "
            "(when recurrence_type='once' or the user gave a single date)."
        ),
    )
    due_date_iso: str | None = Field(
        default=None,
        description=(
            "For one-time tasks only. ISO date (YYYY-MM-DD) when the task is due. "
            "Resolve relative phrases like 'next Friday' or 'tomorrow' using "
            "today's date provided in the prompt. Leave null for recurring chores."
        ),
    )
    shame_after_days: int | None = Field(
        default=None,
        description="Days overdue before public shame kicks in. Defaults to 3.",
    )
    object: str | None = Field(
        default=None,
        description=(
            "The single noun the chore acts on, lowercase, singular. "
            "Examples: 'trash' for 'Take out trash', 'bathroom' for 'Clean main bathroom', "
            "'lawn' for 'Mow lawn', 'plants' for 'Water plants'. Leave null if the "
            "title isn't a clean verb-noun shape."
        ),
    )
    qualifier: str | None = Field(
        default=None,
        description=(
            "Modifier that distinguishes WHICH instance of the object. Lowercase. "
            "Examples: 'main' for 'Clean main bathroom', 'kitchen' for 'Wipe kitchen "
            "counters', 'front' for 'Sweep front porch'. Leave null when the chore "
            "doesn't single out one instance (e.g. 'Take out trash' has no qualifier)."
        ),
    )


class DuplicateCheckDecision(BaseModel):
    """Output of the semantic-dedup specialist."""

    confidence: Literal["high", "medium", "none"] = Field(
        description=(
            "high: this is plainly the same chore as an existing one. "
            "medium: likely overlap, worth asking the user to confirm. "
            "none: no meaningful overlap — safe to create."
        )
    )
    existing_chore_id: str | None = Field(
        default=None,
        description="The matched existing chore's id, or null if confidence is 'none'.",
    )
    reasoning: str = Field(
        default="",
        description="One short sentence (shown to the user when we ask for confirmation).",
    )


class TargetReference(BaseModel):
    """The natural-language phrase the user used to refer to an existing chore."""

    reference: str = Field(
        description=(
            "A brief restatement of the chore the user is referring to, in their "
            "own words. Examples: 'trash', 'the bathrooms', 'taking out the recycling'. "
            "Leave empty string if the user didn't actually reference a chore."
        )
    )


class TargetMatch(BaseModel):
    """Which active chore the user means."""

    confidence: Literal["high", "medium", "none"] = Field(
        description=(
            "high: clearly the right chore — proceed. "
            "medium: likely match but worth confirming first. "
            "none: no candidate plausibly matches."
        )
    )
    chore_id: str | None = Field(
        default=None,
        description="The matched chore's id, or null if confidence is 'none'.",
    )
    reasoning: str = Field(
        default="",
        description="One short sentence explaining the match (shown to the user).",
    )


class ChoreUpdateDraft(BaseModel):
    """What to change on an existing chore."""

    target_reference: str = Field(
        description="Brief restatement of which chore the user wants to change."
    )
    new_title: str | None = None
    new_assignee: Literal["household", "primary", "secondary"] | None = None
    new_recurrence_type: Literal["schedule", "completion"] | None = None
    new_recurrence: ChoreRecurrence | None = None
    new_shame_after_days: int | None = None


# ── Graph state ───────────────────────────────────────────────────────────


class TasksState(TypedDict, total=False):
    # Inputs
    email_body: str

    # Date / chore context
    today_iso: str
    today_day_of_week: str
    timezone_name: str
    active_chores: list[dict]

    # Classification
    action: str
    complexity: str

    # Create branch
    parsed_chore: dict
    completeness_issues: list[str]
    duplicate_check: dict
    created_chore: dict

    # Complete / delete / update branches
    target_reference: dict
    target_match: dict
    completed_chore: dict
    deleted_chore: dict
    update_draft: dict
    updated_chore: dict

    # List branch
    pending_summary: list[dict]

    # Outcome
    outcome: str
    error_message: str

    # Reply
    reply_plain: str
    reply_html: str

    # Usage
    input_tokens: int
    output_tokens: int


# ── Prompts ──────────────────────────────────────────────────────────────


_INTENT_PROMPT = (
    "Classify the user's email into one of these chore-tracker actions:\n"
    "- create: user wants to add a new recurring chore\n"
    "- complete: user is marking a chore done\n"
    "- list: user wants to see what's pending/overdue\n"
    "- delete: user wants to remove a chore from tracking\n"
    "- update: user wants to change an existing chore (assignee, recurrence, etc.)\n"
    "- clarify: the request is unclear or out of scope\n"
    "\n"
    "Email body:\n"
    "{email_body}"
)


_PARSE_PROMPT = (
    "Extract the chore the user wants to add.\n"
    "\n"
    "Today is {today_day_of_week}, {today_iso} ({timezone_name}).\n"
    "\n"
    "Rules:\n"
    "- title: short imperative name, in this exact CANONICAL FORMAT:\n"
    "    • Start with the verb in imperative form ('Take out', 'Clean', "
    "      'Water', 'Replace').\n"
    "    • Sentence case: capitalize only the first word; lowercase the "
    "      rest UNLESS it's a proper noun ('Amazon', 'UPS') or an acronym "
    "      ('HVAC').\n"
    "    • Singular, no leading or interior articles. Write 'Take out "
    "      trash' NOT 'Take out the trash'. Write 'Clean kitchen "
    "      countertops' NOT 'Clean the Kitchen Countertops'.\n"
    "    • Be SPECIFIC when the user names a particular target — use the "
    "      qualifier inline ('Clean main bathroom', 'Clean hallway "
    "      bathroom', NOT just 'Clean bathroom').\n"
    "    • No scheduling phrases ('every Tuesday') — those go in recurrence.\n"
    "  Good examples: 'Take out trash', 'Clean main bathroom', 'Water "
    "  plants', 'Mow lawn', 'Pick up dry cleaning'.\n"
    "  Bad examples: 'Take out the trash', 'clean downstairs bathroom', "
    "  'Clean Kitchen Countertops', 'TRASH'.\n"
    "- assignee: 'household' (default) if either spouse can do it. Use "
    "  'primary' or 'secondary' when the user names a specific person — "
    "  resolve the name using the mapping below.\n"
    "{name_mapping_block}"
    "- recurrence_type:\n"
    "    'schedule' when anchored to days ('every Tuesday', 'weekly');\n"
    "    'completion' when anchored to elapsed time since last done "
    "      ('every 7 days', 'once a week after I do it');\n"
    "    'once' when this is a one-time task with a specific date "
    "      ('pick up the package on Friday', 'call the plumber tomorrow').\n"
    "  Default to 'schedule' when unsure for recurring; default to 'once' "
    "  when a single specific date is given.\n"
    "- recurrence.frequency: DAILY/WEEKLY/MONTHLY/YEARLY (only for recurring).\n"
    "- recurrence.interval: how many of those frequency units between runs. "
    "  Default 1. CRITICAL: when the user says 'every N <unit>', interval=N.\n"
    "    'every 2 days' → frequency=DAILY, interval=2\n"
    "    'every 3 weeks' → frequency=WEEKLY, interval=3\n"
    "    'every other Saturday' → frequency=WEEKLY, interval=2, byday=[SA]\n"
    "    'every 6 months' → frequency=MONTHLY, interval=6\n"
    "    'weekly' / 'every week' → frequency=WEEKLY, interval=1\n"
    "  Do NOT collapse 'every 2 days' to interval=1 — the number is load-bearing.\n"
    "- recurrence.byday: two-letter codes for weekly day-of-week patterns "
    "  (MO TU WE TH FR SA SU). Leave empty for non-weekly or unspecified.\n"
    "- due_date_iso: ISO date (YYYY-MM-DD) for one-time tasks. Resolve "
    "  relative phrases ('next Friday', 'tomorrow', 'this Saturday') using "
    "  today's date above.\n"
    "- shame_after_days: only set if the user explicitly says so.\n"
    "- object: the single noun the chore acts on. Lowercase, singular. "
    "  This is the THING being acted on, not the verb.\n"
    "    'Take out trash' → object='trash'\n"
    "    'Clean main bathroom' → object='bathroom'\n"
    "    'Clean hallway bathroom' → object='bathroom'\n"
    "    'Wipe kitchen counters' → object='counters'\n"
    "    'Mow lawn' → object='lawn'\n"
    "    'Water plants' → object='plants'\n"
    "    'Pick up dry cleaning' → object='dry cleaning'\n"
    "  Leave null only if the title doesn't have a clean verb-noun shape.\n"
    "- qualifier: the modifier that distinguishes WHICH instance of the "
    "  object. Lowercase. Leave null when there's no distinguishing modifier.\n"
    "    'Take out trash' → qualifier=null\n"
    "    'Clean main bathroom' → qualifier='main'\n"
    "    'Clean hallway bathroom' → qualifier='hallway'\n"
    "    'Clean kids bathroom' → qualifier='kids'\n"
    "    'Wipe kitchen counters' → qualifier='kitchen'\n"
    "    'Sweep front porch' → qualifier='front'\n"
    "    'Mow lawn' → qualifier=null\n"
    "  CRITICAL: when the user names a SPECIFIC instance (main/hallway/"
    "  master/kids/front/back/upstairs/downstairs/kitchen), put it here. "
    "  Two chores with different qualifiers are different chores even if "
    "  they share an object.\n"
    "\n"
    "Leave fields null when the user didn't say. Do not invent values.\n"
    "If the user did not describe a chore at all (e.g. just said 'hi'), "
    "leave title null.\n"
    "\n"
    "Email body:\n"
    "{email_body}"
)


_DEDUP_PROMPT = (
    "Decide whether the proposed new chore is already covered by one of the "
    "existing active chores.\n"
    "\n"
    "Proposed new chore:\n"
    "  title: {new_title}\n"
    "  recurrence: {new_recurrence}\n"
    "  assignee: {new_assignee}\n"
    "\n"
    "Existing active chores:\n"
    "{existing_block}\n"
    "\n"
    "Confidence rubric:\n"
    "- high: same underlying activity on the same target. Synonyms or "
    "paraphrases of one already-tracked chore (e.g. 'take the trash out' "
    "vs 'Take out trash'; 'wipe down the main bathroom' vs 'Clean the "
    "main bathroom').\n"
    "- medium: meaningful overlap and you genuinely cannot tell if they are "
    "the same — ask the user.\n"
    "- none: distinct chores. THIS IS THE DEFAULT when in doubt. Examples "
    "that are NOT duplicates:\n"
    "  * Different rooms or fixtures: 'main bathroom' vs 'hallway bathroom' "
    "    vs 'kids' bathroom' vs 'master bathroom' — each room is its own "
    "    chore even though all involve cleaning a bathroom.\n"
    "  * Different scope of the same room: 'wipe down kitchen counters' "
    "    (daily) vs 'deep clean the kitchen' (weekly) — different scope, "
    "    different cadence, different chores.\n"
    "  * Different activities on the same target: 'mow the lawn' vs "
    "    'weed the lawn' — same target, different work.\n"
    "\n"
    "Rule of thumb: if the chores refer to physically different objects, "
    "rooms, or fixtures, they are NOT duplicates regardless of how similar "
    "the wording sounds. Only flag duplicates when the underlying work is "
    "the same on the same target.\n"
    "\n"
    "Quick test: strip the leading verb. If each remaining title has at "
    "least one distinctive word the other doesn't (e.g. 'main' vs "
    "'hallway', 'counters' vs nothing-specific), they're DIFFERENT chores.\n"
    "\n"
    "Reasoning should be one short sentence — it gets shown to the user."
)


_TARGET_PARSE_PROMPT = (
    "Extract which existing chore the user is referring to.\n"
    "\n"
    "Action context: {action}  (complete = marking done; delete = removing from "
    "tracking; update = changing details).\n"
    "\n"
    "Active chores currently being tracked:\n"
    "{existing_block}\n"
    "\n"
    "Return a brief restatement of the chore the user references, in their own "
    "words (e.g. 'the trash', 'taking out recycling', 'bathrooms'). Leave the "
    "reference empty if the user didn't name a specific chore.\n"
    "\n"
    "Email body:\n"
    "{email_body}"
)


_TARGET_MATCH_PROMPT = (
    "Decide which active chore the user is referring to.\n"
    "\n"
    "Action: {action}\n"
    "User's reference: {reference}\n"
    "Original message:\n"
    "{email_body}\n"
    "\n"
    "Candidate chores:\n"
    "{candidates_block}\n"
    "\n"
    "Match against the chore title — paraphrases and partial references are "
    "fine ('trash' matches 'Take out trash'; 'bathrooms' matches 'Clean bathrooms').\n"
    "\n"
    "Confidence rubric:\n"
    "- high: one clear match. Proceed.\n"
    "- medium: likely match worth confirming with the user (e.g. two plausible candidates).\n"
    "- none: no candidate plausibly matches.\n"
    "\n"
    "Reasoning should be one short sentence — it gets shown to the user."
)


_UPDATE_PARSE_PROMPT = (
    "Extract the chore update the user is requesting.\n"
    "\n"
    "Active chores:\n"
    "{existing_block}\n"
    "\n"
    "Today is {today_day_of_week}, {today_iso} ({timezone_name}).\n"
    "\n"
    "Rules:\n"
    "- target_reference: brief restatement of which chore is changing.\n"
    "- Set new_* fields ONLY for what the user explicitly wants to change. "
    "Leave others null.\n"
    "- new_recurrence_type / new_recurrence: only if the user changes the schedule.\n"
    "- new_assignee: one of 'household', 'primary', 'secondary' if the user "
    "  reassigns — resolve named people via the mapping below.\n"
    "{name_mapping_block}"
    "\n"
    "Email body:\n"
    "{email_body}"
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _format_recurrence_human(rule: dict[str, Any], rec_type: str) -> str:
    """Render a recurrence rule + type as readable text for replies.

    Examples:
      ({freq: DAILY, interval: 2}, 'completion') → 'every 2 days (completion-based)'
      ({freq: WEEKLY, byday: [TU]}, 'schedule')  → 'weekly on TU (schedule-based)'
      ({freq: WEEKLY, interval: 2, byday: [SA]}, 'schedule')
          → 'every 2 weeks on SA (schedule-based)'
    """
    freq = (rule.get("frequency") or "").upper()
    interval = int(rule.get("interval") or 1)
    byday = ",".join(rule.get("byday") or [])
    unit = {
        "DAILY": "day",
        "WEEKLY": "week",
        "MONTHLY": "month",
        "YEARLY": "year",
    }.get(freq, freq.lower() or "cycle")
    if interval == 1:
        base = {"DAILY": "daily", "WEEKLY": "weekly", "MONTHLY": "monthly", "YEARLY": "yearly"}.get(
            freq, freq.lower() or "every cycle"
        )
    else:
        base = f"every {interval} {unit}s"
    if byday:
        base += f" on {byday}"
    return f"{base} ({rec_type}-based)"


def _name_mapping_block(name_map: dict[str, str]) -> str:
    """Render a 'primary = <name>, secondary = <name>' hint into the prompt.

    Returns an empty string when no names are configured — the model then
    falls back to literal household/primary/secondary classification.
    """
    if not name_map:
        return ""
    lines = ["  Name → user_id mapping:"]
    for uid in ("primary", "secondary"):
        if uid in name_map:
            lines.append(f"    • '{name_map[uid]}' → {uid}")
    lines.append(
        "  When the user names one of these people, set assignee to the "
        "matching user_id. Treat case-insensitive matches and common "
        "shortenings as a match."
    )
    return "\n".join(lines) + "\n"


def _make_specialist(
    litellm_client: LiteLLMClient, model_name: str, output_type: type
) -> Agent:
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


def _format_existing_block(chores: list[dict]) -> str:
    if not chores:
        return "  (none)"
    lines = []
    for c in chores:
        rule = c.get("recurrence_rule") or {}
        freq = rule.get("frequency") or "?"
        byday = rule.get("byday") or []
        when = f"{freq.lower()}"
        if byday:
            when += f" on {','.join(byday)}"
        lines.append(
            f"  - id={c['id']}  '{c['title']}'  ({when}, "
            f"{c.get('recurrence_type', '?')}-based, assignee={c.get('assignee', '?')})"
        )
    return "\n".join(lines)


def _format_incomplete_text(issues: list[str]) -> str:
    return (
        "I couldn't fully parse your chore request.\n"
        "Issues: " + "; ".join(issues) + ".\n\n"
        "Reply with the chore, who's doing it, and how often, e.g. "
        "'Add chore: take out trash every Tuesday, household'."
    )


def build_tasks_graph(
    store: ChoreStore,
    timezone_name: str,
    litellm_client: LiteLLMClient,
    model_name: str = "local-default",
    now_fn: Any = None,
    assignee_names: dict[str, str] | None = None,
) -> Any:
    """Compile the tasks workflow with deps closed over.

    `assignee_names` maps user_id → display name (e.g. {'primary': 'Alex',
    'secondary': 'Sam'}). When set, the parse prompt teaches the model to
    resolve natural-language references like 'assign to <name>' back to the
    canonical user_id slot.
    """
    name_map = assignee_names or {}
    intent_agent = _make_specialist(litellm_client, model_name, IntentClassification)
    parse_local = _make_specialist(litellm_client, model_name, ChoreDraft)
    parse_cloud = _make_specialist(litellm_client, "cloud-default", ChoreDraft)
    dedup_local = _make_specialist(litellm_client, model_name, DuplicateCheckDecision)
    dedup_cloud = _make_specialist(litellm_client, "cloud-default", DuplicateCheckDecision)
    target_parse_local = _make_specialist(litellm_client, model_name, TargetReference)
    target_parse_cloud = _make_specialist(litellm_client, "cloud-default", TargetReference)
    target_match_local = _make_specialist(litellm_client, model_name, TargetMatch)
    target_match_cloud = _make_specialist(litellm_client, "cloud-default", TargetMatch)
    update_parse_local = _make_specialist(litellm_client, model_name, ChoreUpdateDraft)
    update_parse_cloud = _make_specialist(litellm_client, "cloud-default", ChoreUpdateDraft)

    def _pick_pool(state: TasksState) -> tuple[str, str]:
        if state.get("complexity") == "complex":
            return ("cloud", "cloud-default")
        return ("local", model_name)

    # ── Nodes ────────────────────────────────────────────────────────────

    async def enrich_context(state: TasksState) -> dict:
        tz = ZoneInfo(timezone_name)
        now = now_fn() if now_fn is not None else datetime.now(tz)
        today = now.date()
        active = await store.list_active_chores()
        active_dicts = [
            {
                "id": c.id,
                "title": c.title,
                "assignee": c.assignee,
                "recurrence_type": c.recurrence_type,
                "recurrence_rule": c.recurrence_rule,
                "object": c.object,
                "qualifier": c.qualifier,
            }
            for c in active
        ]
        return {
            "today_iso": today.isoformat(),
            "today_day_of_week": today.strftime("%A"),
            "timezone_name": timezone_name,
            "active_chores": active_dicts,
        }

    async def classify_intent(state: TasksState) -> dict:
        prompt = _INTENT_PROMPT.format(email_body=state["email_body"])
        result = await intent_agent.run(prompt)
        usage = result.usage()
        log.info(
            "tasks_intent_classified",
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

    async def parse_chore_draft(state: TasksState) -> dict:
        prompt = _PARSE_PROMPT.format(
            today_day_of_week=state["today_day_of_week"],
            today_iso=state["today_iso"],
            timezone_name=state["timezone_name"],
            name_mapping_block=_name_mapping_block(name_map),
            email_body=state["email_body"],
        )
        pool, model_label = _pick_pool(state)
        agent = parse_cloud if pool == "cloud" else parse_local
        try:
            result = await agent.run(prompt)
            usage = result.usage()
            draft = result.output
            normalized_title = _normalize_title(draft.title) if draft.title else draft.title
            if normalized_title != draft.title:
                log.info(
                    "tasks_title_normalized",
                    raw=draft.title,
                    normalized=normalized_title,
                )
            draft_dict = draft.model_dump()
            draft_dict["title"] = normalized_title
            log.info(
                "tasks_chore_parsed",
                title=normalized_title,
                assignee=draft.assignee,
                recurrence_type=draft.recurrence_type,
                model=model_label,
            )
            return {
                "parsed_chore": draft_dict,
                "input_tokens": state.get("input_tokens", 0) + (usage.input_tokens or 0),
                "output_tokens": state.get("output_tokens", 0) + (usage.output_tokens or 0),
            }
        except Exception as e:  # noqa: BLE001
            log.error("tasks_chore_parse_failed", error=str(e))
            return {
                "outcome": "incomplete",
                "completeness_issues": [f"parse_error: {e}"],
            }

    async def check_for_duplicates(state: TasksState) -> dict:
        """Dedup — code-side shape comparison first, LLM fallback for legacy.

        Modern parses produce a structured (object, qualifier) shape on
        every chore. When both the new draft AND every existing active
        chore have shape data, comparison is pure code and deterministic.
        When any side is missing shape data (legacy chores from before
        migration 005), fall back to the LLM-based comparison so we don't
        regress on existing data.
        """
        draft = state.get("parsed_chore") or {}
        existing = state.get("active_chores") or []
        new_title = draft.get("title") or ""

        if not new_title or not existing:
            return {
                "duplicate_check": {
                    "confidence": "none",
                    "existing_chore_id": None,
                    "reasoning": "",
                }
            }

        # Code-side shape comparison.
        new_obj = _norm_shape(draft.get("object"))
        new_qual = _norm_shape(draft.get("qualifier"))
        if new_obj and all(c.get("object") for c in existing):
            best: tuple[str, str | None, str] | None = None  # (confidence, chore_id, reason)
            # Rank: high > medium > none. Stop early on a high.
            rank = {"high": 2, "medium": 1, "none": 0}
            for c in existing:
                ex_obj = _norm_shape(c.get("object"))
                ex_qual = _norm_shape(c.get("qualifier"))
                verdict = _shape_verdict(new_obj, new_qual, ex_obj, ex_qual)
                if verdict == "none":
                    continue
                reason = _shape_reason(
                    new_title, c.get("title", c["id"]),
                    new_obj, new_qual, ex_obj, ex_qual, verdict,
                )
                candidate = (verdict, c["id"], reason)
                if best is None or rank[verdict] > rank[best[0]]:
                    best = candidate
                if verdict == "high":
                    break
            if best is None:
                log.info(
                    "tasks_dedup_code_decision",
                    confidence="none",
                    new_object=new_obj, new_qualifier=new_qual,
                )
                return {
                    "duplicate_check": {
                        "confidence": "none",
                        "existing_chore_id": None,
                        "reasoning": "No existing chore shares this object+qualifier.",
                    }
                }
            log.info(
                "tasks_dedup_code_decision",
                confidence=best[0],
                existing_id=best[1],
                new_object=new_obj, new_qualifier=new_qual,
            )
            return {
                "duplicate_check": {
                    "confidence": best[0],
                    "existing_chore_id": best[1],
                    "reasoning": best[2],
                }
            }

        # Fallback: at least one chore is missing structured shape data
        # (legacy DB row from before migration 005). Use the LLM.
        log.info(
            "tasks_dedup_llm_fallback",
            reason="missing_shape_on_new_or_existing",
            new_has_object=bool(new_obj),
            existing_missing_shape=sum(1 for c in existing if not c.get("object")),
        )
        rec = draft.get("recurrence") or {}
        new_rec_desc = (
            f"{(rec.get('frequency') or '?').lower()}"
            + (f" on {','.join(rec.get('byday') or [])}" if rec.get("byday") else "")
        )
        prompt = _DEDUP_PROMPT.format(
            new_title=new_title,
            new_recurrence=new_rec_desc,
            new_assignee=draft.get("assignee") or "household",
            existing_block=_format_existing_block(existing),
        )
        pool, model_label = _pick_pool(state)
        agent = dedup_cloud if pool == "cloud" else dedup_local
        try:
            result = await agent.run(prompt)
            usage = result.usage()
            decision = result.output
            log.info(
                "tasks_dedup_decision",
                confidence=decision.confidence,
                existing_id=decision.existing_chore_id,
                reasoning=decision.reasoning[:120],
                model=model_label,
            )
            return {
                "duplicate_check": {
                    "confidence": decision.confidence,
                    "existing_chore_id": decision.existing_chore_id,
                    "reasoning": decision.reasoning,
                },
                "input_tokens": state.get("input_tokens", 0) + (usage.input_tokens or 0),
                "output_tokens": state.get("output_tokens", 0) + (usage.output_tokens or 0),
            }
        except Exception as e:  # noqa: BLE001
            log.error("tasks_dedup_failed", error=str(e))
            return {
                "duplicate_check": {
                    "confidence": "none",
                    "existing_chore_id": None,
                    "reasoning": f"dedup_failed: {e}",
                }
            }

    async def validate_chore(state: TasksState) -> dict:
        draft = state.get("parsed_chore") or {}
        issues: list[str] = []
        if not draft.get("title"):
            issues.append("no chore title")

        # Infer recurrence_type when the model left it null.
        rec_type = draft.get("recurrence_type")
        due_date = draft.get("due_date_iso")
        if not rec_type:
            rec_type = "once" if due_date else "schedule"
            draft["recurrence_type"] = rec_type

        if rec_type == "once":
            if not due_date:
                issues.append(
                    "no due date given for one-time task (e.g. 'on Friday', "
                    "'tomorrow', '2026-06-01')"
                )
        else:
            rec = draft.get("recurrence")
            if not rec or not rec.get("frequency"):
                issues.append("no recurrence given (e.g. 'every Tuesday', 'weekly')")

        if issues:
            return {"outcome": "incomplete", "completeness_issues": issues}
        return {"parsed_chore": draft}

    async def write_chore(state: TasksState) -> dict:
        draft = state.get("parsed_chore") or {}
        title = draft["title"]
        chore_id = draft.get("id") or _slugify(title)

        # If the slug collides with an existing chore, suffix with -2, -3…
        existing_ids = {c["id"] for c in (state.get("active_chores") or [])}
        if chore_id in existing_ids:
            base = chore_id
            n = 2
            while f"{base}-{n}" in existing_ids:
                n += 1
            chore_id = f"{base}-{n}"

        assignee = draft.get("assignee") or "household"
        rec_type = draft.get("recurrence_type") or "schedule"
        rec_rule = dict(draft.get("recurrence") or {})
        shame = draft.get("shame_after_days") or 3
        due_date = draft.get("due_date_iso") if rec_type == "once" else None
        obj = (draft.get("object") or None)
        if isinstance(obj, str):
            obj = obj.strip().lower() or None
        qual = (draft.get("qualifier") or None)
        if isinstance(qual, str):
            qual = qual.strip().lower() or None
        log.info(
            "tasks_write_chore_start",
            chore_id=chore_id, title=title, assignee=assignee,
            object=obj, qualifier=qual,
        )
        try:
            chore = await store.add_chore(
                id=chore_id,
                title=title,
                description=draft.get("description"),
                assignee=assignee,
                recurrence_type=rec_type,
                recurrence_rule=rec_rule,
                shame_after_days=shame,
                due_date=due_date,
                object=obj,
                qualifier=qual,
            )
            log.info("tasks_write_chore_done", chore_id=chore.id)
            return {
                "created_chore": {
                    "id": chore.id,
                    "title": chore.title,
                    "assignee": chore.assignee,
                    "recurrence_type": chore.recurrence_type,
                    "recurrence_rule": chore.recurrence_rule,
                    "shame_after_days": chore.shame_after_days,
                    "due_date": chore.due_date,
                    "object": chore.object,
                    "qualifier": chore.qualifier,
                },
                "outcome": "created",
            }
        except Exception as e:  # noqa: BLE001
            log.error("tasks_write_failed", error=str(e))
            return {"outcome": "error", "error_message": str(e)}

    async def mark_duplicate_high(state: TasksState) -> dict:
        return {"outcome": "duplicate_existing"}

    async def mark_duplicate_medium(state: TasksState) -> dict:
        return {"outcome": "duplicate_needs_confirmation"}

    # ── Target identification (shared by complete/delete) ─────────────────

    async def parse_target_reference(state: TasksState) -> dict:
        action = state.get("action", "")
        existing = state.get("active_chores") or []
        if not existing:
            return {
                "target_reference": {"reference": ""},
                "outcome": "target_not_found",
            }
        prompt = _TARGET_PARSE_PROMPT.format(
            action=action,
            existing_block=_format_existing_block(existing),
            email_body=state["email_body"],
        )
        pool, model_label = _pick_pool(state)
        agent = target_parse_cloud if pool == "cloud" else target_parse_local
        try:
            result = await agent.run(prompt)
            usage = result.usage()
            ref = result.output
            log.info(
                "tasks_target_parsed",
                action=action,
                reference=ref.reference,
                model=model_label,
            )
            return {
                "target_reference": {"reference": ref.reference},
                "input_tokens": state.get("input_tokens", 0) + (usage.input_tokens or 0),
                "output_tokens": state.get("output_tokens", 0) + (usage.output_tokens or 0),
            }
        except Exception as e:  # noqa: BLE001
            log.error("tasks_target_parse_failed", error=str(e))
            return {
                "target_reference": {"reference": ""},
                "outcome": "target_not_found",
                "error_message": f"parse_error: {e}",
            }

    async def reason_about_target_match(state: TasksState) -> dict:
        """Match the user's natural-language reference to one active chore.

        Direct id match short-circuits (free, deterministic). Everything
        else goes to the LLM specialist with all candidates.
        """
        action = state.get("action", "")
        existing = state.get("active_chores") or []
        target = state.get("target_reference") or {}
        reference = target.get("reference") or ""

        if not existing:
            return {
                "target_match": {
                    "confidence": "none",
                    "chore_id": None,
                    "reasoning": "No active chores being tracked.",
                }
            }

        # Direct id match — user said the slug. Free deterministic shortcut.
        ref_lower = reference.lower().strip()
        for c in existing:
            if c["id"] == ref_lower:
                return {
                    "target_match": {
                        "confidence": "high",
                        "chore_id": c["id"],
                        "reasoning": f"Direct id match: '{c['id']}'.",
                    }
                }

        lines = []
        for c in existing:
            lines.append(
                f"  - id={c['id']}  '{c['title']}'  (assignee={c.get('assignee', '?')})"
            )
        prompt = _TARGET_MATCH_PROMPT.format(
            action=action,
            reference=reference or "(no clear reference)",
            email_body=state["email_body"],
            candidates_block="\n".join(lines),
        )
        pool, model_label = _pick_pool(state)
        agent = target_match_cloud if pool == "cloud" else target_match_local
        try:
            result = await agent.run(prompt)
            usage = result.usage()
            decision = result.output
            log.info(
                "tasks_target_decision",
                action=action,
                confidence=decision.confidence,
                chore_id=(decision.chore_id or ""),
                model=model_label,
            )
            return {
                "target_match": {
                    "confidence": decision.confidence,
                    "chore_id": decision.chore_id,
                    "reasoning": decision.reasoning,
                },
                "input_tokens": state.get("input_tokens", 0) + (usage.input_tokens or 0),
                "output_tokens": state.get("output_tokens", 0) + (usage.output_tokens or 0),
            }
        except Exception as e:  # noqa: BLE001
            log.error("tasks_target_reason_failed", error=str(e))
            return {
                "target_match": {
                    "confidence": "none",
                    "chore_id": None,
                    "reasoning": f"reasoning failed: {e}",
                }
            }

    # ── Complete branch ───────────────────────────────────────────────────

    async def write_completion(state: TasksState) -> dict:
        match = state.get("target_match") or {}
        chore_id = match.get("chore_id")
        if not chore_id:
            return {"outcome": "target_not_found"}
        # The completion is attributed to the inbound source. We don't know
        # the sender precisely here — store the assignee bucket as a sensible
        # default; the inbox/IMAP poller can override via context later.
        existing = state.get("active_chores") or []
        chore = next((c for c in existing if c["id"] == chore_id), None)
        completed_by = (chore or {}).get("assignee") or "household"
        try:
            comp = await store.record_completion(
                chore_id=chore_id,
                completed_by=completed_by,
                completed_via="email",
            )
            return {
                "completed_chore": {
                    "id": chore_id,
                    "title": (chore or {}).get("title", chore_id),
                    "completed_by": comp.completed_by,
                    "completed_at": comp.completed_at,
                },
                "outcome": "completed",
            }
        except Exception as e:  # noqa: BLE001
            log.error("tasks_complete_failed", error=str(e))
            return {"outcome": "error", "error_message": str(e)}

    # ── Delete branch ─────────────────────────────────────────────────────

    async def soft_delete_chore_action(state: TasksState) -> dict:
        match = state.get("target_match") or {}
        chore_id = match.get("chore_id")
        if not chore_id:
            return {"outcome": "target_not_found"}
        existing = state.get("active_chores") or []
        chore = next((c for c in existing if c["id"] == chore_id), None)
        try:
            ok = await store.soft_delete_chore(chore_id)
        except Exception as e:  # noqa: BLE001
            log.error("tasks_delete_failed", error=str(e))
            return {"outcome": "error", "error_message": str(e)}
        if not ok:
            return {"outcome": "target_not_found"}
        return {
            "deleted_chore": {
                "id": chore_id,
                "title": (chore or {}).get("title", chore_id),
            },
            "outcome": "deleted",
        }

    # ── List branch ───────────────────────────────────────────────────────

    async def render_pending_summary(state: TasksState) -> dict:
        tz = ZoneInfo(timezone_name)
        now = now_fn() if now_fn is not None else datetime.now(tz)
        statuses = await store.status_for_all_active(now=now, tz=tz)
        summary: list[dict] = []
        for s in statuses:
            summary.append(
                {
                    "id": s.chore.id,
                    "title": s.chore.title,
                    "assignee": s.chore.assignee,
                    "next_due_iso": s.next_due.isoformat(),
                    "overdue_days": s.overdue_days,
                    "last_completed_at": (
                        s.last_completed_at.isoformat() if s.last_completed_at else None
                    ),
                }
            )
        return {"pending_summary": summary, "outcome": "listed"}

    # ── Update branch ─────────────────────────────────────────────────────

    async def parse_chore_update(state: TasksState) -> dict:
        existing = state.get("active_chores") or []
        if not existing:
            return {"outcome": "target_not_found"}
        prompt = _UPDATE_PARSE_PROMPT.format(
            existing_block=_format_existing_block(existing),
            today_day_of_week=state.get("today_day_of_week", ""),
            today_iso=state.get("today_iso", ""),
            timezone_name=state.get("timezone_name", ""),
            name_mapping_block=_name_mapping_block(name_map),
            email_body=state["email_body"],
        )
        pool, model_label = _pick_pool(state)
        agent = update_parse_cloud if pool == "cloud" else update_parse_local
        try:
            result = await agent.run(prompt)
            usage = result.usage()
            draft = result.output
            log.info(
                "tasks_update_parsed",
                target=draft.target_reference,
                changes=[k for k, v in draft.model_dump().items()
                         if k != "target_reference" and v is not None],
                model=model_label,
            )
            return {
                "update_draft": draft.model_dump(),
                "target_reference": {"reference": draft.target_reference},
                "input_tokens": state.get("input_tokens", 0) + (usage.input_tokens or 0),
                "output_tokens": state.get("output_tokens", 0) + (usage.output_tokens or 0),
            }
        except Exception as e:  # noqa: BLE001
            log.error("tasks_update_parse_failed", error=str(e))
            return {
                "outcome": "incomplete",
                "completeness_issues": [f"parse_error: {e}"],
            }

    async def apply_chore_update(state: TasksState) -> dict:
        match = state.get("target_match") or {}
        chore_id = match.get("chore_id")
        if not chore_id:
            return {"outcome": "target_not_found"}
        draft = state.get("update_draft") or {}
        fields: dict[str, Any] = {}
        if draft.get("new_title"):
            fields["title"] = _normalize_title(draft["new_title"])
        if draft.get("new_assignee"):
            fields["assignee"] = draft["new_assignee"]
        if draft.get("new_recurrence_type"):
            fields["recurrence_type"] = draft["new_recurrence_type"]
        if draft.get("new_recurrence"):
            fields["recurrence_rule"] = draft["new_recurrence"]
        if draft.get("new_shame_after_days") is not None:
            fields["shame_after_days"] = draft["new_shame_after_days"]
        if not fields:
            return {
                "outcome": "incomplete",
                "completeness_issues": ["no changes specified"],
            }
        try:
            updated = await store.update_chore(chore_id, **fields)
        except Exception as e:  # noqa: BLE001
            log.error("tasks_update_failed", error=str(e))
            return {"outcome": "error", "error_message": str(e)}
        if updated is None:
            return {"outcome": "target_not_found"}
        return {
            "updated_chore": {
                "id": updated.id,
                "title": updated.title,
                "assignee": updated.assignee,
                "recurrence_type": updated.recurrence_type,
                "recurrence_rule": updated.recurrence_rule,
                "shame_after_days": updated.shame_after_days,
                "changed_fields": list(fields.keys()),
            },
            "outcome": "updated",
        }

    async def mark_target_needs_confirmation(state: TasksState) -> dict:
        return {"outcome": "target_needs_confirmation"}

    async def mark_target_not_found(state: TasksState) -> dict:
        return {"outcome": "target_not_found"}

    async def mark_clarification(state: TasksState) -> dict:
        return {"outcome": "clarification"}

    async def build_reply(state: TasksState) -> dict:
        from alfred.notifications.calendar_render import (
            render_clarification_html,
            render_clarification_plain,
        )

        outcome = state.get("outcome", "clarification")

        # Guard against stale-state UX bugs: if a side-effect node populated
        # one of the result dicts (created_chore / completed_chore / etc.)
        # but `outcome` ended up empty or pointing at a clarification branch,
        # always render the success reply that matches the populated dict.
        # Mirrors the symptom the user hit where a chore was written but the
        # email reply still said "Need more information".
        result_outcomes = {
            "created_chore": "created",
            "completed_chore": "completed",
            "deleted_chore": "deleted",
            "updated_chore": "updated",
        }
        for key, success in result_outcomes.items():
            if state.get(key) and outcome != success:
                log.warning(
                    "tasks_build_reply_outcome_mismatch",
                    populated=key,
                    state_outcome=outcome,
                    rendering_as=success,
                )
                outcome = success
                break

        log.info(
            "tasks_build_reply",
            outcome=outcome,
            has_created=bool(state.get("created_chore")),
            has_completed=bool(state.get("completed_chore")),
            has_deleted=bool(state.get("deleted_chore")),
            has_updated=bool(state.get("updated_chore")),
        )

        if outcome == "created":
            c = state["created_chore"]
            if c["recurrence_type"] == "once":
                when = f"due {c.get('due_date') or '?'} (one-time)"
            else:
                when = _format_recurrence_human(
                    c.get("recurrence_rule") or {}, c["recurrence_type"]
                )
            txt = (
                f"Added chore '{c['title']}' (id: {c['id']})\n"
                f"  • assignee: {c['assignee']}\n"
                f"  • {when}\n"
                f"  • shame after: {c['shame_after_days']} days overdue"
            )
            return {
                "reply_plain": render_clarification_plain(txt),
                "reply_html": render_clarification_html(txt),
            }

        if outcome == "duplicate_existing":
            dup = state.get("duplicate_check") or {}
            existing_id = dup.get("existing_chore_id") or "?"
            txt = (
                f"This looks like the same chore as '{existing_id}' that's "
                f"already on the list. {dup.get('reasoning', '')}\n\n"
                "Did you mean to update the existing chore? Reply with what "
                "to change, or 'add anyway' if it's actually a separate chore."
            )
            return {
                "reply_plain": render_clarification_plain(txt),
                "reply_html": render_clarification_html(txt),
            }

        if outcome == "duplicate_needs_confirmation":
            dup = state.get("duplicate_check") or {}
            existing_id = dup.get("existing_chore_id") or "?"
            txt = (
                f"I think this might overlap with the existing chore "
                f"'{existing_id}'. {dup.get('reasoning', '')}\n\n"
                "Reply 'add anyway' to create it as a new chore, or describe "
                "the difference so I can update the existing one instead."
            )
            return {
                "reply_plain": render_clarification_plain(txt),
                "reply_html": render_clarification_html(txt),
            }

        if outcome == "completed":
            c = state["completed_chore"]
            txt = (
                f"Marked '{c['title']}' (id: {c['id']}) as done.\n"
                f"  • completed by: {c['completed_by']}\n"
                f"  • at: {c['completed_at']}"
            )
            return {
                "reply_plain": render_clarification_plain(txt),
                "reply_html": render_clarification_html(txt),
            }

        if outcome == "deleted":
            c = state["deleted_chore"]
            txt = (
                f"Removed '{c['title']}' (id: {c['id']}) from tracking.\n\n"
                "You won't get reminders for this chore anymore. Existing "
                "completion history is preserved."
            )
            return {
                "reply_plain": render_clarification_plain(txt),
                "reply_html": render_clarification_html(txt),
            }

        if outcome == "updated":
            c = state["updated_chore"]
            changes = ", ".join(c.get("changed_fields", [])) or "(none)"
            when = _format_recurrence_human(
                c.get("recurrence_rule") or {}, c["recurrence_type"]
            )
            txt = (
                f"Updated '{c['title']}' (id: {c['id']}). Changed: {changes}.\n"
                f"  • assignee: {c['assignee']}\n"
                f"  • recurrence: {when}\n"
                f"  • shame after: {c['shame_after_days']} days overdue"
            )
            return {
                "reply_plain": render_clarification_plain(txt),
                "reply_html": render_clarification_html(txt),
            }

        if outcome == "listed":
            summary = state.get("pending_summary") or []
            if not summary:
                txt = "No chores are currently being tracked."
            else:
                overdue = [s for s in summary if s["overdue_days"] > 0]
                due_today = [s for s in summary if s["overdue_days"] == 0]
                upcoming = [s for s in summary if s["overdue_days"] < 0]

                lines: list[str] = []
                if overdue:
                    lines.append("OVERDUE")
                    for s in sorted(overdue, key=lambda x: -x["overdue_days"]):
                        lines.append(
                            f"  • {s['title']} (id: {s['id']}, {s['assignee']}) — "
                            f"{s['overdue_days']}d overdue"
                        )
                if due_today:
                    if lines:
                        lines.append("")
                    lines.append("DUE TODAY")
                    for s in due_today:
                        lines.append(
                            f"  • {s['title']} (id: {s['id']}, {s['assignee']})"
                        )
                if upcoming:
                    if lines:
                        lines.append("")
                    lines.append("UPCOMING")
                    for s in upcoming:
                        lines.append(
                            f"  • {s['title']} (id: {s['id']}, {s['assignee']}) — "
                            f"due {s['next_due_iso'][:10]}"
                        )
                txt = "\n".join(lines)
            return {
                "reply_plain": render_clarification_plain(txt),
                "reply_html": render_clarification_html(txt),
            }

        if outcome == "target_needs_confirmation":
            match = state.get("target_match") or {}
            chore_id = match.get("chore_id") or "?"
            txt = (
                f"I think you mean the chore '{chore_id}'. "
                f"{match.get('reasoning', '')}\n\n"
                "Reply 'yes' to confirm, or give me more detail to pick a different one."
            )
            return {
                "reply_plain": render_clarification_plain(txt),
                "reply_html": render_clarification_html(txt),
            }

        if outcome == "target_not_found":
            action = state.get("action") or "that"
            ref = (state.get("target_reference") or {}).get("reference") or ""
            tail = f" matching '{ref}'" if ref else ""
            txt = (
                f"I couldn't find a chore{tail} to {action}. "
                "Reply with the chore id or a clearer name."
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
                "Sorry — something went wrong while processing your chore: "
                + state.get("error_message", "unknown error")
            )
            return {
                "reply_plain": render_clarification_plain(txt),
                "reply_html": render_clarification_html(txt),
            }

        # default: clarification
        txt = (
            "I couldn't tell what you wanted to do with the chore list. Try "
            "something like 'Add chore: take out trash every Tuesday, household'."
        )
        return {
            "reply_plain": render_clarification_plain(txt),
            "reply_html": render_clarification_html(txt),
        }

    # ── Edges ────────────────────────────────────────────────────────────

    def route_after_intent(state: TasksState) -> str:
        action = state.get("action", "clarify")
        if action == "create":
            return "parse_chore_draft"
        if action == "complete":
            return "parse_target_reference"
        if action == "delete":
            return "parse_target_reference"
        if action == "update":
            return "parse_chore_update"
        if action == "list":
            return "render_pending_summary"
        return "mark_clarification"

    def route_after_parse(state: TasksState) -> str:
        # Parse may have short-circuited to incomplete on exception.
        if state.get("outcome") == "incomplete":
            return "build_reply"
        return "check_for_duplicates"

    def route_after_dedup(state: TasksState) -> str:
        dup = state.get("duplicate_check") or {}
        conf = dup.get("confidence", "none")
        if conf == "high":
            return "mark_duplicate_high"
        if conf == "medium":
            return "mark_duplicate_medium"
        return "validate_chore"

    def route_after_validate(state: TasksState) -> str:
        return "build_reply" if state.get("outcome") == "incomplete" else "write_chore"

    def route_after_target_parse(state: TasksState) -> str:
        if state.get("outcome") == "target_not_found":
            return "build_reply"
        return "reason_about_target_match"

    def route_after_target_match(state: TasksState) -> str:
        action = state.get("action", "")
        match = state.get("target_match") or {}
        conf = match.get("confidence", "none")
        if conf == "medium":
            return "mark_target_needs_confirmation"
        if conf == "none":
            return "mark_target_not_found"
        # high confidence — dispatch by action
        if action == "complete":
            return "write_completion"
        if action == "delete":
            return "soft_delete_chore_action"
        if action == "update":
            return "apply_chore_update"
        return "mark_clarification"

    def route_after_update_parse(state: TasksState) -> str:
        if state.get("outcome") in {"incomplete", "target_not_found"}:
            return "build_reply"
        return "reason_about_target_match"

    # ── Assemble ─────────────────────────────────────────────────────────

    graph = StateGraph(TasksState)
    for node_name, fn in [
        ("enrich_context", enrich_context),
        ("classify_intent", classify_intent),
        ("parse_chore_draft", parse_chore_draft),
        ("check_for_duplicates", check_for_duplicates),
        ("validate_chore", validate_chore),
        ("write_chore", write_chore),
        ("mark_duplicate_high", mark_duplicate_high),
        ("mark_duplicate_medium", mark_duplicate_medium),
        ("parse_target_reference", parse_target_reference),
        ("reason_about_target_match", reason_about_target_match),
        ("write_completion", write_completion),
        ("soft_delete_chore_action", soft_delete_chore_action),
        ("parse_chore_update", parse_chore_update),
        ("apply_chore_update", apply_chore_update),
        ("render_pending_summary", render_pending_summary),
        ("mark_target_needs_confirmation", mark_target_needs_confirmation),
        ("mark_target_not_found", mark_target_not_found),
        ("mark_clarification", mark_clarification),
        ("build_reply", build_reply),
    ]:
        graph.add_node(node_name, _traced_node(node_name, fn))

    graph.set_entry_point("enrich_context")
    graph.add_edge("enrich_context", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "parse_chore_draft": "parse_chore_draft",
            "parse_target_reference": "parse_target_reference",
            "parse_chore_update": "parse_chore_update",
            "render_pending_summary": "render_pending_summary",
            "mark_clarification": "mark_clarification",
        },
    )
    # Create branch
    graph.add_conditional_edges(
        "parse_chore_draft",
        route_after_parse,
        {
            "build_reply": "build_reply",
            "check_for_duplicates": "check_for_duplicates",
        },
    )
    graph.add_conditional_edges(
        "check_for_duplicates",
        route_after_dedup,
        {
            "mark_duplicate_high": "mark_duplicate_high",
            "mark_duplicate_medium": "mark_duplicate_medium",
            "validate_chore": "validate_chore",
        },
    )
    graph.add_conditional_edges(
        "validate_chore",
        route_after_validate,
        {
            "build_reply": "build_reply",
            "write_chore": "write_chore",
        },
    )
    graph.add_edge("write_chore", "build_reply")
    graph.add_edge("mark_duplicate_high", "build_reply")
    graph.add_edge("mark_duplicate_medium", "build_reply")
    # Complete / delete branch (shared identifier)
    graph.add_conditional_edges(
        "parse_target_reference",
        route_after_target_parse,
        {
            "build_reply": "build_reply",
            "reason_about_target_match": "reason_about_target_match",
        },
    )
    graph.add_conditional_edges(
        "reason_about_target_match",
        route_after_target_match,
        {
            "write_completion": "write_completion",
            "soft_delete_chore_action": "soft_delete_chore_action",
            "apply_chore_update": "apply_chore_update",
            "mark_target_needs_confirmation": "mark_target_needs_confirmation",
            "mark_target_not_found": "mark_target_not_found",
            "mark_clarification": "mark_clarification",
        },
    )
    graph.add_edge("write_completion", "build_reply")
    graph.add_edge("soft_delete_chore_action", "build_reply")
    # Update branch (parse first to know what to change, then identify target)
    graph.add_conditional_edges(
        "parse_chore_update",
        route_after_update_parse,
        {
            "build_reply": "build_reply",
            "reason_about_target_match": "reason_about_target_match",
        },
    )
    graph.add_edge("apply_chore_update", "build_reply")
    # List branch
    graph.add_edge("render_pending_summary", "build_reply")
    # Misc
    graph.add_edge("mark_target_needs_confirmation", "build_reply")
    graph.add_edge("mark_target_not_found", "build_reply")
    graph.add_edge("mark_clarification", "build_reply")
    graph.add_edge("build_reply", END)

    return graph.compile()
