"""Tasks workflow — LangGraph state machine for chore management.

Graph topology and per-node closures live here. Everything else — Pydantic
schemas, prompt strings, deterministic helpers, LLM specialists, reply
rendering — lives in sibling modules:

  schemas.py      typed outputs of LLM specialists
  prompts.py      prompt templates + format helpers
  formatting.py   pure-code helpers (slug, title normalize, shape match…)
  specialists.py  Specialist class encapsulating local/cloud + token bookkeeping
  outcomes.py     Outcome StrEnum — single source of truth for outcome strings
  replies.py      dispatch-table reply builder + assignee humanization
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypedDict
from zoneinfo import ZoneInfo

import structlog
from langgraph.graph import END, StateGraph
from opentelemetry import trace

from alfred.agents.tasks.formatting import (
    format_recurrence_human,
    normalize_shape,
    normalize_title,
    shape_reason,
    shape_verdict,
    slugify,
)
from alfred.agents.tasks.outcomes import Outcome
from alfred.agents.tasks.prompts import (
    DEDUP_PROMPT,
    INTENT_PROMPT,
    PARSE_PROMPT,
    TARGET_MATCH_PROMPT,
    TARGET_PARSE_PROMPT,
    UPDATE_PARSE_PROMPT,
    format_existing_block,
    name_mapping_block,
)
from alfred.agents.tasks.replies import build_payload, resolve_outcome
from alfred.agents.tasks.specialists import SpecialistRegistry
from alfred.notifications.tasks_render import render_reply_html, render_reply_plain

if TYPE_CHECKING:
    from alfred.agents.tasks.store import ChoreStore
    from alfred.routing.clients import LiteLLMClient

log = structlog.get_logger()
tracer = trace.get_tracer("alfred.tasks.workflow")

_OI_SPAN_KIND = "openinference.span.kind"


# ── Graph state ─────────────────────────────────────────────────────────────


class TasksState(TypedDict, total=False):
    """LangGraph state container.

    All keys are optional (`total=False`) — node functions return only
    the slices they touch and LangGraph merges them. Anything not declared
    here is silently dropped by the state machine, so every key a node
    returns MUST appear in this TypedDict.
    """

    # Inputs
    email_body: str

    # Date / chore context (filled by enrich_context)
    today_iso: str
    today_day_of_week: str
    timezone_name: str
    active_chores: list[dict]

    # Classification (filled by classify_intent)
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

    # Outcome (one of the Outcome enum values)
    outcome: str
    error_message: str

    # Reply text (filled by build_reply)
    reply_plain: str
    reply_html: str

    # Cumulative token usage
    input_tokens: int
    output_tokens: int


# ── Tracing ─────────────────────────────────────────────────────────────────


def _traced_node(
    name: str,
    fn: Callable[[TasksState], Any],
) -> Callable[[TasksState], Any]:
    """Wrap a node coroutine so each invocation emits a CHAIN child span.

    The parent AGENT span is set by TasksAgent.run; this adds one CHAIN
    span per node with action/outcome attributes for Phoenix.
    """

    async def wrapper(state: TasksState) -> Any:
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


# ── Graph builder ───────────────────────────────────────────────────────────


def build_tasks_graph(
    store: ChoreStore,
    timezone_name: str,
    litellm_client: LiteLLMClient,
    model_name: str = "local-default",
    now_fn: Any = None,
    assignee_names: dict[str, str] | None = None,
) -> Any:
    """Compile the tasks workflow with all dependencies closed over.

    Args:
        store:           ChoreStore for DB reads/writes.
        timezone_name:   IANA tz name for date math + "today" rendering.
        litellm_client:  Used to construct typed-output specialist agents.
        model_name:      LiteLLM alias for the local model (default
                         "local-default"). Cloud routing uses
                         "cloud-default" automatically when intent
                         classification flags complexity=complex.
        now_fn:          Optional callable returning a tz-aware datetime.
                         Lets the simulation harness pin "today" to a
                         specific date deterministically.
        assignee_names:  Optional user_id → display-name mapping. The
                         parse prompt teaches the model to resolve names
                         (e.g. "assign to Antonio" → primary), and the
                         reply renderer uses these for user-facing text.
    """
    name_map = assignee_names or {}
    specialists = SpecialistRegistry.build(litellm_client, model_name)

    # ── Nodes ────────────────────────────────────────────────────────────

    async def enrich_context(state: TasksState) -> dict[str, Any]:
        """Pure code: load today + active chores from the store."""
        tz = ZoneInfo(timezone_name)
        now = now_fn() if now_fn is not None else datetime.now(tz)
        today = now.date()
        active = await store.list_active_chores()
        return {
            "today_iso": today.isoformat(),
            "today_day_of_week": today.strftime("%A"),
            "timezone_name": timezone_name,
            "active_chores": [
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
            ],
        }

    async def classify_intent(state: TasksState) -> dict[str, Any]:
        """LLM: pick action + complexity for downstream routing."""
        prompt = INTENT_PROMPT.format(email_body=state["email_body"])
        invocation = await specialists.intent.invoke(prompt, state)
        if invocation.output is None:
            return {
                "outcome": Outcome.ERROR.value,
                "error_message": "intent classification failed",
                **invocation.updates,
            }
        log.info(
            "tasks_intent_classified",
            action=invocation.output.action,
            complexity=invocation.output.complexity,
            reasoning=invocation.output.reasoning[:120],
        )
        return {
            "action": invocation.output.action,
            "complexity": invocation.output.complexity,
            **invocation.updates,
        }

    async def parse_chore_draft(state: TasksState) -> dict[str, Any]:
        """LLM: parse the user's email into a typed ChoreDraft."""
        prompt = PARSE_PROMPT.format(
            today_day_of_week=state["today_day_of_week"],
            today_iso=state["today_iso"],
            timezone_name=state["timezone_name"],
            name_mapping_block=name_mapping_block(name_map),
            email_body=state["email_body"],
        )
        invocation = await specialists.parse.invoke(prompt, state)
        if invocation.output is None:
            return {
                "outcome": Outcome.INCOMPLETE.value,
                "completeness_issues": ["parse_error"],
                **invocation.updates,
            }
        draft = invocation.output
        normalized = normalize_title(draft.title) if draft.title else draft.title
        if normalized != draft.title:
            log.info(
                "tasks_title_normalized",
                raw=draft.title, normalized=normalized,
            )
        draft_dict = draft.model_dump()
        draft_dict["title"] = normalized
        log.info(
            "tasks_chore_parsed",
            title=normalized,
            assignee=draft.assignee,
            recurrence_type=draft.recurrence_type,
            model=invocation.model_label,
        )
        return {"parsed_chore": draft_dict, **invocation.updates}

    async def check_for_duplicates(state: TasksState) -> dict[str, Any]:
        """Code-side shape comparison first; LLM fallback for legacy chores."""
        draft = state.get("parsed_chore") or {}
        existing = state.get("active_chores") or []
        new_title = draft.get("title") or ""

        if not new_title or not existing:
            return {"duplicate_check": _no_match("")}

        new_obj = normalize_shape(draft.get("object"))
        new_qual = normalize_shape(draft.get("qualifier"))

        # Code path: every existing chore must have an object for a
        # deterministic comparison. Otherwise fall back to LLM.
        all_have_shape = bool(new_obj) and all(c.get("object") for c in existing)
        if all_have_shape:
            best = _best_shape_match(new_title, new_obj, new_qual, existing)
            if best is None:
                log.info(
                    "tasks_dedup_code_decision",
                    confidence="none",
                    new_object=new_obj, new_qualifier=new_qual,
                )
                return {
                    "duplicate_check": _no_match(
                        "No existing chore shares this object+qualifier."
                    ),
                }
            confidence, chore_id, reason = best
            log.info(
                "tasks_dedup_code_decision",
                confidence=confidence,
                existing_id=chore_id,
                new_object=new_obj, new_qualifier=new_qual,
            )
            return {
                "duplicate_check": {
                    "confidence": confidence,
                    "existing_chore_id": chore_id,
                    "reasoning": reason,
                },
            }

        # Fallback: at least one chore is missing shape data (legacy row
        # from before migration 005). Use the LLM specialist.
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
        prompt = DEDUP_PROMPT.format(
            new_title=new_title,
            new_recurrence=new_rec_desc,
            new_assignee=draft.get("assignee") or "household",
            existing_block=format_existing_block(existing),
        )
        invocation = await specialists.dedup.invoke(prompt, state)
        if invocation.output is None:
            return {
                "duplicate_check": _no_match("dedup specialist failed"),
                **invocation.updates,
            }
        decision = invocation.output
        log.info(
            "tasks_dedup_decision",
            confidence=decision.confidence,
            existing_id=decision.existing_chore_id,
            reasoning=decision.reasoning[:120],
            model=invocation.model_label,
        )
        return {
            "duplicate_check": {
                "confidence": decision.confidence,
                "existing_chore_id": decision.existing_chore_id,
                "reasoning": decision.reasoning,
            },
            **invocation.updates,
        }

    async def validate_chore(state: TasksState) -> dict[str, Any]:
        """Pure code: required-field check, infer recurrence_type from due_date."""
        draft = state.get("parsed_chore") or {}
        issues: list[str] = []
        if not draft.get("title"):
            issues.append("no chore title")
        rec_type = draft.get("recurrence_type")
        due_date = draft.get("due_date_iso")
        if not rec_type:
            rec_type = "once" if due_date else "schedule"
            draft["recurrence_type"] = rec_type
        if rec_type == "once":
            if not due_date:
                issues.append(
                    "no due date given for one-time task "
                    "(e.g. 'on Friday', 'tomorrow', '2026-06-01')"
                )
        else:
            rec = draft.get("recurrence")
            if not rec or not rec.get("frequency"):
                issues.append("no recurrence given (e.g. 'every Tuesday', 'weekly')")
        if issues:
            return {
                "outcome": Outcome.INCOMPLETE.value,
                "completeness_issues": issues,
            }
        return {"parsed_chore": draft}

    async def write_chore(state: TasksState) -> dict[str, Any]:
        """Insert the parsed chore into the store and surface it on state."""
        draft = state.get("parsed_chore") or {}
        title = draft["title"]
        existing_ids = {c["id"] for c in (state.get("active_chores") or [])}
        chore_id = _pick_chore_id(draft.get("id") or slugify(title), existing_ids)
        rec_type = draft.get("recurrence_type") or "schedule"
        log.info(
            "tasks_write_chore_start",
            chore_id=chore_id, title=title,
            assignee=draft.get("assignee") or "household",
            object=normalize_shape(draft.get("object")),
            qualifier=normalize_shape(draft.get("qualifier")),
        )
        try:
            chore = await store.add_chore(
                id=chore_id,
                title=title,
                description=draft.get("description"),
                assignee=draft.get("assignee") or "household",
                recurrence_type=rec_type,
                recurrence_rule=dict(draft.get("recurrence") or {}),
                shame_after_days=draft.get("shame_after_days") or 3,
                due_date=draft.get("due_date_iso") if rec_type == "once" else None,
                object=normalize_shape(draft.get("object")),
                qualifier=normalize_shape(draft.get("qualifier")),
            )
        except Exception as e:  # noqa: BLE001 — surface store errors as outcome
            log.error("tasks_write_failed", error=str(e))
            return {"outcome": Outcome.ERROR.value, "error_message": str(e)}
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
            "outcome": Outcome.CREATED.value,
        }

    async def mark_duplicate_high(_state: TasksState) -> dict[str, Any]:
        return {"outcome": Outcome.DUPLICATE_EXISTING.value}

    async def mark_duplicate_medium(_state: TasksState) -> dict[str, Any]:
        return {"outcome": Outcome.DUPLICATE_NEEDS_CONFIRMATION.value}

    # ── Target identification (shared by complete/delete/update) ─────────

    async def parse_target_reference(state: TasksState) -> dict[str, Any]:
        """LLM: extract the user's natural-language reference to a chore."""
        action = state.get("action", "")
        existing = state.get("active_chores") or []
        if not existing:
            return {
                "target_reference": {"reference": ""},
                "outcome": Outcome.TARGET_NOT_FOUND.value,
            }
        prompt = TARGET_PARSE_PROMPT.format(
            action=action,
            existing_block=format_existing_block(existing),
            email_body=state["email_body"],
        )
        invocation = await specialists.target_parse.invoke(prompt, state)
        if invocation.output is None:
            return {
                "target_reference": {"reference": ""},
                "outcome": Outcome.TARGET_NOT_FOUND.value,
                "error_message": "target reference parse failed",
                **invocation.updates,
            }
        log.info(
            "tasks_target_parsed",
            action=action,
            reference=invocation.output.reference,
            model=invocation.model_label,
        )
        return {
            "target_reference": {"reference": invocation.output.reference},
            **invocation.updates,
        }

    async def reason_about_target_match(state: TasksState) -> dict[str, Any]:
        """Resolve the reference to one active chore (id-shortcut, then LLM)."""
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
                },
            }

        # Direct id shortcut — deterministic, no LLM cost.
        ref_lower = reference.lower().strip()
        for c in existing:
            if c["id"] == ref_lower:
                return {
                    "target_match": {
                        "confidence": "high",
                        "chore_id": c["id"],
                        "reasoning": f"Direct id match: '{c['id']}'.",
                    },
                }

        candidates_block = "\n".join(
            f"  - id={c['id']}  '{c['title']}'  (assignee={c.get('assignee', '?')})"
            for c in existing
        )
        prompt = TARGET_MATCH_PROMPT.format(
            action=action,
            reference=reference or "(no clear reference)",
            email_body=state["email_body"],
            candidates_block=candidates_block,
        )
        invocation = await specialists.target_match.invoke(prompt, state)
        if invocation.output is None:
            return {
                "target_match": {
                    "confidence": "none",
                    "chore_id": None,
                    "reasoning": "target-match specialist failed",
                },
                **invocation.updates,
            }
        decision = invocation.output
        log.info(
            "tasks_target_decision",
            action=action,
            confidence=decision.confidence,
            chore_id=decision.chore_id or "",
            model=invocation.model_label,
        )
        return {
            "target_match": {
                "confidence": decision.confidence,
                "chore_id": decision.chore_id,
                "reasoning": decision.reasoning,
            },
            **invocation.updates,
        }

    # ── Complete branch ──────────────────────────────────────────────────

    async def write_completion(state: TasksState) -> dict[str, Any]:
        match = state.get("target_match") or {}
        chore_id = match.get("chore_id")
        if not chore_id:
            return {"outcome": Outcome.TARGET_NOT_FOUND.value}
        existing = state.get("active_chores") or []
        chore = next((c for c in existing if c["id"] == chore_id), None)
        completed_by = (chore or {}).get("assignee") or "household"
        try:
            comp = await store.record_completion(
                chore_id=chore_id,
                completed_by=completed_by,
                completed_via="email",
            )
        except Exception as e:  # noqa: BLE001
            log.error("tasks_complete_failed", error=str(e))
            return {"outcome": Outcome.ERROR.value, "error_message": str(e)}
        return {
            "completed_chore": {
                "id": chore_id,
                "title": (chore or {}).get("title", chore_id),
                "completed_by": comp.completed_by,
                "completed_at": comp.completed_at,
            },
            "outcome": Outcome.COMPLETED.value,
        }

    # ── Delete branch ────────────────────────────────────────────────────

    async def soft_delete_chore_action(state: TasksState) -> dict[str, Any]:
        match = state.get("target_match") or {}
        chore_id = match.get("chore_id")
        if not chore_id:
            return {"outcome": Outcome.TARGET_NOT_FOUND.value}
        existing = state.get("active_chores") or []
        chore = next((c for c in existing if c["id"] == chore_id), None)
        try:
            ok = await store.soft_delete_chore(chore_id)
        except Exception as e:  # noqa: BLE001
            log.error("tasks_delete_failed", error=str(e))
            return {"outcome": Outcome.ERROR.value, "error_message": str(e)}
        if not ok:
            return {"outcome": Outcome.TARGET_NOT_FOUND.value}
        return {
            "deleted_chore": {
                "id": chore_id,
                "title": (chore or {}).get("title", chore_id),
            },
            "outcome": Outcome.DELETED.value,
        }

    # ── List branch ──────────────────────────────────────────────────────

    async def render_pending_summary(_state: TasksState) -> dict[str, Any]:
        tz = ZoneInfo(timezone_name)
        now = now_fn() if now_fn is not None else datetime.now(tz)
        statuses = await store.status_for_all_active(now=now, tz=tz)
        summary = [
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
            for s in statuses
        ]
        return {"pending_summary": summary, "outcome": Outcome.LISTED.value}

    # ── Update branch ────────────────────────────────────────────────────

    async def parse_chore_update(state: TasksState) -> dict[str, Any]:
        existing = state.get("active_chores") or []
        if not existing:
            return {"outcome": Outcome.TARGET_NOT_FOUND.value}
        prompt = UPDATE_PARSE_PROMPT.format(
            existing_block=format_existing_block(existing),
            today_day_of_week=state.get("today_day_of_week", ""),
            today_iso=state.get("today_iso", ""),
            timezone_name=state.get("timezone_name", ""),
            name_mapping_block=name_mapping_block(name_map),
            email_body=state["email_body"],
        )
        invocation = await specialists.update_parse.invoke(prompt, state)
        if invocation.output is None:
            return {
                "outcome": Outcome.INCOMPLETE.value,
                "completeness_issues": ["update_parse_failed"],
                **invocation.updates,
            }
        draft = invocation.output
        log.info(
            "tasks_update_parsed",
            target=draft.target_reference,
            changes=[
                k for k, v in draft.model_dump().items()
                if k != "target_reference" and v is not None
            ],
            model=invocation.model_label,
        )
        return {
            "update_draft": draft.model_dump(),
            "target_reference": {"reference": draft.target_reference},
            **invocation.updates,
        }

    async def apply_chore_update(state: TasksState) -> dict[str, Any]:
        match = state.get("target_match") or {}
        chore_id = match.get("chore_id")
        if not chore_id:
            return {"outcome": Outcome.TARGET_NOT_FOUND.value}
        fields = _build_update_fields(state.get("update_draft") or {})
        if not fields:
            return {
                "outcome": Outcome.INCOMPLETE.value,
                "completeness_issues": ["no changes specified"],
            }
        try:
            updated = await store.update_chore(chore_id, **fields)
        except Exception as e:  # noqa: BLE001
            log.error("tasks_update_failed", error=str(e))
            return {"outcome": Outcome.ERROR.value, "error_message": str(e)}
        if updated is None:
            return {"outcome": Outcome.TARGET_NOT_FOUND.value}
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
            "outcome": Outcome.UPDATED.value,
        }

    # ── Marker nodes (set outcome only) ──────────────────────────────────

    async def mark_target_needs_confirmation(_s: TasksState) -> dict[str, Any]:
        return {"outcome": Outcome.TARGET_NEEDS_CONFIRMATION.value}

    async def mark_target_not_found(_s: TasksState) -> dict[str, Any]:
        return {"outcome": Outcome.TARGET_NOT_FOUND.value}

    async def mark_clarification(_s: TasksState) -> dict[str, Any]:
        return {"outcome": Outcome.CLARIFICATION.value}

    # ── Reply ────────────────────────────────────────────────────────────

    async def build_reply(state: TasksState) -> dict[str, Any]:
        """Render the user-facing reply via the dispatch table in replies.py."""
        final_outcome, mismatch_key = resolve_outcome(state)
        if mismatch_key is not None:
            log.warning(
                "tasks_build_reply_outcome_mismatch",
                populated=mismatch_key,
                state_outcome=state.get("outcome"),
                rendering_as=final_outcome,
            )
        log.info(
            "tasks_build_reply",
            outcome=final_outcome,
            has_created=bool(state.get("created_chore")),
            has_completed=bool(state.get("completed_chore")),
            has_deleted=bool(state.get("deleted_chore")),
            has_updated=bool(state.get("updated_chore")),
        )
        payload = build_payload(state, name_map)
        return {
            "reply_plain": render_reply_plain(payload),
            "reply_html": render_reply_html(payload),
        }

    # ── Edge predicates ──────────────────────────────────────────────────

    def route_after_intent(state: TasksState) -> str:
        action = state.get("action", "clarify")
        if action == "create":
            return "parse_chore_draft"
        if action in ("complete", "delete"):
            return "parse_target_reference"
        if action == "update":
            return "parse_chore_update"
        if action == "list":
            return "render_pending_summary"
        return "mark_clarification"

    def route_after_parse(state: TasksState) -> str:
        if state.get("outcome") == Outcome.INCOMPLETE.value:
            return "build_reply"
        return "check_for_duplicates"

    def route_after_dedup(state: TasksState) -> str:
        conf = (state.get("duplicate_check") or {}).get("confidence", "none")
        if conf == "high":
            return "mark_duplicate_high"
        if conf == "medium":
            return "mark_duplicate_medium"
        return "validate_chore"

    def route_after_validate(state: TasksState) -> str:
        if state.get("outcome") == Outcome.INCOMPLETE.value:
            return "build_reply"
        return "write_chore"

    def route_after_target_parse(state: TasksState) -> str:
        if state.get("outcome") == Outcome.TARGET_NOT_FOUND.value:
            return "build_reply"
        return "reason_about_target_match"

    def route_after_target_match(state: TasksState) -> str:
        action = state.get("action", "")
        conf = (state.get("target_match") or {}).get("confidence", "none")
        if conf == "medium":
            return "mark_target_needs_confirmation"
        if conf == "none":
            return "mark_target_not_found"
        if action == "complete":
            return "write_completion"
        if action == "delete":
            return "soft_delete_chore_action"
        if action == "update":
            return "apply_chore_update"
        return "mark_clarification"

    def route_after_update_parse(state: TasksState) -> str:
        if state.get("outcome") in (
            Outcome.INCOMPLETE.value,
            Outcome.TARGET_NOT_FOUND.value,
        ):
            return "build_reply"
        return "reason_about_target_match"

    # ── Graph assembly ───────────────────────────────────────────────────

    graph = StateGraph(TasksState)
    for node_name, fn in (
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
    ):
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
        "parse_chore_draft", route_after_parse,
        {"build_reply": "build_reply", "check_for_duplicates": "check_for_duplicates"},
    )
    graph.add_conditional_edges(
        "check_for_duplicates", route_after_dedup,
        {
            "mark_duplicate_high": "mark_duplicate_high",
            "mark_duplicate_medium": "mark_duplicate_medium",
            "validate_chore": "validate_chore",
        },
    )
    graph.add_conditional_edges(
        "validate_chore", route_after_validate,
        {"build_reply": "build_reply", "write_chore": "write_chore"},
    )

    # Complete/delete/update share the target-identification step
    graph.add_conditional_edges(
        "parse_target_reference", route_after_target_parse,
        {
            "build_reply": "build_reply",
            "reason_about_target_match": "reason_about_target_match",
        },
    )
    graph.add_conditional_edges(
        "reason_about_target_match", route_after_target_match,
        {
            "write_completion": "write_completion",
            "soft_delete_chore_action": "soft_delete_chore_action",
            "apply_chore_update": "apply_chore_update",
            "mark_target_needs_confirmation": "mark_target_needs_confirmation",
            "mark_target_not_found": "mark_target_not_found",
            "mark_clarification": "mark_clarification",
        },
    )
    graph.add_conditional_edges(
        "parse_chore_update", route_after_update_parse,
        {
            "build_reply": "build_reply",
            "reason_about_target_match": "reason_about_target_match",
        },
    )

    # Terminal edges
    for src in (
        "write_chore",
        "mark_duplicate_high",
        "mark_duplicate_medium",
        "write_completion",
        "soft_delete_chore_action",
        "apply_chore_update",
        "render_pending_summary",
        "mark_target_needs_confirmation",
        "mark_target_not_found",
        "mark_clarification",
    ):
        graph.add_edge(src, "build_reply")
    graph.add_edge("build_reply", END)

    return graph.compile()


# ── Module-level helpers (pure, stateless) ──────────────────────────────────


def _no_match(reason: str) -> dict[str, Any]:
    """Helper: empty duplicate-check result."""
    return {"confidence": "none", "existing_chore_id": None, "reasoning": reason}


def _pick_chore_id(candidate: str, taken: set[str]) -> str:
    """Pick a free chore id, suffixing -2, -3, ... on slug collisions."""
    if candidate not in taken:
        return candidate
    n = 2
    while f"{candidate}-{n}" in taken:
        n += 1
    return f"{candidate}-{n}"


def _best_shape_match(
    new_title: str,
    new_obj: str | None,
    new_qual: str | None,
    existing: list[dict],
) -> tuple[str, str, str] | None:
    """Return (confidence, chore_id, reason) for the strongest shape match.

    None if no existing chore matches at any confidence level. Stops early
    when a 'high' match is found.
    """
    rank = {"high": 2, "medium": 1, "none": 0}
    best: tuple[str, str, str] | None = None
    for c in existing:
        ex_obj = normalize_shape(c.get("object"))
        ex_qual = normalize_shape(c.get("qualifier"))
        verdict = shape_verdict(new_obj, new_qual, ex_obj, ex_qual)
        if verdict == "none":
            continue
        reason = shape_reason(new_title, c.get("title", c["id"]), new_qual, verdict)
        candidate = (verdict, c["id"], reason)
        if best is None or rank[verdict] > rank[best[0]]:
            best = candidate
        if verdict == "high":
            break
    return best


def _build_update_fields(draft: dict[str, Any]) -> dict[str, Any]:
    """Translate a ChoreUpdateDraft dict into ChoreStore.update_chore kwargs."""
    fields: dict[str, Any] = {}
    if draft.get("new_title"):
        fields["title"] = normalize_title(draft["new_title"])
    if draft.get("new_assignee"):
        fields["assignee"] = draft["new_assignee"]
    if draft.get("new_recurrence_type"):
        fields["recurrence_type"] = draft["new_recurrence_type"]
    if draft.get("new_recurrence"):
        fields["recurrence_rule"] = draft["new_recurrence"]
    if draft.get("new_shame_after_days") is not None:
        fields["shame_after_days"] = draft["new_shame_after_days"]
    return fields


# Re-export so external callers don't have to know module layout.
__all__ = ["TasksState", "build_tasks_graph", "format_recurrence_human"]
