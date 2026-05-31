"""Tasks agent — thin wrapper around the LangGraph chores workflow."""

from __future__ import annotations

from typing import Any

import structlog
from opentelemetry import trace

from alfred.agents.base import AgentBase, AgentContext, AgentResult
from alfred.agents.tasks.store import ChoreStore
from alfred.agents.tasks.workflow import build_tasks_graph
from scaffold.core.models import TokenUsage
from scaffold.routing.clients import LiteLLMClient
from scaffold.storage.database import Database

_tracer = trace.get_tracer("alfred.tasks.agent")
log = structlog.get_logger()


class TasksAgent(AgentBase):
    """Household chore tracker — accepts create/complete/list/delete/update."""

    name = "tasks"
    description = "Track recurring household chores, send reminders, mark completions"
    model = "local-default"

    def __init__(
        self,
        db: Database,
        litellm_client: LiteLLMClient,
        notification_service: Any = None,  # noqa: ARG002 — registry signature compat
        timezone_name: str = "UTC",
        now_fn: Any = None,
        assignee_names: dict[str, str] | None = None,
    ) -> None:
        self._db = db
        self._litellm = litellm_client
        self._store = ChoreStore(db)
        self._graph = build_tasks_graph(
            store=self._store,
            timezone_name=timezone_name,
            litellm_client=litellm_client,
            model_name="local-default",
            now_fn=now_fn,
            assignee_names=assignee_names or {},
        )

    async def run(self, message: str, context: AgentContext) -> AgentResult:  # noqa: ARG002
        with _tracer.start_as_current_span("tasks.workflow") as span:
            span.set_attribute("openinference.span.kind", "AGENT")
            span.set_attribute("alfred.agent", "tasks")
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
        created_chore = final_state.get("created_chore")
        input_tokens = final_state.get("input_tokens", 0)
        output_tokens = final_state.get("output_tokens", 0)

        log.info(
            "tasks_workflow_done",
            outcome=outcome,
            chore_created=bool(created_chore),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return AgentResult(
            message=reply_plain,
            data={
                "model": "local-default",
                "chore": created_chore,
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

    @property
    def store(self) -> ChoreStore:
        return self._store
