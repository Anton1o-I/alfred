"""High-level orchestrator interface wrapping the LangGraph graph."""

from __future__ import annotations

import structlog

from alfred.agents.registry import AgentRegistry
from alfred.orchestrator.graph import build_orchestrator_graph
from alfred.orchestrator.state import OrchestratorState
from scaffold.audit.logger import AuditLogger
from scaffold.budget.policies import BudgetPolicy
from scaffold.budget.tracker import BudgetTracker
from scaffold.core.constants import AgentStatus
from scaffold.core.models import AgentRequest, AgentResponse
from scaffold.notifications.service import NotificationService
from scaffold.routing.clients import LiteLLMClient

log = structlog.get_logger()


class Orchestrator:
    """Central dispatcher wrapping the LangGraph orchestrator graph."""

    def __init__(
        self,
        client: LiteLLMClient,
        registry: AgentRegistry,
        budget_policy: BudgetPolicy,
        budget_tracker: BudgetTracker,
        audit_logger: AuditLogger,
        notification_service: NotificationService,
    ) -> None:
        graph = build_orchestrator_graph(
            client=client,
            registry=registry,
            budget_policy=budget_policy,
            budget_tracker=budget_tracker,
            audit_logger=audit_logger,
            notification_service=notification_service,
        )
        self._graph = graph.compile()
        self._registry = registry

    async def handle(self, request: AgentRequest) -> AgentResponse:
        """Process an incoming request through the orchestrator graph."""
        initial_state: OrchestratorState = {
            "user_message": request.user_message,
            "source": request.source,
            "user_id": request.user_id or "",
            "request_id": request.request_id,
            "should_notify": request.source in ("scheduler", "imessage", "webhook"),
        }

        log.info(
            "orchestrator_handle",
            request_id=request.request_id[:8],
            source=request.source,
            message_preview=request.user_message[:80],
        )

        # Run the graph
        result = await self._graph.ainvoke(initial_state)

        status_map = {
            "success": AgentStatus.SUCCESS,
            "error": AgentStatus.ERROR,
            "budget_exceeded": AgentStatus.BUDGET_EXCEEDED,
            "timeout": AgentStatus.TIMEOUT,
        }

        return AgentResponse(
            request_id=request.request_id,
            agent_name=result.get("target_agent", "orchestrator"),
            status=status_map.get(result.get("status", "error"), AgentStatus.ERROR),
            message=result.get("agent_response", "No response generated."),
            metadata={
                "routing_reasoning": result.get("routing_reasoning", ""),
                "budget_reason": result.get("budget_reason", ""),
                "notification_sent": result.get("notification_sent", False),
            },
            data=result.get("agent_data", {}) or {},
        )
