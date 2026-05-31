"""LangGraph node functions for the orchestrator."""

from __future__ import annotations

import time
from typing import Any

import structlog

from scaffold.agents.base import AgentContext
from scaffold.agents.registry import AgentRegistry
from scaffold.audit.logger import AuditLogger
from scaffold.budget.policies import BudgetPolicy
from scaffold.budget.tracker import BudgetTracker
from scaffold.core.constants import AuditEventType
from scaffold.notifications.service import NotificationService
from scaffold.orchestrator.state import OrchestratorState
from scaffold.routing.clients import LiteLLMClient

log = structlog.get_logger()


def create_intent_classifier(
    client: LiteLLMClient,
    registry: AgentRegistry,
    model: str = "local-default",
):
    """Create the intent classification node.

    `model` is the LiteLLM alias used for the routing call — kept local
    by default since classification is cheap and runs on every request.
    Wired from `Settings.orchestrator.intent_classifier_model` so an
    operator can override without code changes.
    """

    async def classify_intent(state: OrchestratorState) -> dict[str, Any]:
        """Classify the user's intent and select the target agent."""
        agents = registry.get_agent_descriptions()
        if not agents:
            return {
                "target_agent": "",
                "routing_reasoning": "No agents registered",
                "status": "error",
                "error_message": "No agents are currently available",
            }

        agent_list = "\n".join(
            f"- {a['name']}: {a['description']}" for a in agents
        )

        response = await client.complete(
            messages=[{"role": "user", "content": state["user_message"]}],
            model=model,
            system=(
                "You are a request router. Given the user's message, select the most "
                "appropriate agent to handle it. Respond with ONLY the agent name, "
                "nothing else.\n\n"
                f"Available agents:\n{agent_list}\n\n"
                "If no agent fits, respond with: none"
            ),
            max_tokens=50,
        )

        target = response.content.strip().lower()

        # Validate the agent exists
        if registry.get(target) is None:
            # Fallback: keyword matching
            target = _keyword_fallback(state["user_message"], agents)

        # Fail closed — never silently dispatch to an unmatched agent
        if not target or registry.get(target) is None:
            capabilities = ", ".join(a["name"] for a in agents)
            return {
                "target_agent": "",
                "routing_reasoning": "No agent matched the request",
                "status": "unroutable",
                "agent_response": (
                    "I can't help with that request — it doesn't match any of "
                    f"my available capabilities ({capabilities}). Try rephrasing "
                    "or ask about something one of those agents handles."
                ),
            }

        return {
            "target_agent": target,
            "routing_reasoning": f"Classified as '{target}' by intent router",
        }

    return classify_intent


def _keyword_fallback(message: str, agents: list[dict[str, str]]) -> str:
    """Simple keyword-based fallback for intent classification."""
    msg = message.lower()
    keyword_map = {
        "calendar": ["calendar", "schedule", "appointment", "meeting", "event", "busy"],
        "meals": ["meal", "dinner", "lunch", "recipe", "grocery", "cook", "food"],
        "research": ["research", "find", "look up", "compare", "recommend", "search"],
        "news": ["news", "headline", "article", "happening", "current events"],
        "podcast": ["podcast", "episode", "listen", "show", "audio"],
        "tasks": ["task", "todo", "chore", "remind", "maintenance", "fix"],
    }

    agent_names = {a["name"] for a in agents}
    for agent_name, keywords in keyword_map.items():
        if agent_name in agent_names and any(kw in msg for kw in keywords):
            return agent_name

    # No confident match — let the caller fail closed
    return ""


def create_budget_checker(policy: BudgetPolicy, audit: AuditLogger):
    """Create the budget check node."""

    async def check_budget(state: OrchestratorState) -> dict[str, Any]:
        """Check if the request is within budget."""
        target = state.get("target_agent", "")
        request_id = state.get("request_id", "")

        decision = await policy.check_request_allowed(target)

        await audit.log_event(
            event_type=AuditEventType.BUDGET_CHECK,
            request_id=request_id,
            agent_name=target,
            details={
                "allowed": decision.allowed,
                "reason": decision.reason,
                "agent_used_today": decision.agent_used_today,
                "global_used_today": decision.global_used_today,
            },
        )

        if not decision.allowed:
            return {
                "budget_allowed": False,
                "budget_reason": decision.reason,
                "status": "budget_exceeded",
                "agent_response": f"Budget limit reached: {decision.reason}",
            }

        return {
            "budget_allowed": True,
            "budget_reason": decision.reason,
        }

    return check_budget


def create_agent_executor(
    registry: AgentRegistry,
    tracker: BudgetTracker,
    audit: AuditLogger,
):
    """Create the agent execution node."""

    async def execute_agent(state: OrchestratorState) -> dict[str, Any]:
        """Execute the target agent."""
        target = state.get("target_agent", "")
        request_id = state.get("request_id", "")

        agent = registry.get(target)
        if agent is None:
            return {
                "status": "error",
                "error_message": f"Agent '{target}' not found",
                "agent_response": f"Sorry, I don't have an agent named '{target}' available.",
            }

        # Email bodies are PII and routinely contain quoted thread content
        # from third parties; don't persist any of that to the audit log.
        # CLI/webhook messages are user-authored and lower risk — keep a
        # short slice for traceability.
        source = state.get("source", "")
        if source == "email":
            audit_message = "[redacted: source=email]"
        else:
            audit_message = state["user_message"][:80]
        await audit.log_event(
            event_type=AuditEventType.AGENT_DISPATCH,
            request_id=request_id,
            agent_name=target,
            source=source,
            details={"user_message": audit_message},
        )

        start = time.monotonic()
        try:
            context = AgentContext(
                user_id=state.get("user_id"),
                request_id=request_id,
                source=state.get("source", "cli"),
            )
            result = await agent.run(state["user_message"], context)

            elapsed_ms = int((time.monotonic() - start) * 1000)

            await tracker.record_usage(target, request_id, result.usage)

            await audit.log_event(
                event_type=AuditEventType.AGENT_DISPATCH,
                request_id=request_id,
                agent_name=target,
                details={"status": "success", "duration_ms": elapsed_ms},
                duration_ms=elapsed_ms,
                token_usage=result.usage,
            )

            return {
                "status": "success",
                "agent_response": result.message,
                "agent_data": result.data,
            }

        except Exception as e:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            log.error("agent_execution_failed", agent=target, error=str(e))

            await audit.log_event(
                event_type=AuditEventType.ERROR,
                request_id=request_id,
                agent_name=target,
                details={"error": str(e), "duration_ms": elapsed_ms},
                duration_ms=elapsed_ms,
            )

            return {
                "status": "error",
                "error_message": str(e),
                "agent_response": f"Sorry, something went wrong: {e}",
            }

    return execute_agent


def create_notifier(notification_service: NotificationService):
    """Create the notification delivery node."""

    async def send_notification(state: OrchestratorState) -> dict[str, Any]:
        """Send the agent response as a notification if needed."""
        should_notify = state.get("should_notify", False)
        if not should_notify:
            return {"notification_sent": False}

        source = state.get("source", "")
        response = state.get("agent_response", "")
        request_id = state.get("request_id", "")
        user_id = state.get("user_id")

        if not response:
            return {"notification_sent": False}

        # Scheduled tasks notify all family members
        if source == "scheduler":
            await notification_service.send_to_family(
                body=response, request_id=request_id
            )
        elif user_id:
            await notification_service.send_to_user(
                user_id=user_id, body=response, request_id=request_id
            )

        return {"notification_sent": True}

    return send_notification
