"""LangGraph orchestrator graph definition."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from alfred.agents.registry import AgentRegistry
from alfred.notifications.service import NotificationService
from alfred.orchestrator.nodes import (
    create_agent_executor,
    create_budget_checker,
    create_intent_classifier,
    create_notifier,
)
from alfred.orchestrator.state import OrchestratorState
from scaffold.audit.logger import AuditLogger
from scaffold.budget.policies import BudgetPolicy
from scaffold.budget.tracker import BudgetTracker
from scaffold.routing.clients import LiteLLMClient


def should_check_budget(state: OrchestratorState) -> str:
    """Conditional edge: skip budget + execution if the router refused to route."""
    if state.get("status") == "unroutable":
        return "respond"
    return "check_budget"


def should_execute(state: OrchestratorState) -> str:
    """Conditional edge: proceed to execution only if budget allows."""
    if state.get("budget_allowed", False):
        return "execute"
    return "respond"


def should_notify(state: OrchestratorState) -> str:
    """Conditional edge: send notification if source is scheduler or imessage."""
    source = state.get("source", "")
    if source in ("scheduler", "imessage", "webhook"):
        return "notify"
    return "done"


def build_orchestrator_graph(
    client: LiteLLMClient,
    registry: AgentRegistry,
    budget_policy: BudgetPolicy,
    budget_tracker: BudgetTracker,
    audit_logger: AuditLogger,
    notification_service: NotificationService,
) -> StateGraph:
    """Build the LangGraph orchestrator.

    Graph flow:
        classify_intent -> check_budget -> [budget ok?]
            -> yes: execute_agent -> [notify?] -> notify / done
            -> no: respond with budget error
    """
    # Create node functions with injected dependencies
    classify = create_intent_classifier(client, registry)
    check_budget = create_budget_checker(budget_policy, audit_logger)
    execute = create_agent_executor(registry, budget_tracker, audit_logger)
    notify = create_notifier(notification_service)

    # Build graph
    graph = StateGraph(OrchestratorState)

    # Add nodes
    graph.add_node("classify_intent", classify)
    graph.add_node("check_budget", check_budget)
    graph.add_node("execute_agent", execute)
    graph.add_node("notify", notify)

    # Define edges
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        should_check_budget,
        {
            "check_budget": "check_budget",
            "respond": END,
        },
    )

    # Conditional: budget check passes?
    graph.add_conditional_edges(
        "check_budget",
        should_execute,
        {
            "execute": "execute_agent",
            "respond": END,
        },
    )

    # Conditional: should we send a notification?
    graph.add_conditional_edges(
        "execute_agent",
        should_notify,
        {
            "notify": "notify",
            "done": END,
        },
    )

    graph.add_edge("notify", END)

    return graph
