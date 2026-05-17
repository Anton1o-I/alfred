"""LangGraph state definition for the orchestrator."""

from __future__ import annotations

from typing import Any, TypedDict


class OrchestratorState(TypedDict, total=False):
    """State that flows through the LangGraph orchestrator.

    LangGraph passes this state between nodes. Each node reads what it needs
    and writes its outputs back.
    """

    # Input
    user_message: str
    source: str  # "cli", "imessage", "scheduler", "webhook"
    user_id: str
    request_id: str

    # Routing
    target_agent: str  # which agent to dispatch to
    routing_reasoning: str  # why this agent was chosen

    # Execution
    agent_response: str  # final text response from the agent
    agent_data: dict[str, Any]  # structured data from the agent
    status: str  # "success", "error", "budget_exceeded", "timeout"
    error_message: str

    # Budget
    budget_allowed: bool
    budget_reason: str
    tokens_used: int

    # Notification
    should_notify: bool
    notification_sent: bool
