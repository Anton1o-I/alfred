"""Shared Pydantic models that flow through the entire system."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from scaffold.core.constants import AgentStatus, RequestSource


class TokenUsage(BaseModel):
    """Token accounting for a single LLM call."""

    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ToolCall(BaseModel):
    """Record of a single tool invocation."""

    tool_name: str
    arguments: dict[str, Any] = {}
    result: Any = None
    error: str | None = None
    duration_ms: int = 0


class AgentRequest(BaseModel):
    """Inbound request to the system."""

    request_id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: RequestSource
    user_message: str
    user_id: str | None = None
    conversation_id: str | None = None
    metadata: dict[str, Any] = {}


class AgentResponse(BaseModel):
    """Outbound response from an agent."""

    request_id: str
    agent_name: str
    status: AgentStatus
    message: str
    tool_calls: list[ToolCall] = []
    token_usage: list[TokenUsage] = []
    duration_ms: int = 0
    metadata: dict[str, Any] = {}
    data: dict[str, Any] = {}  # Agent-specific structured output (e.g., the event for calendar)


class LLMResponse(BaseModel):
    """Normalized response from any LLM provider."""

    content: str
    tool_calls: list[ToolCall] = []
    usage: TokenUsage
    stop_reason: str | None = None
