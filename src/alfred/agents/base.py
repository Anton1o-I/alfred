"""Agent base class using Pydantic AI for typed tool-calling agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog
from pydantic import BaseModel, Field

from scaffold.core.models import TokenUsage

log = structlog.get_logger()


class AgentResult(BaseModel):
    """Structured result from an agent execution."""

    message: str
    data: dict[str, Any] = {}
    usage: TokenUsage = Field(default_factory=TokenUsage)


@dataclass
class AgentContext:
    """Runtime context available to agent tools during execution."""

    user_id: str | None = None
    request_id: str = ""
    source: str = "cli"
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentBase(ABC):
    """Base class for all alfred agents.

    Concrete agents are typically thin wrappers around a `pydantic_ai.Agent`
    that adapt its result into AgentResult. The LangGraph orchestrator handles
    routing, budget gating, audit logging, and lifecycle.
    """

    name: str
    description: str
    model: str = "local-default"  # LiteLLM model alias

    @abstractmethod
    async def run(
        self, message: str, context: AgentContext
    ) -> AgentResult:
        """Execute the agent's primary task."""
