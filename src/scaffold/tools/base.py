"""Tool base class and protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Base class for all tools available to agents."""

    name: str
    description: str
    requires_confirmation: bool = False

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> Any:
        """Execute the tool with the given arguments."""

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """Return JSON Schema for the tool's parameters (for LLM tool_use)."""
