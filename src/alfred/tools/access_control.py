"""Per-agent tool access control enforcement."""

from __future__ import annotations

from alfred.core.config import AgentConfig
from alfred.core.exceptions import ToolAccessDeniedError


class ToolAccessControl:
    """Enforces per-agent tool allowlists at dispatch time."""

    def __init__(self, agent_configs: dict[str, AgentConfig]) -> None:
        self._allowlists: dict[str, set[str]] = {
            name: set(config.allowed_tools)
            for name, config in agent_configs.items()
        }

    def is_allowed(self, agent_name: str, tool_name: str) -> bool:
        """Check if an agent is allowed to use a tool."""
        allowed = self._allowlists.get(agent_name)
        if allowed is None:
            return False
        return tool_name in allowed

    def get_allowed_tools(self, agent_name: str) -> list[str]:
        """Get the list of tools an agent is allowed to use."""
        return list(self._allowlists.get(agent_name, set()))

    def enforce(self, agent_name: str, tool_name: str) -> None:
        """Raise ToolAccessDeniedError if not allowed."""
        if not self.is_allowed(agent_name, tool_name):
            raise ToolAccessDeniedError(agent_name, tool_name)
