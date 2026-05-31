"""Agent discovery and registration."""

from __future__ import annotations

import structlog

from alfred.agents.base import AgentBase
from scaffold.core.config import AgentConfig

log = structlog.get_logger()


class AgentRegistry:
    """Central catalog of all registered agents."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentBase] = {}

    def register(self, agent: AgentBase, config: AgentConfig | None = None) -> None:
        """Register an agent. Skips if config says disabled."""
        if config and not config.enabled:
            log.info("agent_disabled", agent=agent.name)
            return
        self._agents[agent.name] = agent
        log.info("agent_registered", agent=agent.name)

    def get(self, name: str) -> AgentBase | None:
        """Look up an agent by name."""
        return self._agents.get(name)

    def list_all(self) -> list[AgentBase]:
        """Return all registered agents."""
        return list(self._agents.values())

    def list_names(self) -> list[str]:
        """Return names of all registered agents."""
        return list(self._agents.keys())

    def get_agent_descriptions(self) -> list[dict[str, str]]:
        """Return name+description pairs for intent classification."""
        return [
            {"name": a.name, "description": a.description}
            for a in self._agents.values()
        ]
