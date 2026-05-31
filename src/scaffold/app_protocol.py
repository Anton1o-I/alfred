"""AppProtocol — the surface scaffold modules can assume about the host app.

Scaffold code (registry, HTTP server, scheduler) needs to talk about
"the app" without importing any concrete agent system. This Protocol
declares the minimum attributes it depends on — agent systems supply
their own App class that conforms.

If a scaffold module needs more from the app, add it here and document
why; do not import alfred.app from scaffold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from scaffold.agents.registry import AgentRegistry
    from scaffold.core.config import Settings
    from scaffold.scheduler.registry import RoutineRegistry


class AppProtocol(Protocol):
    """Minimum app surface scaffold dispatchers and triggers use."""

    settings: "Settings"
    agent_registry: "AgentRegistry"
    routine_registry: "RoutineRegistry"
