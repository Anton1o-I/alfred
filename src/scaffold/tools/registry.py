"""Central tool catalog."""

from __future__ import annotations

from alfred.tools.base import Tool


class ToolRegistry:
    """Central catalog of all available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool by name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_all(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_schemas_for_agent(
        self, allowed_tools: list[str]
    ) -> list[dict]:
        """Return tool schemas only for tools in the allowlist."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.get_schema(),
            }
            for tool in self._tools.values()
            if tool.name in allowed_tools
        ]
