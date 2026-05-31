"""Exception hierarchy for the Alfred system."""


class HomeAgentError(Exception):
    """Base for all application exceptions."""


class BudgetExceededError(HomeAgentError):
    """Token budget cap reached."""

    def __init__(self, agent: str, limit_type: str, current: int, limit: int) -> None:
        self.agent = agent
        self.limit_type = limit_type
        self.current = current
        self.limit = limit
        super().__init__(
            f"Budget exceeded for {agent}: {limit_type} "
            f"({current:,}/{limit:,} tokens)"
        )


class AgentTimeoutError(HomeAgentError):
    """Agent execution exceeded its timeout."""

    def __init__(self, agent: str, timeout_seconds: int) -> None:
        self.agent = agent
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Agent '{agent}' timed out after {timeout_seconds}s")


class ModelRoutingError(HomeAgentError):
    """Failed to route to any available model."""


class ConfigurationError(HomeAgentError):
    """Invalid or missing configuration."""
