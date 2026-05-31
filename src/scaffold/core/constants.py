"""Enums and constants for the Alfred system."""

from enum import StrEnum


class RequestSource(StrEnum):
    CLI = "cli"
    IMESSAGE = "imessage"
    SCHEDULER = "scheduler"
    REACTIVE = "reactive"
    EMAIL = "email"


class AgentStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    REFUSED = "refused"


class ModelPreference(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"
    AUTO = "auto"


class TopicPriority(StrEnum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class NotificationChannel(StrEnum):
    IMESSAGE = "imessage"
    EMAIL = "email"


class AuditEventType(StrEnum):
    TOOL_CALL = "tool_call"
    LLM_REQUEST = "llm_request"
    AGENT_DISPATCH = "agent_dispatch"
    BUDGET_CHECK = "budget_check"
    NOTIFICATION = "notification"
    ERROR = "error"


DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_RETRIES = 3
MAX_TOOL_LOOP_ITERATIONS = 10
