"""Model routing — thin wrapper that selects LiteLLM model names.

Actual routing, fallback, and load balancing is handled by LiteLLM Proxy.
This module maps agent preferences to LiteLLM model aliases.
"""

from __future__ import annotations

from typing import Any

import structlog

from scaffold.core.config import AgentConfig
from scaffold.core.models import AgentRequest, LLMResponse
from scaffold.routing.clients import LiteLLMClient

log = structlog.get_logger()

# Keywords that suggest complex reasoning (route to cloud model)
_COMPLEX_KEYWORDS = {
    "analyze", "compare", "research", "investigate", "explain why",
    "plan", "strategy", "evaluate", "summarize article", "pros and cons",
    "deep dive", "comprehensive",
}

# Map preference + complexity to LiteLLM model aliases
_MODEL_MAP = {
    ("local", "simple"): "local-default",
    ("local", "complex"): "local-default",
    ("cloud", "simple"): "cloud-default",
    ("cloud", "complex"): "cloud-strong",
    ("auto", "simple"): "local-default",    # LiteLLM fallback handles failures
    ("auto", "complex"): "cloud-default",
}


class ModelRouter:
    """Selects a LiteLLM model alias based on agent config and request complexity.

    The actual routing, fallback, and retry logic lives in LiteLLM Proxy.
    This is a thin layer that picks the right model name.
    """

    def __init__(self, client: LiteLLMClient) -> None:
        self._client = client

    async def route(
        self,
        request: AgentRequest,
        agent_config: AgentConfig,
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Route to the appropriate LiteLLM model alias."""
        if messages is None:
            messages = [{"role": "user", "content": request.user_message}]

        preference = agent_config.model_preference
        complexity = self._classify_complexity(request, tools)
        model = _MODEL_MAP.get((preference, complexity), "local-default")

        log.info(
            "model_routing",
            preference=preference,
            complexity=complexity,
            model=model,
            request_id=request.request_id[:8],
        )

        return await self._client.complete(
            messages=messages,
            model=model,
            tools=tools,
            max_tokens=max_tokens,
            system=system,
        )

    def _classify_complexity(
        self, request: AgentRequest, tools: list[dict[str, Any]] | None
    ) -> str:
        """Heuristic complexity classifier. Returns 'simple' or 'complex'."""
        message = request.user_message.lower()

        if tools and len(tools) > 3:
            return "complex"
        if len(request.user_message) > 500:
            return "complex"
        if any(kw in message for kw in _COMPLEX_KEYWORDS):
            return "complex"

        return "simple"
