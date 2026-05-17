"""LLM client — talks to LiteLLM Proxy via OpenAI-compatible API."""

from __future__ import annotations

import os

import httpx
import structlog
from openai import AsyncOpenAI

from alfred.core.models import LLMResponse, TokenUsage, ToolCall

log = structlog.get_logger()


class LiteLLMClient:
    """Unified LLM client that talks to the LiteLLM Proxy.

    All model routing, fallback, and cost tracking is handled by LiteLLM.
    This client just speaks the OpenAI-compatible protocol.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url or os.environ.get(
            "LITELLM_BASE_URL", "http://localhost:4000"
        )
        self._api_key = api_key or os.environ.get(
            "LITELLM_MASTER_KEY", "sk-alfred-dev"
        )
        self._client = AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=self._api_key,
        )

    async def complete(
        self,
        messages: list[dict],
        model: str = "local-default",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        """Send a chat completion through LiteLLM Proxy."""
        full_messages = list(messages)
        if system:
            full_messages = [{"role": "system", "content": system}, *full_messages]

        kwargs: dict = {
            "model": model,
            "messages": full_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        raw = await self._client.chat.completions.with_raw_response.create(**kwargs)
        response = raw.parse()
        cost_header = raw.headers.get("x-litellm-response-cost")
        try:
            cost_usd = float(cost_header) if cost_header is not None else 0.0
        except ValueError:
            cost_usd = 0.0

        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                import json

                tool_calls.append(
                    ToolCall(
                        tool_name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments,
                    )
                )

        # Build token usage — LiteLLM normalizes this across providers
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=TokenUsage(
                provider=model.split("/")[0] if "/" in model else "litellm",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost_usd,
            ),
            stop_reason=choice.finish_reason,
        )

    async def health_check(self) -> bool:
        """Check if LiteLLM Proxy is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/health/liveliness")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def _format_tools(self, tools: list[dict]) -> list[dict]:
        """Convert our tool schema format to OpenAI function calling format."""
        formatted = []
        for t in tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", t.get("input_schema", {})),
                },
            })
        return formatted

    async def close(self) -> None:
        await self._client.close()
