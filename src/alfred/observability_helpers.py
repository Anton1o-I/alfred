"""Helpers for emitting consistent OTel attributes on agent spans.

Background: pydantic-ai's instrumentation uses OTel's GenAI semantic
conventions (`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`),
while Phoenix's UI primarily surfaces OpenInference conventions
(`llm.token_count.prompt`, `llm.token_count.completion`). The two
naming schemes don't overlap, so Phoenix shows only total tokens at
parent spans unless we set both names ourselves.

This module provides a context manager that wraps a unit of LLM work
(e.g. one agent.run call) and stamps the span with both naming
conventions when we know the token counts.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span

_tracer = trace.get_tracer("alfred")


@contextmanager
def llm_span(
    name: str,
    *,
    prompt_id: str | None = None,
    model: str | None = None,
    agent_name: str | None = None,
):
    """Open an LLM-flavored span. Set token counts via record_tokens() before exit."""
    attributes: dict[str, Any] = {"openinference.span.kind": "LLM"}
    if prompt_id:
        attributes["alfred.prompt.fqn"] = prompt_id
    if model:
        attributes["llm.model_name"] = model
        attributes["gen_ai.request.model"] = model
    if agent_name:
        attributes["alfred.agent.name"] = agent_name

    with _tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span


def record_tokens(
    span: Span,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Stamp BOTH OpenInference and OTel GenAI token-count attributes on a span.

    Phoenix's UI reads OpenInference; many other OTel viewers read GenAI.
    Setting both costs nothing and makes the span legible everywhere.
    """
    total = input_tokens + output_tokens
    # OpenInference (Phoenix native)
    span.set_attribute("llm.token_count.prompt", input_tokens)
    span.set_attribute("llm.token_count.completion", output_tokens)
    span.set_attribute("llm.token_count.total", total)
    # OTel GenAI semantic conventions
    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
    span.set_attribute("gen_ai.usage.total_tokens", total)
