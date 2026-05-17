"""Arize Phoenix OpenTelemetry instrumentation setup."""

from __future__ import annotations

import os

import structlog

log = structlog.get_logger()


def init_observability() -> None:
    """Initialize OpenTelemetry tracing with Arize Phoenix.

    Call this once at application startup, before any LLM calls.
    Instruments LangChain/LangGraph and LiteLLM automatically.
    """
    endpoint = os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317"
    )

    try:
        from phoenix.otel import register

        tracer_provider = register(
            project_name="alfred",
            endpoint=endpoint,
        )

        # Instrument LangGraph (via LangChain instrumentor — LangGraph builds on it)
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

        # Instrument the OpenAI SDK — every LLM call (local or cloud) goes
        # through LiteLLMClient → openai.AsyncOpenAI, so this captures all
        # of them with token counts, model name, and prompt/response payloads.
        from openinference.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

        # Pydantic AI ships with built-in OTel support; turning it on adds
        # per-agent and per-tool-call spans on top of the OpenAI spans.
        try:
            from pydantic_ai import Agent

            Agent.instrument_all()
        except (ImportError, AttributeError):
            pass

        log.info(
            "observability_initialized",
            endpoint=endpoint,
            project="alfred",
        )
    except ImportError as e:
        log.warning("observability_unavailable", error=str(e))
    except Exception as e:
        log.warning("observability_init_failed", error=str(e))
