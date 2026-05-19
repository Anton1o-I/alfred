"""LLM specialist that produces a one-line celebration for a completed chore.

Wired into the tasks workflow: when the user emails "done — walked the
dog" and the workflow lands `Outcome.COMPLETED`, the `generate_completion_cheer`
node calls `generate_cheer()` with a single `CompletionCheerInput`. The
returned string replaces the static "Chore completed" header in the
reply email.

Design constraints (mirror the shame specialist):
- Local model only (qwen3:14b via `local-default`). No cloud escalation.
- Higher temperature (0.9) for variety.
- One call per completion — the input is always a single chore, never a
  batch (a completion email is one chore).
- All failure modes (network error, malformed JSON, missing key, empty
  content) return `""` so the renderer falls back to the static banner.
- ≤ 100 char trim enforced at the boundary as defense against
  prompt-disobedient output.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from opentelemetry import trace
from pydantic import BaseModel

from alfred.specialists.celebrate.prompts import CHEER_SYSTEM, build_cheer_prompt

log = structlog.get_logger()
_tracer = trace.get_tracer("alfred.specialists.completion_cheer")


# Hard cap applied AFTER the LLM responds — protects rendering from
# prompt-disobedient output. The prompt asks for ≤80; we trim at 100 to
# tolerate small overruns while still bounding banner width.
_CHEER_MAX_CHARS: int = 100

# Temperature is deliberately high. Variety across completions matters
# more than precision — the data the LLM sees (title) is already exact.
_CHEER_TEMPERATURE: float = 0.9

# Local-only by design. Do not change this to a cloud model without
# explicit approval — see CLAUDE.md "Model Changes Need Approval".
_CHEER_MODEL: str = "local-default"


class CompletionCheerInput(BaseModel):
    """Per-completion input fed into the cheer prompt."""

    chore_id: str
    title: str
    assignee: str


class CompletionCheerLine(BaseModel):
    """Structured response shape the model is asked to return."""

    cheer: str


def _trim(cheer: str) -> str:
    """Strip + truncate a single cheer line."""
    cleaned = cheer.strip()
    if len(cleaned) > _CHEER_MAX_CHARS:
        cleaned = cleaned[: _CHEER_MAX_CHARS - 1].rstrip() + "…"
    return cleaned


def _parse_response(content: str) -> str:
    """Parse the model's JSON content into a single cheer string.

    Tolerates a leading/trailing code fence (some local models add one
    despite instructions). Returns `""` on any malformed input or empty
    cheer field.
    """
    body = content.strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        body = "\n".join(lines)
    try:
        raw = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return ""
    try:
        parsed = CompletionCheerLine.model_validate(raw)
    except Exception:  # noqa: BLE001 — pydantic ValidationError + anything malformed
        return ""
    return _trim(parsed.cheer)


async def generate_cheer(
    item: CompletionCheerInput,
    client: Any,
    model_name: str = _CHEER_MODEL,
) -> str:
    """Call the local LLM to produce one celebration line for a completed chore.

    Returns `""` on any failure (network error, malformed JSON, missing
    key, empty content). Callers MUST fall back to the static
    "Chore completed" header when the result is empty.

    Args:
      item: the single completed chore (title, assignee, id).
      client: the `LiteLLMClient` instance from `app.litellm_client`.
      model_name: model alias to call through the proxy. Defaults to the
        local model; do not change without approval.
    """
    prompt = build_cheer_prompt(item)

    with _tracer.start_as_current_span("specialists.completion_cheer") as span:
        span.set_attribute("alfred.completion_cheer.input_title", item.title)
        span.set_attribute("alfred.completion_cheer.model", model_name)
        try:
            # Bypass `LiteLLMClient.complete` so we can pass temperature
            # without expanding its surface. The underlying AsyncOpenAI
            # client is the same one used elsewhere.
            resp = await client._client.chat.completions.create(  # noqa: SLF001
                model=model_name,
                messages=[
                    {"role": "system", "content": CHEER_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=_CHEER_TEMPERATURE,
                max_tokens=256,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "specialists_completion_cheer_call_failed",
                error=str(e),
                input_title=item.title,
            )
            span.set_attribute("alfred.completion_cheer.output_length", 0)
            span.set_attribute("alfred.completion_cheer.error", str(e))
            return ""

        try:
            content = resp.choices[0].message.content or ""
        except (AttributeError, IndexError):
            content = ""

        out = _parse_response(content)
        span.set_attribute("alfred.completion_cheer.output_length", len(out))
        log.info(
            "specialists_completion_cheer_done",
            input_title=item.title,
            output_length=len(out),
        )
        return out
