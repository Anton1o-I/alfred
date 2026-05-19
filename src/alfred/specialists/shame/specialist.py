"""LLM specialist that produces one-line roasts for overdue chores.

Wired into the daily briefing: after chore data is fetched and bucketed,
the briefing calls `generate_roasts()` with one `ChoreRoastInput` per
shame-tier (tier ≥ 1) overdue chore. The returned `{chore_id: roast}`
dict is threaded through to the chore renderers, which substitute the
roast for the static `fallback_label`.

Design constraints:
- Local model only (qwen3:14b via `local-default`). No cloud escalation.
- Higher temperature (0.85) for variety; same chore-set should not yield
  the same lines two days in a row.
- One batched call per briefing — never one call per chore.
- All failure modes (network error, malformed JSON, missing ids, empty
  input) return `{}` so the renderer falls back to the static label.
- ≤ 100 char trim enforced at the boundary as defense against
  prompt-disobedient output.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from opentelemetry import trace
from pydantic import BaseModel, Field

from alfred.specialists.shame.prompts import ROAST_SYSTEM, build_roast_prompt

log = structlog.get_logger()
_tracer = trace.get_tracer("alfred.notifications.shame_roast")


# Hard cap applied AFTER the LLM responds — protects rendering from
# prompt-disobedient output. The prompt asks for ≤80; we trim at 100 to
# tolerate small overruns while still bounding row width.
_ROAST_MAX_CHARS: int = 100

# Temperature is deliberately high. Variety across days matters more than
# precision — the data the LLM sees (titles, days) is already exact.
_ROAST_TEMPERATURE: float = 0.95

# Local-only by design. Do not change this to a cloud model without
# explicit approval — see CLAUDE.md "Model Changes Need Approval".
_ROAST_MODEL: str = "local-default"


class ChoreRoastInput(BaseModel):
    """Per-chore input fed into the roast prompt."""

    chore_id: str
    title: str
    assignee: str
    tier: int
    days: int


class ChoreRoastLine(BaseModel):
    chore_id: str
    roast: str


class ChoreRoastOutput(BaseModel):
    """Structured response shape the model is asked to return."""

    lines: list[ChoreRoastLine] = Field(default_factory=list)


def _trim(roast: str) -> str:
    """Strip + truncate a single roast line."""
    cleaned = roast.strip()
    if len(cleaned) > _ROAST_MAX_CHARS:
        # Truncate to max-1 and append an ellipsis for visual hint.
        cleaned = cleaned[: _ROAST_MAX_CHARS - 1].rstrip() + "…"
    return cleaned


def _parse_response(content: str, valid_ids: set[str]) -> dict[str, str]:
    """Parse the model's JSON content into a `{chore_id: roast}` dict.

    Tolerates a leading/trailing code fence (some local models add one
    despite instructions). Drops any line whose chore_id wasn't in the
    request, and drops any line with an empty roast.
    """
    body = content.strip()
    if body.startswith("```"):
        # Strip first fence line and trailing fence.
        lines = body.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        body = "\n".join(lines)
    try:
        raw = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return {}
    try:
        parsed = ChoreRoastOutput.model_validate(raw)
    except Exception:  # noqa: BLE001 — pydantic ValidationError + anything malformed
        return {}
    out: dict[str, str] = {}
    for line in parsed.lines:
        if line.chore_id not in valid_ids:
            continue
        roast = _trim(line.roast)
        if not roast:
            continue
        out[line.chore_id] = roast
    return out


async def generate_roasts(
    items: list[ChoreRoastInput],
    client: Any,
    model_name: str = _ROAST_MODEL,
) -> dict[str, str]:
    """Call the local LLM to produce one roast per chore.

    Returns `{}` on any failure (empty input, network error, malformed
    JSON, validation error). Callers MUST fall back to the static
    `fallback_label` per chore when a chore_id is missing from the dict.

    Args:
      items: one input per shame-tier overdue chore (tier ≥ 1).
      client: the `LiteLLMClient` instance from `app.litellm_client`.
      model_name: model alias to call through the proxy. Defaults to the
        local model; do not change without approval.
    """
    if not items:
        return {}

    valid_ids = {it.chore_id for it in items}
    prompt = build_roast_prompt(items)

    with _tracer.start_as_current_span("notifications.shame_roast_specialist") as span:
        span.set_attribute("alfred.shame_roast.input_count", len(items))
        span.set_attribute("alfred.shame_roast.model", model_name)
        try:
            # Bypass `LiteLLMClient.complete` so we can pass temperature
            # without expanding its surface. The underlying AsyncOpenAI
            # client is the same one used elsewhere.
            resp = await client._client.chat.completions.create(  # noqa: SLF001
                model=model_name,
                messages=[
                    {"role": "system", "content": ROAST_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=_ROAST_TEMPERATURE,
                max_tokens=1024,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "shame_roast_specialist_call_failed",
                error=str(e),
                input_count=len(items),
            )
            span.set_attribute("alfred.shame_roast.output_count", 0)
            span.set_attribute("alfred.shame_roast.error", str(e))
            return {}

        try:
            content = resp.choices[0].message.content or ""
        except (AttributeError, IndexError):
            content = ""

        out = _parse_response(content, valid_ids)
        span.set_attribute("alfred.shame_roast.output_count", len(out))
        log.info(
            "shame_roast_specialist_complete",
            input_count=len(items),
            output_count=len(out),
        )
        return out
