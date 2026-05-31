"""Typed-output specialist agents for the tasks workflow.

Encapsulates the repeated pattern that used to live inline in every LLM
node:

  1. Build a local + cloud Pydantic AI agent pair for a given output type.
  2. Pick which pool to run against based on `complexity` in state.
  3. Run the prompt, catch errors, accumulate input/output token counts
     into state.
  4. Surface the parsed output plus a state-update dict that the node can
     merge with its own keys.

`Specialist` exposes a single `invoke(prompt, state)` method returning
`(output | None, updates)`. When the call fails, `output` is None and
`updates` only contains the token accumulators (zeroed) — callers decide
how to handle the failure with their own outcome.

`SpecialistRegistry` is a tiny holder that builds all six specialists in
one place so `build_tasks_graph` doesn't have to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from alfred.agents.tasks.prompts import SPECIALIST_SYSTEM_PROMPT
from alfred.agents.tasks.schemas import (
    ChoreDraft,
    ChoreUpdateDraft,
    DuplicateCheckDecision,
    IntentClassification,
    TargetMatch,
    TargetReference,
)

if TYPE_CHECKING:
    from scaffold.routing.clients import LiteLLMClient

log = structlog.get_logger()

# State key names. Single source of truth so the Specialist and the
# TasksState TypedDict can't drift.
_INPUT_TOKENS_KEY = "input_tokens"
_OUTPUT_TOKENS_KEY = "output_tokens"

# Sentinel model alias used for cloud routing. The actual cloud model is
# configured in `config/litellm.yaml`; we just point at the alias here.
_CLOUD_MODEL_ALIAS = "cloud-default"


def _build_agent[T: BaseModel](
    litellm_client: LiteLLMClient,
    model_name: str,
    output_type: type[T],
) -> Agent:
    """Construct a Pydantic AI typed-output agent backed by LiteLLM."""
    model = OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(
            base_url=f"{litellm_client.base_url}/v1",
            api_key=litellm_client._api_key,  # noqa: SLF001 — no public accessor yet
        ),
    )
    return Agent(
        model=model,
        output_type=output_type,
        system_prompt=SPECIALIST_SYSTEM_PROMPT,
    )


def _token_updates(
    state: dict[str, Any],
    input_tokens: int,
    output_tokens: int,
) -> dict[str, int]:
    """Return the state-update dict for token accumulators."""
    return {
        _INPUT_TOKENS_KEY: state.get(_INPUT_TOKENS_KEY, 0) + (input_tokens or 0),
        _OUTPUT_TOKENS_KEY: state.get(_OUTPUT_TOKENS_KEY, 0) + (output_tokens or 0),
    }


@dataclass
class SpecialistInvocation[T: BaseModel]:
    """Result of a single Specialist.invoke call.

    `output` is None when the LLM call raised. `updates` always contains
    the (zero or non-zero) input_tokens / output_tokens accumulator
    updates — merge it into the node's return dict.
    """

    output: T | None
    updates: dict[str, Any]
    model_label: str


class Specialist[T: BaseModel]:
    """Local + cloud agent pair for one output type, plus the invoke path.

    Selection between local and cloud is driven by `state["complexity"]`
    (set by the intent classifier). The cloud agent is always built —
    classify_intent decides whether to use it.
    """

    __slots__ = ("_local", "_cloud", "_local_model", "_log_event")

    def __init__(
        self,
        litellm_client: LiteLLMClient,
        output_type: type[T],
        local_model_name: str,
        log_event: str,
    ) -> None:
        self._local = _build_agent(litellm_client, local_model_name, output_type)
        self._cloud = _build_agent(litellm_client, _CLOUD_MODEL_ALIAS, output_type)
        self._local_model = local_model_name
        self._log_event = log_event

    def _pick(self, state: dict[str, Any]) -> tuple[Agent, str]:
        if state.get("complexity") == "complex":
            return self._cloud, _CLOUD_MODEL_ALIAS
        return self._local, self._local_model

    async def invoke(
        self,
        prompt: str,
        state: dict[str, Any],
    ) -> SpecialistInvocation[T]:
        """Run the prompt and return (output, state_updates).

        On exception, `output` is None and `updates` only contains zero
        token deltas (so accumulators don't reset). The caller decides
        what outcome to set.
        """
        agent, model_label = self._pick(state)
        try:
            result = await agent.run(prompt)
        except Exception as e:  # noqa: BLE001 — broad on purpose, agents raise many types
            log.error(self._log_event + "_failed", error=str(e), model=model_label)
            return SpecialistInvocation(
                output=None,
                updates=_token_updates(state, 0, 0),
                model_label=model_label,
            )
        usage = result.usage()
        return SpecialistInvocation(
            output=result.output,
            updates=_token_updates(
                state,
                usage.input_tokens or 0,
                usage.output_tokens or 0,
            ),
            model_label=model_label,
        )


@dataclass
class SpecialistRegistry:
    """Container for every specialist the workflow uses.

    Built once at graph-compile time, then closed over by the node
    functions. Keeping them in one struct means the nodes touch
    `specialists.parse.invoke(...)` rather than juggling six local/cloud
    pairs in their closure.
    """

    intent: Specialist[IntentClassification]
    parse: Specialist[ChoreDraft]
    dedup: Specialist[DuplicateCheckDecision]
    target_parse: Specialist[TargetReference]
    target_match: Specialist[TargetMatch]
    update_parse: Specialist[ChoreUpdateDraft]

    @classmethod
    def build(
        cls,
        litellm_client: LiteLLMClient,
        local_model_name: str,
    ) -> SpecialistRegistry:
        return cls(
            intent=Specialist(
                litellm_client, IntentClassification, local_model_name,
                "tasks_intent",
            ),
            parse=Specialist(
                litellm_client, ChoreDraft, local_model_name,
                "tasks_chore_parse",
            ),
            dedup=Specialist(
                litellm_client, DuplicateCheckDecision, local_model_name,
                "tasks_dedup",
            ),
            target_parse=Specialist(
                litellm_client, TargetReference, local_model_name,
                "tasks_target_parse",
            ),
            target_match=Specialist(
                litellm_client, TargetMatch, local_model_name,
                "tasks_target_match",
            ),
            update_parse=Specialist(
                litellm_client, ChoreUpdateDraft, local_model_name,
                "tasks_update_parse",
            ),
        )
