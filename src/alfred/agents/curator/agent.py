"""Curator agent — fetches sources, triages with local model, synthesizes top items."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
import yaml
from opentelemetry import trace
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from scaffold.agents.base import AgentBase, AgentContext, AgentResult
from alfred.agents.curator.render import (
    render_comparison,
    render_digest,
    render_digest_html,
)
from alfred.agents.curator.sources import (
    CandidateItem,
    annotate_authorship,
    fetch_all,
)
from alfred.observability_helpers import llm_span, record_tokens
from alfred.prompts import load_prompt
from scaffold.core.models import TokenUsage
from scaffold.routing.clients import LiteLLMClient

log = structlog.get_logger()
_tracer = trace.get_tracer("alfred.curator.agent")


# ── Configuration loading ────────────────────────────────────


@dataclass
class CuratorConfig:
    """Curator-specific config loaded from YAML files."""

    topics: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    user_context: dict[str, Any]
    top_labs: list[str]
    top_authors: list[str]
    output_dir: Path
    max_items_per_digest: int = 12
    triage_score_threshold: float = 3.0  # 1-5 scale; only items >= this go to synthesis
    # Per-source cap on items in a single digest — forces variety so one
    # busy publisher (e.g. OpenAI Blog during a release week) cannot
    # swallow the whole reading list.
    max_items_per_source: int = 3
    # Minimum content length (chars) for an item to even enter triage.
    # Drops quote-posts, link-only shares, and other low-effort fluff.
    min_content_length: int = 200
    # Concurrency caps prevent slamming the local GPU (or hitting cloud rate limits)
    triage_concurrency: int = 4
    synthesis_concurrency: int = 4

    @property
    def topics_by_id(self) -> dict[str, dict[str, Any]]:
        return {t["id"]: t for t in self.topics}

    @classmethod
    def load(cls, config_dir: Path = Path("config")) -> CuratorConfig:
        topics_data = _load_yaml(config_dir / "research_topics.yaml")
        sources_data = _load_yaml(config_dir / "research_sources.yaml")
        user_ctx = _load_yaml(config_dir / "user_context.yaml")
        authors_data = _load_yaml(config_dir / "research_authors.yaml")

        return cls(
            topics=topics_data.get("topics", []),
            sources=sources_data.get("sources", []),
            user_context=user_ctx,
            top_labs=authors_data.get("top_labs", []),
            top_authors=authors_data.get("top_authors", []),
            output_dir=Path("data/digests"),
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        log.warning("curator_config_missing", path=str(path))
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ── Structured outputs from the model ────────────────────────


class TriageDecision(BaseModel):
    """Output of the triage pass — per-item relevance scoring."""

    topic_id: str = Field(description="The topic ID this item best fits, or 'none'")
    relevance: float = Field(ge=0, le=5, description="Relevance score 0-5 against user context")
    rationale: str = Field(description="One sentence justification")


class SynthesisOutput(BaseModel):
    """Output of the synthesis pass — the per-item curated content."""

    summary: str = Field(description="3-5 sentence summary of the item")
    why_it_matters: str = Field(
        description="1-2 sentences on field/industry significance — not personal"
    )


# ── Pydantic AI agent builders ───────────────────────────────


def _build_provider(litellm_client: LiteLLMClient, model_name: str) -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(
            base_url=f"{litellm_client.base_url}/v1",
            api_key=litellm_client._api_key,
        ),
    )


def _build_triage_agent(
    litellm_client: LiteLLMClient,
    model_name: str,
    prompt_version: str | None = None,
) -> tuple[Agent[None, TriageDecision], str]:
    """Build the triage agent. Returns (agent, prompt_fqn) for audit logging."""
    prompt = load_prompt("curator.triage", version=prompt_version)
    model = _build_provider(litellm_client, model_name)
    agent = Agent(
        model=model,
        output_type=TriageDecision,
        system_prompt=prompt.body,
    )
    return agent, prompt.fqn


def _build_synthesis_agent(
    litellm_client: LiteLLMClient,
    model_name: str,
    prompt_version: str | None = None,
) -> tuple[Agent[None, SynthesisOutput], str]:
    """Build the synthesis agent. Returns (agent, prompt_fqn) for audit logging."""
    prompt = load_prompt("curator.synthesis", version=prompt_version)
    model = _build_provider(litellm_client, model_name)
    agent = Agent(
        model=model,
        output_type=SynthesisOutput,
        system_prompt=prompt.body,
    )
    return agent, prompt.fqn


# ── Triage and synthesis passes ──────────────────────────────


async def _triage_item(
    agent: Agent[None, TriageDecision],
    item: CandidateItem,
    user_context: dict[str, Any],
    topics: list[dict[str, Any]],
    prompt_fqn: str = "curator.triage",
    model_name: str = "local-default",
) -> tuple[CandidateItem, TriageDecision | None]:
    topic_menu = "\n".join(
        f"  - {t['id']}: {t['name']} — {t['description'].strip()}" for t in topics
    )
    hints = ", ".join(item.topic_hints) if item.topic_hints else "(none)"
    authors_str = ", ".join(item.authors) if item.authors else "(none listed)"
    matched_authors = (
        ", ".join(item.matched_authors) if item.matched_authors else "(none)"
    )
    matched_labs = (
        ", ".join(item.matched_labs) if item.matched_labs else "(none)"
    )

    prompt = (
        f"USER PROFILE:\n{json.dumps(user_context, indent=2)}\n\n"
        f"AVAILABLE TOPICS:\n{topic_menu}\n\n"
        f"CONTENT TO TRIAGE:\n"
        f"Title: {item.title}\n"
        f"Source: {item.source_name}\n"
        f"Source's topic hints: {hints}\n"
        f"Authors: {authors_str}\n"
        f"AUTHORSHIP SIGNAL — matched top-tier authors: {matched_authors}\n"
        f"AUTHORSHIP SIGNAL — matched top-tier labs in abstract: {matched_labs}\n"
        f"Summary/abstract: {item.summary}\n\n"
        f"Classify into a topic_id from the list above (or 'none' if no fit), "
        f"and score relevance 0-5 against the user profile. For arXiv "
        f"submissions in particular, treat presence of matched top-tier "
        f"authors or labs as a positive signal of substantive work; "
        f"absence of those signals is not disqualifying but should raise "
        f"the bar for high scores."
    )

    try:
        with llm_span(
            "curator.triage",
            prompt_id=prompt_fqn,
            model=model_name,
            agent_name="curator",
        ) as span:
            result = await agent.run(prompt)
            usage = result.usage()
            record_tokens(
                span,
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
            )
        return item, result.output
    except Exception as e:
        log.warning("triage_failed", title=item.title[:60], error=str(e))
        return item, None


async def _synthesize_item(
    agent: Agent[None, SynthesisOutput],
    item: CandidateItem,
    user_context: dict[str, Any],  # accepted but intentionally not used in v1.2+
    prompt_fqn: str = "curator.synthesis",
    model_name: str = "local-default",
) -> tuple[CandidateItem, SynthesisOutput | None]:
    """Synthesize one item. Per synthesis prompt v1.2, no user profile is
    passed — 'why it matters' is framed at the field/industry level, not
    personalized to the reader."""
    prompt = (
        f"CONTENT TO SUMMARIZE:\n"
        f"Title: {item.title}\n"
        f"Source: {item.source_name}\n"
        f"URL: {item.url}\n"
        f"Raw text: {item.summary}\n\n"
        f"Produce the summary and the 'why it matters' line. Faithful to "
        f"the source — do not embellish or invent details."
    )

    try:
        with llm_span(
            "curator.synthesis",
            prompt_id=prompt_fqn,
            model=model_name,
            agent_name="curator",
        ) as span:
            result = await agent.run(prompt)
            usage = result.usage()
            record_tokens(
                span,
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
            )
        return item, result.output
    except Exception as e:
        log.warning("synthesis_failed", title=item.title[:60], error=str(e))
        return item, None


# ── The agent class ──────────────────────────────────────────


class CuratorAgent(AgentBase):
    """Research curator — weekly reading-list digest."""

    name = "curator"
    description = (
        "Personal research curator. Pulls from configured RSS and arXiv sources, "
        "triages against user context, and produces a curated reading-list digest."
    )
    # Synthesis runs through cloud-default (Sonnet). Triage stays local —
    # see _build_triage_agent call below. Bake-off (2026-05-17) showed cloud
    # synthesis produces materially better "Why it matters" framing.
    model = "cloud-default"

    def __init__(
        self,
        litellm_client: LiteLLMClient,
        config: CuratorConfig,
    ) -> None:
        self._llm = litellm_client
        self._config = config

    async def run(self, message: str, context: AgentContext) -> AgentResult:
        """Default conversational invocation: produce a single-model digest via cloud synthesis."""
        return await self.generate_digest(model_name="cloud-default", request_id=context.request_id)

    async def generate_digest(
        self,
        model_name: str,
        request_id: str,
        compare_with: str | None = None,
        max_candidates: int | None = None,
    ) -> AgentResult:
        """Fetch sources, triage, synthesize, render.

        If compare_with is set, also run that second model and emit an A/B file.
        """
        with _tracer.start_as_current_span("curator.digest") as span:
            span.set_attribute("openinference.span.kind", "AGENT")
            span.set_attribute("alfred.agent", "curator")
            span.set_attribute("alfred.request_id", request_id)
            span.set_attribute("alfred.model", model_name)
            if compare_with:
                span.set_attribute("alfred.compare_with", compare_with)
            return await self._generate_digest_inner(
                model_name=model_name,
                request_id=request_id,
                compare_with=compare_with,
                max_candidates=max_candidates,
                span=span,
            )

    async def _generate_digest_inner(
        self,
        *,
        model_name: str,
        request_id: str,
        compare_with: str | None,
        max_candidates: int | None,
        span: trace.Span,
    ) -> AgentResult:
        cfg = self._config
        log.info(
            "curator_start", model=model_name, compare_with=compare_with, sources=len(cfg.sources)
        )

        # 1. Fetch
        candidates = await fetch_all(cfg.sources, lookback_days=7)
        if not candidates:
            return AgentResult(
                message="No candidate items fetched. Are the source URLs reachable?",
                data={"items_fetched": 0},
                usage=TokenUsage(model=model_name),
            )

        # 1a. Drop items below the min content-length bar (quote-posts, fluff)
        pre_filter = len(candidates)
        candidates = [c for c in candidates if len(c.summary) >= cfg.min_content_length]
        log.info(
            "curator_minlen_filter",
            kept=len(candidates),
            dropped=pre_filter - len(candidates),
            threshold=cfg.min_content_length,
        )

        # 1b. Annotate authorship signals for the triage prompt
        for item in candidates:
            annotate_authorship(item, cfg.top_labs, cfg.top_authors)

        if max_candidates is not None and len(candidates) > max_candidates:
            log.info(
                "curator_candidate_cap",
                before=len(candidates),
                after=max_candidates,
            )
            candidates = candidates[:max_candidates]

        # 2. Triage (always local — cheap, fast). Throttled to avoid swamping the GPU.
        triage_agent, triage_prompt_fqn = _build_triage_agent(self._llm, "local-default")
        log.info("curator_using_prompt", phase="triage", prompt=triage_prompt_fqn)
        triage_sem = asyncio.Semaphore(cfg.triage_concurrency)

        async def triage_bounded(item: CandidateItem):
            async with triage_sem:
                return await _triage_item(
                    triage_agent,
                    item,
                    cfg.user_context,
                    cfg.topics,
                    prompt_fqn=triage_prompt_fqn,
                    model_name="local-default",
                )

        triage_results = await asyncio.gather(
            *(triage_bounded(item) for item in candidates)
        )

        # Attach triage outputs; apply topic weight × source quality weight; filter
        topic_weights = {t["id"]: t.get("weight", 1.0) for t in cfg.topics}
        scored: list[CandidateItem] = []
        for item, decision in triage_results:
            if decision is None or decision.topic_id == "none":
                continue
            topic_w = topic_weights.get(decision.topic_id, 1.0)
            item.topic_id = decision.topic_id
            item.relevance = decision.relevance * topic_w * item.quality_weight
            if item.relevance >= cfg.triage_score_threshold:
                scored.append(item)

        # Sort by weighted relevance, then enforce per-source cap, then take top N.
        scored.sort(key=lambda x: x.relevance or 0, reverse=True)
        per_source_counts: dict[str, int] = {}
        capped: list[CandidateItem] = []
        for item in scored:
            count = per_source_counts.get(item.source_name, 0)
            if count >= cfg.max_items_per_source:
                continue
            per_source_counts[item.source_name] = count + 1
            capped.append(item)
            if len(capped) >= cfg.max_items_per_digest:
                break
        top = capped
        log.info(
            "curator_triage_done",
            fetched=len(candidates),
            passed=len(scored),
            surfacing=len(top),
            per_source_cap=cfg.max_items_per_source,
        )

        if not top:
            return AgentResult(
                message=f"Fetched {len(candidates)} items; none met the relevance threshold.",
                data={"items_fetched": len(candidates), "items_surfaced": 0},
                usage=TokenUsage(model=model_name),
            )

        # 3. Synthesize — once per model
        models_to_run = [model_name] + ([compare_with] if compare_with else [])
        items_by_model: dict[str, list[CandidateItem]] = {}
        total_usage = TokenUsage(model=model_name)

        for m in models_to_run:
            synth_agent, synth_prompt_fqn = _build_synthesis_agent(self._llm, m)
            log.info(
                "curator_using_prompt",
                phase="synthesis",
                model=m,
                prompt=synth_prompt_fqn,
            )
            synth_sem = asyncio.Semaphore(cfg.synthesis_concurrency)
            # Deep-copy items so each model gets a clean canvas
            run_items = [_clone_item(it) for it in top]

            async def synth_bounded(
                it: CandidateItem,
                _agent=synth_agent,
                _sem=synth_sem,
                _fqn=synth_prompt_fqn,
                _model=m,
            ):
                async with _sem:
                    return await _synthesize_item(
                        _agent,
                        it,
                        cfg.user_context,
                        prompt_fqn=_fqn,
                        model_name=_model,
                    )

            synth_results = await asyncio.gather(
                *(synth_bounded(it) for it in run_items)
            )
            for item, synth in synth_results:
                if synth is None:
                    continue
                item.curated_summary = synth.summary
                item.why_it_matters = synth.why_it_matters
            items_by_model[m] = [it for it, s in synth_results if s is not None]

        # 4. Render
        generated_at = datetime.now(UTC)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        html: str | None = None
        if compare_with:
            md = render_comparison(items_by_model, cfg.topics_by_id, generated_at)
            outfile = cfg.output_dir / f"{generated_at.strftime('%Y-%m-%d')}-comparison.md"
        else:
            md = render_digest(
                items_by_model[model_name], cfg.topics_by_id, model_name, generated_at
            )
            html = render_digest_html(
                items_by_model[model_name], cfg.topics_by_id, model_name, generated_at
            )
            outfile = cfg.output_dir / f"{generated_at.strftime('%Y-%m-%d')}.md"

        outfile.write_text(md)
        log.info("curator_digest_written", path=str(outfile), items=len(top))
        span.set_attribute("alfred.curator.items_fetched", len(candidates))
        span.set_attribute("alfred.curator.items_surfaced", len(top))
        span.set_attribute("alfred.curator.output_path", str(outfile))

        return AgentResult(
            message=(
                f"Digest written: {outfile} "
                f"({len(top)} items across {len(models_to_run)} model(s))"
            ),
            data={
                "path": str(outfile),
                "items": len(top),
                "models": models_to_run,
                "html": html,
            },
            usage=total_usage,
        )


def _clone_item(item: CandidateItem) -> CandidateItem:
    return CandidateItem(
        title=item.title,
        url=item.url,
        summary=item.summary,
        published=item.published,
        source_name=item.source_name,
        topic_hints=list(item.topic_hints),
        topic_id=item.topic_id,
        relevance=item.relevance,
    )
