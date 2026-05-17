"""Render curator output as a markdown digest."""

from __future__ import annotations

from datetime import datetime

from alfred.agents.curator.sources import CandidateItem


def render_digest(
    items: list[CandidateItem],
    topics_by_id: dict[str, dict],
    model_name: str,
    generated_at: datetime,
) -> str:
    """Render a list of curated items as a single markdown digest, grouped by topic."""
    lines: list[str] = []
    lines.append(f"# Alfred Research Digest — {generated_at.strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append(
        f"_Synthesized by `{model_name}` on {generated_at.strftime('%Y-%m-%d %H:%M %Z')}._"
    )
    lines.append(f"_Total items surfaced: {len(items)}._")
    lines.append("")

    if not items:
        lines.append("_No items met the relevance threshold this run._")
        return "\n".join(lines)

    grouped: dict[str, list[CandidateItem]] = {}
    for item in items:
        grouped.setdefault(item.topic_id or "uncategorized", []).append(item)

    # Sort topics by configured order; uncategorized last
    ordered_topic_ids = [tid for tid in topics_by_id if tid in grouped]
    if "uncategorized" in grouped:
        ordered_topic_ids.append("uncategorized")

    for tid in ordered_topic_ids:
        topic = topics_by_id.get(tid, {"name": "Uncategorized"})
        topic_items = sorted(grouped[tid], key=lambda x: x.relevance or 0, reverse=True)
        lines.append(f"## {topic['name']}")
        lines.append("")
        for item in topic_items:
            lines.append(f"### {item.title}")
            lines.append(f"_{item.source_name} · [link]({item.url})_")
            if item.relevance is not None:
                lines.append(f"_Relevance: {item.relevance:.2f}_")
            lines.append("")
            if item.curated_summary:
                lines.append(item.curated_summary)
                lines.append("")
            if item.why_it_matters:
                lines.append(f"**Why it matters:** {item.why_it_matters}")
                lines.append("")
        lines.append("")

    return "\n".join(lines)


def render_comparison(
    items_by_model: dict[str, list[CandidateItem]],
    topics_by_id: dict[str, dict],
    generated_at: datetime,
) -> str:
    """Render side-by-side digests from multiple models for A/B comparison."""
    lines: list[str] = []
    lines.append(f"# Alfred Research Digest — A/B Comparison — {generated_at.strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append(f"Generated at {generated_at.strftime('%Y-%m-%d %H:%M %Z')}.")
    lines.append("Same candidate items, summarized through each model below.")
    lines.append("")
    for model_name, items in items_by_model.items():
        lines.append("---")
        lines.append("")
        lines.append(f"# Model: `{model_name}`")
        lines.append("")
        digest = render_digest(items, topics_by_id, model_name, generated_at)
        # Strip the inner header (we just wrote our own model header above)
        digest_lines = digest.split("\n")
        for ln in digest_lines:
            if ln.startswith("# Alfred Research Digest"):
                continue
            lines.append(ln)
    return "\n".join(lines)
