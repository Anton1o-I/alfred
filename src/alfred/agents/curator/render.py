"""Render curator output — markdown for disk, HTML for email."""

from __future__ import annotations

from datetime import datetime
from html import escape

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


# ── HTML rendering for email ──────────────────────────────────────────────
# Inline styles only — most email clients (Gmail, Outlook, Apple Mail) strip
# <style> blocks and external CSS. Colors and spacing are tuned for both
# light and dark mode legibility in modern web/mobile mail clients.

_FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)


def _style(d: dict[str, str]) -> str:
    return "; ".join(f"{k}: {v}" for k, v in d.items())


def render_digest_html(
    items: list[CandidateItem],
    topics_by_id: dict[str, dict],
    model_name: str,
    generated_at: datetime,
) -> str:
    """Render a curated digest as a standalone, email-safe HTML document."""
    title = f"Alfred Research Digest — {generated_at.strftime('%Y-%m-%d')}"
    subtitle = (
        f"Synthesized by <code>{escape(model_name)}</code> on "
        f"{generated_at.strftime('%Y-%m-%d %H:%M %Z')} · "
        f"{len(items)} item{'s' if len(items) != 1 else ''}"
    )

    body_style = _style({
        "margin": "0",
        "padding": "0",
        "background": "#f5f5f7",
        "font-family": _FONT_STACK,
        "color": "#1d1d1f",
    })
    container_style = _style({
        "max-width": "640px",
        "margin": "0 auto",
        "padding": "32px 24px 48px",
        "background": "#ffffff",
    })
    h1_style = _style({
        "margin": "0 0 6px",
        "font-size": "26px",
        "font-weight": "600",
        "letter-spacing": "-0.01em",
        "color": "#111",
    })
    subtitle_style = _style({
        "margin": "0 0 32px",
        "font-size": "13px",
        "color": "#86868b",
    })

    parts: list[str] = [
        '<!doctype html><html><head><meta charset="utf-8">',
        f"<title>{escape(title)}</title></head>",
        f'<body style="{body_style}">',
        f'<div style="{container_style}">',
        f'<h1 style="{h1_style}">{escape(title)}</h1>',
        f'<p style="{subtitle_style}">{subtitle}</p>',
    ]

    if not items:
        empty_style = _style({"font-style": "italic", "color": "#86868b"})
        parts.append(
            f'<p style="{empty_style}">No items met the relevance threshold this run.</p>'
        )
    else:
        grouped: dict[str, list[CandidateItem]] = {}
        for item in items:
            grouped.setdefault(item.topic_id or "uncategorized", []).append(item)
        ordered_topic_ids = [tid for tid in topics_by_id if tid in grouped]
        if "uncategorized" in grouped:
            ordered_topic_ids.append("uncategorized")

        for tid in ordered_topic_ids:
            topic = topics_by_id.get(tid, {"name": "Uncategorized"})
            topic_items = sorted(
                grouped[tid], key=lambda x: x.relevance or 0, reverse=True
            )
            parts.append(_render_topic_section_html(topic["name"], topic_items))

    footer_style = _style({
        "margin": "40px 0 0",
        "padding-top": "16px",
        "border-top": "1px solid #e5e5ea",
        "font-size": "12px",
        "color": "#86868b",
        "text-align": "center",
    })
    parts.append(f'<p style="{footer_style}">Alfred · weekly research digest</p>')
    parts.append("</div></body></html>")
    return "".join(parts)


def _render_topic_section_html(name: str, items: list[CandidateItem]) -> str:
    topic_h_style = _style({
        "margin": "32px 0 16px",
        "padding-top": "16px",
        "border-top": "1px solid #e5e5ea",
        "font-size": "11px",
        "font-weight": "600",
        "letter-spacing": "0.08em",
        "text-transform": "uppercase",
        "color": "#6e6e73",
    })
    out = [f'<h2 style="{topic_h_style}">{escape(name)}</h2>']
    for item in items:
        out.append(_render_item_html(item))
    return "".join(out)


def _render_item_html(item: CandidateItem) -> str:
    wrapper_style = _style({"margin": "0 0 28px"})
    title_style = _style({
        "margin": "0 0 4px",
        "font-size": "17px",
        "font-weight": "600",
        "line-height": "1.35",
        "color": "#111",
    })
    title_link_style = _style({
        "color": "#111",
        "text-decoration": "none",
    })
    meta_style = _style({
        "margin": "0 0 12px",
        "font-size": "13px",
        "color": "#86868b",
    })
    summary_style = _style({
        "margin": "0 0 12px",
        "font-size": "15px",
        "line-height": "1.6",
        "color": "#1d1d1f",
    })
    callout_style = _style({
        "margin": "0",
        "padding": "12px 14px",
        "border-left": "3px solid #007aff",
        "background": "#f5f7fb",
        "font-size": "14px",
        "line-height": "1.55",
        "color": "#1d1d1f",
        "border-radius": "0 4px 4px 0",
    })
    callout_label_style = _style({
        "font-weight": "600",
        "color": "#0040a0",
    })

    rel = (
        f' · Relevance {item.relevance:.2f}' if item.relevance is not None else ""
    )
    parts = [
        f'<div style="{wrapper_style}">',
        f'<h3 style="{title_style}">'
        f'<a href="{escape(item.url)}" style="{title_link_style}">'
        f"{escape(item.title)}</a></h3>",
        f'<p style="{meta_style}">{escape(item.source_name)}{rel}</p>',
    ]
    if item.curated_summary:
        parts.append(
            f'<p style="{summary_style}">{escape(item.curated_summary)}</p>'
        )
    if item.why_it_matters:
        parts.append(
            f'<p style="{callout_style}">'
            f'<span style="{callout_label_style}">Why it matters:</span> '
            f"{escape(item.why_it_matters)}</p>"
        )
    parts.append("</div>")
    return "".join(parts)


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
