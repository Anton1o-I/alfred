"""Source fetchers — pulls candidate items from RSS feeds and arXiv categories."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import feedparser
import httpx
import structlog

log = structlog.get_logger()


@dataclass
class CandidateItem:
    """A piece of content the curator might surface in a digest."""

    title: str
    url: str
    summary: str  # raw description/abstract from the feed; not yet curated
    published: datetime | None
    source_name: str
    topic_hints: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    quality_weight: float = 1.0  # per-source multiplier; lower for noisy feeds like arXiv
    # Filled in later by the triage/synthesis steps:
    topic_id: str | None = None
    relevance: float | None = None
    curated_summary: str | None = None
    why_it_matters: str | None = None
    # Authorship signals (populated during fetch):
    matched_authors: list[str] = field(default_factory=list)
    matched_labs: list[str] = field(default_factory=list)


async def fetch_rss(
    name: str,
    url: str,
    topic_hints: list[str],
    quality_weight: float = 1.0,
) -> list[CandidateItem]:
    """Pull recent items from a standard RSS/Atom feed.

    Uses a real browser User-Agent because Substack, Cloudflare, and many
    media properties 403 the default `python-httpx` UA outright.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
    }
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            content = resp.content
    except Exception as e:
        log.warning("source_fetch_failed", source=name, error=str(e))
        return []

    parsed = feedparser.parse(content)
    items: list[CandidateItem] = []
    for entry in parsed.entries:
        published = _coerce_datetime(entry)
        items.append(
            CandidateItem(
                title=getattr(entry, "title", "(untitled)"),
                url=getattr(entry, "link", ""),
                summary=_clean(
                    getattr(entry, "summary", "") or getattr(entry, "description", "")
                ),
                published=published,
                source_name=name,
                topic_hints=list(topic_hints),
                authors=_extract_authors(entry),
                quality_weight=quality_weight,
            )
        )
    return items


async def fetch_arxiv(
    category: str,
    topic_hints: list[str],
    quality_weight: float = 0.6,
    max_items: int = 40,
) -> list[CandidateItem]:
    """Pull recent submissions from an arXiv category via its RSS feed.

    arXiv is a preprint firehose — default quality_weight is 0.6 to
    reflect that items must clear a higher bar than vetted sources.
    """
    url = f"http://export.arxiv.org/rss/{category}"
    items = await fetch_rss(f"arXiv {category}", url, topic_hints, quality_weight)
    return items[:max_items]


async def fetch_all(sources: list[dict], lookback_days: int = 7) -> list[CandidateItem]:
    """Fetch from all configured sources concurrently; filter to lookback window."""
    tasks = []
    for src in sources:
        qw = src.get("quality_weight", 1.0)
        if src["type"] == "rss":
            tasks.append(
                fetch_rss(src["name"], src["url"], src.get("topic_hints", []), qw)
            )
        elif src["type"] == "arxiv":
            tasks.append(
                fetch_arxiv(src["category"], src.get("topic_hints", []), qw)
            )
        else:
            log.warning("unknown_source_type", type=src["type"])

    results = await asyncio.gather(*tasks, return_exceptions=True)
    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

    all_items: list[CandidateItem] = []
    for r in results:
        if isinstance(r, Exception):
            log.warning("source_task_exception", error=str(r))
            continue
        for item in r:
            if item.published is None or item.published >= cutoff:
                all_items.append(item)

    log.info("sources_fetched", total_items=len(all_items), sources=len(sources))
    return all_items


def _coerce_datetime(entry) -> datetime | None:
    """Parse a feedparser entry's published/updated timestamp into a UTC datetime."""
    for attr in ("published_parsed", "updated_parsed"):
        ts = getattr(entry, attr, None)
        if ts:
            try:
                return datetime(*ts[:6], tzinfo=UTC)
            except (TypeError, ValueError):
                continue
    return None


def _clean(text: str, max_chars: int = 4000) -> str:
    """Strip HTML tags crudely and truncate. Good enough for triage."""
    import re

    no_tags = re.sub(r"<[^>]+>", " ", text)
    collapsed = re.sub(r"\s+", " ", no_tags).strip()
    return collapsed[:max_chars]


def _extract_authors(entry) -> list[str]:
    """Pull author names from a feedparser entry. Handles RSS and Atom shapes,
    and arXiv's typical 'Name1, Name2, Name3' single-field format."""
    raw: list[str] = []

    # Atom-style: entry.authors = [{name: "..."}, ...]
    if hasattr(entry, "authors") and entry.authors:
        for a in entry.authors:
            name = a.get("name") if isinstance(a, dict) else None
            if name:
                raw.append(name)

    # RSS-style: entry.author = "Name1, Name2" (often used by arXiv)
    elif hasattr(entry, "author") and entry.author:
        # arXiv puts the full author list as one comma-separated string
        # like "First Last, Other Name, Third Person"
        for chunk in str(entry.author).split(","):
            cleaned = chunk.strip()
            if cleaned:
                raw.append(cleaned)

    return raw


def annotate_authorship(
    item: CandidateItem,
    top_labs: list[str],
    top_authors: list[str],
) -> None:
    """Stamp matched_authors and matched_labs onto an item in place.

    Author match is exact (case-insensitive). Lab match is substring against
    title+summary (where affiliations usually surface, e.g. "we at Anthropic").
    """
    author_set = {a.lower() for a in top_authors}
    item.matched_authors = [a for a in item.authors if a.lower() in author_set]

    haystack = f"{item.title} {item.summary}".lower()
    item.matched_labs = [lab for lab in top_labs if lab.lower() in haystack]
