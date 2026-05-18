"""Deterministic pure-code helpers used by the tasks workflow.

None of these touch the LLM, the DB, or the network. They're separated
out so they can be unit-tested in isolation and reused without dragging
in the whole graph module.
"""

from __future__ import annotations

import re
from typing import Final

# Articles dropped from titles after the first (verb) word during
# normalization. "Take out the trash" → "Take out trash".
_ARTICLES: Final[frozenset[str]] = frozenset({"the", "a", "an"})

# Trailing punctuation stripped from titles.
_TITLE_TRAILING_PUNCT: Final[str] = ".!?,;:"

# Mapping from RRULE frequency to its English unit + adjective.
_FREQ_UNIT: Final[dict[str, str]] = {
    "DAILY": "day",
    "WEEKLY": "week",
    "MONTHLY": "month",
    "YEARLY": "year",
}
_FREQ_ADJECTIVE: Final[dict[str, str]] = {
    "DAILY": "daily",
    "WEEKLY": "weekly",
    "MONTHLY": "monthly",
    "YEARLY": "yearly",
}


def slugify(s: str) -> str:
    """Slugify a string for use as a chore id.

    Replaces runs of non-alphanumeric characters with hyphens, lowercases,
    and strips leading/trailing hyphens. Falls back to "chore" on empty
    input so we always return a usable slug.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return slug or "chore"


def normalize_title(s: str) -> str:
    """Canonicalize a chore title so dedup compares apples to apples.

    Rules:
      - Strip leading/trailing whitespace and trailing punctuation.
      - Drop articles ('the', 'a', 'an') that appear AFTER the first word,
        so 'Take out the trash' → 'Take out trash'. Articles in the
        leading position are left alone (rare in imperatives, but conservative).
      - Capitalize the first letter; leave the rest of the casing intact so
        proper nouns ('Amazon', 'UPS') and the LLM's chosen casing survive.

    Empty input returns the same empty/None value untouched.
    """
    if not s:
        return s
    cleaned = s.strip().rstrip(_TITLE_TRAILING_PUNCT)
    parts = cleaned.split()
    if len(parts) > 1:
        filtered = [parts[0]] + [w for w in parts[1:] if w.lower() not in _ARTICLES]
        cleaned = " ".join(filtered)
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def format_recurrence_human(rule: dict, rec_type: str) -> str:
    """Render a recurrence rule + type as readable text for replies.

    Examples:
      ({freq: DAILY, interval: 2}, 'completion')
          → 'every 2 days (completion-based)'
      ({freq: WEEKLY, byday: [TU]}, 'schedule')
          → 'weekly on TU (schedule-based)'
      ({freq: WEEKLY, interval: 2, byday: [SA]}, 'schedule')
          → 'every 2 weeks on SA (schedule-based)'
    """
    freq = (rule.get("frequency") or "").upper()
    interval = int(rule.get("interval") or 1)
    byday = ",".join(rule.get("byday") or [])
    unit = _FREQ_UNIT.get(freq, freq.lower() or "cycle")
    if interval == 1:
        base = _FREQ_ADJECTIVE.get(freq, freq.lower() or "every cycle")
    else:
        base = f"every {interval} {unit}s"
    if byday:
        base += f" on {byday}"
    return f"{base} ({rec_type}-based)"


def format_incomplete_text(issues: list[str]) -> str:
    """Body text for replies when the parse couldn't fill required fields."""
    return (
        "I couldn't fully parse your chore request.\n"
        "Issues: " + "; ".join(issues) + ".\n\n"
        "Reply with the chore, who's doing it, and how often, e.g. "
        "'Add chore: take out trash every Tuesday, household'."
    )


# ── Shape (object/qualifier) comparison for deterministic dedup ─────────────


def normalize_shape(s: str | None) -> str | None:
    """Lowercase + strip a shape field. Returns None for empty input."""
    if not s:
        return None
    out = s.strip().lower()
    return out or None


def shape_verdict(
    new_obj: str | None,
    new_qual: str | None,
    ex_obj: str | None,
    ex_qual: str | None,
) -> str:
    """Pure-code dedup comparison. Returns 'high' | 'medium' | 'none'.

    Truth table (after normalize_shape):
      different object               → none
      same object, both qualifiers, same → high
      same object, both qualifiers, diff → none
      same object, one qualified, one not → medium
      same object, neither qualified → high
    """
    if not new_obj or not ex_obj:
        return "none"
    if new_obj != ex_obj:
        return "none"
    if new_qual and ex_qual:
        return "high" if new_qual == ex_qual else "none"
    if new_qual or ex_qual:
        return "medium"
    return "high"


def shape_reason(
    new_title: str,
    ex_title: str,
    new_qual: str | None,
    verdict: str,
) -> str:
    """One-sentence explanation of a shape comparison verdict.

    Shown to the user in the reply and logged for debugging.
    """
    if verdict == "high":
        return f"'{new_title}' is the same chore as '{ex_title}' (same target)."
    if verdict == "medium":
        unq = new_title if not new_qual else ex_title
        q = ex_title if not new_qual else new_title
        return (
            f"'{unq}' might be the same chore as '{q}' — one of them is more "
            "specific. Reply to confirm or split them apart."
        )
    return f"'{new_title}' and '{ex_title}' target different things."
