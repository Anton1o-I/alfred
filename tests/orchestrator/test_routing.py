"""Router fail-closed behavior — never silently dispatch to a random agent."""

from __future__ import annotations

from alfred.orchestrator.nodes import _keyword_fallback


def test_keyword_fallback_returns_empty_when_no_match() -> None:
    """An unmatched request must NOT silently route to agents[0]."""
    agents = [
        {"name": "calendar", "description": "schedules"},
        {"name": "meals", "description": "food"},
    ]
    assert _keyword_fallback("what is the meaning of life?", agents) == ""


def test_keyword_fallback_matches_calendar() -> None:
    agents = [{"name": "calendar", "description": "schedules"}]
    assert _keyword_fallback("any meetings today?", agents) == "calendar"


def test_keyword_fallback_skips_unregistered_agent() -> None:
    """Even if keyword matches, the agent must be registered."""
    agents = [{"name": "calendar", "description": "schedules"}]
    # 'recipe' would match meals, but meals isn't registered
    assert _keyword_fallback("give me a recipe", agents) == ""
