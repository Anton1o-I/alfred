"""Unit tests for shame-tier resolution and the briefing-split predicate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from alfred.agents.tasks.renderers import (
    TIER_NONE,
    ShameTierTable,
    _should_split_for_shame,
    shame_assignees,
    shame_tier,
)

# Mirrors the production YAML so tier ranges stay realistic in tests.
TIERS: list[dict[str, Any]] = [
    {"tier": 1, "min_days": 2, "max_days": 5, "fallback_label": "{days}d overdue",
     "row_prefix": "·"},
    {"tier": 2, "min_days": 6, "max_days": 10, "fallback_label": "{days}d overdue",
     "row_prefix": "!"},
    {"tier": 3, "min_days": 11, "max_days": None,
     "fallback_label": "{days}d overdue", "row_prefix": "!!"},
]


def _table() -> ShameTierTable:
    return ShameTierTable(TIERS)


@pytest.mark.parametrize(
    ("overdue_days", "expected"),
    [
        (-1, TIER_NONE),  # not yet due
        (0, TIER_NONE),   # due today, not overdue
        (1, TIER_NONE),   # eligible but below tier 1's min_days=2
        (2, 1),           # every-2-days chore, 4d since done → tier 1 fires
        (5, 1),
        (6, 2),
        (10, 2),
        (11, 3),
        (60, 3),
    ],
)
def test_shame_tier_table(overdue_days: int, expected: int) -> None:
    assert shame_tier(overdue_days, table=_table()) == expected


def test_shame_tier_returns_none_when_no_table() -> None:
    assert shame_tier(20, table=None) == TIER_NONE


def test_shame_tier_returns_none_when_no_range_matches() -> None:
    # Gap table: only tier 2 (7-13d). Days 1-6 should fall through.
    table = ShameTierTable(
        [{"tier": 2, "min_days": 7, "max_days": 13, "fallback_label": "{days}d"}]
    )
    assert shame_tier(5, table=table) == TIER_NONE
    assert shame_tier(10, table=table) == 2


# ── _should_split_for_shame ─────────────────────────────────────────────────


@dataclass
class _FakeChore:
    assignee: str
    title: str = "x"


@dataclass
class _FakeStatus:
    overdue_days: int
    chore: _FakeChore


def test_split_false_when_no_table() -> None:
    statuses = [_FakeStatus(20, _FakeChore("primary"))]
    assert _should_split_for_shame(statuses, None) is False


def test_split_false_when_no_overdue() -> None:
    statuses = [_FakeStatus(-1, _FakeChore("primary"))]
    assert _should_split_for_shame(statuses, _table()) is False


def test_split_false_for_tier_1_only() -> None:
    statuses = [_FakeStatus(3, _FakeChore("primary"))]  # tier 1
    assert _should_split_for_shame(statuses, _table()) is False


def test_split_true_at_tier_2() -> None:
    statuses = [_FakeStatus(8, _FakeChore("primary"))]
    assert _should_split_for_shame(statuses, _table()) is True


def test_split_true_at_tier_3() -> None:
    statuses = [_FakeStatus(30, _FakeChore("primary"))]
    assert _should_split_for_shame(statuses, _table()) is True


def test_split_ignores_household_assignee() -> None:
    statuses = [_FakeStatus(30, _FakeChore("household"))]
    assert _should_split_for_shame(statuses, _table()) is False


def test_split_mixed_tiers() -> None:
    statuses = [
        _FakeStatus(3, _FakeChore("primary")),       # tier 1
        _FakeStatus(8, _FakeChore("secondary")),     # tier 2 → triggers split
        _FakeStatus(30, _FakeChore("household")),    # excluded
    ]
    assert _should_split_for_shame(statuses, _table()) is True


# ── User-reported scenario: every-2-days chore, 4 days since last done ──────


def test_short_interval_chore_now_shames() -> None:
    """Regression: 'every 2 days' chore last done 4 days ago.

    next_due = last_done + 2 = 2 days ago → overdue_days = 2.
    Previously the global shame_after_days=3 floor swallowed this. After
    the gate change, tier 1 (min_days=2) fires.
    """
    assert shame_tier(2, table=_table()) == 1


def test_shame_assignees_includes_1_day_overdue() -> None:
    """shame_assignees previously gated on shame_after_days; now any overdue."""
    statuses = [
        _FakeStatus(1, _FakeChore("primary")),
        _FakeStatus(0, _FakeChore("secondary")),
        _FakeStatus(10, _FakeChore("household")),  # excluded
    ]
    assert shame_assignees(statuses) == {"primary"}
