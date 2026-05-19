"""Unit tests for shame-tier resolution and the briefing-split predicate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pytest

from alfred.agents.tasks.renderers import (
    TIER_NONE,
    ShameTierTable,
    _should_split_for_shame,
    shame_tier,
)


# Mirrors the production YAML so tier ranges stay realistic in tests.
TIERS: list[dict[str, Any]] = [
    {"tier": 1, "min_days": 3, "max_days": 6, "fallback_label": "{days}d pending",
     "row_prefix": "·"},
    {"tier": 2, "min_days": 7, "max_days": 13, "fallback_label": "{days}d overdue",
     "row_prefix": "!"},
    {"tier": 3, "min_days": 14, "max_days": None,
     "fallback_label": "{days}d — two weeks pending", "row_prefix": "!!"},
]


def _table() -> ShameTierTable:
    return ShameTierTable(TIERS)


@pytest.mark.parametrize(
    ("overdue_days", "shame_after_days", "expected"),
    [
        (-1, 3, TIER_NONE),  # not yet due
        (0, 3, TIER_NONE),   # due today
        (1, 3, TIER_NONE),   # below shame_after_days
        (2, 3, TIER_NONE),   # still below threshold
        (3, 3, 1),           # first tier kicks in exactly at threshold
        (5, 3, 1),
        (6, 3, 1),
        (7, 3, 2),
        (10, 3, 2),
        (13, 3, 2),
        (14, 3, 3),
        (60, 3, 3),
        # High shame_after_days suppresses tiering until later.
        (5, 7, TIER_NONE),
        (7, 7, 2),
    ],
)
def test_shame_tier_table(
    overdue_days: int, shame_after_days: int, expected: int
) -> None:
    assert (
        shame_tier(
            overdue_days,
            shame_after_days=shame_after_days,
            table=_table(),
        )
        == expected
    )


def test_shame_tier_returns_none_when_no_table() -> None:
    assert shame_tier(20, shame_after_days=3, table=None) == TIER_NONE


def test_shame_tier_returns_none_when_no_range_matches() -> None:
    # Gap table: only tier 2 (7-13d). Days 3-6 should fall through.
    table = ShameTierTable(
        [{"tier": 2, "min_days": 7, "max_days": 13, "fallback_label": "{days}d"}]
    )
    assert shame_tier(5, shame_after_days=3, table=table) == TIER_NONE
    assert shame_tier(10, shame_after_days=3, table=table) == 2


# ── _should_split_for_shame ─────────────────────────────────────────────────


@dataclass
class _FakeChore:
    assignee: str
    shame_after_days: int = 3
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
    statuses = [_FakeStatus(5, _FakeChore("primary"))]  # tier 1
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
        _FakeStatus(5, _FakeChore("primary")),       # tier 1
        _FakeStatus(8, _FakeChore("secondary")),     # tier 2 → triggers split
        _FakeStatus(30, _FakeChore("household")),    # excluded
    ]
    assert _should_split_for_shame(statuses, _table()) is True


# Suppress unused-import warning for datetime — kept available for future fixtures.
_ = datetime
