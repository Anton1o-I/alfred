"""Smoke tests for BudgetPolicy — token and dollar cap enforcement."""

from __future__ import annotations

import pytest

from scaffold.core.config import BudgetConfig
from scaffold.budget.policies import BudgetPolicy
from scaffold.budget.tracker import BudgetTracker
from scaffold.core.models import TokenUsage
from scaffold.storage.database import Database


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "test.db"))
    await database.initialize()
    yield database
    await database.close()


async def test_allowed_when_no_spend(db: Database) -> None:
    policy = BudgetPolicy(BudgetConfig(daily_agent_usd_limits={"x": 1.0}), db)
    decision = await policy.check_request_allowed("x")
    assert decision.allowed is True


async def test_refuses_when_agent_usd_cap_exceeded(db: Database) -> None:
    cfg = BudgetConfig(daily_agent_usd_limits={"x": 0.0001})
    policy = BudgetPolicy(cfg, db)
    tracker = BudgetTracker(db)

    await tracker.record_usage(
        "x",
        "req-1",
        TokenUsage(model="cloud-default", input_tokens=10, output_tokens=5, estimated_cost_usd=0.001),
    )

    decision = await policy.check_request_allowed("x")
    assert decision.allowed is False
    assert "daily $ limit" in decision.reason


async def test_refuses_when_token_cap_exceeded(db: Database) -> None:
    cfg = BudgetConfig(daily_agent_limits={"x": 100})
    policy = BudgetPolicy(cfg, db)
    tracker = BudgetTracker(db)

    await tracker.record_usage(
        "x",
        "req-1",
        TokenUsage(model="local-default", input_tokens=80, output_tokens=30),
    )

    decision = await policy.check_request_allowed("x")
    assert decision.allowed is False
    assert "token limit" in decision.reason


async def test_zero_limits_mean_unlimited(db: Database) -> None:
    cfg = BudgetConfig(
        global_daily_usd_limit=0.0,
        global_monthly_usd_limit=0.0,
    )
    policy = BudgetPolicy(cfg, db)
    tracker = BudgetTracker(db)

    await tracker.record_usage(
        "x",
        "req-1",
        TokenUsage(model="cloud-default", input_tokens=1000, output_tokens=500, estimated_cost_usd=100.0),
    )

    decision = await policy.check_request_allowed("x")
    # No per-agent caps set; agent_usd_limit=0 means unlimited
    assert decision.allowed is True
