"""Token spend recording and summary queries."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from scaffold.core.models import TokenUsage
from scaffold.storage.database import Database


class SpendSummary(BaseModel):
    """Spend summary for a time period."""

    period: str
    total_tokens: int
    total_cost_usd: float
    by_agent: dict[str, int]
    by_provider: dict[str, int]


class BudgetTracker:
    """Records token spend after LLM calls complete."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def record_usage(
        self, agent_name: str, request_id: str, usage: TokenUsage
    ) -> None:
        """Record a completed LLM call's token usage."""
        await self._db.execute(
            "INSERT INTO budget_usage "
            "(agent_name, request_id, provider, model, "
            " input_tokens, output_tokens, estimated_cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                agent_name,
                request_id,
                usage.provider,
                usage.model,
                usage.input_tokens,
                usage.output_tokens,
                usage.estimated_cost_usd,
            ),
        )

    async def get_spend_summary(
        self, period: Literal["today", "week", "month"] = "today"
    ) -> SpendSummary:
        """Get aggregated spend for a time period."""
        date_filter = {
            "today": "date(created_at) = date('now')",
            "week": "created_at >= datetime('now', '-7 days')",
            "month": "strftime('%Y-%m', created_at) = strftime('%Y-%m', 'now')",
        }[period]

        total_row = await self._db.fetch_one(
            f"SELECT COALESCE(SUM(input_tokens + output_tokens), 0) as tokens, "
            f"       COALESCE(SUM(estimated_cost_usd), 0) as cost "
            f"FROM budget_usage WHERE {date_filter}",
        )

        by_agent_rows = await self._db.fetch_all(
            f"SELECT agent_name, SUM(input_tokens + output_tokens) as tokens "
            f"FROM budget_usage WHERE {date_filter} GROUP BY agent_name",
        )

        by_provider_rows = await self._db.fetch_all(
            f"SELECT provider, SUM(input_tokens + output_tokens) as tokens "
            f"FROM budget_usage WHERE {date_filter} GROUP BY provider",
        )

        return SpendSummary(
            period=period,
            total_tokens=total_row["tokens"] if total_row else 0,
            total_cost_usd=total_row["cost"] if total_row else 0.0,
            by_agent={r["agent_name"]: r["tokens"] for r in by_agent_rows},
            by_provider={r["provider"]: r["tokens"] for r in by_provider_rows},
        )
