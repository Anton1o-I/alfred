"""Budget policy evaluation — check-before-dispatch."""

from __future__ import annotations

from pydantic import BaseModel

from alfred.core.config import BudgetConfig
from scaffold.storage.database import Database


class BudgetDecision(BaseModel):
    """Result of a budget check."""

    allowed: bool
    reason: str
    agent_used_today: int
    agent_limit: int
    global_used_today: int
    global_daily_limit: int


class BudgetPolicy:
    """Evaluates whether a request is within budget."""

    def __init__(self, config: BudgetConfig, db: Database) -> None:
        self._config = config
        self._db = db

    async def check_request_allowed(
        self, agent_name: str, estimated_tokens: int = 0
    ) -> BudgetDecision:
        """Check all budget levels: per-agent + global, tokens + dollars."""
        agent_today = await self.get_agent_usage_today(agent_name)
        global_today = await self.get_global_usage_today()
        global_month = await self.get_global_usage_this_month()
        agent_cost_today = await self.get_agent_cost_today(agent_name)
        global_cost_today = await self.get_global_cost_today()
        global_cost_month = await self.get_global_cost_this_month()

        agent_limit = self._config.daily_agent_limits.get(agent_name, 0)
        agent_usd_limit = self._config.daily_agent_usd_limits.get(agent_name, 0.0)

        decision_base = dict(
            agent_used_today=agent_today,
            agent_limit=agent_limit,
            global_used_today=global_today,
            global_daily_limit=self._config.global_daily_limit,
        )

        # Token caps (0 = unlimited). Tokens are tracked for visibility but
        # dollar caps are the real safety net for cloud spend; local Ollama
        # has $0 cost, so token quotas there are just artificial friction.
        if agent_limit > 0 and agent_today + estimated_tokens > agent_limit:
            return BudgetDecision(
                allowed=False,
                reason=f"Agent '{agent_name}' daily token limit reached "
                f"({agent_today:,}/{agent_limit:,})",
                **decision_base,
            )
        if (
            self._config.global_daily_limit > 0
            and global_today + estimated_tokens > self._config.global_daily_limit
        ):
            return BudgetDecision(
                allowed=False,
                reason=f"Global daily token limit reached "
                f"({global_today:,}/{self._config.global_daily_limit:,})",
                **decision_base,
            )
        if (
            self._config.global_monthly_limit > 0
            and global_month + estimated_tokens > self._config.global_monthly_limit
        ):
            return BudgetDecision(
                allowed=False,
                reason=f"Global monthly token limit reached "
                f"({global_month:,}/{self._config.global_monthly_limit:,})",
                **decision_base,
            )

        # Dollar caps (0 = unlimited)
        if agent_usd_limit > 0 and agent_cost_today >= agent_usd_limit:
            return BudgetDecision(
                allowed=False,
                reason=f"Agent '{agent_name}' daily $ limit reached "
                f"(${agent_cost_today:.4f}/${agent_usd_limit:.2f})",
                **decision_base,
            )
        if (
            self._config.global_daily_usd_limit > 0
            and global_cost_today >= self._config.global_daily_usd_limit
        ):
            return BudgetDecision(
                allowed=False,
                reason=f"Global daily $ limit reached "
                f"(${global_cost_today:.4f}/${self._config.global_daily_usd_limit:.2f})",
                **decision_base,
            )
        if (
            self._config.global_monthly_usd_limit > 0
            and global_cost_month >= self._config.global_monthly_usd_limit
        ):
            return BudgetDecision(
                allowed=False,
                reason=f"Global monthly $ limit reached "
                f"(${global_cost_month:.4f}/${self._config.global_monthly_usd_limit:.2f})",
                **decision_base,
            )

        return BudgetDecision(
            allowed=True, reason="Within budget", **decision_base
        )

    async def get_agent_usage_today(self, agent_name: str) -> int:
        """Total tokens used by an agent today."""
        row = await self._db.fetch_one(
            "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) as total "
            "FROM budget_usage "
            "WHERE agent_name = ? AND date(created_at) = date('now')",
            (agent_name,),
        )
        return row["total"] if row else 0

    async def get_global_usage_today(self) -> int:
        """Total tokens used across all agents today."""
        row = await self._db.fetch_one(
            "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) as total "
            "FROM budget_usage "
            "WHERE date(created_at) = date('now')",
        )
        return row["total"] if row else 0

    async def get_global_usage_this_month(self) -> int:
        """Total tokens used across all agents this month."""
        row = await self._db.fetch_one(
            "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) as total "
            "FROM budget_usage "
            "WHERE strftime('%Y-%m', created_at) = strftime('%Y-%m', 'now')",
        )
        return row["total"] if row else 0

    async def get_agent_cost_today(self, agent_name: str) -> float:
        row = await self._db.fetch_one(
            "SELECT COALESCE(SUM(estimated_cost_usd), 0.0) as total "
            "FROM budget_usage "
            "WHERE agent_name = ? AND date(created_at) = date('now')",
            (agent_name,),
        )
        return float(row["total"]) if row else 0.0

    async def get_global_cost_today(self) -> float:
        row = await self._db.fetch_one(
            "SELECT COALESCE(SUM(estimated_cost_usd), 0.0) as total "
            "FROM budget_usage "
            "WHERE date(created_at) = date('now')",
        )
        return float(row["total"]) if row else 0.0

    async def get_global_cost_this_month(self) -> float:
        row = await self._db.fetch_one(
            "SELECT COALESCE(SUM(estimated_cost_usd), 0.0) as total "
            "FROM budget_usage "
            "WHERE strftime('%Y-%m', created_at) = strftime('%Y-%m', 'now')",
        )
        return float(row["total"]) if row else 0.0
