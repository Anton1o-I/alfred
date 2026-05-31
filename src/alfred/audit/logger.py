"""Structured audit logging — dual-writes to SQLite and structlog."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import structlog

from scaffold.storage.database import Database
from scaffold.core.models import TokenUsage

log = structlog.get_logger()


class AuditLogger:
    """Records all system events to SQLite and structured log output."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def log_event(
        self,
        event_type: str,
        request_id: str,
        agent_name: str | None = None,
        source: str | None = None,
        details: dict[str, Any] | None = None,
        token_usage: TokenUsage | None = None,
        duration_ms: int = 0,
    ) -> None:
        """Write one audit row and emit a structured log line."""
        now = datetime.now(UTC).isoformat()
        details_json = json.dumps(details or {})

        input_tokens = token_usage.input_tokens if token_usage else 0
        output_tokens = token_usage.output_tokens if token_usage else 0
        cost = token_usage.estimated_cost_usd if token_usage else 0.0

        await self._db.execute(
            "INSERT INTO audit_log "
            "(request_id, timestamp, event_type, agent_name, source, details, "
            " token_usage_input, token_usage_output, cost_usd, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                request_id,
                now,
                event_type,
                agent_name,
                source,
                details_json,
                input_tokens,
                output_tokens,
                cost,
                duration_ms,
            ),
        )

        log.info(
            "audit_event",
            event_type=event_type,
            request_id=request_id[:8],
            agent=agent_name,
            source=source,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=f"{cost:.6f}",
            duration_ms=duration_ms,
        )

    async def get_events(
        self,
        request_id: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query audit log for dashboard/debugging."""
        conditions: list[str] = []
        params: list[Any] = []

        if request_id:
            conditions.append("request_id = ?")
            params.append(request_id)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM audit_log {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        return await self._db.fetch_all(sql, tuple(params))
