"""CRUD for dynamic topic interests, shared across agents."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from scaffold.storage.database import Database
from scaffold.core.constants import TopicPriority


class TopicInterest(BaseModel):
    """A tracked topic of interest."""

    id: int | None = None
    name: str
    priority: TopicPriority = TopicPriority.NORMAL
    added_by: str | None = None
    active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None


class TopicStore:
    """CRUD for dynamic topic interests, backed by SQLite."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(
        self,
        name: str,
        priority: TopicPriority = TopicPriority.NORMAL,
        added_by: str | None = None,
    ) -> TopicInterest:
        """Add a new topic. Reactivates if it was previously soft-deleted."""
        existing = await self.get(name)
        if existing:
            if not existing.active:
                await self._db.execute(
                    "UPDATE topics SET active = 1, priority = ?, "
                    "updated_at = datetime('now') WHERE name = ?",
                    (priority, name),
                )
            return await self.get(name)  # type: ignore[return-value]

        await self._db.execute(
            "INSERT INTO topics (name, priority, added_by) VALUES (?, ?, ?)",
            (name, priority, added_by),
        )
        return await self.get(name)  # type: ignore[return-value]

    async def remove(self, name: str) -> bool:
        """Soft-delete a topic (sets active=0)."""
        result = await self._db.execute(
            "UPDATE topics SET active = 0, updated_at = datetime('now') "
            "WHERE name = ? AND active = 1",
            (name,),
        )
        return result.rowcount > 0

    async def update_priority(
        self, name: str, priority: TopicPriority
    ) -> TopicInterest | None:
        """Update a topic's priority level."""
        await self._db.execute(
            "UPDATE topics SET priority = ?, updated_at = datetime('now') "
            "WHERE name = ? AND active = 1",
            (priority, name),
        )
        return await self.get(name)

    async def list_active(
        self, priority: TopicPriority | None = None
    ) -> list[TopicInterest]:
        """List all active topics, optionally filtered by priority."""
        if priority:
            rows = await self._db.fetch_all(
                "SELECT * FROM topics WHERE active = 1 AND priority = ? "
                "ORDER BY name",
                (priority,),
            )
        else:
            rows = await self._db.fetch_all(
                "SELECT * FROM topics WHERE active = 1 ORDER BY name",
            )
        return [TopicInterest(**row) for row in rows]

    async def get(self, name: str) -> TopicInterest | None:
        """Get a topic by name."""
        row = await self._db.fetch_one(
            "SELECT * FROM topics WHERE name = ?", (name,)
        )
        if row is None:
            return None
        return TopicInterest(**row)
