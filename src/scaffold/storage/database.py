"""Async SQLite connection manager with migration support."""

from __future__ import annotations

import contextlib
import importlib.resources
import os
from pathlib import Path

import aiosqlite

_DB_FILE_MODE = 0o600


class Database:
    """Async SQLite connection manager with migration support."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create DB file if needed, run pending migrations."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._run_migrations()
        # Audit log + chore history hold PII; default umask leaves the DB
        # world-readable. Tighten after WAL has materialised the sidecars.
        for path in (
            self.db_path,
            f"{self.db_path}-wal",
            f"{self.db_path}-shm",
        ):
            with contextlib.suppress(FileNotFoundError):
                os.chmod(path, _DB_FILE_MODE)

    async def _run_migrations(self) -> None:
        """Run all SQL migration files in order."""
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS _migrations ("
            "  name TEXT PRIMARY KEY,"
            "  applied_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )
        await self._db.commit()

        migrations_dir = (
            Path(importlib.resources.files("scaffold")) / "storage" / "migrations"
        )
        if not migrations_dir.exists():
            return

        applied = {
            row[0]
            for row in await self._db.execute_fetchall(
                "SELECT name FROM _migrations"
            )
        }

        for sql_file in sorted(migrations_dir.glob("*.sql")):
            if sql_file.name in applied:
                continue
            sql = sql_file.read_text()
            await self._db.executescript(sql)
            await self._db.execute(
                "INSERT INTO _migrations (name) VALUES (?)", (sql_file.name,)
            )
            await self._db.commit()

    async def execute(self, sql: str, params: tuple = ()) -> aiosqlite.Cursor:
        """Execute a SQL statement."""
        assert self._db is not None, "Database not initialized"
        cursor = await self._db.execute(sql, params)
        await self._db.commit()
        return cursor

    async def fetch_one(self, sql: str, params: tuple = ()) -> dict | None:
        """Fetch a single row as a dict."""
        assert self._db is not None, "Database not initialized"
        cursor = await self._db.execute(sql, params)
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    async def fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        """Fetch all rows as dicts."""
        assert self._db is not None, "Database not initialized"
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
