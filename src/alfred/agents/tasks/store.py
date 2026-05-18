"""Chore data layer — pure code, no LLM.

Encapsulates all chore + completion CRUD plus the "what's due / overdue"
queries that both the workflow and the daily briefing read. Centralized
so the briefing and the agent see the same notion of status.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import structlog

if TYPE_CHECKING:
    from alfred.storage.database import Database

log = structlog.get_logger()


@dataclass
class Chore:
    id: str
    title: str
    description: str | None
    assignee: str
    recurrence_type: str  # "schedule" | "completion" | "once"
    recurrence_rule: dict[str, Any]
    shame_after_days: int
    active: bool
    created_at: str
    due_date: str | None = None  # ISO date for one-time tasks
    # Structured shape — used by dedup for deterministic code comparison.
    # Both nullable so legacy chores (created before migration 005) fall
    # through to the LLM-based dedup fallback.
    object: str | None = None
    qualifier: str | None = None


@dataclass
class Completion:
    id: int
    chore_id: str
    completed_at: str
    completed_by: str
    completed_via: str


@dataclass
class ChoreStatus:
    """A chore plus its computed next-due date and overdue-days."""

    chore: Chore
    last_completed_at: datetime | None
    next_due: datetime
    overdue_days: int  # negative when not yet due


class ChoreStore:
    """All chore reads/writes go through here. Constructor takes the Database."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # ── Chore CRUD ────────────────────────────────────────────────────────

    async def add_chore(
        self,
        *,
        id: str,
        title: str,
        assignee: str,
        recurrence_type: str,
        recurrence_rule: dict[str, Any],
        description: str | None = None,
        shame_after_days: int = 3,
        due_date: str | None = None,
        object: str | None = None,  # noqa: A002 — column name, shadowing builtin is intentional
        qualifier: str | None = None,
    ) -> Chore:
        await self._db.execute(
            "INSERT INTO chores "
            "(id, title, description, assignee, recurrence_type, "
            " recurrence_rule, shame_after_days, due_date, "
            " object, qualifier, active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
            (
                id,
                title,
                description,
                assignee,
                recurrence_type,
                json.dumps(recurrence_rule),
                shame_after_days,
                due_date,
                object,
                qualifier,
            ),
        )
        log.info(
            "chore_added",
            id=id, assignee=assignee, recurrence_type=recurrence_type,
            due_date=due_date, object=object, qualifier=qualifier,
        )
        chore = await self.get_chore(id)
        assert chore is not None
        return chore

    async def get_chore(self, chore_id: str) -> Chore | None:
        row = await self._db.fetch_one(
            "SELECT * FROM chores WHERE id = ?", (chore_id,)
        )
        return _row_to_chore(row) if row else None

    async def list_active_chores(self) -> list[Chore]:
        rows = await self._db.fetch_all(
            "SELECT * FROM chores WHERE active = 1 ORDER BY id"
        )
        return [_row_to_chore(r) for r in rows]

    async def update_chore(
        self,
        chore_id: str,
        **fields: Any,
    ) -> Chore | None:
        """Update any subset of {title, description, assignee, recurrence_type,
        recurrence_rule, shame_after_days, active}."""
        allowed = {
            "title", "description", "assignee", "recurrence_type",
            "recurrence_rule", "shame_after_days", "active", "due_date",
            "object", "qualifier",
        }
        sets: list[str] = []
        params: list[Any] = []
        for k, v in fields.items():
            if k not in allowed:
                continue
            sets.append(f"{k} = ?")
            params.append(json.dumps(v) if k == "recurrence_rule" else v)
        if not sets:
            return await self.get_chore(chore_id)
        params.append(chore_id)
        await self._db.execute(
            f"UPDATE chores SET {', '.join(sets)} WHERE id = ?",
            tuple(params),
        )
        log.info("chore_updated", id=chore_id, fields=list(fields.keys()))
        return await self.get_chore(chore_id)

    async def soft_delete_chore(self, chore_id: str) -> bool:
        chore = await self.get_chore(chore_id)
        if chore is None or not chore.active:
            return False
        await self._db.execute(
            "UPDATE chores SET active = 0 WHERE id = ?", (chore_id,)
        )
        log.info("chore_soft_deleted", id=chore_id)
        return True

    # ── Completion CRUD ───────────────────────────────────────────────────

    async def record_completion(
        self,
        *,
        chore_id: str,
        completed_by: str,
        completed_via: str,
    ) -> Completion:
        cursor = await self._db.execute(
            "INSERT INTO chore_completions (chore_id, completed_by, completed_via) "
            "VALUES (?, ?, ?)",
            (chore_id, completed_by, completed_via),
        )
        log.info(
            "chore_completed",
            chore_id=chore_id,
            completed_by=completed_by,
            via=completed_via,
        )
        return Completion(
            id=cursor.lastrowid or 0,
            chore_id=chore_id,
            completed_at=datetime.now().isoformat(),
            completed_by=completed_by,
            completed_via=completed_via,
        )

    async def last_completion(self, chore_id: str) -> Completion | None:
        row = await self._db.fetch_one(
            "SELECT * FROM chore_completions WHERE chore_id = ? "
            "ORDER BY completed_at DESC LIMIT 1",
            (chore_id,),
        )
        if not row:
            return None
        return Completion(
            id=row["id"],
            chore_id=row["chore_id"],
            completed_at=row["completed_at"],
            completed_by=row["completed_by"],
            completed_via=row["completed_via"],
        )

    async def completions_in_window(
        self, start: datetime, end: datetime
    ) -> list[Completion]:
        rows = await self._db.fetch_all(
            "SELECT * FROM chore_completions "
            "WHERE completed_at >= ? AND completed_at < ? "
            "ORDER BY completed_at DESC",
            (start.isoformat(), end.isoformat()),
        )
        return [
            Completion(
                id=r["id"],
                chore_id=r["chore_id"],
                completed_at=r["completed_at"],
                completed_by=r["completed_by"],
                completed_via=r["completed_via"],
            )
            for r in rows
        ]

    # ── Status (next-due + overdue) ───────────────────────────────────────

    async def status_for_all_active(
        self, now: datetime, tz: ZoneInfo
    ) -> list[ChoreStatus]:
        """Return every active chore with its computed next-due and overdue-days.

        One-time chores that already have a completion are skipped — they're
        done and shouldn't surface in the briefing or pending list anymore.
        """
        chores = await self.list_active_chores()
        out: list[ChoreStatus] = []
        for c in chores:
            last = await self.last_completion(c.id)
            if c.recurrence_type == "once" and last is not None:
                continue
            last_dt = _parse_db_dt(last.completed_at, tz) if last else None
            next_due = _compute_next_due(c, last_dt, now, tz)
            overdue = (now.date() - next_due.date()).days
            out.append(
                ChoreStatus(
                    chore=c,
                    last_completed_at=last_dt,
                    next_due=next_due,
                    overdue_days=overdue,
                )
            )
        return out


# ── helpers ──────────────────────────────────────────────────────────────


def _row_to_chore(row: dict[str, Any]) -> Chore:
    return Chore(
        id=row["id"],
        title=row["title"],
        description=row["description"],
        assignee=row["assignee"],
        recurrence_type=row["recurrence_type"],
        recurrence_rule=json.loads(row["recurrence_rule"]),
        shame_after_days=row["shame_after_days"],
        active=bool(row["active"]),
        created_at=row["created_at"],
        due_date=_row_get(row, "due_date"),
        object=_row_get(row, "object"),
        qualifier=_row_get(row, "qualifier"),
    )


def _row_get(row: dict[str, Any], key: str) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def _parse_db_dt(s: str, tz: ZoneInfo) -> datetime:
    """SQLite default `datetime('now')` writes 'YYYY-MM-DD HH:MM:SS' (UTC).
    Tolerate either that or full ISO 8601 with tz."""
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.fromisoformat(s.replace(" ", "T"))
    if dt.tzinfo is None:
        from datetime import UTC

        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(tz)


def _compute_next_due(
    chore: Chore, last_completed: datetime | None, now: datetime, tz: ZoneInfo
) -> datetime:
    """Compute when this chore is next due.

    Schedule-based: walk the recurrence rule forward from today, return the next
    occurrence (today inclusive if it matches).

    Completion-based: anchor = last_completed (or chore.created_at if never).
    Next due = anchor + interval-derived delta.
    """
    rule = chore.recurrence_rule or {}
    if chore.recurrence_type == "once":
        # One-time: due_date is the answer (start of day in local tz).
        if chore.due_date:
            try:
                d = datetime.fromisoformat(chore.due_date).date()
            except ValueError:
                d = now.date()
            return datetime.combine(d, datetime.min.time(), tzinfo=tz)
        # Fallback: treat as due today if due_date is missing.
        return now.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    if chore.recurrence_type == "completion":
        anchor = last_completed or _parse_db_dt(chore.created_at, tz)
        return anchor + _interval_delta(rule)
    # schedule-based
    return _next_schedule_occurrence(rule, now, tz)


def _interval_delta(rule: dict[str, Any]) -> timedelta:
    """How long between completion-based runs."""
    freq = (rule.get("frequency") or "DAILY").upper()
    interval = int(rule.get("interval") or 1)
    if freq == "DAILY":
        return timedelta(days=interval)
    if freq == "WEEKLY":
        return timedelta(weeks=interval)
    if freq == "MONTHLY":
        return timedelta(days=30 * interval)  # approximation
    if freq == "YEARLY":
        return timedelta(days=365 * interval)
    return timedelta(days=interval)


_DAY_MAP = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}


def _next_schedule_occurrence(
    rule: dict[str, Any], now: datetime, tz: ZoneInfo
) -> datetime:
    """Walk forward from today and return the next datetime that satisfies the rule.

    Only handles DAILY / WEEKLY with optional BYDAY. MONTHLY/YEARLY fall back
    to interval-based math from now.
    """
    freq = (rule.get("frequency") or "DAILY").upper()
    # interval is honored implicitly via _interval_delta for non-weekly-byday paths
    byday = [d.upper() for d in (rule.get("byday") or [])]
    byday_ints = {_DAY_MAP[d] for d in byday if d in _DAY_MAP}

    today = now.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    horizon = today + timedelta(days=400)  # safety cap

    if freq == "WEEKLY" and byday_ints:
        d = today
        while d < horizon:
            if d.weekday() in byday_ints:
                return d
            d = d + timedelta(days=1)
    if freq == "DAILY":
        return today  # daily means due today every day
    if freq == "WEEKLY":
        # Weekly with no BYDAY → repeat every `interval` weeks from today
        return today
    if freq in {"MONTHLY", "YEARLY"}:
        return today + _interval_delta(rule)
    return today
