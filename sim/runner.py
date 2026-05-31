"""Scenario runner — calendar and tasks agents against in-memory backends.

Reads YAML scenario files, runs each through the appropriate agent against
in-memory state (real LLM, fake I/O), checks the result against the
declared expectations, and prints a pass/fail report.

Scenarios live in `tests/scenarios/*.yaml`. Each entry declares an
optional `kind: calendar | tasks` (default calendar). Calendar scenarios
seed `initial_calendar`; tasks scenarios seed `initial_chores` and
`initial_completions` into an in-memory SQLite DB.

Real LLM calls happen against the configured local model — slower than
unit tests (~5s/case) and non-deterministic, but they catch real prompt
and routing regressions.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import structlog
import yaml

from alfred.agents.base import AgentContext
from alfred.agents.calendar.agent import CalendarAgent, CalendarConfig
from alfred.agents.tasks.agent import TasksAgent
from alfred.routing.clients import LiteLLMClient
from scaffold.storage.database import Database
from sim.in_memory_calendar import InMemoryCalendarClient

log = structlog.get_logger()


@dataclass
class Scenario:
    """A single scenario — one email + expected outcome."""

    name: str
    email: str
    today: str  # ISO date like "2026-05-17"
    kind: str = "calendar"  # "calendar" | "tasks"
    initial_calendar: list[dict[str, Any]] = field(default_factory=list)
    initial_chores: list[dict[str, Any]] = field(default_factory=list)
    initial_completions: list[dict[str, Any]] = field(default_factory=list)
    expect: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    scenario: Scenario
    passed: bool
    failures: list[str]
    outcome: str | None
    elapsed_seconds: float
    input_tokens: int
    output_tokens: int


def load_scenarios(path: Path) -> list[Scenario]:
    """Load scenarios from a YAML file or directory of YAML files."""
    files: list[Path] = (
        sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml"))
        if path.is_dir()
        else [path]
    )

    scenarios: list[Scenario] = []
    for f in files:
        with open(f) as fh:
            raw = yaml.safe_load(fh) or []
        for entry in raw:
            scenarios.append(
                Scenario(
                    name=entry["name"],
                    email=entry["email"],
                    today=entry["today"],
                    kind=entry.get("kind", "calendar"),
                    initial_calendar=entry.get("initial_calendar") or [],
                    initial_chores=entry.get("initial_chores") or [],
                    initial_completions=entry.get("initial_completions") or [],
                    expect=entry.get("expect") or {},
                )
            )
    return scenarios


async def run_scenarios(
    scenarios: list[Scenario],
    litellm_client: LiteLLMClient,
    timezone: str = "America/Chicago",
    verbose: bool = False,
) -> list[ScenarioResult]:
    """Run a list of scenarios sequentially. Each builds its own agent +
    client pair so the in-memory calendar is fresh per case."""
    import time

    results: list[ScenarioResult] = []
    for sc in scenarios:
        tz = ZoneInfo(timezone)
        fixed_now = datetime.fromisoformat(sc.today).replace(
            hour=12, minute=0, second=0, tzinfo=tz
        )
        t0 = time.time()
        ctx = AgentContext(request_id=f"sim-{sc.name}", source="simulation")

        client: InMemoryCalendarClient | None = None
        db: Database | None = None

        try:
            if sc.kind == "tasks":
                db = await _make_in_memory_db(sc)
                agent = TasksAgent(
                    db=db,
                    litellm_client=litellm_client,
                    timezone_name=timezone,
                    now_fn=lambda fn=fixed_now: fn,
                    # Fixed persona-neutral test identities so scenarios can
                    # assert that natural-language assignee references resolve
                    # correctly without leaking real names into the repo.
                    assignee_names={"primary": "Alex", "secondary": "Sam"},
                )
                result = await agent.run(sc.email, ctx)
                failures = await _check_task_expectations(sc, result, db)
            else:
                client = InMemoryCalendarClient(
                    calendar_name="Alfred", timezone=timezone
                )
                client.seed(sc.initial_calendar)
                cfg = CalendarConfig(Path("config"))
                cfg.timezone = timezone
                agent = CalendarAgent(
                    google_client=client,
                    litellm_client=litellm_client,
                    calendar_config=cfg,
                    now_fn=lambda fn=fixed_now: fn,
                )
                result = await agent.run(sc.email, ctx)
                failures = _check_expectations(
                    scenario=sc, result=result, client=client
                )
        except Exception as e:  # noqa: BLE001
            results.append(
                ScenarioResult(
                    scenario=sc,
                    passed=False,
                    failures=[f"agent run raised: {e}"],
                    outcome=None,
                    elapsed_seconds=time.time() - t0,
                    input_tokens=0,
                    output_tokens=0,
                )
            )
            if db is not None:
                await db.close()
            continue
        finally:
            pass

        elapsed = time.time() - t0
        results.append(
            ScenarioResult(
                scenario=sc,
                passed=not failures,
                failures=failures,
                outcome=(result.data or {}).get("outcome"),
                elapsed_seconds=elapsed,
                input_tokens=result.usage.input_tokens if result.usage else 0,
                output_tokens=result.usage.output_tokens if result.usage else 0,
            )
        )
        if verbose:
            log.info(
                "scenario_done",
                name=sc.name,
                kind=sc.kind,
                passed=not failures,
                outcome=(result.data or {}).get("outcome"),
                elapsed=round(elapsed, 2),
            )
        if db is not None:
            await db.close()
    return results


async def _make_in_memory_db(sc: Scenario) -> Database:
    """Build a fresh in-memory SQLite DB and seed it with scenario chores."""
    # SQLite ":memory:" databases are per-connection. Use a uuid file-uri
    # cached in shared memory so all connections in this process can see
    # the seeded rows. For our simple usage a fresh `:memory:` is enough
    # since we use a single Database connection per scenario.
    db_path = f"file:sim-{uuid.uuid4().hex}?mode=memory&cache=shared"
    db = Database(db_path)
    # aiosqlite supports URIs but Database.__init__ doesn't pass uri=True,
    # so fall back to plain ":memory:" — it works because the Database
    # holds a single long-lived connection.
    db.db_path = ":memory:"
    await db.initialize()

    for c in sc.initial_chores:
        rec_rule = c.get("recurrence_rule") or {}
        if isinstance(rec_rule, str):
            rec_rule = json.loads(rec_rule)
        await db.execute(
            "INSERT INTO chores "
            "(id, title, description, assignee, recurrence_type, "
            " recurrence_rule, shame_after_days, active, created_at, due_date, "
            " object, qualifier) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                c["id"],
                c["title"],
                c.get("description"),
                c.get("assignee", "household"),
                c.get("recurrence_type", "schedule"),
                json.dumps(rec_rule),
                c.get("shame_after_days", 3),
                1 if c.get("active", True) else 0,
                c.get("created_at") or "2026-01-01 00:00:00",
                c.get("due_date"),
                c.get("object"),
                c.get("qualifier"),
            ),
        )
    for comp in sc.initial_completions:
        await db.execute(
            "INSERT INTO chore_completions "
            "(chore_id, completed_at, completed_by, completed_via) "
            "VALUES (?, ?, ?, ?)",
            (
                comp["chore_id"],
                comp.get("completed_at") or "2026-01-01 00:00:00",
                comp.get("completed_by", "household"),
                comp.get("completed_via", "test"),
            ),
        )
    return db


async def _check_task_expectations(
    scenario: Scenario, result: Any, db: Database
) -> list[str]:
    """Verify expectations for tasks-kind scenarios."""
    failures: list[str] = []
    expect = scenario.expect
    data = result.data or {}
    outcome = data.get("outcome")

    if "outcome" in expect and expect["outcome"] != outcome:
        failures.append(f"outcome: expected {expect['outcome']!r}, got {outcome!r}")

    if "active_count_after" in expect:
        row = await db.fetch_one(
            "SELECT COUNT(*) AS n FROM chores WHERE active = 1"
        )
        actual = row["n"] if row else 0
        if actual != expect["active_count_after"]:
            failures.append(
                f"active_count_after: expected {expect['active_count_after']}, got {actual}"
            )

    if "completion_count_after" in expect:
        row = await db.fetch_one("SELECT COUNT(*) AS n FROM chore_completions")
        actual = row["n"] if row else 0
        if actual != expect["completion_count_after"]:
            failures.append(
                f"completion_count_after: expected {expect['completion_count_after']}, got {actual}"
            )

    if "completed_chore_id" in expect:
        row = await db.fetch_one(
            "SELECT chore_id FROM chore_completions ORDER BY id DESC LIMIT 1"
        )
        actual = row["chore_id"] if row else None
        if actual != expect["completed_chore_id"]:
            failures.append(
                f"completed_chore_id: expected {expect['completed_chore_id']!r}, got {actual!r}"
            )

    if "chore_active" in expect:
        for chore_id, expected_active in expect["chore_active"].items():
            row = await db.fetch_one(
                "SELECT active FROM chores WHERE id = ?", (chore_id,)
            )
            actual = bool(row["active"]) if row else False
            if actual != expected_active:
                failures.append(
                    f"chore_active[{chore_id}]: expected {expected_active}, got {actual}"
                )

    reply = result.message or ""
    if "reply_contains" in expect:
        for needle in expect["reply_contains"]:
            if needle.lower() not in reply.lower():
                failures.append(f"reply_contains: {needle!r} not in reply")
    if "reply_excludes" in expect:
        for needle in expect["reply_excludes"]:
            if needle.lower() in reply.lower():
                failures.append(f"reply_excludes: {needle!r} found in reply")

    if (
        "max_input_tokens" in expect
        and result.usage
        and result.usage.input_tokens > expect["max_input_tokens"]
    ):
        failures.append(
            f"max_input_tokens: used {result.usage.input_tokens} "
            f"(cap {expect['max_input_tokens']})"
        )

    return failures


def _check_expectations(
    scenario: Scenario,
    result: Any,
    client: InMemoryCalendarClient,
) -> list[str]:
    """Return a list of failure strings — empty list means the scenario passed."""
    failures: list[str] = []
    expect = scenario.expect
    data = result.data or {}
    outcome = data.get("outcome")

    if "outcome" in expect and expect["outcome"] != outcome:
        failures.append(f"outcome: expected {expect['outcome']!r}, got {outcome!r}")

    if "calendar_count_after" in expect:
        actual = len(client.events)
        if actual != expect["calendar_count_after"]:
            failures.append(
                f"calendar_count_after: expected {expect['calendar_count_after']}, got {actual}"
            )

    created = data.get("event") or {}
    if "event_summary_contains" in expect:
        summary = (created.get("summary") or "").lower()
        for needle in expect["event_summary_contains"]:
            if needle.lower() not in summary:
                failures.append(
                    f"event_summary_contains: {needle!r} not in {created.get('summary')!r}"
                )
    if "event_summary_excludes" in expect:
        summary = (created.get("summary") or "").lower()
        for needle in expect["event_summary_excludes"]:
            if needle.lower() in summary:
                failures.append(
                    f"event_summary_excludes: {needle!r} found in {created.get('summary')!r}"
                )
    if (
        "event_start_iso" in expect
        and created
        and created.get("start_iso") != expect["event_start_iso"]
    ):
        failures.append(
            f"event_start_iso: expected {expect['event_start_iso']}, "
            f"got {created.get('start_iso')}"
        )
    if "has_recurrence" in expect:
        has = bool(created.get("recurrence"))
        if has != expect["has_recurrence"]:
            failures.append(
                f"has_recurrence: expected {expect['has_recurrence']}, got {has}"
            )

    reply = result.message or ""
    if "reply_contains" in expect:
        for needle in expect["reply_contains"]:
            if needle.lower() not in reply.lower():
                failures.append(f"reply_contains: {needle!r} not in reply")
    if "reply_excludes" in expect:
        for needle in expect["reply_excludes"]:
            if needle.lower() in reply.lower():
                failures.append(f"reply_excludes: {needle!r} found in reply")

    if (
        "max_input_tokens" in expect
        and result.usage
        and result.usage.input_tokens > expect["max_input_tokens"]
    ):
        failures.append(
            f"max_input_tokens: used {result.usage.input_tokens} "
            f"(cap {expect['max_input_tokens']})"
        )

    return failures


def print_report(results: list[ScenarioResult]) -> None:
    """Pretty-print a pass/fail table to stdout."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    print()
    print(f"Ran {total} scenarios — {passed} passed, {failed} failed")
    print("─" * 80)

    name_w = max((len(r.scenario.name) for r in results), default=20)
    name_w = max(name_w, 20)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        outcome = r.outcome or "—"
        print(
            f"  [{status}]  {r.scenario.name:<{name_w}}  "
            f"outcome={outcome:<18}  {r.elapsed_seconds:>5.2f}s  "
            f"tok={r.input_tokens:>5}/{r.output_tokens:<4}"
        )
        for fail in r.failures:
            print(f"            └─ {fail}")
    print()
