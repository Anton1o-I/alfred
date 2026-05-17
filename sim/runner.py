"""Calendar scenario runner.

Reads YAML scenario files, runs each through the calendar agent against
an in-memory calendar (real LLM, fake I/O), checks the result against the
declared expectations, and prints a pass/fail report.

Scenarios live in `tests/scenarios/*.yaml`. Real LLM calls happen against
the configured local model — they're slower than unit tests (~5s/case)
and non-deterministic, but they catch real prompt and routing regressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import structlog
import yaml

from alfred.agents.base import AgentContext
from alfred.agents.calendar.agent import CalendarAgent, CalendarConfig
from alfred.routing.clients import LiteLLMClient
from sim.in_memory_calendar import InMemoryCalendarClient

log = structlog.get_logger()


@dataclass
class Scenario:
    """A single scenario — one email + expected outcome."""

    name: str
    email: str
    today: str  # ISO date like "2026-05-17"
    initial_calendar: list[dict[str, Any]] = field(default_factory=list)
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
                    initial_calendar=entry.get("initial_calendar") or [],
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
        client = InMemoryCalendarClient(calendar_name="Alfred", timezone=timezone)
        client.seed(sc.initial_calendar)

        cfg = CalendarConfig(Path("config"))
        cfg.timezone = timezone

        # Pin "today" to noon local on the given date so relative-date
        # reasoning resolves deterministically across runs.
        tz = ZoneInfo(timezone)
        fixed_now = datetime.fromisoformat(sc.today).replace(
            hour=12, minute=0, second=0, tzinfo=tz
        )
        agent = CalendarAgent(
            google_client=client,
            litellm_client=litellm_client,
            calendar_config=cfg,
            now_fn=lambda fn=fixed_now: fn,
        )

        t0 = time.time()
        ctx = AgentContext(request_id=f"sim-{sc.name}", source="simulation")
        try:
            result = await agent.run(sc.email, ctx)
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
            continue
        elapsed = time.time() - t0

        failures = _check_expectations(
            scenario=sc, result=result, client=client
        )
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
                passed=not failures,
                outcome=(result.data or {}).get("outcome"),
                elapsed=round(elapsed, 2),
            )
    return results


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
