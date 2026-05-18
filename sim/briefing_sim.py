"""Snapshot simulator for the daily briefing email.

Runs the real render path (incl. real local LLM narrative) against a fixed
set of (events, chores) fixtures, dumps `.md` (plain) + `.html` per scenario
to `data/sim_briefings/<scenario>.{md,html}`. Lets us eyeball the actual
rendered output before deploying changes — the user requested this loop:
build → simulate → review → iterate.

Run it with: `uv run alfred sim briefing`.

The fixtures are intentionally hand-written, not loaded from YAML, because
they're cheap to read inline and there are few of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import structlog

from alfred.agents.calendar.briefing import (
    _DAILY_PROMPT,
    DailyNarrative,
    _make_narrative_agent,
    _render_chores_for_prompt,
    _render_email,
    _render_events_html,
    _render_events_plain,
    analyze_day,
)
from alfred.agents.tasks.store import Chore, ChoreStatus
from alfred.notifications.tasks_render import (
    categorize_statuses,
    render_chores_html,
    render_chores_plain,
)
from alfred.routing.clients import LiteLLMClient

log = structlog.get_logger()


# ── Fixture builders ──────────────────────────────────────────────────────


def _event(
    *,
    summary: str,
    start: str,
    end: str,
    location: str | None = None,
) -> dict[str, Any]:
    """Build a flattened event dict matching what `_fetch_events_for_window` emits."""
    return {
        "start_iso": start,
        "end_iso": end,
        "summary": summary,
        "location": location,
    }


def _chore(
    *,
    title: str,
    assignee: str,
    overdue_days: int,
    shame_after_days: int = 3,
) -> ChoreStatus:
    """Build a ChoreStatus directly — bypasses the store for sim purposes."""
    now = datetime.now(ZoneInfo("America/Phoenix"))
    return ChoreStatus(
        chore=Chore(
            id=f"sim-{title.lower().replace(' ', '-')}",
            title=title,
            description=None,
            assignee=assignee,
            recurrence_type="schedule",
            recurrence_rule={"every_days": 7},
            shame_after_days=shame_after_days,
            active=True,
            created_at=now.isoformat(),
        ),
        last_completed_at=None,
        next_due=now - timedelta(days=overdue_days),
        overdue_days=overdue_days,
    )


@dataclass
class Scenario:
    name: str
    description: str
    tomorrow: date
    events: list[dict[str, Any]]
    chores: list[ChoreStatus]


def _build_scenarios() -> list[Scenario]:
    """Six scenarios spanning the lead-section decision space."""
    tz = ZoneInfo("America/Phoenix")
    tomorrow = (datetime.now(tz) + timedelta(days=1)).date()

    def at(hour: int, minute: int = 0) -> str:
        dt = datetime.combine(tomorrow, datetime.min.time(), tzinfo=tz).replace(
            hour=hour, minute=minute
        )
        return dt.isoformat()

    return [
        Scenario(
            name="01_empty_calendar_overdue_chores",
            description="No events tomorrow; two chores are overdue.",
            tomorrow=tomorrow,
            events=[],
            chores=[
                _chore(title="Take out trash", assignee="alex", overdue_days=4),
                _chore(title="Water plants", assignee="alex", overdue_days=2),
            ],
        ),
        Scenario(
            name="02_empty_calendar_due_today",
            description="No events; three chores due today.",
            tomorrow=tomorrow,
            events=[],
            chores=[
                _chore(title="Empty dishwasher", assignee="household", overdue_days=0),
                _chore(title="Laundry", assignee="alex", overdue_days=0),
                _chore(title="Sweep kitchen", assignee="household", overdue_days=0),
            ],
        ),
        Scenario(
            name="03_busy_day_no_chores",
            description="Five back-to-back events; no chores tracked.",
            tomorrow=tomorrow,
            events=[
                _event(summary="Team standup", start=at(9), end=at(9, 30)),
                _event(
                    summary="1:1 with Jamie",
                    start=at(10),
                    end=at(10, 45),
                    location="Zoom",
                ),
                _event(summary="Design review", start=at(11), end=at(12)),
                _event(
                    summary="Lunch with Pat",
                    start=at(12, 30),
                    end=at(13, 30),
                    location="Cafe",
                ),
                _event(summary="Sprint planning", start=at(14), end=at(15, 30)),
            ],
            chores=[],
        ),
        Scenario(
            name="04_busy_day_one_overdue",
            description="Three events + one overdue chore.",
            tomorrow=tomorrow,
            events=[
                _event(summary="Dentist", start=at(8), end=at(9), location="Dental Smiles"),
                _event(summary="Project sync", start=at(11), end=at(12)),
                _event(summary="Soccer practice", start=at(17), end=at(18, 30)),
            ],
            chores=[
                _chore(title="Replace HVAC filter", assignee="alex", overdue_days=6),
            ],
        ),
        Scenario(
            name="05_light_day_few_chores",
            description="One short event; chores due today and tomorrow.",
            tomorrow=tomorrow,
            events=[
                _event(summary="Coffee with Sam", start=at(15), end=at(15, 45)),
            ],
            chores=[
                _chore(title="Pay credit card", assignee="alex", overdue_days=0),
                _chore(title="Pick up dry cleaning", assignee="alex", overdue_days=-1),
            ],
        ),
        Scenario(
            name="06_packed_day_multiple_overdue",
            description="Six events with a tight transition + 3 overdue chores.",
            tomorrow=tomorrow,
            events=[
                _event(summary="Gym", start=at(7), end=at(8)),
                _event(summary="Sprint review", start=at(9), end=at(10)),
                _event(summary="Customer call", start=at(10, 5), end=at(10, 45)),
                _event(summary="Architecture sync", start=at(11), end=at(12, 30)),
                _event(summary="School pickup", start=at(15), end=at(15, 30)),
                _event(summary="Dinner with family", start=at(18, 30), end=at(20)),
            ],
            chores=[
                _chore(title="Take out trash", assignee="alex", overdue_days=5),
                _chore(title="Reply to landlord", assignee="alex", overdue_days=3),
                _chore(title="Refill prescription", assignee="alex", overdue_days=1),
            ],
        ),
    ]


# ── Runner ────────────────────────────────────────────────────────────────


async def _render_one(
    scenario: Scenario,
    litellm_client: LiteLLMClient,
    tz: ZoneInfo,
) -> tuple[str, str, DailyNarrative]:
    """Run one scenario through the real prompt + render path."""
    analysis = analyze_day(scenario.events, tz)
    categorized = categorize_statuses(scenario.chores)

    agent = _make_narrative_agent(litellm_client, DailyNarrative, "local-default")
    prompt = _DAILY_PROMPT.format(
        label=scenario.tomorrow.strftime("%A, %B %-d, %Y"),
        tz=tz.key,
        event_count=analysis["event_count"],
        events_block=_render_events_plain(scenario.events, tz),
        total_minutes=analysis["total_minutes"],
        first_start=analysis["first_start"] or "—",
        last_end=analysis["last_end"] or "—",
        open_blocks=analysis["open_blocks"] or "(none)",
        tight_transitions=analysis["tight_transitions"] or "(none)",
        chores_block=_render_chores_for_prompt(categorized),
    )
    result = await agent.run(prompt)
    narrative = result.output

    has_chores = any(
        categorized[k] for k in ("overdue", "due_today", "due_tomorrow")
    )
    chores_plain = render_chores_plain(categorized) if has_chores else None
    chores_html = render_chores_html(categorized) if has_chores else None

    headline = f"TOMORROW · {scenario.tomorrow.strftime('%a, %B %-d')}"
    plain, html = _render_email(
        headline=headline,
        narrative=narrative,
        events_section_plain=_render_events_plain(scenario.events, tz),
        events_section_html=_render_events_html(scenario.events, tz),
        observations=narrative.observations,
        chores_section_plain=chores_plain,
        chores_section_html=chores_html,
        lead_section=narrative.lead_section,
    )
    return plain, html, narrative


async def run_briefing_sim(output_dir: Path | None = None) -> int:
    """Render every scenario, write outputs to disk, print a summary.

    Returns 0 on success, 1 if any scenario errored.
    """
    out = output_dir or Path("data/sim_briefings")
    out.mkdir(parents=True, exist_ok=True)

    litellm_client = LiteLLMClient()  # picks up LITELLM_BASE_URL + LITELLM_MASTER_KEY from env
    tz = ZoneInfo("America/Phoenix")

    scenarios = _build_scenarios()
    failures = 0
    print(f"Rendering {len(scenarios)} briefing scenarios → {out}/")
    print()
    for s in scenarios:
        try:
            plain, html, narrative = await _render_one(s, litellm_client, tz)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {s.name}: {e}")
            failures += 1
            continue
        (out / f"{s.name}.md").write_text(plain)
        (out / f"{s.name}.html").write_text(html)
        lead = narrative.lead_section
        n_obs = len(narrative.observations)
        print(
            f"  ✓ {s.name:<40s}  lead={lead:<6s}  obs={n_obs}  "
            f"greeting={narrative.greeting[:40]!r}"
        )
    print()
    print(f"Done. {len(scenarios) - failures}/{len(scenarios)} ok.")
    print(f"Outputs: {out}/")
    return 1 if failures else 0
