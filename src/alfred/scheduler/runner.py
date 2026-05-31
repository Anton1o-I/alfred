"""APScheduler runner — legacy in-process cron, being retired in favor of n8n.

Both paths now share the same RoutineRegistry dispatch, so behavior is
identical whether the trigger comes from APScheduler or an n8n HTTP
POST. Once n8n has been running cleanly for a cutover window, this
module and `alfred-scheduler.service` get deleted.
"""

from __future__ import annotations

import asyncio
import signal
from typing import TYPE_CHECKING

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from scaffold.core.constants import RequestSource
from scaffold.core.models import AgentRequest

if TYPE_CHECKING:
    from alfred.app import App
    from scaffold.core.config import ScheduledTaskConfig

log = structlog.get_logger()


async def run_routine(
    app: App,
    task: ScheduledTaskConfig,
    compare: bool = False,  # noqa: ARG001 — kept for CLI compat; ignored
    max_items: int | None = None,  # noqa: ARG001 — kept for CLI compat; ignored
) -> None:
    """Execute a scheduled task by name.

    For tasks whose agent_name is registered as a routine, dispatch via
    the RoutineRegistry. For everything else (un-registered agent names),
    fall back to the orchestrator path used by CLI/iMessage requests.
    """
    log.info("routine_start", task=task.name, agent=task.agent_name)
    registry = app.routine_registry

    if task.agent_name in registry.list_names():
        mode = (task.message or "").strip().lower() or None
        try:
            result = await registry.dispatch(app, task.agent_name, mode)
        except Exception as e:  # noqa: BLE001
            log.exception(
                "routine_failed", task=task.name, agent=task.agent_name, error=str(e)
            )
            return
        log.info(
            "routine_finish",
            task=task.name,
            agent=task.agent_name,
            status="success",
            **(result or {}),
        )
        return

    # Generic fallback for agents not registered as a routine: send through
    # the orchestrator the same way a CLI or iMessage request would.
    request = AgentRequest(
        user_message=task.message,
        source=RequestSource.SCHEDULER,
        user_id=None,
    )
    response = await app.orchestrator.handle(request)
    log.info(
        "routine_finish",
        task=task.name,
        agent=task.agent_name,
        status=response.status,
        request_id=response.request_id[:8],
    )


def build_scheduler(app: App) -> AsyncIOScheduler:
    """Build an AsyncIOScheduler with one cron job per enabled task in settings."""
    scheduler = AsyncIOScheduler()

    enabled = [t for t in app.settings.scheduled_tasks if t.enabled]
    if not enabled:
        log.warning("scheduler_no_enabled_tasks")
        return scheduler

    known_routines = set(app.routine_registry.list_names())

    for task in enabled:
        if (
            task.agent_name not in known_routines
            and app.agent_registry.get(task.agent_name) is None
        ):
            log.warning(
                "scheduler_skip_task",
                task=task.name,
                reason="agent_not_registered",
                agent=task.agent_name,
            )
            continue

        scheduler.add_job(
            run_routine,
            trigger=CronTrigger.from_crontab(task.cron),
            args=[app, task],
            id=task.name,
            name=task.name,
            misfire_grace_time=300,
            coalesce=True,
            max_instances=1,
        )
        log.info("scheduler_task_registered", task=task.name, cron=task.cron)

    return scheduler


async def run_scheduler(app: App) -> None:
    """Run the scheduler in foreground until SIGINT/SIGTERM."""
    scheduler = build_scheduler(app)
    scheduler.start()
    log.info("scheduler_started", jobs=[j.id for j in scheduler.get_jobs()])

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    try:
        await stop.wait()
    finally:
        log.info("scheduler_stopping")
        scheduler.shutdown(wait=False)


async def run_routine_by_name(
    app: App,
    name: str,
    compare: bool = False,  # noqa: ARG001 — kept for CLI compat
    max_items: int | None = None,  # noqa: ARG001 — kept for CLI compat
) -> int:
    """Fire a single scheduled task once by name. Returns 0 on success, 1 on error."""
    for task in app.settings.scheduled_tasks:
        if task.name == name:
            await run_routine(app, task)
            return 0
    log.error("routine_not_found", name=name)
    return 1
