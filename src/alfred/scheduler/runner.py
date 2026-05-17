"""APScheduler-based runner for cron-style scheduled routines."""

from __future__ import annotations

import asyncio
import signal
from typing import TYPE_CHECKING

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from alfred.core.constants import RequestSource
from alfred.core.models import AgentRequest

if TYPE_CHECKING:
    from alfred.app import App
    from alfred.core.config import ScheduledTaskConfig

log = structlog.get_logger()


async def run_routine(app: App, task: ScheduledTaskConfig, compare: bool = False) -> None:
    """Execute a single scheduled routine through the orchestrator (or direct, for compare)."""
    log.info("routine_start", task=task.name, agent=task.agent_name, compare=compare)

    # Special path: curator A/B compare bypasses the orchestrator and calls the
    # agent's generate_digest() directly, since the orchestrator only routes
    # plain user messages.
    if compare and task.agent_name == "curator":
        from alfred.agents.curator import CuratorAgent

        agent = app.agent_registry.get("curator")
        if not isinstance(agent, CuratorAgent):
            log.error("curator_not_registered_for_compare")
            return
        result = await agent.generate_digest(
            model_name="local-default",
            request_id="compare-run",
            compare_with="cloud-default",
        )
        log.info("routine_finish", task=task.name, status="success", path=result.data.get("path"))
        return

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

    for task in enabled:
        if app.agent_registry.get(task.agent_name) is None:
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


async def run_routine_by_name(app: App, name: str, compare: bool = False) -> int:
    """Fire a single named routine once. Returns 0 on success, 1 on error."""
    for task in app.settings.scheduled_tasks:
        if task.name == name:
            await run_routine(app, task, compare=compare)
            return 0
    log.error("routine_not_found", name=name)
    return 1
