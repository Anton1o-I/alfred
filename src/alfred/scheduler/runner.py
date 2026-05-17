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


async def _email_curator_digest(
    app: App,
    digest_path: str,
    html_body: str | None,
    request_id: str,
) -> None:
    """Send the curator's digest to all family recipients (multipart text + html)."""
    from datetime import UTC, datetime
    from pathlib import Path

    from alfred.notifications.signature import (
        append_to_body,
        append_to_html,
        render_signature,
    )

    body = Path(digest_path).read_text()
    subject = f"Alfred Research Digest — {datetime.now(UTC).strftime('%Y-%m-%d')}"
    cfg = app.settings.notifications
    from_name = cfg.email_from_names_by_agent.get("curator")
    tagline = cfg.email_taglines_by_agent.get("curator", "")
    sig_plain, sig_html = render_signature(from_name or "", tagline)
    body = append_to_body(body, sig_plain)
    html_body_with_sig = append_to_html(html_body, sig_html) if html_body else None

    results = await app.notification_service.send_to_family(
        body=body,
        subject=subject,
        html_body=html_body_with_sig,
        request_id=request_id,
        from_name=from_name,
        agent_name="curator",
    )
    for r in results:
        if not r.success:
            log.warning(
                "curator_email_failed", channel=r.channel, error=r.error
            )


async def run_routine(
    app: App,
    task: ScheduledTaskConfig,
    compare: bool = False,
    max_items: int | None = None,
) -> None:
    """Execute a single scheduled routine through the orchestrator (or direct, for compare)."""
    log.info("routine_start", task=task.name, agent=task.agent_name, compare=compare)

    # Inbox poller: fetch unread messages, dispatch to orchestrator, reply.
    if task.agent_name == "inbox":
        if app.inbox_poller is None:
            log.warning("inbox_poller_not_configured", task=task.name)
            return
        counts = await app.inbox_poller.process_once(app)
        log.info("routine_finish", task=task.name, status="success", **counts)
        return

    # Calendar briefings: daily next-day reminder + Sunday week preview.
    # `task.message` selects the mode ("daily" | "weekly").
    if task.agent_name == "briefing":
        from alfred.agents.calendar.briefing import (
            run_daily_briefing,
            run_weekly_preview,
        )

        mode = (task.message or "").strip().lower()
        if mode == "daily":
            result = await run_daily_briefing(app)
        elif mode == "weekly":
            result = await run_weekly_preview(app)
        else:
            log.warning("briefing_unknown_mode", task=task.name, mode=mode)
            return
        log.info("routine_finish", task=task.name, status="success", **result)
        return

    # Curator goes direct (not through orchestrator) so we can email the
    # actual digest body with a proper subject. Non-compare runs also email;
    # compare runs just write the A/B file (review-only, no email).
    if task.agent_name == "curator":
        from alfred.agents.curator import CuratorAgent

        agent = app.agent_registry.get("curator")
        if not isinstance(agent, CuratorAgent):
            log.error("curator_not_registered")
            return
        result = await agent.generate_digest(
            model_name="cloud-default",
            request_id=f"{task.name}-direct",
            compare_with="local-default" if compare else None,
            max_candidates=max_items,
        )
        path = result.data.get("path")
        log.info("routine_finish", task=task.name, status="success", path=path)
        if not compare and path:
            await _email_curator_digest(
                app,
                path,
                html_body=result.data.get("html"),
                request_id=f"{task.name}-direct",
            )
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

    # Pseudo-agents that route through dedicated paths in run_routine
    # rather than the agent registry.
    pseudo_agents = {"inbox", "briefing"}

    for task in enabled:
        if (
            task.agent_name not in pseudo_agents
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
    app: App, name: str, compare: bool = False, max_items: int | None = None
) -> int:
    """Fire a single named routine once. Returns 0 on success, 1 on error."""
    for task in app.settings.scheduled_tasks:
        if task.name == name:
            await run_routine(app, task, compare=compare, max_items=max_items)
            return 0
    log.error("routine_not_found", name=name)
    return 1
