"""Inbox-poll routine — registered with RoutineRegistry under "inbox"."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from alfred.app import App

log = structlog.get_logger()


async def run_inbox_poll(app: App) -> dict:
    """Fetch unread messages, dispatch each to the orchestrator, send replies."""
    if app.inbox_poller is None:
        log.warning("inbox_poller_not_configured")
        return {"status": "skipped", "reason": "not_configured"}
    counts = await app.inbox_poller.process_once(app)
    return {"status": "success", **counts}
