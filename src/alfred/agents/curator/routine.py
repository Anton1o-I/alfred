"""Curator end-to-end routine: generate digest, then email it.

Registered with the RoutineRegistry under the name "curator". Wraps
`CuratorAgent.generate_digest` plus the email-delivery step. The
parent OTel span ensures the whole run lands in one Phoenix trace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from opentelemetry import trace

from alfred.agents.curator import CuratorAgent
from alfred.agents.curator.email import email_curator_digest

if TYPE_CHECKING:
    from alfred.app import App

log = structlog.get_logger()
_tracer = trace.get_tracer("alfred.curator.routine")


async def run_curator_routine(app: App) -> dict:
    """Fetch + triage + synthesize + render + email. Single Phoenix trace."""
    agent = app.agent_registry.get("curator")
    if not isinstance(agent, CuratorAgent):
        log.error("curator_not_registered")
        return {"status": "skipped", "reason": "curator_not_registered"}

    request_id = "curator-routine"
    with _tracer.start_as_current_span("routine.curator") as span:
        span.set_attribute("alfred.task", "curator")
        result = await agent.generate_digest(
            model_name="cloud-default",
            request_id=request_id,
            compare_with=None,
        )
        path = result.data.get("path")
        if not path:
            return {"status": "no_digest"}
        await email_curator_digest(
            app,
            path,
            html_body=result.data.get("html"),
            request_id=request_id,
        )
    return {"status": "success", "path": path}
