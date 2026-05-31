"""Curator digest email step.

Lives next to the curator agent because it's curator-specific behavior:
the subject line, persona name, tagline, and recipient routing tag all
encode curator knowledge that the generic notification service shouldn't
have. Moved out of the legacy scheduler so the agent owns its delivery.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from opentelemetry import trace

from alfred.signature import append_to_body, append_to_html, render_signature

if TYPE_CHECKING:
    from alfred.app import App

log = structlog.get_logger()
_tracer = trace.get_tracer("alfred.curator.email")


async def email_curator_digest(
    app: App,
    digest_path: str,
    html_body: str | None,
    request_id: str,
) -> None:
    """Send the curator's digest to all family recipients (multipart text + html)."""
    with _tracer.start_as_current_span("curator.email") as span:
        span.set_attribute("alfred.request_id", request_id)
        span.set_attribute("alfred.curator.digest_path", digest_path)
        span.set_attribute("alfred.curator.html_present", bool(html_body))

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
        successes = sum(1 for r in results if r.success)
        failures = sum(1 for r in results if not r.success)
        span.set_attribute("alfred.curator.recipients", len(results))
        span.set_attribute("alfred.curator.delivered", successes)
        span.set_attribute("alfred.curator.failed", failures)
        for r in results:
            if not r.success:
                log.warning(
                    "curator_email_failed", channel=r.channel, error=r.error
                )
