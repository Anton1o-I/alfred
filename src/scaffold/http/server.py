"""FastAPI app exposing the routine-trigger surface for n8n (or any caller).

Endpoints:
  GET  /healthz           — liveness probe
  GET  /routines          — list registered routine names + modes
  POST /routines/{name}   — dispatch the named routine in the background

Auth: every request must carry `X-Alfred-Token: <shared_secret>` matching
the value the server was started with. Misses return 401. The secret is
read at startup from an env var so it never lives in code; n8n sends
the same value from its HTTP Request node.

Long-running routines (curator can take 5+ minutes) are dispatched via
FastAPI's BackgroundTasks so the HTTP response returns within
milliseconds — n8n's HTTP node sees a 202 and moves on. Failure of the
background task is visible in Alfred's logs / Phoenix / notification_log,
not in the HTTP response.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, status
from pydantic import BaseModel

from scaffold.scheduler.registry import RoutineNotFoundError

if TYPE_CHECKING:
    from scaffold.app_protocol import AppProtocol

log = structlog.get_logger()


class TriggerBody(BaseModel):
    """Optional payload for POST /routines/{name}.

    `mode` selects the sub-routine for mode-bearing entries (e.g.
    "briefing" → "daily" / "morning" / "weekly"). Any other keys are
    ignored today; reserved for future per-call parameters.
    """

    mode: str | None = None


def create_routine_app(app: "AppProtocol", api_token: str) -> FastAPI:
    """Build the FastAPI app bound to a particular wired application.

    Pass the already-constructed Alfred (or other agent system) App; the
    handlers reach into `app.routine_registry` to dispatch.
    """
    if not api_token:
        raise ValueError(
            "create_routine_app requires a non-empty api_token; set "
            "ALFRED_ROUTINE_API_KEY in .env"
        )

    fastapi_app = FastAPI(
        title="Agent routine trigger",
        version="1.0",
        docs_url="/docs",
        redoc_url=None,
    )

    def _require_token(token: str | None) -> None:
        if not token or token != api_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid X-Alfred-Token",
            )

    @fastapi_app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @fastapi_app.get("/routines")
    async def list_routines(
        x_alfred_token: str | None = Header(default=None),
    ) -> dict[str, list[dict[str, object]]]:
        _require_token(x_alfred_token)
        registry = app.routine_registry
        out: list[dict[str, object]] = []
        for name in registry.list_names():
            entry: dict[str, object] = {"name": name}
            modes = registry.modes_for(name)
            if modes is not None:
                entry["modes"] = modes
            out.append(entry)
        return {"routines": out}

    @fastapi_app.post(
        "/routines/{name}",
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def trigger_routine(
        name: str,
        background: BackgroundTasks,
        body: TriggerBody | None = None,
        x_alfred_token: str | None = Header(default=None),
    ) -> dict[str, str]:
        _require_token(x_alfred_token)
        registry = app.routine_registry
        mode = body.mode if body else None

        if name not in registry.list_names():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown routine '{name}'",
            )
        modes = registry.modes_for(name)
        if modes is not None and (not mode or mode not in modes):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"routine '{name}' requires a mode in {modes}; "
                    f"got {mode!r}"
                ),
            )

        log.info("routine_triggered", name=name, mode=mode)
        background.add_task(_run_routine_safely, app, name, mode)
        return {"status": "accepted", "routine": name}

    return fastapi_app


async def _run_routine_safely(
    app: "AppProtocol", name: str, mode: str | None
) -> None:
    """Run a routine and swallow exceptions — the HTTP response is long gone."""
    try:
        await app.routine_registry.dispatch(app, name, mode)
        log.info("routine_completed", name=name, mode=mode)
    except RoutineNotFoundError as e:
        log.error("routine_dispatch_not_found", name=name, mode=mode, error=str(e))
    except Exception as e:  # noqa: BLE001
        log.exception("routine_failed", name=name, mode=mode, error=str(e))
