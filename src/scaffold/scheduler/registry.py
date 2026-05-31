"""RoutineRegistry — pluggable dispatch table for scheduled-task names.

Generic infrastructure: agent systems register their routines at startup
in `app.py`, and the HTTP routine endpoint (or any other trigger source)
dispatches by name through this registry. Replaces the hardcoded if-chain
in the legacy `scheduler/runner.py:run_routine`.

A "routine" is any async callable that takes an App and an optional mode
string. Some routines (like a daily briefing) are single-shot; others
(like a "briefing" pseudo-agent) branch on a sub-mode passed through the
trigger body. The registry supports both shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable

import structlog

if TYPE_CHECKING:
    from scaffold.app_protocol import AppProtocol

log = structlog.get_logger()

RoutineFn = Callable[["AppProtocol"], Awaitable[dict | None]]
ModeRoutineFn = Callable[["AppProtocol"], Awaitable[dict | None]]


class RoutineNotFoundError(KeyError):
    """Raised when a trigger names a routine that wasn't registered."""


class RoutineRegistry:
    """Name → callable lookup for scheduled / triggered routines.

    Two registration shapes:
      - `register(name, fn)` for a single-shot routine.
      - `register_modes(name, {mode: fn, ...})` for a pseudo-agent that
        branches on a `mode` string passed in the trigger body. Used by
        the `briefing` family (daily / morning / weekly).
    """

    def __init__(self) -> None:
        self._routines: dict[str, RoutineFn] = {}
        self._mode_tables: dict[str, dict[str, ModeRoutineFn]] = {}

    def register(self, name: str, fn: RoutineFn) -> None:
        if name in self._routines or name in self._mode_tables:
            raise ValueError(f"Routine '{name}' already registered")
        self._routines[name] = fn
        log.info("routine_registered", name=name, kind="single")

    def register_modes(
        self, name: str, modes: dict[str, ModeRoutineFn]
    ) -> None:
        if name in self._routines or name in self._mode_tables:
            raise ValueError(f"Routine '{name}' already registered")
        if not modes:
            raise ValueError(f"Routine '{name}' needs at least one mode")
        self._mode_tables[name] = dict(modes)
        log.info("routine_registered", name=name, kind="modes", modes=list(modes))

    def list_names(self) -> list[str]:
        """All registered routine names (single-shot + mode-bearing)."""
        return sorted([*self._routines.keys(), *self._mode_tables.keys()])

    def modes_for(self, name: str) -> list[str] | None:
        """Return the mode names for a mode-bearing routine, else None."""
        table = self._mode_tables.get(name)
        return sorted(table.keys()) if table else None

    async def dispatch(
        self,
        app: "AppProtocol",
        name: str,
        mode: str | None = None,
    ) -> dict | None:
        """Run the routine registered for `name`. Returns whatever it returns.

        Raises `RoutineNotFoundError` if `name` is unknown or if a
        mode-bearing routine is invoked without a (valid) mode.
        """
        if name in self._routines:
            if mode:
                log.warning(
                    "routine_mode_ignored",
                    name=name,
                    mode=mode,
                    reason="single-shot routine does not accept modes",
                )
            return await self._routines[name](app)

        table = self._mode_tables.get(name)
        if table is None:
            raise RoutineNotFoundError(name)
        if not mode:
            raise RoutineNotFoundError(
                f"Routine '{name}' requires a mode (one of: {sorted(table)})"
            )
        fn = table.get(mode)
        if fn is None:
            raise RoutineNotFoundError(
                f"Routine '{name}' has no mode '{mode}' "
                f"(available: {sorted(table)})"
            )
        return await fn(app)
