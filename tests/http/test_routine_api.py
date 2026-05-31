"""Tests for the routine-trigger HTTP surface.

Covers: auth gating, listing, single-shot dispatch, mode dispatch,
unknown routine, missing mode. Uses FastAPI's TestClient against a
stub App that only carries a RoutineRegistry — the rest of the App
surface isn't touched by these endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from scaffold.http.server import create_routine_app
from scaffold.scheduler.registry import RoutineRegistry


@dataclass
class _StubApp:
    routine_registry: RoutineRegistry


@pytest.fixture
def registry() -> RoutineRegistry:
    r = RoutineRegistry()
    r.calls = []  # type: ignore[attr-defined]

    async def curator(app: Any) -> dict:  # noqa: ARG001
        r.calls.append(("curator", None))  # type: ignore[attr-defined]
        return {"ran": "curator"}

    async def daily(app: Any) -> dict:  # noqa: ARG001
        r.calls.append(("briefing", "daily"))  # type: ignore[attr-defined]
        return {"ran": "daily"}

    async def morning(app: Any) -> dict:  # noqa: ARG001
        r.calls.append(("briefing", "morning"))  # type: ignore[attr-defined]
        return {"ran": "morning"}

    r.register("curator", curator)
    r.register_modes("briefing", {"daily": daily, "morning": morning})
    return r


@pytest.fixture
def client(registry: RoutineRegistry) -> TestClient:
    app = create_routine_app(_StubApp(routine_registry=registry), api_token="secret")
    # raise_server_exceptions=False so background-task errors don't fail the test
    return TestClient(app, raise_server_exceptions=False)


def test_create_app_requires_token() -> None:
    with pytest.raises(ValueError, match="api_token"):
        create_routine_app(_StubApp(routine_registry=RoutineRegistry()), api_token="")


def test_healthz_no_auth(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_routines_requires_token(client: TestClient) -> None:
    assert client.get("/routines").status_code == 401
    assert client.get("/routines", headers={"X-Alfred-Token": "wrong"}).status_code == 401


def test_list_routines_with_token(client: TestClient) -> None:
    r = client.get("/routines", headers={"X-Alfred-Token": "secret"})
    assert r.status_code == 200
    body = r.json()
    names = [e["name"] for e in body["routines"]]
    assert names == ["briefing", "curator"]
    briefing = next(e for e in body["routines"] if e["name"] == "briefing")
    assert briefing["modes"] == ["daily", "morning"]
    curator = next(e for e in body["routines"] if e["name"] == "curator")
    assert "modes" not in curator


def test_trigger_requires_token(client: TestClient) -> None:
    r = client.post("/routines/curator")
    assert r.status_code == 401


def test_trigger_unknown_routine_404(client: TestClient) -> None:
    r = client.post(
        "/routines/nope", headers={"X-Alfred-Token": "secret"}
    )
    assert r.status_code == 404


def test_trigger_single_shot_accepted(
    client: TestClient, registry: RoutineRegistry
) -> None:
    r = client.post(
        "/routines/curator", headers={"X-Alfred-Token": "secret"}
    )
    assert r.status_code == 202
    assert r.json() == {"status": "accepted", "routine": "curator"}
    # background task ran by the time TestClient returns
    assert ("curator", None) in registry.calls  # type: ignore[attr-defined]


def test_trigger_mode_required_for_pseudo_agent(
    client: TestClient,
) -> None:
    r = client.post(
        "/routines/briefing", headers={"X-Alfred-Token": "secret"}
    )
    assert r.status_code == 400
    assert "mode" in r.json()["detail"]


def test_trigger_mode_must_be_known(client: TestClient) -> None:
    r = client.post(
        "/routines/briefing",
        headers={"X-Alfred-Token": "secret"},
        json={"mode": "weekly"},  # not registered in this fixture
    )
    assert r.status_code == 400


def test_trigger_mode_dispatch(
    client: TestClient, registry: RoutineRegistry
) -> None:
    r = client.post(
        "/routines/briefing",
        headers={"X-Alfred-Token": "secret"},
        json={"mode": "morning"},
    )
    assert r.status_code == 202
    assert ("briefing", "morning") in registry.calls  # type: ignore[attr-defined]


def test_background_failure_does_not_break_response(
    client: TestClient,
) -> None:
    """If the routine raises, the HTTP 202 has already returned."""
    bad = RoutineRegistry()

    async def boom(app: Any) -> dict:  # noqa: ARG001
        raise RuntimeError("boom")

    bad.register("boom", boom)
    app = create_routine_app(_StubApp(routine_registry=bad), api_token="s")
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/routines/boom", headers={"X-Alfred-Token": "s"})
    assert r.status_code == 202
