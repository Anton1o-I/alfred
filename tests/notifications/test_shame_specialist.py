"""Unit tests for the shame-roast specialist.

Covers prompt assembly, happy-path parsing, and the failure modes that
must return `{}` so the renderer falls back to the static label.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from alfred.specialists.shame.prompts import (
    ROAST_FEW_SHOT,
    ROAST_SYSTEM,
    build_roast_prompt,
)
from alfred.specialists.shame.specialist import (
    ChoreRoastInput,
    _parse_response,
    _trim,
    generate_roasts,
)


def _items() -> list[ChoreRoastInput]:
    return [
        ChoreRoastInput(
            chore_id="trash", title="Take out trash", assignee="primary",
            tier=2, days=8,
        ),
        ChoreRoastInput(
            chore_id="dog", title="Walk the dog", assignee="secondary",
            tier=1, days=3,
        ),
    ]


# ── Prompt assembly ────────────────────────────────────────────────────────


def test_build_roast_prompt_includes_system_voice_anchors() -> None:
    # System prompt is sent separately, but the prompt body still embeds
    # the few-shot anchors and asks for JSON.
    prompt = build_roast_prompt(_items())
    assert "Take out trash" in prompt
    assert "Walk the dog" in prompt
    assert "trash" in prompt
    assert "JSON" in prompt
    # Few-shot examples appear (use one well-known anchor).
    assert any(ex["title"] in prompt for ex in ROAST_FEW_SHOT)


def test_system_prompt_calls_out_voice_rules() -> None:
    # Smoke check the system prompt's key constraints survive edits.
    assert "ONE line" in ROAST_SYSTEM
    assert "80 characters" in ROAST_SYSTEM
    assert "JSON" in ROAST_SYSTEM


def test_few_shot_covers_all_three_tiers() -> None:
    tiers = {ex["tier"] for ex in ROAST_FEW_SHOT}
    assert tiers == {1, 2, 3}


# ── Parsing ────────────────────────────────────────────────────────────────


def test_parse_response_happy_path() -> None:
    content = json.dumps({
        "lines": [
            {"chore_id": "trash", "roast": "The kitchen has begun composing its own smell."},
            {"chore_id": "dog", "roast": "The dog has prepared a brief sigh."},
        ]
    })
    out = _parse_response(content, valid_ids={"trash", "dog"})
    assert out["trash"].startswith("The kitchen")
    assert out["dog"].startswith("The dog")


def test_parse_response_strips_markdown_fence() -> None:
    content = "```json\n" + json.dumps({"lines": [
        {"chore_id": "trash", "roast": "ok"}
    ]}) + "\n```"
    out = _parse_response(content, valid_ids={"trash"})
    assert out == {"trash": "ok"}


def test_parse_response_drops_unknown_chore_ids() -> None:
    content = json.dumps({"lines": [
        {"chore_id": "ghost", "roast": "nope"},
        {"chore_id": "trash", "roast": "yes"},
    ]})
    out = _parse_response(content, valid_ids={"trash"})
    assert out == {"trash": "yes"}


def test_parse_response_drops_empty_roasts() -> None:
    content = json.dumps({"lines": [
        {"chore_id": "trash", "roast": "   "},
        {"chore_id": "dog", "roast": "ok"},
    ]})
    out = _parse_response(content, valid_ids={"trash", "dog"})
    assert out == {"dog": "ok"}


def test_parse_response_malformed_json_returns_empty() -> None:
    assert _parse_response("not json", valid_ids={"x"}) == {}
    assert _parse_response('{"lines": [bad', valid_ids={"x"}) == {}


def test_parse_response_schema_mismatch_returns_empty() -> None:
    # Missing required "roast" field.
    content = json.dumps({"lines": [{"chore_id": "trash"}]})
    assert _parse_response(content, valid_ids={"trash"}) == {}


# ── Trim ───────────────────────────────────────────────────────────────────


def test_trim_enforces_100_char_limit() -> None:
    long = "a" * 200
    out = _trim(long)
    assert len(out) <= 100


def test_trim_short_unchanged() -> None:
    assert _trim("  hi  ") == "hi"


# ── generate_roasts (mocked client) ────────────────────────────────────────


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = type("M", (), {"content": content})()


class _FakeResp:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str | None = None, raise_exc: Exception | None = None) -> None:
        self._content = content
        self._raise = raise_exc
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeResp:  # noqa: D401
        self.calls.append(kwargs)
        if self._raise:
            raise self._raise
        return _FakeResp(self._content or "")


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeAsyncOpenAI:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _FakeChat(completions)


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self._client = _FakeAsyncOpenAI(completions)


@pytest.mark.asyncio
async def test_generate_roasts_empty_input_returns_empty() -> None:
    out = await generate_roasts([], _FakeClient(_FakeCompletions(content="")))
    assert out == {}


@pytest.mark.asyncio
async def test_generate_roasts_happy_path() -> None:
    payload = json.dumps({"lines": [
        {"chore_id": "trash", "roast": "Smells like decisions deferred."},
        {"chore_id": "dog", "roast": "The dog drafts a memo."},
    ]})
    fc = _FakeCompletions(content=payload)
    out = await generate_roasts(_items(), _FakeClient(fc))
    assert out["trash"].startswith("Smells")
    assert out["dog"].startswith("The dog")
    # Confirm temperature actually got passed.
    assert fc.calls[0]["temperature"] == pytest.approx(0.95)
    # Confirm system + user messages.
    msgs = fc.calls[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


@pytest.mark.asyncio
async def test_generate_roasts_network_error_returns_empty() -> None:
    fc = _FakeCompletions(raise_exc=RuntimeError("connection refused"))
    out = await generate_roasts(_items(), _FakeClient(fc))
    assert out == {}


@pytest.mark.asyncio
async def test_generate_roasts_malformed_json_returns_empty() -> None:
    fc = _FakeCompletions(content="not even close to json")
    out = await generate_roasts(_items(), _FakeClient(fc))
    assert out == {}


@pytest.mark.asyncio
async def test_generate_roasts_empty_content_returns_empty() -> None:
    fc = _FakeCompletions(content="")
    out = await generate_roasts(_items(), _FakeClient(fc))
    assert out == {}


@pytest.mark.asyncio
async def test_generate_roasts_missing_chore_ids_returns_empty() -> None:
    payload = json.dumps({"lines": [
        {"chore_id": "unknown", "roast": "nope"},
    ]})
    fc = _FakeCompletions(content=payload)
    out = await generate_roasts(_items(), _FakeClient(fc))
    assert out == {}


@pytest.mark.asyncio
async def test_generate_roasts_enforces_100_char_trim() -> None:
    long = "x" * 250
    payload = json.dumps({"lines": [
        {"chore_id": "trash", "roast": long},
    ]})
    fc = _FakeCompletions(content=payload)
    out = await generate_roasts(_items(), _FakeClient(fc))
    assert len(out["trash"]) <= 100
