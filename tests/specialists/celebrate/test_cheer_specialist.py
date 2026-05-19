"""Unit tests for the completion-cheer specialist.

Covers prompt assembly, happy-path parsing, and the failure modes that
must return `""` so the renderer falls back to the static banner.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from alfred.specialists.celebrate.prompts import (
    CHEER_FEW_SHOT,
    CHEER_SYSTEM,
    build_cheer_prompt,
)
from alfred.specialists.celebrate.specialist import (
    CompletionCheerInput,
    _parse_response,
    _trim,
    generate_cheer,
)


def _item() -> CompletionCheerInput:
    return CompletionCheerInput(
        chore_id="dog",
        title="Walk the dog",
        assignee="primary",
    )


# ── Prompt assembly ────────────────────────────────────────────────────────


def test_build_cheer_prompt_includes_title_and_json_instruction() -> None:
    prompt = build_cheer_prompt(_item())
    assert "Walk the dog" in prompt
    assert "dog" in prompt
    assert "JSON" in prompt
    # Few-shot examples appear (use one well-known anchor).
    assert any(str(ex["title"]) in prompt for ex in CHEER_FEW_SHOT)


def test_system_prompt_calls_out_voice_rules() -> None:
    assert "ONE line" in CHEER_SYSTEM
    assert "95 characters" in CHEER_SYSTEM
    assert "JSON" in CHEER_SYSTEM
    # Reverse-roast directive is the core voice rule — guard against drift.
    assert "REVERSE" in CHEER_SYSTEM
    # Affirmation prefix is mandatory; guard against drift.
    assert "Well done" in CHEER_SYSTEM
    assert "Bravo" in CHEER_SYSTEM


def test_few_shot_covers_diverse_categories() -> None:
    # Sanity check: at least 12 anchors and they span multiple chore types.
    assert len(CHEER_FEW_SHOT) >= 12
    titles = " ".join(str(ex["title"]).lower() for ex in CHEER_FEW_SHOT)
    # A handful of categories that the prompt explicitly calls out.
    assert "dog" in titles or "cat" in titles  # pets
    assert "bill" in titles  # bills
    assert "lawn" in titles or "gutters" in titles  # outdoor
    assert "plants" in titles  # plants


# ── Parsing ────────────────────────────────────────────────────────────────


def test_parse_response_happy_path() -> None:
    content = json.dumps({"cheer": "The dog has filed a five-star review."})
    assert _parse_response(content) == "The dog has filed a five-star review."


def test_parse_response_strips_markdown_fence() -> None:
    content = "```json\n" + json.dumps({"cheer": "ok"}) + "\n```"
    assert _parse_response(content) == "ok"


def test_parse_response_malformed_json_returns_empty() -> None:
    assert _parse_response("not json") == ""
    assert _parse_response('{"cheer": [bad') == ""


def test_parse_response_missing_cheer_key_returns_empty() -> None:
    content = json.dumps({"not_cheer": "nope"})
    assert _parse_response(content) == ""


def test_parse_response_empty_cheer_returns_empty() -> None:
    content = json.dumps({"cheer": "   "})
    assert _parse_response(content) == ""


# ── Trim ───────────────────────────────────────────────────────────────────


def test_trim_enforces_100_char_limit() -> None:
    long = "a" * 200
    out = _trim(long)
    assert len(out) <= 100


def test_trim_short_unchanged() -> None:
    assert _trim("  hi  ") == "hi"


# ── generate_cheer (mocked client) ─────────────────────────────────────────


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

    async def create(self, **kwargs: Any) -> _FakeResp:
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
async def test_generate_cheer_happy_path() -> None:
    payload = json.dumps({"cheer": "The dog has filed an enthusiastic review."})
    fc = _FakeCompletions(content=payload)
    out = await generate_cheer(_item(), _FakeClient(fc))
    assert out.startswith("The dog")
    # Confirm temperature actually got passed.
    assert fc.calls[0]["temperature"] == pytest.approx(0.9)
    # Confirm system + user messages.
    msgs = fc.calls[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


@pytest.mark.asyncio
async def test_generate_cheer_network_error_returns_empty() -> None:
    fc = _FakeCompletions(raise_exc=RuntimeError("connection refused"))
    out = await generate_cheer(_item(), _FakeClient(fc))
    assert out == ""


@pytest.mark.asyncio
async def test_generate_cheer_malformed_json_returns_empty() -> None:
    fc = _FakeCompletions(content="not even close to json")
    out = await generate_cheer(_item(), _FakeClient(fc))
    assert out == ""


@pytest.mark.asyncio
async def test_generate_cheer_empty_content_returns_empty() -> None:
    fc = _FakeCompletions(content="")
    out = await generate_cheer(_item(), _FakeClient(fc))
    assert out == ""


@pytest.mark.asyncio
async def test_generate_cheer_missing_key_returns_empty() -> None:
    payload = json.dumps({"not_cheer": "nope"})
    fc = _FakeCompletions(content=payload)
    out = await generate_cheer(_item(), _FakeClient(fc))
    assert out == ""


@pytest.mark.asyncio
async def test_generate_cheer_enforces_100_char_trim() -> None:
    long = "x" * 250
    payload = json.dumps({"cheer": long})
    fc = _FakeCompletions(content=payload)
    out = await generate_cheer(_item(), _FakeClient(fc))
    assert len(out) <= 100
