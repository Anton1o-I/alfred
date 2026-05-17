"""Prompt registry smoke tests."""

from __future__ import annotations

import pytest

from alfred.prompts import (
    PromptNotFoundError,
    list_prompts,
    list_versions,
    load_prompt,
)


def test_load_active_version() -> None:
    p = load_prompt("curator.triage")
    assert p.id == "curator.triage"
    assert p.status == "active"
    # Active version should be the latest structured prompt
    assert p.version >= "1.2"
    assert "<role>" in p.body
    # v1.2+ adds authorship signal handling
    assert "AUTHORSHIP" in p.body


def test_load_specific_version() -> None:
    p = load_prompt("curator.triage", version="1.0")
    assert p.version == "1.0"
    assert p.status == "deprecated"
    assert "<role>" not in p.body  # sparse legacy prompt


def test_fqn_format() -> None:
    p = load_prompt("curator.synthesis")
    assert p.fqn == f"curator.synthesis@{p.version}"


def test_unknown_prompt_raises() -> None:
    with pytest.raises(PromptNotFoundError):
        load_prompt("does.not.exist")


def test_unknown_version_raises() -> None:
    with pytest.raises(PromptNotFoundError):
        load_prompt("curator.triage", version="99.0")


def test_list_prompts_returns_all_registered() -> None:
    prompts = list_prompts()
    ids = {p.id for p in prompts}
    assert "curator.triage" in ids
    assert "curator.synthesis" in ids


def test_list_versions_returns_in_file_order() -> None:
    versions = list_versions("curator.triage")
    assert versions[0] == "1.0"
    assert "1.1" in versions
    assert "1.2" in versions
