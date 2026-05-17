"""Filesystem-backed prompt registry with version pinning.

Prompts live in src/alfred/prompts/prompts/<prompt_id>.yaml. Each file
declares an id, description, model compatibility, and a list of versions.
Exactly one version per file is marked `status: active`; that's the
default returned by `load_prompt(id)`.

Why a registry rather than inline strings:
- Versioned changes are visible in git diff at the prompt level
- Audit log can record which prompt + version drove a given agent run
- A/B comparisons can pin specific versions per call
- Non-engineers can review/edit prompts without touching agent code

Why filesystem-backed (vs DB or external service) for now:
- Single user, single host — no concurrency or ACL needs
- Git is the audit/rollback story
- Easy to load in tests, easy to grep
- Migrating to LangSmith/PromptLayer later is a swap of this module only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROMPTS_DIR = Path(__file__).parent / "prompts"


class PromptNotFoundError(LookupError):
    """Raised when load_prompt() can't find a prompt id or version."""


@dataclass(frozen=True)
class Prompt:
    """A loaded prompt — the actual text plus its identifying metadata."""

    id: str
    version: str
    body: str
    description: str
    author: str
    created: str
    changelog: str
    status: str  # "active" | "deprecated" | "experimental"
    model_compatibility: list[str] = field(default_factory=list)

    @property
    def fqn(self) -> str:
        """Fully-qualified name for audit logs and traces."""
        return f"{self.id}@{self.version}"


@dataclass(frozen=True)
class PromptMetadata:
    """Metadata-only view used by `list_prompts()`."""

    id: str
    description: str
    active_version: str
    all_versions: list[str]


def load_prompt(prompt_id: str, version: str | None = None) -> Prompt:
    """Load a prompt by id. If version is omitted, returns the version marked active."""
    record = _load_record(prompt_id)
    versions: list[dict[str, Any]] = record.get("versions", [])
    if not versions:
        raise PromptNotFoundError(f"Prompt '{prompt_id}' has no versions defined")

    if version is None:
        active = [v for v in versions if v.get("status") == "active"]
        if not active:
            raise PromptNotFoundError(
                f"Prompt '{prompt_id}' has no version marked status: active"
            )
        if len(active) > 1:
            raise PromptNotFoundError(
                f"Prompt '{prompt_id}' has multiple active versions: "
                f"{[v['version'] for v in active]}"
            )
        chosen = active[0]
    else:
        matches = [v for v in versions if str(v.get("version")) == str(version)]
        if not matches:
            raise PromptNotFoundError(
                f"Prompt '{prompt_id}' has no version '{version}'. "
                f"Available: {[v['version'] for v in versions]}"
            )
        chosen = matches[0]

    return Prompt(
        id=prompt_id,
        version=str(chosen["version"]),
        body=chosen["body"].rstrip() + "\n",
        description=record.get("description", ""),
        author=chosen.get("author", "unknown"),
        created=chosen.get("created", ""),
        changelog=chosen.get("changelog", ""),
        status=chosen.get("status", "experimental"),
        model_compatibility=list(record.get("model_compatibility", [])),
    )


def list_prompts() -> list[PromptMetadata]:
    """List every prompt available in the registry."""
    out: list[PromptMetadata] = []
    for path in sorted(PROMPTS_DIR.glob("*.yaml")):
        prompt_id = path.stem
        record = _load_record(prompt_id)
        versions: list[dict[str, Any]] = record.get("versions", [])
        active = next(
            (str(v["version"]) for v in versions if v.get("status") == "active"),
            "(none)",
        )
        out.append(
            PromptMetadata(
                id=prompt_id,
                description=record.get("description", ""),
                active_version=active,
                all_versions=[str(v["version"]) for v in versions],
            )
        )
    return out


def list_versions(prompt_id: str) -> list[str]:
    """List all version strings for a given prompt id."""
    record = _load_record(prompt_id)
    return [str(v["version"]) for v in record.get("versions", [])]


@lru_cache(maxsize=64)
def _load_record(prompt_id: str) -> dict[str, Any]:
    path = PROMPTS_DIR / f"{prompt_id}.yaml"
    if not path.exists():
        raise PromptNotFoundError(f"Prompt file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if data.get("id") != prompt_id:
        raise PromptNotFoundError(
            f"Prompt file {path} declares id '{data.get('id')}' but filename is '{prompt_id}'"
        )
    return data


def _clear_cache() -> None:
    """Test helper — clear the @lru_cache so prompt edits are picked up."""
    _load_record.cache_clear()
