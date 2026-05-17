"""Prompt registry — versioned, externalized system prompts.

Public API:
    load_prompt(prompt_id) -> Prompt          # returns active version
    load_prompt(prompt_id, version="1.0") -> Prompt
    list_prompts() -> list[PromptMetadata]
    list_versions(prompt_id) -> list[str]
"""

from alfred.prompts.registry import (
    Prompt,
    PromptMetadata,
    PromptNotFoundError,
    list_prompts,
    list_versions,
    load_prompt,
)

__all__ = [
    "Prompt",
    "PromptMetadata",
    "PromptNotFoundError",
    "list_prompts",
    "list_versions",
    "load_prompt",
]
