"""Workflow outcomes — single source of truth for the strings.

`Outcome` is a `StrEnum` so values can be compared to plain strings (which
is how LangGraph state stores them) while still being IDE-discoverable and
typo-proof. Use the enum at every callsite that produces or consumes an
outcome; the str equality keeps interop with the TypedDict and Phoenix
span attributes free.
"""

from __future__ import annotations

from enum import StrEnum


class Outcome(StrEnum):
    """All terminal outcomes the tasks workflow can produce."""

    # Create branch
    CREATED = "created"
    DUPLICATE_EXISTING = "duplicate_existing"
    DUPLICATE_NEEDS_CONFIRMATION = "duplicate_needs_confirmation"

    # Complete / delete / update / list branches
    COMPLETED = "completed"
    DELETED = "deleted"
    UPDATED = "updated"
    LISTED = "listed"

    # Target-resolution outcomes (shared by complete/delete/update)
    TARGET_NEEDS_CONFIRMATION = "target_needs_confirmation"
    TARGET_NOT_FOUND = "target_not_found"

    # Error / clarification outcomes
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"
    CLARIFICATION = "clarification"
    ERROR = "error"


# Outcomes that indicate a successful side-effect (chore written / completed
# / deleted / updated / listed). build_reply's guard uses this to detect the
# "side effect happened but outcome got clobbered" race and prefer the
# success branch over the default clarification fallback.
SUCCESS_OUTCOMES: frozenset[Outcome] = frozenset(
    {
        Outcome.CREATED,
        Outcome.COMPLETED,
        Outcome.DELETED,
        Outcome.UPDATED,
        Outcome.LISTED,
    }
)
