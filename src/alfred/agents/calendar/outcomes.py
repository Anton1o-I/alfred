"""Calendar workflow outcomes — single source of truth for the strings.

Mirrors the tasks-agent pattern (`agents/tasks/outcomes.py`). Use the
enum at every callsite that produces or consumes an outcome; the str
equality keeps interop with the TypedDict + Phoenix span attributes
free.
"""

from __future__ import annotations

from enum import StrEnum


class Outcome(StrEnum):
    """All terminal outcomes the calendar workflow can produce."""

    # Create branch
    CREATED = "created"

    # Delete branch
    DELETED = "deleted"
    DELETE_NOT_FOUND = "delete_not_found"
    DELETE_NEEDS_CONFIRMATION = "delete_needs_confirmation"

    # Update branch
    UPDATED = "updated"
    UPDATE_NOT_FOUND = "update_not_found"
    UPDATE_NEEDS_CONFIRMATION = "update_needs_confirmation"
    UPDATE_NO_CHANGE = "update_no_change"

    # Validation / fallback
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"
    CLARIFICATION = "clarification"
    ERROR = "error"
