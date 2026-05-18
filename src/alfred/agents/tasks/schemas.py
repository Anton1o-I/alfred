"""Pydantic schemas for the tasks workflow's LLM specialists.

Each `BaseModel` here is the typed output of exactly one specialist agent.
Keeping the schemas separate from the graph logic makes them easy to scan,
reuse from tests, and update independently.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Concrete assignee values the workflow understands. Kept here (not Outcome)
# because they overlap with RecipientConfig.user_id slots, not with workflow
# outcomes.
Assignee = Literal["household", "primary", "secondary"]
RecurrenceType = Literal["schedule", "completion", "once"]
Confidence = Literal["high", "medium", "none"]


class TargetReference(BaseModel):
    """The natural-language phrase the user used to refer to an existing chore.

    Defined here (above IntentClassification) so the router schema can
    reference it for its optional list_target field without a forward ref.
    """

    reference: str = Field(
        description=(
            "A brief restatement of the chore the user is referring to, in their "
            "own words. Examples: 'trash', 'the bathrooms', 'taking out the recycling'. "
            "Leave empty string if the user didn't actually reference a chore."
        )
    )


class IntentClassification(BaseModel):
    """First-pass router decision — what does the user want to do?"""

    action: Literal["create", "complete", "list", "delete", "update", "clarify"] = (
        Field(description="What the user wants to do.")
    )
    complexity: Literal["simple", "complex"] = Field(
        default="simple",
        description=(
            "How hard the request is to reason about. simple: one action, "
            "explicit chore name or single recurrence rule. complex: multiple "
            "chores in one email, ambiguous identification of an existing "
            "chore, or unusual recurrence phrasing. When in doubt, prefer simple."
        ),
    )
    list_target: TargetReference | None = Field(
        default=None,
        description=(
            "When action='list' AND the user scoped the query to a specific "
            "person ('what does Alex owe?', 'show me Sam's chores', 'what "
            "household chores are pending?'), capture the assignee phrase here. "
            "Leave null for unscoped queries ('what chores do I have', 'list "
            "everything pending'). Use exactly the user's wording in the "
            "reference field (e.g. 'Alex', 'my spouse', 'household'). Only "
            "set this for action='list' — leave null otherwise."
        ),
    )
    reasoning: str = Field(default="", description="One short sentence.")


class ChoreRecurrence(BaseModel):
    """How the chore repeats. Parallels the calendar RecurrenceRule shape."""

    frequency: Literal["DAILY", "WEEKLY", "MONTHLY", "YEARLY"] = Field(
        description="How often the chore repeats."
    )
    interval: int = Field(default=1, description="Repeat every N (e.g., 2 = every other).")
    byday: list[str] = Field(
        default_factory=list,
        description=(
            "Days of the week for weekly patterns. Two-letter codes: "
            "MO TU WE TH FR SA SU."
        ),
    )


class ChoreDraft(BaseModel):
    """Parsed chore from the user's message."""

    id: str | None = Field(
        default=None,
        description=(
            "Optional short stable id the user gave (e.g. 'trash', 'bathrooms'). "
            "Leave null if the user didn't name one — the system will slugify the title."
        ),
    )
    title: str | None = Field(
        default=None,
        description=(
            "Concise chore name. Examples: 'Take out trash', 'Clean bathrooms'. "
            "Leave null if the user did not actually describe a chore."
        ),
    )
    description: str | None = None
    assignee: Assignee | None = Field(
        default=None,
        description=(
            "Who is responsible. 'household' if either spouse can do it; "
            "'primary' for the message owner; 'secondary' for the other spouse. "
            "Leave null if the user didn't say — the system will default to household."
        ),
    )
    recurrence_type: RecurrenceType | None = Field(
        default=None,
        description=(
            "'schedule' = anchored to calendar dates ('every Tuesday'). "
            "'completion' = anchored to last completion ('every 7 days after I do it'). "
            "'once' = a single one-time task with a specific due date. "
            "Leave null if the user didn't specify and the system will infer "
            "(once if due_date_iso is set, else schedule)."
        ),
    )
    recurrence: ChoreRecurrence | None = Field(
        default=None,
        description=(
            "Required for recurring chores. Leave null for one-time tasks "
            "(when recurrence_type='once' or the user gave a single date)."
        ),
    )
    due_date_iso: str | None = Field(
        default=None,
        description=(
            "For one-time tasks only. ISO date (YYYY-MM-DD) when the task is due. "
            "Resolve relative phrases like 'next Friday' or 'tomorrow' using "
            "today's date provided in the prompt. Leave null for recurring chores."
        ),
    )
    shame_after_days: int | None = Field(
        default=None,
        description="Days overdue before public shame kicks in. Defaults to 3.",
    )
    object: str | None = Field(
        default=None,
        description=(
            "The single noun the chore acts on, lowercase, singular. "
            "Examples: 'trash' for 'Take out trash', 'bathroom' for 'Clean main bathroom', "
            "'lawn' for 'Mow lawn', 'plants' for 'Water plants'. Leave null if the "
            "title isn't a clean verb-noun shape."
        ),
    )
    qualifier: str | None = Field(
        default=None,
        description=(
            "Modifier that distinguishes WHICH instance of the object. Lowercase. "
            "Examples: 'main' for 'Clean main bathroom', 'kitchen' for 'Wipe kitchen "
            "counters', 'front' for 'Sweep front porch'. Leave null when the chore "
            "doesn't single out one instance (e.g. 'Take out trash' has no qualifier)."
        ),
    )


class DuplicateCheckDecision(BaseModel):
    """LLM-fallback dedup result. The deterministic shape-based path returns
    the same shape via code, not via this schema."""

    confidence: Confidence = Field(
        description=(
            "high: this is plainly the same chore as an existing one. "
            "medium: likely overlap, worth asking the user to confirm. "
            "none: no meaningful overlap — safe to create."
        )
    )
    existing_chore_id: str | None = Field(
        default=None,
        description="The matched existing chore's id, or null if confidence is 'none'.",
    )
    reasoning: str = Field(
        default="",
        description="One short sentence (shown to the user when we ask for confirmation).",
    )


class TargetMatch(BaseModel):
    """Which active chore the user means."""

    confidence: Confidence = Field(
        description=(
            "high: clearly the right chore — proceed. "
            "medium: likely match but worth confirming first. "
            "none: no candidate plausibly matches."
        )
    )
    chore_id: str | None = Field(
        default=None,
        description="The matched chore's id, or null if confidence is 'none'.",
    )
    reasoning: str = Field(
        default="",
        description="One short sentence explaining the match (shown to the user).",
    )


class ChoreUpdateDraft(BaseModel):
    """What to change on an existing chore."""

    target_reference: str = Field(
        description="Brief restatement of which chore the user wants to change."
    )
    new_title: str | None = None
    new_assignee: Assignee | None = None
    new_recurrence_type: Literal["schedule", "completion"] | None = None
    new_recurrence: ChoreRecurrence | None = None
    new_shame_after_days: int | None = None
