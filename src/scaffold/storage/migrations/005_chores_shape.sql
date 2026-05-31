-- Structured chore shape: extract verb/object/qualifier at parse time so
-- dedup can do a deterministic code comparison instead of a fuzzy LLM
-- judgment. See src/alfred/agents/tasks/workflow.py:check_for_duplicates.

ALTER TABLE chores ADD COLUMN object TEXT;
ALTER TABLE chores ADD COLUMN qualifier TEXT;
