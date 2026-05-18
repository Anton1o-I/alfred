-- Household chore tracking — recurring tasks with completion history.
-- See docs/tasks-agent.md and src/alfred/agents/tasks/store.py for the
-- read/write surface. Source of truth for chores; no external sync.

CREATE TABLE IF NOT EXISTS chores (
    id TEXT PRIMARY KEY,                          -- short slug ("trash", "bathrooms")
    title TEXT NOT NULL,
    description TEXT,
    assignee TEXT NOT NULL,                       -- "household" | user_id
    recurrence_type TEXT NOT NULL,                -- "schedule" | "completion"
    recurrence_rule TEXT NOT NULL,                -- JSON serialization of RecurrenceRule
    shame_after_days INTEGER NOT NULL DEFAULT 3,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chore_completions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chore_id TEXT NOT NULL REFERENCES chores(id) ON DELETE CASCADE,
    completed_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
    completed_by TEXT NOT NULL,
    completed_via TEXT NOT NULL                   -- "email" | "imessage" | "cli"
);

CREATE INDEX IF NOT EXISTS idx_chore_completions_chore
    ON chore_completions(chore_id, completed_at DESC);

CREATE INDEX IF NOT EXISTS idx_chores_active
    ON chores(active, assignee);
