-- Add one-time-task support to chores.
-- recurrence_type can now be "once"; due_date holds the target ISO date.
-- recurrence_rule stays NOT NULL for SQL simplicity ("{}" for once-type).

ALTER TABLE chores ADD COLUMN due_date TEXT;
