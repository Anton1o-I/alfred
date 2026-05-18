"""Prompt strings for the tasks workflow's LLM specialists.

Each constant is a template consumed by exactly one node. Format helpers
(`name_mapping_block`, `format_existing_block`) live here too so the prompt
surface area is in one place — anything an operator might want to tweak to
adjust agent behavior is in this module.
"""

from __future__ import annotations

INTENT_PROMPT = (
    "Classify the user's email into one of these chore-tracker actions:\n"
    "- create: user wants to add a new recurring chore\n"
    "- complete: user is marking a chore done\n"
    "- list: user wants to see what's pending/overdue\n"
    "- delete: user wants to remove a chore from tracking\n"
    "- update: user wants to change an existing chore (assignee, recurrence, etc.)\n"
    "- clarify: the request is unclear or out of scope\n"
    "\n"
    "Email body:\n"
    "{email_body}"
)


PARSE_PROMPT = (
    "Extract the chore the user wants to add.\n"
    "\n"
    "Today is {today_day_of_week}, {today_iso} ({timezone_name}).\n"
    "\n"
    "Rules:\n"
    "- title: short imperative name, in this exact CANONICAL FORMAT:\n"
    "    • Start with the verb in imperative form ('Take out', 'Clean', "
    "      'Water', 'Replace').\n"
    "    • Sentence case: capitalize only the first word; lowercase the "
    "      rest UNLESS it's a proper noun ('Amazon', 'UPS') or an acronym "
    "      ('HVAC').\n"
    "    • Singular, no leading or interior articles. Write 'Take out "
    "      trash' NOT 'Take out the trash'. Write 'Clean kitchen "
    "      countertops' NOT 'Clean the Kitchen Countertops'.\n"
    "    • Be SPECIFIC when the user names a particular target — use the "
    "      qualifier inline ('Clean main bathroom', 'Clean hallway "
    "      bathroom', NOT just 'Clean bathroom').\n"
    "    • No scheduling phrases ('every Tuesday') — those go in recurrence.\n"
    "  Good examples: 'Take out trash', 'Clean main bathroom', 'Water "
    "  plants', 'Mow lawn', 'Pick up dry cleaning'.\n"
    "  Bad examples: 'Take out the trash', 'clean downstairs bathroom', "
    "  'Clean Kitchen Countertops', 'TRASH'.\n"
    "- assignee: 'household' (default) if either spouse can do it. Use "
    "  'primary' or 'secondary' when the user names a specific person — "
    "  resolve the name using the mapping below.\n"
    "{name_mapping_block}"
    "- recurrence_type:\n"
    "    'schedule' when anchored to days ('every Tuesday', 'weekly');\n"
    "    'completion' when anchored to elapsed time since last done "
    "      ('every 7 days', 'once a week after I do it');\n"
    "    'once' when this is a one-time task with a specific date "
    "      ('pick up the package on Friday', 'call the plumber tomorrow').\n"
    "  Default to 'schedule' when unsure for recurring; default to 'once' "
    "  when a single specific date is given.\n"
    "- recurrence.frequency: DAILY/WEEKLY/MONTHLY/YEARLY (only for recurring).\n"
    "- recurrence.interval: how many of those frequency units between runs. "
    "  Default 1. CRITICAL: when the user says 'every N <unit>', interval=N.\n"
    "    'every 2 days' → frequency=DAILY, interval=2\n"
    "    'every 3 weeks' → frequency=WEEKLY, interval=3\n"
    "    'every other Saturday' → frequency=WEEKLY, interval=2, byday=[SA]\n"
    "    'every 6 months' → frequency=MONTHLY, interval=6\n"
    "    'weekly' / 'every week' → frequency=WEEKLY, interval=1\n"
    "  Do NOT collapse 'every 2 days' to interval=1 — the number is load-bearing.\n"
    "- recurrence.byday: two-letter codes for weekly day-of-week patterns "
    "  (MO TU WE TH FR SA SU). Leave empty for non-weekly or unspecified.\n"
    "- due_date_iso: ISO date (YYYY-MM-DD) for one-time tasks. Resolve "
    "  relative phrases ('next Friday', 'tomorrow', 'this Saturday') using "
    "  today's date above.\n"
    "- shame_after_days: only set if the user explicitly says so.\n"
    "- object: the single noun the chore acts on. Lowercase, singular. "
    "  This is the THING being acted on, not the verb.\n"
    "    'Take out trash' → object='trash'\n"
    "    'Clean main bathroom' → object='bathroom'\n"
    "    'Clean hallway bathroom' → object='bathroom'\n"
    "    'Wipe kitchen counters' → object='counters'\n"
    "    'Mow lawn' → object='lawn'\n"
    "    'Water plants' → object='plants'\n"
    "    'Pick up dry cleaning' → object='dry cleaning'\n"
    "  Leave null only if the title doesn't have a clean verb-noun shape.\n"
    "- qualifier: the modifier that distinguishes WHICH instance of the "
    "  object. Lowercase. Leave null when there's no distinguishing modifier.\n"
    "    'Take out trash' → qualifier=null\n"
    "    'Clean main bathroom' → qualifier='main'\n"
    "    'Clean hallway bathroom' → qualifier='hallway'\n"
    "    'Clean kids bathroom' → qualifier='kids'\n"
    "    'Wipe kitchen counters' → qualifier='kitchen'\n"
    "    'Sweep front porch' → qualifier='front'\n"
    "    'Mow lawn' → qualifier=null\n"
    "  CRITICAL: when the user names a SPECIFIC instance (main/hallway/"
    "  master/kids/front/back/upstairs/downstairs/kitchen), put it here. "
    "  Two chores with different qualifiers are different chores even if "
    "  they share an object.\n"
    "\n"
    "Leave fields null when the user didn't say. Do not invent values.\n"
    "If the user did not describe a chore at all (e.g. just said 'hi'), "
    "leave title null.\n"
    "\n"
    "Email body:\n"
    "{email_body}"
)


DEDUP_PROMPT = (
    "Decide whether the proposed new chore is already covered by one of the "
    "existing active chores.\n"
    "\n"
    "Proposed new chore:\n"
    "  title: {new_title}\n"
    "  recurrence: {new_recurrence}\n"
    "  assignee: {new_assignee}\n"
    "\n"
    "Existing active chores:\n"
    "{existing_block}\n"
    "\n"
    "Confidence rubric:\n"
    "- high: same underlying activity on the same target. Synonyms or "
    "paraphrases of one already-tracked chore (e.g. 'take the trash out' "
    "vs 'Take out trash'; 'wipe down the main bathroom' vs 'Clean the "
    "main bathroom').\n"
    "- medium: meaningful overlap and you genuinely cannot tell if they are "
    "the same — ask the user.\n"
    "- none: distinct chores. THIS IS THE DEFAULT when in doubt. Examples "
    "that are NOT duplicates:\n"
    "  * Different rooms or fixtures: 'main bathroom' vs 'hallway bathroom' "
    "    vs 'kids' bathroom' vs 'master bathroom' — each room is its own "
    "    chore even though all involve cleaning a bathroom.\n"
    "  * Different scope of the same room: 'wipe down kitchen counters' "
    "    (daily) vs 'deep clean the kitchen' (weekly) — different scope, "
    "    different cadence, different chores.\n"
    "  * Different activities on the same target: 'mow the lawn' vs "
    "    'weed the lawn' — same target, different work.\n"
    "\n"
    "Rule of thumb: if the chores refer to physically different objects, "
    "rooms, or fixtures, they are NOT duplicates regardless of how similar "
    "the wording sounds. Only flag duplicates when the underlying work is "
    "the same on the same target.\n"
    "\n"
    "Quick test: strip the leading verb. If each remaining title has at "
    "least one distinctive word the other doesn't (e.g. 'main' vs "
    "'hallway', 'counters' vs nothing-specific), they're DIFFERENT chores.\n"
    "\n"
    "Reasoning should be one short sentence — it gets shown to the user."
)


TARGET_PARSE_PROMPT = (
    "Extract which existing chore the user is referring to.\n"
    "\n"
    "Action context: {action}  (complete = marking done; delete = removing from "
    "tracking; update = changing details).\n"
    "\n"
    "Active chores currently being tracked:\n"
    "{existing_block}\n"
    "\n"
    "Return a brief restatement of the chore the user references, in their own "
    "words (e.g. 'the trash', 'taking out recycling', 'bathrooms'). Leave the "
    "reference empty if the user didn't name a specific chore.\n"
    "\n"
    "Email body:\n"
    "{email_body}"
)


TARGET_MATCH_PROMPT = (
    "Decide which active chore the user is referring to.\n"
    "\n"
    "Action: {action}\n"
    "User's reference: {reference}\n"
    "Original message:\n"
    "{email_body}\n"
    "\n"
    "Candidate chores:\n"
    "{candidates_block}\n"
    "\n"
    "Match against the chore title — paraphrases and partial references are "
    "fine ('trash' matches 'Take out trash'; 'bathrooms' matches 'Clean bathrooms').\n"
    "\n"
    "Confidence rubric:\n"
    "- high: one clear match. Proceed.\n"
    "- medium: likely match worth confirming with the user (e.g. two plausible candidates).\n"
    "- none: no candidate plausibly matches.\n"
    "\n"
    "Reasoning should be one short sentence — it gets shown to the user."
)


UPDATE_PARSE_PROMPT = (
    "Extract the chore update the user is requesting.\n"
    "\n"
    "Active chores:\n"
    "{existing_block}\n"
    "\n"
    "Today is {today_day_of_week}, {today_iso} ({timezone_name}).\n"
    "\n"
    "Rules:\n"
    "- target_reference: brief restatement of which chore is changing.\n"
    "- Set new_* fields ONLY for what the user explicitly wants to change. "
    "Leave others null.\n"
    "- new_recurrence_type / new_recurrence: only if the user changes the schedule.\n"
    "- new_assignee: one of 'household', 'primary', 'secondary' if the user "
    "  reassigns — resolve named people via the mapping below.\n"
    "{name_mapping_block}"
    "\n"
    "Email body:\n"
    "{email_body}"
)


SPECIALIST_SYSTEM_PROMPT = (
    "/no_think\n"
    "You produce strictly-typed structured output. Fill the schema "
    "directly without preamble, explanation, or chain-of-thought."
)


# ── Prompt-shape helpers ───────────────────────────────────────────────────


def name_mapping_block(name_map: dict[str, str]) -> str:
    """Render a 'primary = <name>, secondary = <name>' hint into the prompt.

    Returns an empty string when no names are configured — the model then
    falls back to literal household/primary/secondary classification.
    """
    if not name_map:
        return ""
    lines = ["  Name → user_id mapping:"]
    for uid in ("primary", "secondary"):
        if uid in name_map:
            lines.append(f"    • '{name_map[uid]}' → {uid}")
    lines.append(
        "  When the user names one of these people, set assignee to the "
        "matching user_id. Treat case-insensitive matches and common "
        "shortenings as a match."
    )
    return "\n".join(lines) + "\n"


def format_existing_block(chores: list[dict]) -> str:
    """Render the active chore list for inclusion in dedup/target prompts."""
    if not chores:
        return "  (none)"
    lines = []
    for c in chores:
        rule = c.get("recurrence_rule") or {}
        freq = rule.get("frequency") or "?"
        byday = rule.get("byday") or []
        when = f"{freq.lower()}"
        if byday:
            when += f" on {','.join(byday)}"
        lines.append(
            f"  - id={c['id']}  '{c['title']}'  ({when}, "
            f"{c.get('recurrence_type', '?')}-based, assignee={c.get('assignee', '?')})"
        )
    return "\n".join(lines)
