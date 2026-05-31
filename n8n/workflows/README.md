# n8n workflows — Alfred routine triggers

One workflow per `scheduled_tasks` entry in `config/settings.yaml`. Each
is a 2-node graph: **Schedule Trigger** (cron) → **HTTP Request** to
`POST http://host.docker.internal:8765/routines/{name}` carrying the
shared `X-Alfred-Token` header.

The JSON files in this directory are version-controlled snapshots; n8n
stores the live workflows in its own SQLite DB inside the `n8n_data`
volume. Re-export from the UI after edits and commit the file.

## One-time setup

### 1. Create the Header Auth credential

n8n needs to send `X-Alfred-Token: <secret>` on every request. The
secret lives in `.env` as `ALFRED_ROUTINE_API_KEY` — same value the
`alfred-server` service reads at boot.

In the n8n UI:

1. **Credentials** (left sidebar) → **+ Add Credential** → search **Header Auth**.
2. **Name:** `Alfred routine API` (must match the name in the workflow JSON below).
3. **Header Auth → Name:** `X-Alfred-Token`
4. **Header Auth → Value:** paste the value of `ALFRED_ROUTINE_API_KEY` from `.env`.
5. **Save**.

You only do this once. All Alfred workflows reuse the same credential.

### 2. Import a workflow

1. **Workflows** (left sidebar) → **+ Add Workflow** → **⋮ menu** → **Import from File**.
2. Pick a `.json` file from this directory (e.g. `alfred-weekly-preview.json`).
3. n8n will warn the credential ID inside the JSON is unresolved — open the
   **HTTP Request node** → **Credential for Header Auth** dropdown →
   select **Alfred routine API**. Save.
4. Top-right toggle: **Inactive → Active** when ready to fire on cron.
   Leave **Inactive** until you've manually tested it (step 3 below).

### 3. Manual smoke test

Before flipping a workflow Active:

1. Open the workflow.
2. Click **Execute Workflow** (top toolbar).
3. The Schedule Trigger node lights up; the HTTP node fires.
4. Confirm `202 Accepted` in the n8n node output panel.
5. `journalctl -u alfred-server -f` should show `routine_triggered` →
   `routine_completed`.
6. The corresponding email should arrive.

If anything fails, the n8n execution log shows the full request/response.

### 4. Cutover one routine at a time

For each workflow you build:

1. Set the workflow **Active**.
2. Edit `config/settings.yaml`: flip `enabled: false` on the matching
   `scheduled_tasks` entry.
3. `sudo systemctl restart alfred-scheduler`.
4. Watch the next fire window — only the n8n journal entry should
   appear, not the APScheduler one.

Once all five workflows are running off n8n:

```bash
sudo systemctl disable --now alfred-scheduler
```

…and the legacy `src/alfred/scheduler/` code can be deleted in a final
cleanup commit.

## Workflow inventory

| File | Routine | Cron | Mode |
|---|---|---|---|
| `alfred-weekly-preview.json` | briefing | `0 18 * * 0` (Sun 18:00) | weekly |
| _TODO_ | briefing | `0 5 * * *`  (daily 05:00) | morning |
| _TODO_ | briefing | `0 19 * * *` (daily 19:00) | daily |
| _TODO_ | curator  | `0 4 * * 1`  (Mon 04:00) | — |
| _TODO_ | inbox    | `* * * * *`  (every min) | — |

Build the rest in the UI using `alfred-weekly-preview.json` as the
template. Once each is wired, **Workflow → ⋮ → Download** the JSON and
drop it next to this README so the repo stays the source of truth.
