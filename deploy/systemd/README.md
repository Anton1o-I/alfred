# systemd units

Two unit files live here:

- **`alfred-scheduler.service`** — legacy in-process APScheduler. Being
  retired in favor of n8n + `alfred-server`. Still functional today.
- **`alfred-server.service`** — new HTTP routine trigger server. Serves
  `POST /routines/{name}` for n8n (or any other caller) to fire scheduled
  routines on cron.

During cutover, both can run side-by-side: APScheduler keeps firing the
cron-style routines while you build out the n8n workflows. Once n8n is
proven, stop+disable `alfred-scheduler`.

## alfred-scheduler (legacy)

Runs `alfred scheduler` as a long-lived service so all the cron-style routines
(inbox poll every 5 min, daily briefing at 7am, Sunday weekly preview, weekly
curator digest, etc.) actually fire on their own.

## Install (one-time)

```bash
# 1. Symlink the unit file (lets us edit it in the repo without copying)
sudo ln -s /home/ant/dev/home-agent/deploy/systemd/alfred-scheduler.service \
    /etc/systemd/system/alfred-scheduler.service

# 2. Tell systemd about it
sudo systemctl daemon-reload

# 3. Start it now AND enable it at boot
sudo systemctl enable --now alfred-scheduler
```

## Common commands

```bash
# Live logs
journalctl -u alfred-scheduler -f

# Recent logs (with timestamps)
journalctl -u alfred-scheduler -n 200 --no-pager

# Status (running? last restart? crash count?)
systemctl status alfred-scheduler

# Restart after a code change
sudo systemctl restart alfred-scheduler

# Stop (e.g., to run alfred manually for debugging)
sudo systemctl stop alfred-scheduler

# Disable (don't start on boot)
sudo systemctl disable alfred-scheduler
```

## What's covered

- **Process crash** → automatic restart after 10s (capped at 5 attempts/min so
  a broken build doesn't pin the CPU)
- **Server reboot** → starts automatically on boot (via `multi-user.target`)
- **Network outage at boot** → waits for `network-online.target` before starting
- **Docker stack not up** → service still starts; routines needing the LiteLLM
  proxy will fail with clear errors until the stack is reachable

## Editing the unit

Edit `deploy/systemd/alfred-scheduler.service` in the repo, then:

```bash
sudo systemctl daemon-reload
sudo systemctl restart alfred-scheduler
```

The symlinked file picks up changes without re-copying.

## Log retention

The unit sets `StandardOutput=journal` / `StandardError=journal`, so every
`structlog` line from Alfred ends up in **journald**. By default journald
keeps logs until disk pressure (no time limit). Recommend capping it.

Edit `/etc/systemd/journald.conf` (or drop a file in
`/etc/systemd/journald.conf.d/`):

```ini
[Journal]
# Keep at most 30 days of logs.
MaxRetentionSec=30day

# Don't let the journal use more than 1 GB total.
SystemMaxUse=1G

# Force persistent storage (default on most distros, but explicit is fine).
Storage=persistent
```

Then reload journald:

```bash
sudo systemctl restart systemd-journald
```

Verify what journald is using:

```bash
journalctl --disk-usage
```

Other things to know:
- **`data/alfred.db` audit_log table** is *not* rotated by journald — it's a
  separate persistent record of each agent request (tokens, cost, status).
  Useful for budget review and regression hunting; no automatic cleanup
  today. Trim manually with SQL if it grows.
- **Phoenix traces** live in the `alfred-phoenix` container (`data/phoenix/`)
  and have their own retention via Phoenix's own config. They're the
  richest debugging surface (per-LLM-call spans) but also the heaviest.

## alfred-server (new — for n8n cutover)

Runs `alfred serve` as a long-lived service. Listens on `0.0.0.0:8765`
(bound by the unit's `ExecStart`); n8n in the docker compose stack
reaches it via `http://host.docker.internal:8765` thanks to the
`extra_hosts: ["host.docker.internal:host-gateway"]` line in
`docker-compose.yml`.

### Prereqs

1. Set `ALFRED_ROUTINE_API_KEY` in `.env`. Generate with
   `openssl rand -hex 32`. The service refuses to start without it.
2. n8n container needs to be restarted to pick up the `extra_hosts`
   change: `docker compose up -d n8n`.

### Install (one-time)

```bash
sudo ln -s /home/ant/dev/home-agent/deploy/systemd/alfred-server.service \
    /etc/systemd/system/alfred-server.service
sudo systemctl daemon-reload
sudo systemctl enable --now alfred-server
```

### Commands

Mirror `alfred-scheduler`'s — just swap the unit name:

```bash
journalctl -u alfred-server -f                # live logs
systemctl status alfred-server                # state + crash count
sudo systemctl restart alfred-server          # after code change
curl http://127.0.0.1:8765/healthz            # liveness probe
```

### Cutover sequencing

1. Install `alfred-server` (both services running). Confirm
   `curl /healthz` returns `200`.
2. Inside n8n: build one workflow (start with `weekly_preview`, lowest
   blast radius — Sunday only). Trigger it manually once; confirm the
   email arrives.
3. Disable the corresponding APScheduler entry by setting
   `enabled: false` in `config/settings.yaml` and restarting
   `alfred-scheduler`. Now only n8n fires that routine.
4. Repeat for the other 4 routines (`weekly_research_curator`,
   `daily_briefing`, `morning_briefing`, `inbox_poll`). Watch the
   journal for each to confirm n8n is the only trigger source.
5. Once all 5 are running off n8n for 24-48h:
   `sudo systemctl disable --now alfred-scheduler`. Optionally delete
   `scheduled_tasks` from settings.yaml and drop the legacy
   `src/alfred/scheduler/` code.
