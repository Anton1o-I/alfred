# systemd: alfred-scheduler

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
