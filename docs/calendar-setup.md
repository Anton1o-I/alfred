# Google Calendar Setup

This guide walks you through connecting the Home Agent to your family's Google Calendars.

## Overview

The calendar agent needs:
1. A Google Cloud project with the Calendar API enabled
2. OAuth2 credentials (a one-time browser login)
3. Your wife's calendar shared with your Google account

After setup, the agent can read and write events on both calendars through a single authenticated session.

## Step 1: Create Google Cloud Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or use an existing one)
3. Enable the **Google Calendar API**:
   - Go to APIs & Services > Library
   - Search for "Google Calendar API"
   - Click Enable
4. Create OAuth credentials:
   - Go to APIs & Services > Credentials
   - Click "Create Credentials" > "OAuth client ID"
   - Application type: **Desktop app**
   - Name: "Home Agent"
   - Click Create
5. Download the credentials JSON file
6. Save it as `config/google_credentials.json`

## Step 2: Authenticate

Run the OAuth flow from your Mac Mini (needs a browser):

```bash
uv run alfred google-auth
```

This opens a browser window. Sign in with your Google account and grant calendar access. The refresh token is saved to `config/google_token.json`.

## Step 3: Share Your Wife's Calendar

1. Your wife opens Google Calendar (web)
2. Settings (gear icon) > Settings
3. Under her calendar, click "Share with specific people"
4. Add your email address
5. Set permission to **"Make changes to events"**
6. Save

## Step 4: Find Calendar IDs

After sharing is set up, list all calendars accessible to your account:

```bash
uv run alfred list-calendars
```

This shows something like:
```
Accessible calendars:

  My Calendar (PRIMARY)
    ID: primary
    Access: owner

  Wife's Calendar
    ID: wife@gmail.com
    Access: writer
```

## Step 5: Update Configuration

Edit `config/calendar.yaml` with your actual names and calendar IDs:

```yaml
family:
  - name: "Andres"          # Your name
    calendar_id: "primary"
  - name: "Maria"            # Your wife's name
    calendar_id: "wife@gmail.com"  # From step 4

timezone: "America/Chicago"  # Your timezone
```

## Step 6: Verify

```bash
uv run alfred ask "What's on my calendar this week?"
```

## Security Notes

- `config/google_credentials.json` — OAuth client secrets. Already in `.gitignore`.
- `config/google_token.json` — Your refresh token. Already in `.gitignore`. Treat like a password.
- The agent only has access to calendars you explicitly share.
- OAuth tokens can be revoked at any time from [Google Account > Security > Third-party apps](https://myaccount.google.com/permissions).

## Troubleshooting

**"No valid Google Calendar credentials"**
Run `alfred google-auth` again. The token may have expired or been revoked.

**Wife's calendar not showing up**
Make sure she shared it with your Google account (Step 3), then run `alfred list-calendars` to verify.

**Wrong timezone on events**
Update the `timezone` field in `config/calendar.yaml`. Use IANA timezone names (e.g., `America/Chicago`, `America/New_York`, `US/Pacific`).
