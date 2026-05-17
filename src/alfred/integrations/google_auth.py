"""Shared Google OAuth helper.

Holds the consent flow used by `alfred google-auth` and the credential loader
used by any Google-API channel (Gmail, Calendar, …). One token file at
`config/google_token.json` carries whatever scopes were granted at the most
recent consent — re-run the CLI to grant additional scopes.
"""

from __future__ import annotations

from pathlib import Path

import structlog
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

log = structlog.get_logger()

GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"
GMAIL_MODIFY_SCOPE = "https://www.googleapis.com/auth/gmail.modify"
CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar"
DEFAULT_SCOPES: list[str] = [GMAIL_SEND_SCOPE, GMAIL_MODIFY_SCOPE, CALENDAR_SCOPE]


def run_consent_flow(
    credentials_path: Path,
    token_path: Path,
    scopes: list[str],
    open_browser: bool = True,
) -> Credentials:
    """Run the one-time browser-based consent flow and persist tokens.

    For Desktop OAuth clients. If running headless / over SSH, set
    `open_browser=False` and forward the printed localhost URL through the
    SSH session (e.g. `ssh -L 8080:localhost:8080 homelab`).
    """
    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Missing OAuth client file at {credentials_path}. "
            "Download it from Google Cloud Console → Credentials → your OAuth "
            "Client ID → Download icon."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), scopes=scopes)
    creds = flow.run_local_server(port=0, open_browser=open_browser)

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    log.info("google_oauth_token_saved", path=str(token_path), scopes=scopes)
    return creds


def load_credentials(
    token_path: Path,
    credentials_path: Path,
    scopes: list[str],
) -> Credentials | None:
    """Load and refresh stored credentials. Returns None if no token yet."""
    if not token_path.exists():
        return None

    creds = Credentials.from_authorized_user_file(str(token_path), scopes)
    if creds.valid:
        return creds

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json())
        log.info("google_oauth_token_refreshed", path=str(token_path))
        return creds

    return None
