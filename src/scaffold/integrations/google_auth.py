"""Shared Google OAuth helper.

Holds the consent flow used by `alfred google-auth` and the credential loader
used by any Google-API channel (Gmail, Calendar, …). One token file at
`config/google_token.json` carries whatever scopes were granted at the most
recent consent — re-run the CLI to grant additional scopes.
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

_TOKEN_FILE_MODE = 0o600


def _write_token(path: Path, body: str) -> None:
    """Write the OAuth token file with restrictive perms (0600).

    Default umask leaves it 0644 — group/other readable. The refresh token
    carries gmail.modify + calendar scope, so any local UID could impersonate
    Alfred. chmod after write closes the window on the existing file even if
    it was created with a more permissive mode previously.
    """
    path.write_text(body)
    os.chmod(path, _TOKEN_FILE_MODE)

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
    _write_token(token_path, creds.to_json())
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
        _write_token(token_path, creds.to_json())
        log.info("google_oauth_token_refreshed", path=str(token_path))
        return creds

    return None
