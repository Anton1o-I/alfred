"""Configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from alfred.core.constants import ModelPreference, NotificationChannel


class LiteLLMConfig(BaseModel):
    """LiteLLM Proxy connection settings."""

    base_url: str = "http://localhost:4000"
    api_key_env: str = "LITELLM_MASTER_KEY"


class BudgetConfig(BaseModel):
    """Token and dollar budget limits."""

    per_request_max_tokens: int = 8192
    daily_agent_limits: dict[str, int] = {}
    global_daily_limit: int = 500_000
    global_monthly_limit: int = 10_000_000

    # Dollar caps (0 = unlimited). Enforced in addition to token caps.
    daily_agent_usd_limits: dict[str, float] = {}
    global_daily_usd_limit: float = 5.0
    global_monthly_usd_limit: float = 100.0


class AgentConfig(BaseModel):
    """Declaration of a single agent."""

    name: str
    description: str
    enabled: bool = True
    allowed_tools: list[str] = []
    model_preference: ModelPreference = ModelPreference.AUTO
    timeout_seconds: int = 60
    max_retries: int = 3


class RecipientConfig(BaseModel):
    """A notification recipient (family member)."""

    user_id: str
    preferred_channel: NotificationChannel = NotificationChannel.IMESSAGE
    phone: str | None = None
    email: str | None = None
    email_env: str | None = None  # If set, takes precedence over `email` (env-var indirection)


class NotificationConfig(BaseModel):
    """Notification system configuration."""

    imessage_enabled: bool = True
    bluebubbles_url_env: str = "BLUEBUBBLES_URL"
    bluebubbles_password_env: str = "BLUEBUBBLES_PASSWORD"

    email_enabled: bool = False
    # "gmail" (OAuth, 7-day Testing limit) or "icloud" (SMTP, app password).
    email_provider: str = "icloud"
    email_from_name: str = "Alfred"  # Default display name in the From header
    # Per-agent From-name overrides — keyed by agent name. Empty falls back to default.
    email_from_names_by_agent: dict[str, str] = {}
    # Per-agent signature taglines (one short line below the persona name).
    # Empty string for an agent → no signature appended.
    email_taglines_by_agent: dict[str, str] = {}

    # Gmail-specific (when email_provider == "gmail")
    sendgrid_api_key_env: str = "SENDGRID_API_KEY"  # legacy, unused
    email_from_address: str = ""  # Must match the OAuth-authorized account

    # iCloud-specific (when email_provider == "icloud"). Values come from env
    # vars so the alias address never lands in a tracked file.
    #
    # iCloud's three protocols disagree on which identifier they accept:
    #   SMTP / IMAP → the @icloud.com alias address (ALFRED_ICLOUD_EMAIL)
    #   CalDAV      → the primary Apple ID email (ALFRED_ICLOUD_USERNAME)
    # We split them so each protocol uses what works.
    icloud_email_env: str = "ALFRED_ICLOUD_EMAIL"
    icloud_username_env: str = "ALFRED_ICLOUD_USERNAME"  # CalDAV only
    icloud_app_password_env: str = "ALFRED_ICLOUD_APP_PASSWORD"

    recipients: list[RecipientConfig] = []


class ScheduledTaskConfig(BaseModel):
    """A scheduled job definition."""

    name: str
    cron: str
    agent_name: str
    message: str
    enabled: bool = True
    notification_channel: NotificationChannel | None = None


class InboxConfig(BaseModel):
    """Inbound-email gating: who Alfred will accept commands from over email."""

    authorized_senders: list[str] = []  # Lower-cased on load; exact-match required
    enabled: bool = False  # Set true once OAuth has gmail.modify scope


class Settings(BaseModel):
    """Root configuration object."""

    agents: dict[str, AgentConfig] = {}
    litellm: LiteLLMConfig = LiteLLMConfig()
    budget: BudgetConfig = BudgetConfig()
    notifications: NotificationConfig = NotificationConfig()
    inbox: InboxConfig = InboxConfig()
    scheduled_tasks: list[ScheduledTaskConfig] = []
    database_path: str = "data/alfred.db"
    log_level: str = "INFO"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning empty dict if it doesn't exist."""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_settings(config_dir: Path = Path("config")) -> Settings:
    """Load and merge all YAML config files, validate, return Settings."""
    settings_data = _load_yaml(config_dir / "settings.yaml")
    agents_data = _load_yaml(config_dir / "agents.yaml")
    notifications_data = _load_yaml(config_dir / "notifications.yaml")

    # Build AgentConfig objects from agents.yaml, injecting the key as the name
    agents = {}
    for key, val in agents_data.items():
        if isinstance(val, dict):
            val["name"] = key
            agents[key] = AgentConfig(**val)

    # Merge scheduled tasks from settings if present
    scheduled_tasks = [
        ScheduledTaskConfig(**t) for t in settings_data.pop("scheduled_tasks", [])
    ]

    inbox_data = settings_data.pop("inbox", {})
    if "authorized_senders" in inbox_data:
        # Authorized senders can also be env-var refs (e.g. "${ALFRED_ALLOWED_SENDER_1}")
        # to keep personal addresses out of tracked files.
        resolved: list[str] = []
        for s in inbox_data["authorized_senders"]:
            if isinstance(s, str) and s.startswith("${") and s.endswith("}"):
                val = os.environ.get(s[2:-1], "").strip().lower()
                if val:
                    resolved.append(val)
            elif s:
                resolved.append(s.lower())
        inbox_data["authorized_senders"] = resolved

    # Resolve email_env -> email indirection on each recipient
    for r in notifications_data.get("recipients", []):
        env_key = r.get("email_env")
        if env_key and not r.get("email"):
            r["email"] = os.environ.get(env_key, "").strip() or None

    return Settings(
        agents=agents,
        litellm=LiteLLMConfig(**settings_data.pop("litellm", {})),
        budget=BudgetConfig(**settings_data.pop("budget", {})),
        notifications=NotificationConfig(**notifications_data),
        inbox=InboxConfig(**inbox_data),
        scheduled_tasks=scheduled_tasks,
        **settings_data,
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return cached settings. Call load_settings() first."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def init_settings(config_dir: Path = Path("config")) -> Settings:
    """Load settings and cache them."""
    global _settings
    _settings = load_settings(config_dir)
    return _settings
