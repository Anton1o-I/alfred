"""Application bootstrap — wires all components together."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from alfred.agents.registry import AgentRegistry
from alfred.audit.logger import AuditLogger
from alfred.budget.policies import BudgetPolicy
from alfred.budget.tracker import BudgetTracker
from alfred.core.config import Settings, init_settings
from alfred.notifications.channels.imessage import BlueBubblesClient
from alfred.notifications.service import NotificationService
from alfred.orchestrator.orchestrator import Orchestrator
from alfred.routing.clients import LiteLLMClient
from alfred.storage.database import Database
from scaffold.tools.registry import ToolRegistry
from alfred.topics.store import TopicStore

log = structlog.get_logger()


@dataclass
class App:
    """Fully wired application instance."""

    settings: Settings
    db: Database
    orchestrator: Orchestrator
    topic_store: TopicStore
    budget_tracker: BudgetTracker
    budget_policy: BudgetPolicy
    notification_service: NotificationService
    agent_registry: AgentRegistry
    tool_registry: ToolRegistry
    litellm_client: LiteLLMClient
    inbox_poller: Any = None  # InboxPoller | None — typed loosely to keep import chain quiet

    async def shutdown(self) -> None:
        """Clean up resources."""
        await self.litellm_client.close()
        await self.db.close()
        log.info("app_shutdown")


async def create_app(config_dir: Path = Path("config")) -> App:
    """Bootstrap the application: load config, initialize all services, wire together.

    Sequence:
    1. Load settings from YAML + env
    2. Initialize observability (Phoenix/OTel)
    3. Initialize Database (run migrations)
    4. Create core services (audit, budget, tools, topics)
    5. Create LiteLLM client + notification service
    6. Create agent registry (agents register themselves)
    7. Create orchestrator (LangGraph graph)
    8. Return wired App
    """
    # 1. Config
    settings = init_settings(config_dir)

    # 2. Observability (best-effort, non-blocking)
    from alfred.observability import init_observability

    init_observability()

    # 3. Database
    db = Database(settings.database_path)
    await db.initialize()
    log.info("database_initialized", path=settings.database_path)

    # 4. Core services
    audit_logger = AuditLogger(db)
    budget_policy = BudgetPolicy(settings.budget, db)
    budget_tracker = BudgetTracker(db)
    tool_registry = ToolRegistry()
    topic_store = TopicStore(db)

    # 5. LiteLLM client
    import os

    litellm_client = LiteLLMClient(
        base_url=settings.litellm.base_url,
        api_key=os.environ.get(settings.litellm.api_key_env, ""),
    )

    # 6. Notification service
    channels: dict = {}
    if settings.notifications.imessage_enabled:
        bb_url = os.environ.get(settings.notifications.bluebubbles_url_env, "")
        bb_pass = os.environ.get(settings.notifications.bluebubbles_password_env, "")
        if bb_url:
            channels["imessage"] = BlueBubblesClient(bb_url, bb_pass)

    if settings.notifications.email_enabled:
        email_channel = _build_email_channel(settings, config_dir)
        if email_channel is not None:
            channels["email"] = email_channel

    notification_service = NotificationService(
        channels=channels,
        config=settings.notifications,
        audit_logger=audit_logger,
        db=db,
    )

    # 7. Agent registry — register concrete agents
    agent_registry = AgentRegistry()
    _register_agents(
        agent_registry, settings, litellm_client, notification_service, config_dir, db
    )

    # 8. Orchestrator
    orchestrator = Orchestrator(
        client=litellm_client,
        registry=agent_registry,
        budget_policy=budget_policy,
        budget_tracker=budget_tracker,
        audit_logger=audit_logger,
        notification_service=notification_service,
    )

    # 9. Inbox poller (optional — iCloud IMAP with app-specific password)
    inbox_poller: Any = None
    if settings.inbox.enabled and settings.inbox.authorized_senders:
        from alfred.inbox.poller import InboxPoller

        icloud_addr = os.environ.get(settings.notifications.icloud_email_env, "").strip()
        icloud_pw = os.environ.get(
            settings.notifications.icloud_app_password_env, ""
        ).strip()
        if icloud_addr and icloud_pw:
            # IMAP auth = the alias address itself, same as SMTP.
            inbox_poller = InboxPoller(
                username=icloud_addr,
                app_password=icloud_pw,
                authorized_senders=settings.inbox.authorized_senders,
            )
        else:
            log.warning("inbox_poller_disabled_missing_env")

    log.info("app_initialized", agents=agent_registry.list_names())

    return App(
        settings=settings,
        db=db,
        orchestrator=orchestrator,
        topic_store=topic_store,
        budget_tracker=budget_tracker,
        budget_policy=budget_policy,
        notification_service=notification_service,
        agent_registry=agent_registry,
        tool_registry=tool_registry,
        litellm_client=litellm_client,
        inbox_poller=inbox_poller,
    )


def _build_email_channel(settings: Settings, config_dir: Path) -> Any:
    """Pick the email channel based on notifications.email_provider."""
    cfg = settings.notifications
    provider = cfg.email_provider.lower()

    if provider == "icloud":
        from alfred.notifications.channels.icloud_email import IcloudEmailClient

        addr = os.environ.get(cfg.icloud_email_env, "").strip()
        pw = os.environ.get(cfg.icloud_app_password_env, "").strip()
        # SMTP auth = the alias address itself; the primary doesn't work here.
        username = addr or None
        if not addr or not pw:
            log.warning(
                "email_channel_disabled_missing_env",
                provider="icloud",
                missing=[
                    k
                    for k, v in [
                        (cfg.icloud_email_env, addr),
                        (cfg.icloud_app_password_env, pw),
                    ]
                    if not v
                ],
            )
            return None
        return IcloudEmailClient(
            from_address=addr,
            app_password=pw,
            from_name=cfg.email_from_name,
            smtp_username=username,
        )

    if provider == "gmail":
        from alfred.notifications.channels.email import EmailClient

        return EmailClient(
            credentials_path=config_dir / "google_credentials.json",
            token_path=config_dir / "google_token.json",
            from_name=cfg.email_from_name,
            from_address=cfg.email_from_address,
        )

    log.warning("email_provider_unknown", provider=provider)
    return None


def _build_calendar_client(
    calendar_config: Any, settings: Settings
) -> Any:
    """Pick the calendar client based on calendar.yaml provider field."""
    provider = calendar_config.provider.lower()

    if provider == "icloud":
        from alfred.agents.calendar.icloud_client import IcloudCalendarClient

        addr = os.environ.get(settings.notifications.icloud_email_env, "").strip()
        pw = os.environ.get(settings.notifications.icloud_app_password_env, "").strip()
        username = (
            os.environ.get(settings.notifications.icloud_username_env, "").strip()
            or addr
        )
        if not addr or not pw:
            log.warning(
                "calendar_client_disabled_missing_env",
                provider="icloud",
            )
            return None
        return IcloudCalendarClient(
            username=username,
            app_password=pw,
            calendar_name=calendar_config.icloud_calendar_name,
            timezone=calendar_config.timezone,
        )

    if provider == "google":
        from alfred.agents.calendar.google_client import GoogleCalendarClient

        return GoogleCalendarClient(
            calendar_ids=calendar_config.calendar_ids,
            timezone=calendar_config.timezone,
        )

    log.warning("calendar_provider_unknown", provider=provider)
    return None


def _register_agents(
    registry: AgentRegistry,
    settings: Settings,
    litellm_client: LiteLLMClient,
    notification_service: NotificationService,
    config_dir: Path,
    db: Database,
) -> None:
    """Register all concrete agents that are enabled in config."""
    # Calendar agent
    cal_config = settings.agents.get("calendar")
    if cal_config and cal_config.enabled:
        try:
            from alfred.agents.calendar.agent import CalendarAgent, CalendarConfig

            calendar_config = CalendarConfig(config_dir)
            client = _build_calendar_client(calendar_config, settings)
            if client is None:
                log.warning("calendar_agent_init_skipped", reason="no_client")
            else:
                agent = CalendarAgent(
                    google_client=client,
                    litellm_client=litellm_client,
                    calendar_config=calendar_config,
                    notification_service=notification_service,
                )
                registry.register(agent, cal_config)
        except Exception as e:
            log.warning("calendar_agent_init_failed", error=str(e))

    # Curator agent (research/learning reading-list digest)
    cur_config = settings.agents.get("curator")
    if cur_config and cur_config.enabled:
        try:
            from alfred.agents.curator import CuratorAgent, CuratorConfig

            curator_config = CuratorConfig.load(config_dir)
            agent = CuratorAgent(
                litellm_client=litellm_client,
                config=curator_config,
            )
            registry.register(agent, cur_config)
        except Exception as e:
            log.warning("curator_agent_init_failed", error=str(e))

    # Tasks agent (household chores with shame mode)
    tasks_config = settings.agents.get("tasks")
    if tasks_config and tasks_config.enabled:
        try:
            from alfred.agents.calendar.agent import CalendarConfig
            from alfred.agents.tasks.agent import TasksAgent

            tz_name = CalendarConfig(config_dir).timezone
            # Build user_id → display name map from recipient configs so the
            # tasks workflow can resolve natural-language assignee references
            # ("assign to <name>") back to the canonical user_id.
            assignee_names: dict[str, str] = {}
            for r in settings.notifications.recipients:
                if r.name_env:
                    name = os.environ.get(r.name_env, "").strip()
                    if name:
                        assignee_names[r.user_id] = name
            agent = TasksAgent(
                db=db,
                litellm_client=litellm_client,
                notification_service=notification_service,
                timezone_name=tz_name,
                assignee_names=assignee_names,
            )
            registry.register(agent, tasks_config)
        except Exception as e:
            log.warning("tasks_agent_init_failed", error=str(e))
