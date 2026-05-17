"""Application bootstrap — wires all components together."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
from alfred.tools.access_control import ToolAccessControl
from alfred.tools.registry import ToolRegistry
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
    ToolAccessControl(settings.agents)  # validates allowlists at startup
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
        from alfred.notifications.channels.email import EmailClient

        channels["email"] = EmailClient(
            credentials_path=config_dir / "google_credentials.json",
            token_path=config_dir / "google_token.json",
            from_name=settings.notifications.email_from_name,
            from_address=settings.notifications.email_from_address,
        )

    notification_service = NotificationService(
        channels=channels,
        config=settings.notifications,
        audit_logger=audit_logger,
        db=db,
    )

    # 7. Agent registry — register concrete agents
    agent_registry = AgentRegistry()
    _register_agents(
        agent_registry, settings, litellm_client, notification_service, config_dir
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
    )


def _register_agents(
    registry: AgentRegistry,
    settings: Settings,
    litellm_client: LiteLLMClient,
    notification_service: NotificationService,
    config_dir: Path,
) -> None:
    """Register all concrete agents that are enabled in config."""
    # Calendar agent
    cal_config = settings.agents.get("calendar")
    if cal_config and cal_config.enabled:
        try:
            from alfred.agents.calendar.agent import CalendarAgent, CalendarConfig
            from alfred.agents.calendar.google_client import GoogleCalendarClient

            calendar_config = CalendarConfig(config_dir)
            google_client = GoogleCalendarClient(
                calendar_ids=calendar_config.calendar_ids,
                timezone=calendar_config.timezone,
            )
            agent = CalendarAgent(
                google_client=google_client,
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
