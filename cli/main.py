"""CLI entry point for Alfred system."""

from __future__ import annotations

import asyncio

import click
import structlog

log = structlog.get_logger()


@click.group()
def main() -> None:
    """Alfred — Multi-agent personal assistant for family productivity."""


@main.command()
def chat() -> None:
    """Interactive conversational mode."""
    asyncio.run(_chat_loop())


@main.command()
@click.argument("message")
def ask(message: str) -> None:
    """Send a one-shot question to Alfred."""
    asyncio.run(_ask(message))


@main.command()
def budget() -> None:
    """Show current token spend summary."""
    asyncio.run(_show_budget())


@main.command()
def scheduler() -> None:
    """Run the scheduler in foreground (long-lived). Honors enabled tasks in settings.yaml."""
    asyncio.run(_run_scheduler())


@main.command(name="run-routine")
@click.argument("name")
def run_routine(name: str) -> None:
    """Fire a single named routine once (for systemd timers, ad-hoc testing)."""
    raise SystemExit(asyncio.run(_run_routine_once(name)))


@main.command()
def topics() -> None:
    """List active topic interests."""
    asyncio.run(_show_topics())


@main.command()
@click.argument("name")
@click.option("--priority", default="normal", type=click.Choice(["high", "normal", "low"]))
def add_topic(name: str, priority: str) -> None:
    """Add a topic of interest."""
    asyncio.run(_add_topic(name, priority))


@main.command()
@click.argument("name")
def remove_topic(name: str) -> None:
    """Remove a topic of interest."""
    asyncio.run(_remove_topic(name))


@main.command()
def health() -> None:
    """Check health of all services (LiteLLM, BlueBubbles, etc.)."""
    asyncio.run(_health_check())


@main.command()
def google_auth() -> None:
    """Run the Google Calendar OAuth2 flow (one-time setup)."""
    from alfred.agents.calendar.google_client import GoogleCalendarClient

    click.echo("Starting Google Calendar OAuth2 flow...")
    click.echo("A browser window will open. Sign in and grant calendar access.\n")
    try:
        GoogleCalendarClient.run_oauth_flow()
        click.echo("\nAuthentication successful! Token saved to config/google_token.json")
        click.echo("\nNext steps:")
        click.echo("  1. Share your wife's calendar with your Google account")
        click.echo("  2. Run 'alfred list-calendars' to find the calendar IDs")
        click.echo("  3. Add the IDs to config/calendar.yaml")
    except FileNotFoundError as e:
        click.echo(f"\nError: {e}")
        click.echo("See docs/calendar-setup.md for instructions.")
    except Exception as e:
        click.echo(f"\nAuthentication failed: {e}")


@main.command()
def list_calendars() -> None:
    """List all Google Calendars accessible to the authenticated account."""
    from alfred.agents.calendar.google_client import GoogleCalendarClient

    try:
        client = GoogleCalendarClient()
        calendars = client.list_calendars()
        click.echo("Accessible calendars:\n")
        for cal in calendars:
            primary = " (PRIMARY)" if cal["primary"] else ""
            click.echo(f"  {cal['summary']}{primary}")
            click.echo(f"    ID: {cal['id']}")
            click.echo(f"    Access: {cal['access_role']}")
            click.echo()
    except RuntimeError as e:
        click.echo(f"Error: {e}")
        click.echo("Run 'alfred google-auth' first.")


# ── Implementation ───────────────────────────────────────────


async def _get_app():
    from alfred.app import create_app

    return await create_app()


async def _chat_loop() -> None:
    from alfred.core.constants import RequestSource
    from alfred.core.models import AgentRequest

    app = await _get_app()
    click.echo("Alfred interactive mode. Type 'quit' to exit.\n")

    try:
        while True:
            try:
                user_input = click.prompt("You", prompt_suffix="> ")
            except (EOFError, KeyboardInterrupt):
                break

            if user_input.strip().lower() in ("quit", "exit", "q"):
                break

            request = AgentRequest(
                source=RequestSource.CLI,
                user_message=user_input,
                user_id="andres",
            )

            response = await app.orchestrator.handle(request)
            click.echo(f"\n{response.agent_name}: {response.message}\n")

    finally:
        await app.shutdown()
        click.echo("Goodbye.")


async def _ask(message: str) -> None:
    from alfred.core.constants import RequestSource
    from alfred.core.models import AgentRequest

    app = await _get_app()
    try:
        request = AgentRequest(
            source=RequestSource.CLI,
            user_message=message,
            user_id="andres",
        )
        response = await app.orchestrator.handle(request)
        click.echo(f"[{response.agent_name}] {response.message}")
    finally:
        await app.shutdown()


async def _show_budget() -> None:
    app = await _get_app()
    try:
        for period in ("today", "week", "month"):
            summary = await app.budget_tracker.get_spend_summary(period)
            click.echo(f"\n── {period.upper()} ──")
            click.echo(f"  Total tokens: {summary.total_tokens:,}")
            click.echo(f"  Total cost:   ${summary.total_cost_usd:.4f}")
            if summary.by_agent:
                click.echo("  By agent:")
                for agent, tokens in summary.by_agent.items():
                    click.echo(f"    {agent}: {tokens:,}")
    finally:
        await app.shutdown()


async def _run_scheduler() -> None:
    from alfred.scheduler.runner import run_scheduler

    app = await _get_app()
    try:
        await run_scheduler(app)
    finally:
        await app.shutdown()


async def _run_routine_once(name: str) -> int:
    from alfred.scheduler.runner import run_routine_by_name

    app = await _get_app()
    try:
        return await run_routine_by_name(app, name)
    finally:
        await app.shutdown()


async def _show_topics() -> None:
    app = await _get_app()
    try:
        active = await app.topic_store.list_active()
        if not active:
            click.echo("No active topics. Add one with: alfred add-topic <name>")
            return
        click.echo("Active topics:")
        for t in active:
            click.echo(f"  [{t.priority}] {t.name}")
    finally:
        await app.shutdown()


async def _add_topic(name: str, priority: str) -> None:
    app = await _get_app()
    try:
        from alfred.core.constants import TopicPriority

        topic = await app.topic_store.add(name, TopicPriority(priority), added_by="cli")
        click.echo(f"Added topic: [{topic.priority}] {topic.name}")
    finally:
        await app.shutdown()


async def _remove_topic(name: str) -> None:
    app = await _get_app()
    try:
        removed = await app.topic_store.remove(name)
        if removed:
            click.echo(f"Removed topic: {name}")
        else:
            click.echo(f"Topic not found or already removed: {name}")
    finally:
        await app.shutdown()


async def _health_check() -> None:
    app = await _get_app()
    try:
        litellm_ok = await app.litellm_client.health_check()
        click.echo(f"LiteLLM Proxy:  {'OK' if litellm_ok else 'UNAVAILABLE'}")

        for name, channel in app.notification_service._channels.items():
            if hasattr(channel, "health_check"):
                ok = await channel.health_check()
                click.echo(f"{name}:  {'OK' if ok else 'UNAVAILABLE'}")

        click.echo(f"Database:       OK ({app.settings.database_path})")
        agents = ", ".join(app.agent_registry.list_names()) or "none registered"
        click.echo(f"Agents:         {agents}")
    finally:
        await app.shutdown()


if __name__ == "__main__":
    main()
