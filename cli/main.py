"""CLI entry point for Alfred system."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import click
import structlog
from dotenv import load_dotenv

# Load .env before anything else reads os.environ. Quoted values (e.g. for
# passwords containing '#') are stripped of their quotes automatically.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

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
@click.option(
    "--compare",
    is_flag=True,
    help="For curator: run synthesis through both local and cloud, write A/B file.",
)
@click.option(
    "--max-items",
    type=int,
    default=None,
    help="For curator: cap fetched candidates before triage (cost/safety bound).",
)
def run_routine(name: str, compare: bool, max_items: int | None) -> None:
    """Fire a single named routine once (for systemd timers, ad-hoc testing)."""
    raise SystemExit(asyncio.run(_run_routine_once(name, compare=compare, max_items=max_items)))


@main.command()
@click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to a YAML scenario file or directory. Default: tests/scenarios/",
)
@click.option(
    "--scenario",
    "scenario_name",
    default=None,
    help="Run only the named scenario.",
)
@click.option("--verbose", "-v", is_flag=True)
def simulate(
    file_path: Path | None, scenario_name: str | None, verbose: bool
) -> None:
    """Run calendar scenarios against an in-memory calendar.

    Real LLM (Qwen by default), fake I/O. Useful for catching prompt /
    routing regressions before hitting iCloud. Scenarios live in
    tests/scenarios/*.yaml.
    """
    asyncio.run(_run_simulation(file_path, scenario_name, verbose))


async def _run_simulation(
    file_path: Path | None, scenario_name: str | None, verbose: bool
) -> None:
    from sim.runner import load_scenarios, print_report, run_scenarios

    target = file_path or (Path(__file__).resolve().parent.parent / "tests" / "scenarios")
    scenarios = load_scenarios(target)
    if scenario_name:
        scenarios = [s for s in scenarios if s.name == scenario_name]
        if not scenarios:
            click.echo(f"No scenario found matching name: {scenario_name}")
            return

    # Build only what we need (LiteLLMClient) — avoid opening the real
    # data/alfred.db so the simulate command can run alongside the
    # scheduler without contention.
    import os

    from alfred.core.config import init_settings
    from alfred.observability import init_observability
    from alfred.routing.clients import LiteLLMClient

    settings = init_settings(Path("config"))
    init_observability()
    litellm_client = LiteLLMClient(
        base_url=settings.litellm.base_url,
        api_key=os.environ.get(settings.litellm.api_key_env, ""),
    )
    try:
        results = await run_scenarios(
            scenarios,
            litellm_client=litellm_client,
            verbose=verbose,
        )
        print_report(results)
    finally:
        await litellm_client.close()


@main.command(name="simulate-briefing")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to write rendered .md and .html outputs. Default: data/sim_briefings/",
)
def simulate_briefing(output_dir: Path | None) -> None:
    """Render the daily briefing across a handful of scenarios.

    Calls the real local LLM, dumps .md + .html per scenario so you can
    eyeball the output before deploying briefing changes.
    """
    asyncio.run(_run_briefing_sim(output_dir))


async def _run_briefing_sim(output_dir: Path | None) -> None:
    from sim.briefing_sim import run_briefing_sim

    from alfred.observability import init_observability

    init_observability()
    code = await run_briefing_sim(output_dir)
    if code:
        raise SystemExit(code)


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


@main.group()
def prompts() -> None:
    """Inspect the versioned prompt registry."""


@prompts.command(name="list")
def prompts_list() -> None:
    """List all prompts with their active version."""
    from alfred.prompts import list_prompts

    for p in list_prompts():
        click.echo(f"{p.id:30s}  active={p.active_version:6s}  versions={p.all_versions}")
        click.echo(f"  {p.description.strip()}")


@prompts.command(name="show")
@click.argument("prompt_id")
@click.option("--version", default=None, help="Specific version (default: active).")
def prompts_show(prompt_id: str, version: str | None) -> None:
    """Show the full body of a prompt at a specific version."""
    from alfred.prompts import load_prompt

    p = load_prompt(prompt_id, version=version)
    click.echo(f"# {p.fqn}  ({p.status})")
    click.echo(f"# author: {p.author}    created: {p.created}")
    click.echo(f"# changelog: {p.changelog.strip()}")
    click.echo()
    click.echo(p.body)


@main.command()
@click.option(
    "--no-browser",
    is_flag=True,
    help="Don't open a browser (SSH case — forward the printed localhost URL).",
)
def google_auth(no_browser: bool) -> None:
    """Run the Google OAuth consent flow. Grants Gmail + Calendar scopes in one consent."""
    from pathlib import Path

    from alfred.integrations.google_auth import DEFAULT_SCOPES, run_consent_flow

    config_dir = Path(__file__).resolve().parent.parent / "config"

    click.echo("Starting Google OAuth flow (Gmail send + modify + Calendar)...")
    click.echo("Sign in as Alfred's gmail account when the browser opens.\n")
    try:
        creds = run_consent_flow(
            credentials_path=config_dir / "google_credentials.json",
            token_path=config_dir / "google_token.json",
            scopes=DEFAULT_SCOPES,
            open_browser=not no_browser,
        )
        click.echo("\nAuthentication successful.")
        click.echo(f"Granted scopes: {creds.scopes}")
        click.echo(f"Token saved to: {config_dir / 'google_token.json'}")
    except FileNotFoundError as e:
        click.echo(f"\nError: {e}")
    except Exception as e:
        click.echo(f"\nAuthentication failed: {e}")


@main.command()
def list_calendars() -> None:
    """List accessible calendars on the configured provider (iCloud or Google)."""
    from alfred.core.config import load_settings

    settings = load_settings()
    config_dir = Path(__file__).resolve().parent.parent / "config"
    # Reuse the same provider-selection logic the app uses.
    from alfred.agents.calendar.agent import CalendarConfig
    from alfred.app import _build_calendar_client

    cal_cfg = CalendarConfig(config_dir)
    client = _build_calendar_client(cal_cfg, settings)
    if client is None:
        click.echo(f"Calendar client unavailable for provider '{cal_cfg.provider}'.")
        click.echo("Check .env values and config/calendar.yaml.")
        return
    try:
        calendars = client.list_calendars()
        click.echo(f"Accessible calendars (provider: {cal_cfg.provider}):\n")
        for cal in calendars:
            primary = " (PRIMARY)" if cal.get("primary") else ""
            click.echo(f"  {cal['summary']}{primary}")
            click.echo(f"    ID: {cal['id']}")
            click.echo(f"    Access: {cal.get('access_role', '?')}")
            click.echo()
    except Exception as e:
        click.echo(f"Error: {e}")


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
                user_id=os.environ.get("ALFRED_USER_ID", "primary"),
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
            user_id=os.environ.get("ALFRED_USER_ID", "primary"),
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


async def _run_routine_once(
    name: str, compare: bool = False, max_items: int | None = None
) -> int:
    from alfred.scheduler.runner import run_routine_by_name

    app = await _get_app()
    try:
        return await run_routine_by_name(app, name, compare=compare, max_items=max_items)
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
