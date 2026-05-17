# Alfred

A local-first, multi-agent personal assistant for family productivity. Conversational when you need it, autonomous when you don't.

Alfred orchestrates a set of specialized AI agents — calendar, meal planning, research, news, podcast summarization, household tasks — behind a single conversational and scheduled interface. The system is designed to run on a home server, keep your data on your own hardware by default, and dispatch to a paid cloud model only when local inference cannot do the job and budget allows.

## What it does

| Agent | Purpose |
|---|---|
| Calendar Manager | Weekly digests, daily briefings, conflict detection across family members, natural-language scheduling |
| Meal Planner | Weekly meal plans, grocery lists, recipe suggestions based on what you have |
| Research Agent | Product comparisons, local area lookups, summaries of long articles |
| News Synthesizer | Daily or weekly news digests filtered by dynamic topics of interest |
| Podcast Summarizer | Transcribes podcasts locally on GPU, summarizes episodes matching tracked topics |
| Task Manager | Shared household to-dos, maintenance reminders, chore tracking |

Agents are reachable through three interfaces:

- A command-line interface (`alfred ask "..."`, `alfred chat`)
- Scheduled routines that fire on cron (weekly digests, daily briefings)
- iMessage relay through BlueBubbles on a Mac Mini (planned; deferred until that hardware is in place)

## Architecture

```
                    ┌──────────────────────────────────────────────┐
   CLI / cron ────▶ │   LangGraph Orchestrator (Alfred)            │
   iMessage  ────▶  │                                              │
                    │   intent routing → budget gate → agent       │
                    │   dispatch → optional notification           │
                    └────────────┬─────────────────────────────────┘
                                 │
                ┌────────────────┼──────────────────────────────┐
                │                │                              │
                ▼                ▼                              ▼
        ┌──────────────┐  ┌──────────────┐         ┌─────────────────────┐
        │ Pydantic AI  │  │ Tool ACL      │        │ LiteLLM Proxy       │
        │ Agents       │──│ enforcement   │───────▶│ (model gateway)     │
        │ (calendar,   │  │ per-agent     │        │                     │
        │  meals, ...) │  │ allowlist     │        │  routes to:         │
        └──────────────┘  └───────────────┘        │  • Ollama (local)   │
                                                   │  • Anthropic Claude │
                                                   └──────────┬──────────┘
                                                              │
        ┌──────────────────────────────────────────┐          │
        │ Arize Phoenix (OpenTelemetry traces,     │◀─────────┘
        │ token + cost accounting per call)        │
        └──────────────────────────────────────────┘
```

Key architectural rules:

- Every LLM call flows through the LiteLLM Proxy. Agents never reach a provider SDK directly. This is what makes "swap Phi-4 for Mistral" or "route this kind of request to Claude instead" a config change rather than a code change.
- Each agent has an explicit tool allowlist enforced by the orchestrator. The calendar agent cannot call the web; the research agent cannot modify the calendar.
- Budget enforcement is dual-layer. Per-call cost is read from LiteLLM's response header and recorded in a SQLite ledger. Both token caps and dollar caps are checked before dispatch, per agent and globally.
- The intent router fails closed. If no agent confidently matches a request, the orchestrator returns a polite refusal rather than silently dispatching to a default agent.
- Audio never leaves the home network. faster-whisper runs on the local GPU; transcripts stay on the device.

## Technology choices

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12+ | Pydantic, async, AI ecosystem |
| Orchestration | LangGraph | State machine + conditional edges for routing/budget/dispatch |
| Per-agent framework | Pydantic AI | Typed tools from Python signatures, framework-managed tool loop, no hand-rolled provider format conversion |
| Model gateway | LiteLLM Proxy (Docker) | Provider-neutral, OpenAI-compatible API, built-in cost calculation, fallback chains |
| Local inference | Ollama (host) + Phi-4 14B | ~9 GB VRAM, strong reasoning, 5-minute model TTL matches job-shaped workload |
| Speech-to-text | Speaches (faster-whisper, Docker) | GPU-accelerated, OpenAI-compatible `/v1/audio/transcriptions`, 5-minute model TTL |
| Cloud LLM | Anthropic Claude (Sonnet 4.6 default, Opus 4.7 for hard reasoning) | Reserved for requests the router classifies as requiring it |
| Observability | Arize Phoenix + OpenTelemetry | Traces every LLM call with tokens, cost, and full prompt/response |
| Workflow triggers | n8n (Docker) | Cron + webhook receivers; planned for iMessage and RSS ingestion |
| Storage | SQLite via `aiosqlite` | Audit log, budget ledger, topic interest store |
| Configuration | YAML in `config/`, secrets in `.env` | Human-readable, version-controlled, secrets segregated |

## Cost control

Two budget axes are enforced before any agent dispatch:

- **Token caps** per agent per day, plus global daily and monthly limits.
- **Dollar caps** per agent per day, plus global daily and monthly limits. Local Ollama calls cost zero by definition, so the dollar caps effectively constrain cloud (Claude) spend.

Caps are configured in `config/settings.yaml`. If any cap is hit, the orchestrator refuses the request with a message naming which budget was exhausted.

Cost is sourced from LiteLLM's `x-litellm-response-cost` HTTP header on every gateway response — LiteLLM maintains pricing for every model it routes to, so Alfred never owns a price table.

## Deployment shape

The system is designed as two physical machines:

- A small always-on host (intended to be a Mac Mini) runs the orchestrator process, the CLI, the scheduler, and the BlueBubbles iMessage relay.
- A GPU host (RTX 4090) runs Ollama and Speaches for local inference and transcription.

Currently both run on the GPU host. The split is intended to land when the Mac Mini is in place. The architecture treats this as a deployment-time concern — `OLLAMA_BASE_URL` and `WHISPER_BASE_URL` are environment variables, not code constants.

## Repository layout

```
src/alfred/
  orchestrator/      LangGraph graph, nodes, state
  agents/            AgentBase + concrete agents (calendar, ...)
  routing/           LiteLLM proxy client (thin OpenAI SDK wrapper)
  core/              Pydantic models, config, constants, exceptions
  budget/            Token + dollar enforcement policy and ledger
  tools/             Tool base class, registry, per-agent ACL
  notifications/     Notification routing (iMessage, email)
  topics/            Dynamic topic interest store shared across agents
  scheduler/         APScheduler-based cron runner
  audit/             SQLite audit log for every orchestrator event
  storage/           Database + migrations
  observability.py   Phoenix + OpenTelemetry instrumentation
  app.py             Application bootstrap (wires everything)

cli/main.py          Click CLI entry point
config/              YAML configuration (agents, budget, models, calendar, etc.)
tests/               Pytest suite
docs/                Setup guides and infrastructure decisions
docker-compose.yml   LiteLLM, Phoenix, n8n, Whisper services
```

## Getting started

Prerequisites: Python 3.12+, `uv`, Docker with NVIDIA Container Toolkit, an RTX-class GPU, and an Ollama install on the host.

```bash
# Install dependencies
uv sync

# Configure secrets
cp .env.example .env
# edit .env — at minimum, set ANTHROPIC_API_KEY and LITELLM_MASTER_KEY

# Bring up the supporting services
docker compose up -d

# Pull the local model on the GPU host
ollama pull phi4:14b-q4_K_M

# Smoke test
uv run alfred ask "What can you help me with?"
```

Calendar setup requires a Google Cloud OAuth client — see [`docs/calendar-setup.md`](docs/calendar-setup.md).

## CLI

```
alfred ask <message>         One-shot question to the orchestrator
alfred chat                  Interactive conversational mode
alfred budget                Show token and dollar spend by period
alfred health                Check whether LiteLLM, DB, and agents are alive
alfred topics                List active topics of interest
alfred add-topic <name>      Add a topic (priority: high/normal/low)
alfred remove-topic <name>   Remove a topic
alfred scheduler             Run the scheduler in foreground (long-lived)
alfred run-routine <name>    Fire a single named routine once (for cron/systemd)
alfred google-auth           Run the Google Calendar OAuth flow (one-time)
alfred list-calendars        List calendars accessible to the authenticated account
```

## Security posture

- The repository is public; `CLAUDE.md` defines a blocking secrets-audit step before any commit or push.
- Secrets live in `.env` (gitignored). `.env.example` carries only variable names and safe placeholders.
- Google OAuth tokens (`config/google_credentials.json`, `config/google_token.json`) and the SQLite database are gitignored.
- Tool ACLs prevent cross-agent capability escalation. Per-agent scope-refusal language in system prompts prevents the LLM from answering off-topic requests even if no tool would be invoked.
- LiteLLM Proxy holds the only outbound API keys; agent code never sees the upstream Anthropic key.

## Status

Early development. The orchestrator, the calendar agent (Pydantic AI), the scheduler, budget enforcement, and Whisper integration are working. The remaining Phase 1 work is Google Calendar OAuth + a real weekly digest, and BlueBubbles iMessage delivery once the Mac Mini is provisioned. Meal planning, research, news, podcast, and task agents are scaffolded but not yet implemented.

See [`PRODUCT_VISION.md`](PRODUCT_VISION.md) for the full product scope and phased rollout, and [`CLAUDE.md`](CLAUDE.md) for the working agreement and repository conventions.
