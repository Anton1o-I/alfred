# Home Agent Infrastructure: Model Gateway & Agent Framework Research Report

*Date: 2026-04-15*

## Context

We have a greenfield multi-agent home assistant with core infrastructure already built (config, storage, budget tracking, audit logging, tool ACL, notification service, topic store). The current model routing (`routing/model_router.py`, `routing/clients.py`) and agent framework (`agents/base.py`) are hand-rolled. The user wants to replace these with production-grade solutions: a proper model gateway for multi-provider LLM routing, and an established agent framework for orchestration.

The user also wants Arize Phoenix for observability and n8n for workflow triggering.

---

## 1. Model Gateway: LiteLLM Proxy (Recommended)

**What:** Self-hostable OpenAI-compatible proxy that routes to 100+ LLM providers through a single API endpoint.

**Why LiteLLM wins over alternatives:**

| Requirement | LiteLLM | OpenRouter | Portkey | LangChain abstractions |
|---|---|---|---|---|
| Self-hostable | Yes (Docker/pip) | No (cloud only) | Partial (gateway OSS, analytics cloud) | N/A (not a gateway) |
| Ollama + Anthropic | Native | No local models | Secondary Ollama support | Yes, but no routing |
| Budget **enforcement** | Hard limits per key/user | No | Cloud only | No |
| Cost tracking per request | Built-in `/spend` endpoint | Basic | Cloud dashboard | No |
| Tool use across providers | Normalizes to OpenAI format | Yes | Yes | Per-provider |
| Fallback/retry/load balancing | Model groups with policies | No | Yes | No |
| Python SDK | `litellm.completion()` or proxy | SDK | SDK | SDK |

**How it fits:**
- Run LiteLLM Proxy on the Mac Mini (Docker container or pip install)
- Configure two model groups: `local` (Ollama on homelab) and `cloud` (Anthropic API)
- All agents talk to `http://localhost:4000/v1/...` using the standard OpenAI Python SDK
- Routing rules, fallbacks, budget caps, and cost tracking all configured in LiteLLM's YAML config
- **Replaces:** `routing/model_router.py`, `routing/clients.py`, the hardcoded Anthropic pricing table, and our custom budget enforcement logic for cloud spend

**Alternatives considered and rejected:**
- **OpenRouter** — cloud-only, cannot route to local Ollama
- **Portkey** — advanced features require cloud dashboard, not fully self-hostable
- **Martian** — cloud-only, opaque routing, no local model support
- **Ollama OpenAI-compat API alone** — only solves local serving, no multi-provider routing
- **vLLM** — serving engine (Ollama alternative), not a gateway

---

## 2. Agent Framework: LangGraph + Pydantic AI (Recommended)

After evaluating 10 frameworks, the recommendation is a **two-layer approach**:

### LangGraph — The Orchestrator

**What:** LangChain's graph-based agent orchestration framework. Each agent is a node; routing is conditional edges; state persists via checkpointing.

**Why it wins for orchestration:**
- **Purpose-built for the orchestrator pattern** — central router dispatching to specialized agents with stateful workflows
- **Multi-model per node** — assign Claude Sonnet to research agent, local Ollama to grocery agent, different model per graph node
- **Tool restrictions at graph level** — each agent node only has access to its declared tools
- **Durable execution** — checkpointing means a crashed agent resumes from last state (critical for scheduled background agents)
- **Both autonomous and workflow patterns** — individual nodes can run ReAct tool loops, while the graph handles orchestration, routing, and handoffs
- **Best-in-class observability** — native integration with Phoenix (OpenTelemetry), Langfuse, and LangSmith

**Downsides:** Steeper learning curve than alternatives. Graph theory + state machine concepts required. LangChain dependency layer.

### Pydantic AI — The Agent Builder

**What:** Type-safe, model-agnostic framework for building individual agents. Built on Pydantic.

**Why it complements LangGraph:**
- Build each specialized agent (calendar, meals, research, news, podcast, tasks) as a Pydantic AI agent
- Each agent declares its own tools with full type safety
- 25+ LLM providers including Anthropic and Ollama natively
- Clean integration with LangGraph as nodes
- Pydantic validation ensures structured outputs from agents

### Frameworks rejected and why:

| Framework | Rejection Reason |
|---|---|
| **Anthropic Agent SDK** | Claude-only by design. Cannot use Ollama for local agents. Great for Claude-specific agents but not as orchestration framework |
| **CrewAI** | Role-based DSL too opinionated. No fine-grained tool restrictions or budget controls per agent |
| **AutoGen/AG2** | Fragmented after Microsoft merger. Near-zero security mechanisms |
| **OpenAI Agents SDK** | Async execution unavailable for non-OpenAI models. OpenAI-ecosystem-centric |
| **Haystack** | Pipeline framework, not agent orchestration. Good *inside* an agent (e.g., RAG), not *around* agents |
| **Google ADK** | Vertex AI deployment bias. Less flexible than graph-based routing. Watching but not mature enough |
| **Smolagents** | Lean and impressive benchmarks, but no checkpointing, limited guardrails, basic observability. Good backup option if LangGraph is too heavy |

---

## 3. Observability: Arize Phoenix (Recommended)

**What:** Open-source LLM observability platform (9.3k GitHub stars, very active). Built entirely on OpenTelemetry.

**Why Phoenix over Langfuse:**
- **OpenTelemetry-native** — not a proprietary SDK, uses standard OTel spans
- **Deep Python integration** — pip-installable, runs in-process or containerized
- **Built-in eval tooling** — not just tracing, includes experiment tracking and prompt management
- **Native integrations** for LangGraph, Pydantic AI, LiteLLM, Anthropic, and Ollama

**What it provides:**
- Multi-agent tracing: each agent call is a span within a trace, parent-child relationships map orchestrator-to-sub-agent flows
- Per-agent, per-model, per-request token/cost tracking out of the box
- Latency breakdown at every span
- Input/output capture for debugging

**Deployment:** Docker container on the Mac Mini, or `pip install arize-phoenix` to run alongside the agent system.

---

## 4. Workflow Triggers: n8n (Recommended as glue layer)

**What:** Self-hostable workflow automation (184k GitHub stars). Visual builder with 400+ integrations.

**Role: Trigger and glue layer, NOT the primary orchestrator.**

**What n8n is great at:**
- Cron/schedule triggers (daily news digest at 7 AM, weekly calendar digest Sunday 6 PM)
- Event triggers (new Google Calendar invite -> webhook to agent system)
- RSS feed monitoring (new podcast episode -> trigger podcast summarizer)
- Simple non-AI automations (data fetching, file handling)
- Native nodes for Google Calendar, RSS, iCalendar, HTTP webhooks

**What n8n is NOT good at:**
- Complex multi-agent reasoning with handoffs and shared state
- Fine-grained token tracking and debugging
- Programmatic control over agent loops
- Streaming responses

**How it fits:** n8n fires webhooks that trigger the LangGraph orchestrator. It handles "when" and "what triggers"; LangGraph handles "how to reason about it."

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────┐
│                    n8n (Docker)                       │
│  Cron triggers, RSS monitors, Calendar webhooks      │
│  -> fires HTTP webhooks to LangGraph orchestrator    │
└──────────────────────┬──────────────────────────────┘
                       │ webhook
┌──────────────────────▼──────────────────────────────┐
│               LangGraph Orchestrator                  │
│  Intent Router -> Conditional Edges -> Agent Nodes    │
│                                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │
│  │ Calendar │ │   Meal   │ │ Research │ │  News  │  │
│  │  Agent   │ │ Planner  │ │  Agent   │ │ Synth  │  │
│  │(Pydantic │ │(Pydantic │ │(Pydantic │ │(Pydantic│  │
│  │   AI)    │ │   AI)    │ │   AI)    │ │  AI)   │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘  │
│  ┌──────────┐ ┌──────────┐                           │
│  │ Podcast  │ │  Tasks   │                           │
│  │Summarizer│ │  Agent   │                           │
│  │(Pydantic │ │(Pydantic │                           │
│  │   AI)    │ │   AI)    │                           │
│  └──────────┘ └──────────┘                           │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              LiteLLM Proxy (Docker)                   │
│  Single endpoint: http://localhost:4000/v1/...        │
│  Routes to:                                           │
│    -> Ollama (homelab RTX 4090) for simple tasks      │
│    -> Anthropic Claude API for complex reasoning      │
│  Budget enforcement, cost tracking, fallback logic    │
└─────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│            Arize Phoenix (Docker)                     │
│  OpenTelemetry traces from all layers                 │
│  Per-agent cost/latency dashboards                    │
│  Multi-agent workflow visualization                   │
└─────────────────────────────────────────────────────┘
```

### Machine Deployment

**Mac Mini:**
- n8n (Docker) — workflow triggers
- LangGraph orchestrator + Pydantic AI agents (Python process)
- LiteLLM Proxy (Docker) — model gateway
- Arize Phoenix (Docker) — observability
- BlueBubbles — iMessage relay
- SQLite — local storage

**Homelab Server (RTX 4090):**
- Ollama — local LLM inference
- faster-whisper — Whisper STT

---

## What Changes in the Codebase

### Replaced by LiteLLM:
- `src/home_agent/routing/model_router.py` — LiteLLM handles routing, fallback, complexity-based model selection
- `src/home_agent/routing/clients.py` — single OpenAI-compat client replaces both OllamaClient and AnthropicClient
- Hardcoded pricing table in AnthropicClient — LiteLLM tracks cost natively
- Cloud-side budget enforcement — LiteLLM enforces hard limits per API key

### Replaced by LangGraph + Pydantic AI:
- `src/home_agent/agents/base.py` — AgentBase ABC and tool loop replaced by Pydantic AI agent definitions
- `src/home_agent/agents/registry.py` — LangGraph graph definition handles agent registration
- `src/home_agent/orchestrator/` — LangGraph graph IS the orchestrator
- `src/home_agent/routing/intent_classifier.py` (not yet built) — LangGraph conditional edges handle intent routing

### Replaced by Arize Phoenix:
- Portions of `src/home_agent/audit/logger.py` — Phoenix captures traces/spans automatically via OTel instrumentation

### Kept as-is (framework-agnostic):
- `src/home_agent/core/` — config, models, exceptions, constants
- `src/home_agent/storage/` — SQLite database, migrations
- `src/home_agent/budget/` — our local budget tracking (complements LiteLLM's enforcement)
- `src/home_agent/tools/` — tool base class, registry, ACL (adapts to Pydantic AI tool definitions)
- `src/home_agent/notifications/` — BlueBubbles/email service
- `src/home_agent/topics/` — topic interest store
- `config/` — YAML config files (extended with LiteLLM config)

---

## New Dependencies

```
# Add
langgraph>=0.3
langchain-core>=0.3
langchain-openai>=0.3          # talks to LiteLLM via OpenAI-compat
pydantic-ai>=0.2
arize-phoenix[otel]>=14.0
openinference-instrumentation-langchain
openinference-instrumentation-litellm

# Keep
pydantic, pydantic-settings, pyyaml, structlog, aiosqlite, apscheduler, click, httpx

# Remove / downgrade
anthropic                       # accessed via LiteLLM, not directly
```

---

## Implementation Sequence

1. **Set up LiteLLM Proxy** — Docker compose, configure Ollama + Anthropic model groups, test routing
2. **Set up Arize Phoenix** — Docker compose, verify traces from LiteLLM calls
3. **Set up n8n** — Docker compose, basic cron triggers that fire webhooks
4. **Refactor agents to Pydantic AI** — convert AgentBase subclasses to Pydantic AI agent definitions with typed tools
5. **Build LangGraph orchestrator** — graph with intent router, agent nodes, conditional edges, state management
6. **Wire n8n -> LangGraph** — webhook triggers for scheduled tasks (news digest, calendar digest, podcast check)
7. **Wire notifications** — keep existing BlueBubbles service, connect to LangGraph output nodes
8. **Instrument with Phoenix** — add OTel spans across LangGraph + LiteLLM + Pydantic AI

## Verification

- LiteLLM: `curl http://localhost:4000/v1/models` returns both local and cloud models
- Phoenix: open `http://localhost:6006`, see traces for test requests
- n8n: open `http://localhost:5678`, cron trigger fires and hits orchestrator webhook
- End-to-end: send "What's on my calendar this week?" via CLI -> LangGraph routes to calendar agent -> Pydantic AI agent calls tools -> response delivered via BlueBubbles -> full trace visible in Phoenix
