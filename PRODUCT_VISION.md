# Alfred — Product Vision

A multi-agent personal assistant for family productivity. Conversational when you need it, autonomous when you don't.

---

## Core Principles

1. **Family-first** — Designed for a household, not a single user. Shared context, individual preferences.
2. **Secure by default** — Every agent is sandboxed. Token budgets, tool permissions, and execution limits are enforced at the orchestrator level.
3. **Incrementally useful** — Each agent delivers standalone value. No "build everything before anything works."
4. **Observable** — Every action the system takes is logged, auditable, and explainable.

---

## Interaction Model

| Mode | Description |
|------|-------------|
| **Conversational** | On-demand requests via CLI, iMessage (via BlueBubbles relay on Mac Mini), or chat interface |
| **Voice** | Voice commands via Whisper STT on homelab, hands-free interaction from kitchen/rooms |
| **Scheduled** | Cron-based routines that run autonomously and push notifications (e.g., weekly calendar digest every Sunday evening) |
| **Reactive** | Event-driven triggers (e.g., new calendar invite detected → conflict check → notify) |

---

## Agents

### 1. Orchestrator (Router)
The central dispatcher. All requests — conversational or scheduled — flow through it.

**Responsibilities:**
- Intent classification and agent routing
- Model routing: decides whether a request requires Claude API (complex reasoning) or can be handled by the local LLM on the homelab (simple/routine tasks) — key lever for cost control
- Request validation and sanitization
- Token budget enforcement (per-request and per-agent daily caps)
- Tool access control (each agent has an explicit allowlist)
- Execution timeout enforcement
- Conversation memory management
- Fallback handling when an agent can't fulfill a request

### 2. Family Calendar Manager
**Purpose:** Keep the family in sync on what's happening this week and beyond.

**Capabilities:**
- Connect to Google Calendar (multiple accounts — yours + wife's)
- Weekly digest notification: "Here's what's on this week" (pushed Sunday evening or configurable)
- Daily morning briefing: today's schedule + any changes since yesterday
- Conflict detection: flag double-bookings across family members
- Natural language scheduling: "Schedule a dentist appointment for Tuesday afternoon"
- Event coordination: "Find a time that works for both of us this weekend"
- Travel time awareness: flag tight gaps between events at different locations

**Integrations:** Google Calendar API, notification service (email/SMS/Telegram)

### 3. Grocery & Meal Planner
**Purpose:** Reduce the cognitive load of "what's for dinner" and "what do we need from the store."

**Capabilities:**
- Meal plan generation for the week based on preferences, dietary needs, and what's in season
- Grocery list generation from meal plans
- Shared grocery list that both family members can add to conversationally
- Recipe suggestions based on what you already have
- Budget-conscious options: flag sales, suggest substitutions
- Recurring staples: auto-add items you always buy
- Prep instructions and timing coordination for complex meals

**Integrations:** Recipe APIs, grocery store APIs (Instacart/Kroger), shared list storage

### 4. Research & Information Agent
**Purpose:** On-demand research assistant for household decisions and curiosity.

**Capabilities:**
- Local area lookup: restaurants, services, events, activities
- Product research and comparison (appliances, tools, gear)
- Travel planning: flights, hotels, itineraries, activity suggestions
- News/topic monitoring: "Tell me if anything happens with [topic]"
- Price tracking and deal alerts
- Summarize long articles, reviews, or documents
- Answer household questions: "How do I fix a running toilet?" with sourced answers

**Integrations:** Web search, web scraping, location APIs

### 5. Household Task Manager (Suggested)
**Purpose:** Track and coordinate recurring household responsibilities.

**Capabilities:**
- Shared to-do lists with assignment and reminders
- Recurring task scheduling (chores, maintenance)
- Home maintenance calendar: HVAC filter changes, gutter cleaning, appliance service dates
- Bill and subscription tracking with renewal reminders
- Package delivery tracking
- "Who's doing what" visibility for the household

**Integrations:** Todoist/Google Tasks, package tracking APIs, reminder/notification service

### 6. News Synthesizer
**Purpose:** Stay informed on topics you care about without doomscrolling.

**Capabilities:**
- Scheduled news sweeps across configurable sources (RSS feeds, news APIs, specific sites)
- Topic-based filtering: only surface stories matching your interest list
- Synthesis, not aggregation: produce a concise briefing that explains what happened and why it matters, not a list of headlines
- Deduplication: merge coverage of the same story from multiple sources into one summary
- Trend detection: flag when a topic you track is getting unusual coverage volume
- **Dynamic topic management**: add/remove/prioritize topics conversationally ("Start tracking AI regulation" / "Drop crypto news")
- Configurable delivery: daily morning digest, real-time alerts for high-priority topics, or weekly roundup
- Source attribution: always link back to original articles

**Integrations:** RSS/Atom feeds, news APIs (NewsAPI, GNews, or Brave News), web scraping, notification service

### 7. Podcast Summarizer
**Purpose:** Get the key insights from podcasts without listening to every episode.

**Capabilities:**
- Monitor subscribed podcast feeds for new episodes
- Download audio and transcribe locally via Whisper on the homelab GPU (no audio leaves the network)
- Generate structured summaries: key topics, main arguments, notable quotes, action items
- Topic-based filtering: only process episodes that match your interest list, skip the rest
- **Dynamic topic/show management**: add/remove podcasts and topic filters conversationally ("Add Huberman Lab" / "Only summarize episodes about sleep")
- Highlight extraction: flag segments that are especially relevant to your tracked topics
- Scheduled delivery: batch summaries into a morning or evening digest
- Full transcript available on demand

**Integrations:** Podcast RSS feeds, faster-whisper (homelab GPU), notification service, local storage for transcripts

### Shared: Topic Interest System
The News Synthesizer and Podcast Summarizer share a **dynamic topic interest store** managed by the orchestrator.

- Add/remove topics via conversation: "I'm interested in home automation, AI policy, and marathon training"
- Priority levels: high (real-time alerts), normal (included in digests), low (weekly roundup only)
- Topics are available to any agent that needs them (research agent can also use them for proactive suggestions)
- Stored in the config layer, version-controlled and auditable

---

## Architecture

Two physical machines handle different concerns: the Mac Mini coordinates everything; the homelab server does the heavy compute.

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                        ENTRY POINTS (Mac Mini)                       │
  │                                                                      │
  │   iMessage ──► BlueBubbles     CLI ──► Terminal     Cron Scheduler   │
  │                    │                       │               │         │
  │                    └───────────────────────┴───────────────┘         │
  └────────────────────────────────┬─────────────────────────────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │           MAC MINI — Orchestrator        │
              │                                          │
              │   - Intent Router                        │
              │   - Model Router (local vs. cloud)       │
              │   - Auth & Guardrails                    │
              │   - Token Budget Manager                 │
              │   - Audit Logger                         │
              └──────────┬──────────────────┬────────────┘
                         │                  │
          ┌──────────────▼──────┐   ┌───────▼──────────────────────┐
          │   CLAUDE API (cloud) │   │  HOMELAB SERVER — RTX 4090   │
          │                     │   │                               │
          │  Complex reasoning  │   │  Ollama (local LLM inference) │
          │  tasks only,        │   │  faster-whisper (Whisper STT) │
          │  budget-gated       │   │  Heavy compute workloads      │
          └─────────────────────┘   └───────────────────────────────┘
                         │                  │
                         └────────┬─────────┘
                                  │
     ┌──────────┬──────────┬────┼────┬──────────┬──────────┬──────────┐
     ▼          ▼          ▼         ▼          ▼          ▼          ▼
  ┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
  │Calendar││ Meal   ││Research││ Tasks  ││  News  ││Podcast ││ Voice  │
  │ Agent  ││Planner ││ Agent  ││ Agent  ││ Synth  ││Summary ││Pipeline│
  └───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘
      │         │         │         │         │         │         │
      └─────────┴─────────┴────┬────┴─────────┴─────────┴─────────┘
                                       │
     ┌─────────────────────────────────▼───────────────────────────┐
     │                   Tool / Integration Layer                   │
     │  Google Calendar, Web Search, Grocery APIs, RSS/News APIs,  │
     │  Podcast Feeds, Topic Interest Store, Storage                │
     └──────────────────────────────────────────────────────────────┘
                                       │
                     ┌─────────────────▼──────────────────┐
                     │   RESPONSE DELIVERY (Mac Mini)      │
                     │   BlueBubbles ──► iMessage          │
                     │   SendGrid ──► Email digests        │
                     └─────────────────────────────────────┘
```

### Machine Responsibilities

**Mac Mini (Orchestrator host)**
- Runs the orchestrator process and all agent logic
- Hosts BlueBubbles for iMessage relay (send and receive)
- Runs the cron scheduler and CLI entry point
- Routes requests to homelab or Claude API based on task complexity
- Always-on, low power draw

**Homelab Server (RTX 4090)**
- Runs Ollama for local LLM inference (Llama 3, Mistral, Qwen, or equivalent)
- Runs faster-whisper for local Whisper STT on GPU
- Handles any other compute-heavy tasks (embeddings, batch processing)
- Reachable from Mac Mini over local network only

**Claude API (cloud)**
- Reserved for requests the model router classifies as requiring strong reasoning
- Budget-gated: orchestrator enforces daily/monthly spend limits before dispatching

---

## Security & Guardrails

### Token Spend Control
- **Per-request budget**: Each conversational request has a max token ceiling
- **Per-agent daily budget**: Each agent has a daily token allowance; orchestrator refuses requests when exhausted
- **Global daily/monthly cap**: Hard stop across all agents to prevent runaway costs
- **Spend dashboard**: Running total visible at any time via conversational query

### Tool Access Control
- **Principle of least privilege**: Each agent declares its required tools; orchestrator enforces the allowlist
- **Read-only by default**: Agents that only need to read data (calendar, finance) cannot write unless explicitly granted
- **No cross-agent tool sharing**: Calendar agent cannot invoke web search tools; research agent cannot modify calendar
- **Human-in-the-loop for destructive actions**: Any write/delete/send operation requires user confirmation (configurable per-action)

### Agent Containment
- **Execution timeouts**: Hard kill after configurable max duration per agent invocation
- **No self-modification**: Agents cannot modify their own prompts, tools, or permissions
- **Output size limits**: Responses are capped to prevent context window abuse
- **Retry limits**: Max 3 retries on failure before escalating to user
- **Sandboxed execution**: Agents run in isolated contexts; no shared mutable state except through the orchestrator

### Audit & Observability
- **Full action log**: Every tool call, API request, and agent decision is logged with timestamp and context
- **Cost attribution**: Token spend tracked per agent, per request, per day
- **Alert on anomalies**: Notify if any agent exceeds expected usage patterns
- **Conversation history**: All interactions stored for review (with retention policy)

### Network Security & Data Locality
- **Local network only**: The Mac Mini and homelab server communicate exclusively over the home LAN; no ports are exposed to the public internet
- **Local-first data policy**: Voice audio captured via Whisper STT and all routine queries are processed entirely on local hardware and never leave the home network
- **Cloud API as last resort**: Requests are dispatched to the Claude API only when the model router determines the local LLM is insufficient for the task; the decision criteria and routing log are auditable
- **No audio exfiltration**: faster-whisper transcription runs on the homelab GPU; raw audio is discarded after transcription and is never transmitted off-network

---

## Notification Strategy

| Notification | Frequency | Channel | Default Time |
|---|---|---|---|
| Weekly calendar digest | Weekly | iMessage + Email | Sunday 6:00 PM |
| Daily morning briefing | Daily | iMessage | 7:00 AM |
| Calendar conflict alert | Real-time | iMessage | Immediate |
| Grocery list ready | Weekly | iMessage | After meal plan generated |
| Maintenance reminder | As scheduled | iMessage | Morning of |
| Research result ready | On completion | iMessage | Immediate |
| News digest | Daily or weekly | iMessage + Email | Configurable (default 7:00 AM) |
| Breaking topic alert | Real-time | iMessage | Immediate |
| Podcast summaries | As episodes drop | iMessage + Email | Batched into evening digest |

---

## Tech Stack (Proposed)

| Layer | Technology | Rationale |
|---|---|---|
| Language | Python 3.12+ | Rich AI/ML ecosystem, fast prototyping |
| AI Backbone | Claude API (Anthropic) | Tool use, long context, reliable reasoning |
| Agent Framework | Claude Agent SDK or custom | Multi-agent orchestration with tool control |
| Scheduler | APScheduler or system cron | Reliable scheduled task execution |
| Storage | SQLite (local) → PostgreSQL (if scaled) | Simple start, easy migration path |
| Notifications | BlueBubbles (iMessage) + Email (SendGrid) | Native iMessage via Mac Mini relay, email for long-form digests |
| Local LLM | Ollama + open model (Llama 3, Mistral, Qwen) | Zero-cost inference for routine tasks, full data privacy |
| Speech-to-Text | faster-whisper on homelab GPU | Local voice input, no audio leaves the network |
| Hardware | Mac Mini (orchestrator) + homelab server w/ RTX 4090 (inference) | Two-machine split: coordination vs. compute |
| Calendar | Google Calendar API | Primary family calendar platform |
| Web Search | Tavily or Brave Search API | Structured search results for research agent |
| Config | YAML/TOML files | Human-readable, version-controlled configuration |

---

## Phased Rollout

### Phase 1 — Foundation
- Orchestrator with intent routing and guardrails
- Local LLM setup on homelab via Ollama (model selection, health check, fallback behavior)
- Model router in orchestrator: classify requests as local vs. cloud, enforce budget gate before any Claude API dispatch
- Calendar agent: read calendars, generate weekly digest
- BlueBubbles integration for iMessage send/receive on Mac Mini
- Basic CLI interface
- Token budget enforcement and audit logging

### Phase 2 — Meal Planning & Grocery
- Meal planner agent with preference learning
- Grocery list generation and shared list
- Recipe integration

### Phase 3 — Research & Media Intelligence
- Research agent with web search
- Local area lookups
- Dynamic topic interest system (shared config, conversational management)
- News synthesizer: RSS/news API ingestion, topic filtering, daily digest generation
- Podcast summarizer: feed monitoring, Whisper transcription on homelab GPU, structured summaries
- Monitoring/alerting for tracked topics (real-time alerts for high-priority topics)

### Phase 4 — Voice & Household Management
- Whisper STT integration on homelab GPU (faster-whisper)
- Voice input pipeline: capture → transcribe on homelab → route through orchestrator → respond via iMessage
- Task manager agent
- Email digests as secondary notification channel

### Phase 5 — Polish & Expand
- Cross-agent workflows (meal plan → calendar blocks → grocery list)
- Preference learning and personalization
- Family member profiles with individual notification preferences
- Podcast transcript search ("What did Huberman say about cold exposure?")
