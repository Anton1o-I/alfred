# OpenAlfred — Product Specification

**Version:** 0.1 (planning)
**Status:** Pre-development
**Date:** 2026-05-17

---

## 1. What Is OpenAlfred

OpenAlfred is an open-source, self-hosted personal agent platform built on the bones of Alfred — a working multi-agent home assistant. The goal is to turn a single-household homelab project into something a technically capable person can clone, configure, and run without writing Python.

The shortest possible summary: **a composable agent runtime you operate yourself, with real LLM routing, built-in observability, and a YAML authoring layer so you can wire new agents without touching code.**

What it is not: a SaaS product, a no-code drag-and-drop builder for non-technical users, or a thin wrapper around a cloud agent API.

---

## 2. Positioning

| Dimension | OpenAlfred |
|---|---|
| **Deployment** | Self-hosted. Docker Compose brings the full stack up. |
| **Data** | Stays on your hardware. Cloud LLMs are opt-in per agent, never default. |
| **Authoring** | YAML-first for single-shot agents and graph-style workflows; Python escape hatch for anything complex. |
| **LLM access** | Local (Ollama) by default; cloud (Anthropic, OpenAI, etc.) via LiteLLM, budget-gated. |
| **Target user** | Developer or technical power user. Homelab owner, not enterprise IT. |

**What it replaces or extends:**

- Replaces: cobbling together n8n automations, raw GPT API calls, and custom scripts for personal productivity tasks.
- Extends: gives Ollama and LiteLLM users a structured, observable agent runtime on top of their existing inference stack.
- Not competing with: Home Assistant (different domain), AutoGPT / CrewAI (different trust and cost model), ChatGPT (different deployment model).

---

## 3. User Personas

### P1 — The Homelab Developer
Has a machine running Ollama and maybe a few Docker containers. Comfortable editing YAML and reading Python. Wants a personal assistant that runs on their hardware, costs near zero for routine tasks, and doesn't send data to the cloud unless they explicitly say so. Will read a README if it's good. Will not read 40 pages of docs.

**Installs via:** `git clone` + `docker compose up` + `alfred setup` wizard.

### P2 — The YAML Power User
Technical enough to write structured config but not a software engineer. Might be a data analyst, a researcher, or an ops person with homelab curiosity. Wants to define a new agent — "summarize my RSS feeds every Monday" — without writing a Python class. Will tolerate YAML schemas and a short worked example. Will not debug LangGraph internals.

**Installs via:** same as P1. Extends the platform by writing agent YAML files.

### P3 — The Python Contributor
A developer who wants to add a new tool integration (e.g., Todoist, Notion, Home Assistant) or a new agent with complex multi-step logic. Understands Python, async, Pydantic. Reads the architecture. Opens a PR.

**Engages via:** the Python agent/tool API, existing agents as reference implementations, CONTRIBUTING guide.

---

## 4. Core Value Propositions

**1. Real agent graphs, not toy demos.**
The calendar agent already runs a non-trivial LangGraph state machine with typed LLM outputs, conditional routing, cost-tier selection, and end-to-end OTel tracing. OpenAlfred's runtime generalizes this pattern — not rebuilds it.

**2. Local-first, cloud-optional.**
Routine classification and extraction runs on Qwen 3 14B via Ollama at zero marginal cost. Cloud spend is explicitly opted into per agent with hard USD caps. This is already the architecture; OpenAlfred just makes it configurable without code.

**3. Clone-to-running in one session.**
The setup path — Docker Compose + setup wizard + two filled `.env` values — should produce a working agent system in under an hour for P1. Every decision that would block that is a blocker for Tier 1.

**4. Declarative agent authoring.**
The main differentiator vs. running Alfred yourself: you can define a new agent in YAML and the runtime composes it. No Python required for the common case.

**5. Observable by default.**
Every workflow run produces a full Phoenix trace tree with one span per graph node. Token spend and USD cost tracked per agent per day. Available via `alfred budget` and the Phoenix UI on day one.

---

## 5. Functional Requirements

### 5.1 Core Runtime (must exist before anything else)

| ID | Requirement |
|---|---|
| FR-01 | LangGraph orchestrator routes requests to registered agents by name. |
| FR-02 | LiteLLM Proxy is the single gateway for all LLM calls; no agent calls a model API directly. |
| FR-03 | Per-agent USD caps enforced before any cloud model call. Hard stop, not a soft warning. |
| FR-04 | Every workflow node emits an OTel CHAIN span under a parent AGENT span. Traces visible in Phoenix. |
| FR-05 | `alfred health` verifies LiteLLM, Ollama, and Phoenix reachability and prints a status table. |
| FR-06 | `alfred budget` shows per-agent USD spend for the current day and month. |
| FR-07 | Alfred itself runs inside Docker Compose alongside LiteLLM, Ollama, Phoenix, and n8n. (Currently runs as a bare systemd process — must containerize.) |

### 5.2 Built-in Agents (shipped with the platform)

| Agent | Status in Alfred today | OpenAlfred Tier |
|---|---|---|
| Calendar (email-driven create/delete/digest) | Fully built, LangGraph graph | Tier 1 |
| Research Curator (weekly reading-list digest) | Fully built, Pydantic AI | Tier 1 |
| News Synthesizer | Stubbed in config | Tier 2 |
| Meal Planner | Stubbed in config | Tier 2 |
| Tasks | Stubbed in config | Tier 2 |
| Podcast Summarizer | Stubbed in config | Tier 3 |

The two working agents (calendar, curator) ship in Tier 1. Stubs are present in `config/agents.yaml` and well-defined in the product vision; they become real in Tier 2.

### 5.3 Email Transport

| ID | Requirement |
|---|---|
| FR-08 | iCloud SMTP/IMAP/CalDAV supported on day one (the existing path). |
| FR-09 | Gmail SMTP/IMAP supported in Tier 2. OAuth is the hard part — documented in a provider guide. |
| FR-10 | Per-agent display names and taglines in outbound email (already works; must survive the generalization). |
| FR-11 | Per-recipient subscriptions filter which agents reach which address (already works). |
| FR-12 | Inbound allowlist lives in env vars, never in tracked YAML. |

### 5.4 Scheduling

| ID | Requirement |
|---|---|
| FR-13 | Cron-style scheduled tasks defined in `config/settings.yaml` and executed by the Alfred scheduler process. |
| FR-14 | At minimum: inbox poll (every minute), daily briefing (7am), weekly preview (Sunday 6pm), weekly curator digest (Monday 4am). |
| FR-15 | New scheduled tasks added via YAML edit + restart, not code. |

### 5.5 Declarative Agent Authoring (the differentiator — see Section 7)

| ID | Requirement |
|---|---|
| FR-16 | A single-shot agent definable in YAML: input, system prompt, output schema (Pydantic field list), model preference, tool allowlist. |
| FR-17 | A graph-style workflow definable in YAML: named nodes, edges, conditional routing rules. Each node is one of: llm-call, tool-call, built-in, or router. |
| FR-18 | Built-in node types: `date_enrichment`, `email_send`, `calendar_query`, `calendar_write`, `rss_fetch`, `web_search`. Extensible by adding Python tool classes. |
| FR-19 | The YAML runtime assembles a real LangGraph graph at startup. No separate engine — it's the same runtime as the hand-written agents. |
| FR-20 | The existing calendar agent must be expressible in YAML (as a validation exercise; not required to migrate it). |
| FR-21 | YAML agents appear in `alfred health`, `alfred budget`, and Phoenix traces identically to code-defined agents. |

### 5.6 Setup Experience

| ID | Requirement |
|---|---|
| FR-22 | `alfred setup` is an interactive CLI wizard that walks through: LLM provider (local/cloud/both), email provider, notification preferences, timezone. Writes `.env` and edits `config/*.yaml`. |
| FR-23 | Running `docker compose up -d && alfred setup && alfred health` on a fresh clone produces a working system. |
| FR-24 | `.env.example` documents every required variable with a comment explaining where to get the value. (Already largely true; must be complete.) |

---

## 6. Non-Functional Requirements

### 6.1 Privacy and Data Locality

- All data processed locally by default. Ollama runs on the same host as Alfred (or a LAN-adjacent host).
- Cloud API calls happen only when the agent's `model_preference` is `cloud` or the intent classifier tags a request as `complex` with cloud escalation enabled.
- No telemetry or usage reporting to any external service. Phoenix is self-hosted.
- Audio (Whisper STT) never leaves the local network.
- Personal identity (email addresses, names, phone numbers) lives exclusively in `.env`, never in tracked YAML or code.

### 6.2 Cost Control

- USD caps per agent per day enforced before any cloud model invocation — not after.
- Global daily and monthly USD caps as a safety net.
- Local Ollama calls cost $0; token quotas for local models are visibility-only, not enforced by default.
- `alfred budget` must be accurate to the last completed workflow run.

### 6.3 Deployment Portability

- The full stack (Alfred + LiteLLM + Phoenix + n8n + Ollama) runs via a single `docker compose up -d`.
- Alfred must be containerized by Tier 1. (Currently runs as a bare Python process under systemd — this is the most significant infrastructure change in the whole plan.)
- Tested on: Linux (amd64). Mac/ARM support is a Tier 2 nice-to-have, not a blocker.
- The homelab-specific parts (Whisper via Speaches, BlueBubbles iMessage relay) remain optional and clearly documented as such.

### 6.4 Observability

- Phoenix UI accessible at `http://localhost:6006` with no additional configuration.
- Every LLM call produces a trace: model, input tokens, output tokens, latency, cost estimate.
- Every workflow run produces a parent AGENT span with CHAIN child spans per node.
- `alfred health` checks: LiteLLM proxy reachable, Ollama reachable, Phoenix reachable, configured agents enabled.

### 6.5 Reliability of the Simulation Harness

- The `alfred simulate` command and `tests/scenarios/*.yaml` pattern is the primary regression guard.
- New agents defined via YAML must be testable through the same harness (scenario file + fake I/O client).
- Tier 1 target: calendar agent scenario suite stays at >= 95% pass rate across any model supported in the default config.

---

## 7. The Low-Code Authoring System

This is the feature that separates OpenAlfred from "a well-organized Alfred fork." Everything else is polish on existing bones. The authoring system is genuinely new engineering.

### 7.1 The Problem

Today, adding a new agent means:
1. Write a Python class extending `BaseAgent`.
2. Define Pydantic schemas for typed LLM outputs.
3. Build a LangGraph `StateGraph` with nodes and edges.
4. Register the agent in `app.py`.
5. Add it to `config/agents.yaml` for tool allowlists and budget.

Steps 2–4 require understanding LangGraph internals and Pydantic AI's typed-output pattern. That's a steep entry tax for something like "fetch RSS feeds, filter by topic, summarize each item, email me the result."

### 7.2 Design Principle

The authoring system is a **YAML front-end that compiles to the same LangGraph + Pydantic AI runtime** already running in production. It does not introduce a new execution model or a second graph engine. A YAML-defined agent is indistinguishable from a Python-defined agent at runtime.

This means the authoring system has a clear scope ceiling: it can express what the runtime already knows how to do. For anything outside that boundary, the escape hatch is a Python node class, registered in the tool registry and referenced by name from YAML.

### 7.3 Single-Shot Agent Format

The simplest form — one LLM call in, typed output out:

```yaml
# config/custom_agents/headline_classifier.yaml
kind: agent
name: headline_classifier
description: "Classify a news headline by topic and sentiment."
model_preference: local
output_schema:
  topic: str
  sentiment:
    type: enum
    values: [positive, negative, neutral]
  confidence: float
system_prompt: |
  You classify news headlines. Return a structured result only.
tools: []
budget_usd_daily: 0.50
```

The runtime constructs a Pydantic model from `output_schema`, wraps it in a Pydantic AI agent, registers it in the agent registry, and exposes it identically to a hand-written agent.

The schema language needs to cover: `str`, `int`, `float`, `bool`, `enum` (with values), `list[str]`, and `optional[T]`. Nested objects are a Tier 3 consideration; they add complexity without unblocking the target use cases.

### 7.4 Graph-Style Workflow Format

A multi-step workflow with branching:

```yaml
kind: workflow
name: rss_digest
description: "Fetch RSS feeds, filter by topics, summarize, email."
model_preference: local
budget_usd_daily: 1.00
schedule: "0 7 * * 1"  # Monday 7am

state:
  raw_items: list
  filtered_items: list
  summaries: list
  topics: list

nodes:
  - name: load_topics
    type: builtin
    builtin: topic_store_read

  - name: fetch_feeds
    type: tool
    tool: rss_fetch
    input:
      feeds: "{{ config.feeds }}"
    output_key: raw_items

  - name: filter_by_topic
    type: builtin
    builtin: topic_filter
    input:
      items: "{{ state.raw_items }}"
      topics: "{{ state.topics }}"
    output_key: filtered_items

  - name: summarize
    type: llm
    prompt_template: |
      Summarize this article in 2-3 sentences. Focus on what's new.
      Title: {{ item.title }}
      Content: {{ item.content }}
    output_schema:
      summary: str
      relevance: float
    foreach: "{{ state.filtered_items }}"
    output_key: summaries

  - name: send_digest
    type: builtin
    builtin: email_send
    input:
      subject: "Weekly Research Digest"
      template: digest_email
      data: "{{ state.summaries }}"

edges:
  - from: load_topics
    to: fetch_feeds
  - from: fetch_feeds
    to: filter_by_topic
  - from: filter_by_topic
    to: summarize
  - from: summarize
    to: send_digest
  - from: send_digest
    to: END
```

### 7.5 Node Types

| Type | What it does | Requires |
|---|---|---|
| `llm` | Runs a Pydantic AI agent with the given prompt and schema. Picks local or cloud model based on `model_preference` or a `complexity` tag from a prior classification node. | `prompt_template`, `output_schema` |
| `tool` | Calls a registered Python tool function by name. | `tool` (name), `input` |
| `builtin` | Calls a first-class built-in primitive by name. | `builtin` (name), `input` |
| `router` | Reads a field from state and routes to different next nodes based on its value. | `on` (state field), `routes` (value → node name map) |
| `python` | Calls an arbitrary async Python function by dotted import path. The escape hatch. | `callable` (e.g., `alfred.tools.custom.my_fn`) |

### 7.6 Built-in Primitives (Tier 1 starting set)

These cover the use cases of the two existing agents and the most common next agents:

- `date_enrichment` — enriches state with `today_iso`, `timezone_name`, `upcoming_days` table.
- `topic_store_read` — loads active topics from the topic interest store.
- `topic_filter` — filters a list of items against topics using keyword matching.
- `rss_fetch` — fetches and parses one or more RSS/Atom feeds.
- `calendar_query` — lists events in a time window.
- `calendar_write` — creates or deletes an event.
- `email_send` — renders a template and sends via the configured email provider.
- `email_receive` — reads the inbox (for email-triggered workflows).

### 7.7 Routing Between Model Tiers

The calendar workflow's complexity-routing pattern should be available to YAML workflows. A `router` node reading `complexity` from state can select `local-default` or `cloud-default` for downstream `llm` nodes. The `output_schema` of a classification `llm` node should be able to include a `complexity: enum[simple, complex]` field. The runtime then picks the right model tier for subsequent `llm` nodes automatically.

This needs to be explicit in the YAML spec — not implicit magic — so authors understand when they're spending cloud tokens.

### 7.8 The Calendar Agent as Validation Target

The calendar agent (`src/alfred/agents/calendar/workflow.py`) is the worked example. It has: pure-code nodes (date enrichment, validation, conflict check), LLM specialist nodes (intent classification, event parsing, delete matching), conditional routers, confidence-based branching, and a final rendering node. All of these patterns must be expressible in the YAML format before the authoring system can be declared complete.

The calendar agent does not need to be migrated to YAML. Its Python implementation is production-tested and should remain authoritative. The requirement is: given the YAML format, a competent author could reproduce it.

### 7.9 Open Questions

**Q1: Schema definition language.** A simple field list (`name: type`) works for the MVP. Pydantic's full schema is the obvious target — should the YAML schema DSL just be a subset of Pydantic model syntax, or something simpler that compiles to Pydantic? Recommendation: start with the flat field list. Add nested objects only when a real use case demands it.

**Q2: Template syntax for prompts.** Jinja2 is the natural choice (`{{ state.field }}`). It's well-understood, available as a dependency, and consistent with n8n's expression syntax. Alternative: Python f-strings via `eval()` — simpler but a security risk in a public repo. **Recommendation: Jinja2, sandboxed.**

**Q3: `foreach` semantics.** The `foreach` pattern (run an llm node once per item in a list) is extremely common (summarize each article, classify each event). It needs to be a first-class node attribute rather than a workflow-level loop primitive. Open question: does foreach run in parallel or sequentially? For Tier 1, sequential is correct — simpler and avoids rate-limit issues.

**Q4: How does a YAML agent get triggered?** Three trigger types to support: scheduled (cron in the YAML), email-driven (inbound email to the allowlist inbox triggers the workflow), and on-demand (via CLI `alfred ask`). For Tier 1: scheduled and on-demand. Email-driven in Tier 2.

**Q5: Where do YAML agent files live?** Two options: a `config/custom_agents/` directory (clear, convention-over-configuration), or a `plugins/` directory (signals extensibility intent). Recommendation: `config/custom_agents/`. It aligns with the existing convention that `config/` is where you configure things.

---

## 8. Out of Scope

These are explicitly deferred, not forgotten:

- **SaaS or hosted offering.** OpenAlfred is self-hosted. There is no managed version in v1. A Render/Fly.io one-click template is a Tier 3 exploration, not a product decision.
- **Multi-tenant operation.** One instance per household. Multi-user within a household (multiple recipients, per-user preferences) is in scope; running one server for many unrelated households is not.
- **A web UI for authoring agents.** The YAML authoring system is text-based. A visual graph editor is plausible as a community contribution in Tier 3 but is not part of the platform commitment.
- **A PyPI package.** Useful eventually. Not before the Docker Compose deployment story is solid.
- **Windows support.** Not tested, not documented, not a priority.
- **Mobile app.** Notifications via email are the delivery channel. A mobile app is a different product.
- **Voice input as a first-class interface.** Whisper STT exists and is documented, but it requires the Speaches container and a homelab-adjacent GPU. It's an optional extension, not a core feature.
- **Plugin marketplace.** Community-contributed agents as installable packages. Tier 3 exploration.

---

## 9. Success Metrics

These are per-tier definitions of "done," not aspirational KPIs.

### Tier 1

- A person who has never seen the codebase can follow the README, run `docker compose up -d && alfred setup`, and have the calendar and curator agents working within 60 minutes.
- The calendar simulation harness passes >= 95% of scenarios against the default local model (Qwen 3 14B via Ollama).
- `alfred health` correctly reports the status of all containers and LLM backends.
- Zero personal details (email addresses, names, IPs) exist in any tracked file.
- A YAML single-shot agent defined in `config/custom_agents/` is loaded and invokable via `alfred ask`.

### Tier 2

- Gmail SMTP/IMAP documented and tested as an alternative to iCloud.
- At least two additional agents enabled (meals or news recommended — they have the clearest user value).
- A YAML graph-style workflow with at least three node types (llm, tool, builtin) is shipped as a worked example in `examples/`.
- ISSUE templates, a LICENSE file (MIT recommended), and a CONTRIBUTING guide exist.
- `alfred health` covers all enabled agents, not just the runtime stack.

### Tier 3

- PyPI package `openalfred` installable via `pip install openalfred`.
- Plugin system: Python packages can register new tool classes and built-in primitives.
- At least one hosted-template path documented (Render or Fly.io).
- Three or more community-contributed worked examples in `examples/`.
