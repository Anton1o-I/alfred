# OpenAlfred — Implementation Plan

**Version:** 0.1 (May 2026)
**Status:** Planning artifact — not yet in development
**Companion:** [`openalfred-product-spec.md`](./openalfred-product-spec.md)

---

## Overview

Four phases map to the three tiers in the spec, with an explicit Phase 0 that is mostly already done or nearly done.

| Phase | Maps to | Rough duration | Primary output |
|---|---|---|---|
| Phase 0 | Pre-work | 2-3 weeks | Alfred containerized; repo clean for public |
| Phase 1 | Tier 1 | 6-8 weeks | Clone-and-run for a homelab developer |
| Phase 2 | Tier 2 | 4-6 weeks | Documentation, multi-provider, community scaffolding |
| Phase 3 | Low-code authoring | 8-12 weeks | YAML agent definition runtime |
| Phase 4 | Tier 3 | Ongoing | PyPI, plugin system, deployment templates |

Phases 0-2 are sequential. Phase 3 can start after Phase 1 is complete — it does not block Phase 2. Phase 4 is ongoing and parallelizable with everything.

---

## Phase 0 — Foundations for Public Release (2-3 weeks)

This phase is about making the existing codebase safe and ready to be public, not adding features. Most of it is polish on things that already work.

### Scope

**0.1 Containerize Alfred itself**
Alfred currently runs as a Python process under systemd, outside the Docker compose stack. Add an `alfred` service to `docker-compose.yml` that builds from a `Dockerfile` in the repo root. The Alfred container connects to LiteLLM, Phoenix, and Ollama via Docker networking. Systemd becomes an optional wrapper for the compose stack (`docker compose up`), not a requirement for Alfred itself. This is the single biggest structural change in Phase 0 — everything else is config and docs.

**0.2 Audit and scrub the tracked codebase**
Verify zero personal details are in tracked files. This is already mostly true (the `*_env` pattern is in place), but needs a systematic pass: grep for LAN IPs, real email addresses, full names, any hardcoded timezone that isn't a config reference. Update `.gitignore` to definitively exclude `data/`, `*.db`, `config/google_token.json`, `config/google_credentials.json`.

**0.3 Write `CONTRIBUTING.md` and `LICENSE`**
Choose a license (recommendation: MIT — permissive, compatible with Pydantic AI/LangGraph, no friction for homelab use). `CONTRIBUTING.md` needs: how to set up a dev environment, how to run the simulation harness, how to add a new agent, the secrets audit requirement before every commit.

**0.4 Write a real README**
The current README is not written for a stranger. Rewrite it to cover: what OpenAlfred is in two sentences, system requirements, `docker compose up` quickstart, link to full setup guide. The README is the first thing a potential user sees — it needs to answer "is this for me?" in under 30 seconds.

**0.5 Add secrets audit to CI**
Run `git diff --cached` scanning in a GitHub Actions workflow that runs on every pull request. Fail the PR if any of the patterns from `CLAUDE.md`'s audit checklist are found in the diff. This codifies what is currently a manual rule.

### Success criteria
- `git clone` + `docker compose up` starts all services on a machine with no prior Alfred setup
- `ruff check`, `mypy`, and `uv run pytest` all pass in CI
- Secrets audit CI passes with no false positives on main
- Zero tracked files contain real email addresses, LAN IPs, or personal names (verified by grep)
- LICENSE and CONTRIBUTING.md exist and are accurate

### Dependencies
None. This phase has no blockers.

### Effort notes
0.1 (containerizing Alfred) is the real work here — maybe 2-3 days. Writing a Dockerfile for a Python/uv project is straightforward, but wiring the network connections (Alfred → LiteLLM, Alfred → Phoenix, Alfred → Ollama) and ensuring the scheduler and inbox poller work correctly inside a container needs careful testing. Everything else in Phase 0 is an hour or two each.

---

## Phase 1 — Tier 1: "A Friend Can Install It" (6-8 weeks)

The deliverable: a homelab developer can clone the repo, run a setup wizard, and have a working calendar agent within 60 minutes. This is the MVP.

### Scope

**1.1 Setup wizard (`alfred setup`)**
Interactive CLI that asks:
1. Which email provider? (iCloud / Gmail / SMTP generic)
2. API keys (ANTHROPIC_API_KEY if using cloud, or confirm local-only)
3. Which agents to enable? (calendar / curator / both)
4. Primary recipient email address
5. Timezone
6. Ollama model to use (default: qwen3:14b)

Writes `.env` from answers. Runs `alfred health` at the end. Does not require the user to understand config file structure. Must handle the case where Docker is not running and give a clear error.

**1.2 Health check command (`alfred health`)**
Per-service checks with pass/warn/fail:
- LiteLLM proxy: HTTP GET `/health/liveliness`
- Ollama: model loaded and responding
- Calendar: CalDAV connection and auth (if calendar agent enabled)
- Email: SMTP auth (if email configured)
- Phoenix: OTLP endpoint reachable
- Budget: current spend vs. cap per agent

Each failure includes a one-line fix hint (e.g., "LiteLLM not reachable — is the compose stack running? Try `docker compose up -d litellm`"). No false positives.

**1.3 `.env.example` as a complete reference**
Every variable documented with: what it does, where to find the value, whether it's required or optional, and a safe placeholder. The current `.env.example` is mostly there; this is a completion pass.

**1.4 Simulation harness in CI**
The harness currently runs manually. Add a GitHub Actions workflow that runs `alfred simulate` against the existing scenarios on every pull request. Fail the PR if pass rate drops below 95%. This is the regression guard for prompt and routing changes — it needs to be automatic.

**1.5 Per-agent scenario suites**
The curator agent has no simulation harness. Add at least 5 scenarios covering: normal operation (topics match, digest produced), no matching items, source fetch failure, synthesis failure (fallback behavior). The calendar harness already has scenarios; extend it with any gaps identified in Phase 0's audit.

**1.6 Gmail support (email provider)**
iCloud SMTP is the current default. Gmail SMTP is probably the most common alternative (app-specific password or OAuth — recommend app password first, defer OAuth to Tier 2 to avoid the 7-day token churn problem). Document the Gmail setup path in the README and wizard.

**1.7 CPU-only mode**
Document how to run without a GPU: switch Ollama to a smaller model (recommendation: qwen2.5:3b), disable Whisper or run a CPU-compatible image. The compose file should have a `compose.cpu.yml` override that swaps GPU-requiring services. This is not full support — just a documented path for users without NVIDIA hardware.

### Success criteria
- A homelab developer with Docker installed reaches a working calendar agent in under 60 minutes following only the README and wizard
- `alfred health` returns actionable output for every failure mode, no false positives
- Simulation harness runs automatically on every PR and blocks merges that drop below 95%
- Curator agent has a scenario suite
- Gmail SMTP documented and tested
- CPU-only override documented

### Dependencies
- Phase 0 must be complete (Alfred containerized, CI in place)

### Effort notes
The setup wizard is 2-3 days of real work — the interactive CLI, the `.env` generation, and especially the edge cases (what if Ollama isn't running, what if the iCloud password is wrong). The health check is 1-2 days. CI for the simulation harness is a few hours. Gmail support is an afternoon. The rest is documentation.

---

## Phase 2 — Tier 2: "A Reasonable Stranger Can Install It" (4-6 weeks)

Phase 1 gets a technical homelab user to a working setup. Phase 2 makes the project legible to someone who found it on GitHub and is evaluating whether to try it.

### Scope

**2.1 Per-agent documentation**
One `docs/agents/<agent-name>.md` per built-in agent covering: what it does, configuration options, required `.env` vars, example output, how to disable it, known limitations. Calendar and curator ship documentation here. Scaffolded-but-not-implemented agents get a stub with "planned, not implemented" notice.

**2.2 Architecture documentation**
A `docs/architecture.md` that explains: the LangGraph + Pydantic AI pattern in plain English, how model routing works, how budget enforcement works, how the simulation harness works. This document is for the developer who wants to understand the codebase before adding to it. Diagrams encouraged (ASCII is fine).

**2.3 GitHub community files**
- `ISSUE_TEMPLATE/bug_report.md`: reproduction steps, logs, OS, Docker version
- `ISSUE_TEMPLATE/new_agent_request.md`: what the agent does, expected inputs/outputs, proposed tools
- `ISSUE_TEMPLATE/provider_request.md`: which email/calendar/LLM provider, API docs link
- `PULL_REQUEST_TEMPLATE.md`: checklist covering secrets audit, simulation harness pass, docs updated

**2.4 Changelog**
`CHANGELOG.md` with an initial entry documenting what exists at the point of first public release. Format: keep-a-changelog. Required for anyone evaluating whether to depend on this project.

**2.5 Generic SMTP support**
Beyond iCloud and Gmail: a generic SMTP configuration path for any provider (host, port, username, password). This covers Fastmail, Protonmail Bridge, self-hosted Postfix, etc. One config path that all providers use under the hood; iCloud and Gmail are just pre-configured profiles.

**2.6 Log retention documentation**
The systemd README already covers journald retention. Add a section to the main docs covering: what logs exist (`journald`, `data/alfred.db` audit log, Phoenix traces), how large they get, how to clean them. Not implementing automatic retention — just documenting the manual process clearly.

### Success criteria
- README rated "clear and complete" by three people who didn't write it
- Every built-in agent has a documentation page
- GitHub issue templates exist and cover the three main request types
- Generic SMTP works with Fastmail (tested)
- CHANGELOG.md exists and is accurate

### Dependencies
- Phase 1 complete

### Effort notes
This phase is almost entirely writing. The generic SMTP path is an afternoon of code. Everything else is documentation work — probably 2-3 weeks of focused writing across the full scope.

---

## Phase 3 — Low-Code Agent Authoring (8-12 weeks)

This is the most technically ambitious phase. It builds the YAML agent definition runtime described in the spec's Section 6. It is genuinely hard to do well — the design choices made here will constrain the platform for a long time.

### Scope

**3.1 Single-shot agent YAML runtime**
Load `kind: agent` files from `config/agents/`. For each:
- Validate the schema at startup (prompt, output_schema, schedule/trigger, delivery)
- Resolve named output schemas from the built-in schema library
- At schedule time or trigger time: render the prompt with Jinja2, call `_make_specialist`, write result, deliver via configured channel
- Write to audit log and emit OTel spans using the same patterns as hand-coded agents

This is the simpler half of Phase 3 and should ship first. A useful agent (e.g., a daily "what's on my calendar today" briefing) can be written as a single-shot YAML agent.

**3.2 Graph-style workflow YAML runtime**
Load `kind: workflow` files from `config/agents/`. For each:
- Parse nodes and edges into a LangGraph StateGraph
- For each node, instantiate the appropriate executor: `llm_call` → `_make_specialist`, `builtin` → registered built-in function, `tool_call` → tool registry lookup, `code` → Python dotted-path import
- For edges with conditions, generate routing functions that evaluate the condition against state
- Compile and register the graph

This is where most of the complexity lives. The condition evaluation, state schema generation, and error handling during graph compilation need careful design.

**3.3 Built-in primitive library**
Implement the initial set from the spec: `date_enrich`, `calendar_query`, `calendar_create`, `calendar_delete`, `send_email`, `http_fetch`, `rss_fetch`. Each built-in is a standalone async function with a typed signature. They live in `src/alfred/builtins/` and register themselves via a decorator.

**3.4 Named schema library**
A set of pre-built Pydantic schemas that YAML agents can reference by name without writing Python: `IntentClassification`, `EventDraft`, `DeleteTarget`, `DigestItem`, `WeatherSummary`, `CalendarReply`, etc. These are the schemas the existing agents already use, extracted into a named registry. New schemas can be added by placing them in `src/alfred/schemas/named/`.

**3.5 Prompt library**
Named prompts in `config/prompts/<name>.md`. The runtime looks up prompts by name when an `llm_call` node specifies `prompt_template: <name>`. The existing calendar and curator prompts are migrated here as the seed library.

**3.6 Calendar agent declarative worked example**
Write `config/agents/calendar_workflow_declarative.yaml` that expresses the calendar agent's LangGraph workflow in the graph YAML schema. Run it against the existing scenario suite. This is the acceptance test for the authoring system. The Python workflow stays as canonical — this is proof of completeness, not a migration.

**3.7 Documentation: "How to author an agent in YAML"**
A step-by-step guide covering: single-shot vs. graph shape, available built-ins, named schemas, named prompts, scenario YAML for the harness, how to add a Python escape hatch when needed. This doc is the primary on-ramp for Persona 2 (YAML power user).

### Critical path within Phase 3

3.1 (single-shot runtime) → 3.3 (built-ins) → 3.4 (schema library) → 3.5 (prompt library) → 3.2 (graph runtime) → 3.6 (calendar worked example) → 3.7 (documentation)

3.2 blocks 3.6. 3.6 blocks shipping Phase 3. Do not ship Phase 3 without the worked example passing the scenario suite.

### Open questions that must be resolved before implementation starts

1. **State schema generation strategy** (spec Section 6.7 item 2). Decision needed before writing the graph runtime. Recommendation: auto-generate from YAML field list; write the TypedDict to `data/generated_schemas/` at startup.

2. **Condition expression language scope** (spec Section 6.7 item 3). Must be decided before implementing edge routing. Recommendation: equality/inequality only for v1.

3. **Where new schemas live for custom agents** (Python path vs. discovery convention). Recommendation: `src/alfred/schemas/custom/` with automatic discovery of any `BaseModel` subclass defined there.

### Success criteria
- A single-shot agent defined in YAML runs on schedule and delivers output via email
- A graph-style workflow defined in YAML executes correctly with at least 3 node types
- Calendar agent declarative YAML passes all existing scenarios
- `alfred health` reports status for YAML-defined agents
- "How to author an agent in YAML" documentation exists and covers the full workflow

### Dependencies
- Phase 1 complete (need the containerized Alfred and CI in place)
- The three open questions above must be resolved before starting 3.2

### Effort notes
This is the most genuinely hard phase. The single-shot runtime (3.1) is a week of work. The graph runtime (3.2) is 3-4 weeks — not because the individual parts are complex, but because the edge cases compound: what happens when a node fails, how does the graph handle cycles (it shouldn't allow them, needs validation), how does the `code` escape hatch handle import errors at startup vs. at runtime. Plan for the calendar worked example (3.6) to reveal gaps in the runtime — budget 1-2 weeks of iteration after the first attempt.

---

## Phase 4 — Tier 3: Ecosystem (Ongoing)

Phase 4 items are not a sequential project — they are things that can be worked on independently once Phase 1 is stable. No single person needs to do all of them.

### Scope

**4.1 PyPI package**
Package the platform as `openalfred` on PyPI. This requires: clean public API surface, `pyproject.toml` metadata, release workflow via GitHub Actions. The CLI (`alfred` command) should be installable via `pip install openalfred` and `uvx openalfred`.

**4.2 Plugin system**
A formal plugin registration mechanism: a plugin is a Python package that registers agents, tools, built-ins, and schemas with OpenAlfred's registries. The plugin is installed alongside OpenAlfred (e.g., `pip install openalfred-plugin-spotify`). OpenAlfred discovers plugins at startup via entry points. This replaces the current convention of "put things in `src/alfred/schemas/custom/`" with a proper extensibility contract.

**4.3 Render/Fly.io deployment template**
A one-click deployment template for cloud hosting. This is for users who don't have a homelab but want to run OpenAlfred on a cheap VPS. Covers: single-service deployment (Alfred + LiteLLM + cloud Ollama alternative), config via environment variables, no GPU required. Use cloud Ollama hosting or switch to a cloud-only model config.

**4.4 Multi-LLM provider documentation**
Documented, tested configurations for: Ollama (already done), Anthropic Claude (already done), OpenAI, Groq, Mistral. LiteLLM supports all of these; it's mostly a `litellm.yaml` template and a README section per provider.

**4.5 Community agent examples**
A `examples/agents/` directory with YAML-defined agent examples contributed by the community. Seed it with 3-5 examples: daily news headline digest, GitHub notifications summary, simple todo reminder, home weather briefing.

### Dependencies
- Phase 3 must be complete before 4.1 (PyPI) and 4.5 (community agents), since the YAML authoring system is a core part of the public API

---

## Risks and Mitigations

**Risk 1: Graph YAML runtime is harder than estimated.**
The condition routing, state schema generation, and error handling in the graph runtime are where complexity accumulates. If it takes longer than 4 weeks, cut scope: ship the single-shot runtime first (3.1) and defer the graph runtime (3.2) to a follow-on. Single-shot agents cover a meaningful slice of use cases.

**Risk 2: The calendar declarative worked example reveals the YAML schema is not expressive enough.**
The calendar workflow has non-trivial routing logic (confidence-based delete routing, complexity-based model selection). If these can't be expressed cleanly in the YAML schema without multiple `code` escape hatches, that's a signal the schema needs to be revisited before shipping Phase 3. The worked example (3.6) is the test — run it before declaring Phase 3 done.

**Risk 3: The simulation harness becomes a bottleneck in CI.**
Per-scenario time at ~5s adds up. As more agents and scenarios are added, CI time grows. Mitigation: run scenarios in parallel (the harness already supports this in principle); add a `--fast` mode that runs a curated subset for PR checks and the full suite only on main.

**Risk 4: Email provider diversity creates long tail of support burden.**
Every new email provider is a potential source of auth edge cases, TLS quirks, and SMTP configuration gotchas. Mitigation: invest in the generic SMTP path (Phase 2) and document it well. Resist adding provider-specific code for every provider — document the generic path clearly so users can self-serve.

**Risk 5: Ollama model quality variation breaks the simulation harness.**
The harness runs against the configured local model. If a user upgrades their Ollama model or a new model release changes behavior, scenarios that were passing may fail. Mitigation: pin model versions in CI; document that the harness results are model-dependent; include model name and version in pass/fail output.

---

## Testing Strategy

### Simulation harness (existing, extend)
The `sim/` package with `tests/scenarios/*.yaml` is the primary regression guard. Real LLM calls, fake I/O. Currently covers the calendar agent. Extensions:
- Add a curator agent scenario runner (same pattern as calendar's `InMemoryCalendarClient` — needs an `InMemorySourceFetcher`)
- Add scenarios for every new agent shape in Phase 1.5
- Add scenarios for YAML-defined agents in Phase 3 (the YAML runtime should be testable the same way as hand-coded agents — same `InMemory*` fake I/O, same scenario format)
- Run in CI on every PR; block merges below 95% pass rate

### Unit tests
Node functions in the calendar workflow are pure async functions. They are unit-testable with mocked state dicts. Add unit tests for: the date enrichment node (edge cases: DST transitions, end of year), the validation node (each completeness issue), the routing functions (each edge condition). These are fast and deterministic — they belong in the standard `pytest` suite, not the simulation harness.

### Integration tests
`alfred health` is the integration test surface for the running stack. A `tests/integration/` directory can hold tests that require live services (LiteLLM, Ollama) — marked with `pytest.mark.integration` and excluded from default CI runs. Run manually before releases.

### YAML schema validation tests
Once the YAML authoring runtime exists, add tests in `tests/authoring/` that validate: well-formed YAML loads without error, malformed YAML (missing required fields, unknown node types, cycles in edges) raises a clear error at startup, not at runtime.

---

## Phase 1 Sprint Plan: First 2-3 Weeks

These are the tasks to start immediately after Phase 0 is complete.

**Week 1: Containerize Alfred and validate CI**
- Write `Dockerfile` for Alfred (Python 3.12, uv, copy source)
- Add `alfred` service to `docker-compose.yml`
- Wire Alfred container networking: LiteLLM via service name, Phoenix OTLP endpoint, Ollama via `extra_hosts` or service name
- Verify the scheduler and inbox poller run correctly inside the container
- Add GitHub Actions workflows: `ruff check`, `mypy`, `pytest`, secrets audit
- Verify all pass on main with no false positives in the secrets audit

**Week 2: Setup wizard and health check**
- Implement `alfred setup` interactive CLI (Click prompts, `.env` file writer)
- Implement `alfred health` with per-service checks and fix hints
- Test the wizard end-to-end on a clean machine (Docker installed, nothing else)
- Document the result in a "Quickstart" section in the README

**Week 3: Simulation harness in CI + curator scenarios**
- Add GitHub Actions workflow for `alfred simulate` (runs on PR, blocks below 95%)
- Write 5 curator agent scenarios (normal, no items, source failure, synthesis failure, topic filtering)
- Implement `InMemorySourceFetcher` for the curator simulation runner (mirrors `InMemoryCalendarClient`)
- Verify curator scenario runner works and passes in CI

After Week 3, Phase 0 + the first three Phase 1 items are complete. The repo is public-ready. A homelab developer can clone it and reach a working setup. The simulation harness is a CI gate. The project is in a state where a second contributor could meaningfully help.
