# Project Backlog

Tracked work items not yet scheduled. Newest at top.

---

## Guardrails so the Anthropic-Sonnet 429 cascade can't recur

**Status:** The original incident is already mitigated — but no test
prevents regression.

**What actually happened (2026-05-17 01:58 UTC / 2026-05-16 18:58 MST):**
Phoenix shows 117 ERROR spans clustered around 01:57:49 UTC, every
one a 500 wrapping a Sonnet `rate_limit_error: 30,000 input tokens
per minute (model: claude-sonnet-4-6)`. **It was not legitimate cloud
traffic hitting the cap** — it was a cascade:

1. Ollama unreachable: `api_base` was set to `${OLLAMA_BASE_URL:=...}`
   (shell-default syntax LiteLLM doesn't expand), so `local-default`
   resolved to a literal URL containing `${...}` and every call 500'd.
2. LiteLLM had `fallbacks: [{local-default: [cloud-default]}, {local-fast:
   [cloud-default]}]` — every failed local call silently retried against
   Anthropic Sonnet.
3. Sonnet's **Tier 1 input-TPM (30k)** blew up first — not RPM, not
   output TPM. Routine traffic was suddenly hammering cloud.
4. LiteLLM router `num_retries: 2` amplified each failure to 3 attempts.

**Already fixed in commit f3ed20d** (Sun 2026-05-17 15:48 PT, 21h after
the incident, as part of an unrelated Qwen 3 upgrade):
- `config/litellm.yaml:57` is now `fallbacks: []` with a comment locking
  in the [No Cloud Fallback] policy.
- `api_base: os.environ/OLLAMA_BASE_URL` (correct LiteLLM env syntax).

**Tier 1 caps for reference:** 50 RPM / 30k input TPM / 8k output TPM.

**Why this is still backlog-worthy:** the config matches policy today,
but nothing stops a future PR from re-adding the fallback or from
silently re-introducing the shell-default URL syntax. Guardrails:

- **Config-shape test.** Parse `config/litellm.yaml` in CI and assert
  `router_settings.fallbacks == []`. Single check, prevents the exact
  regression that caused the incident.
- **Smoke test for env-var expansion.** Boot LiteLLM in CI against
  the committed config with a stub `OLLAMA_BASE_URL` and verify the
  resolved `api_base` doesn't contain `${`. Cheap, catches the other
  half of the cascade.
- **Retry policy review.** With no fallback, `num_retries: 2` against
  a 429-returning upstream is a 3× amplifier. Either drop to 0 (fail
  fast — user sees the error) or add exponential backoff with jitter.
  Probably drop to 0 for cloud; keep 1 retry for local.
- **Pre-flight input-token estimation.** Even with policy correct,
  routine traffic shouldn't be sending ~10k-input-token requests. Add
  a check in `src/alfred/budget/` that warns/errors when a single call
  exceeds, say, 5k input tokens — that's an in-prompt issue, not a
  rate-limit one.
- **Phoenix alert / log signal.** Currently 429s show up as buried
  ERROR spans. Add a structured log line (`event_type=rate_limited`
  with provider + model + retry-after) so future incidents are
  greppable.

**Out of scope for now:**
- Multi-key load balancing, tier upgrade, queueing layer — premature
  until we're actually exercising the cap with real cloud traffic.
- Automatic cloud→local fallback — explicitly against policy.

**Acceptance:**
- CI fails if `fallbacks` is re-added or if the LiteLLM config can't
  resolve env vars cleanly.
- Cloud retries on 429 drop to 0 (or backoff-with-cap) — config diff
  + one trace shown.
- A test prompt > 5k input tokens trips the new budget warning.
