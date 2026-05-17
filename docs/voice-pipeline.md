# Voice Pipeline — iMessage Voice Memos through Alfred

Status: design. No implementation has landed. Targets Phase 4 of
[`PRODUCT_VISION.md`](../PRODUCT_VISION.md) ("Voice & Household Management"). Builds on the
Speaches/faster-whisper service already defined in [`docker-compose.yml`](../docker-compose.yml)
and the BlueBubbles relay scaffolded in `src/alfred/notifications/channels/imessage.py`.

---

## 1. Overview & Motivation

Family members already send iMessage voice memos casually ("hey, add carrots to the grocery
list", "what's on the calendar tomorrow?"). The voice pipeline makes Alfred a first-class
recipient of those memos. The flow is: an iMessage voice memo arrives at the Mac Mini's
Messages app, BlueBubbles relays a `new-message` webhook to Alfred, Alfred downloads the
audio attachment from BlueBubbles, ships the bytes to the local Speaches/faster-whisper
service on the homelab GPU, takes the returned transcript, constructs a normal
`AgentRequest(source=RequestSource.IMESSAGE)`, and dispatches through the existing
orchestrator. The reply is sent back via the same BlueBubbles channel the orchestrator
already uses for text-sourced iMessage requests. Nothing in the orchestrator, agents, or
notification routing needs to change — voice is just another way to produce a text
`user_message`.

---

## 2. Architecture

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │                          Mac Mini                                   │
   │                                                                     │
   │   ┌────────────┐   voice memo   ┌──────────────────────┐            │
   │   │ Messages   │ ─────────────▶ │ BlueBubbles server   │            │
   │   │ (iMessage) │                │  - REST API          │            │
   │   └────────────┘                │  - Webhook emitter   │            │
   │                                 └──────────┬───────────┘            │
   │                                            │ POST /webhooks/        │
   │                                            │   bluebubbles          │
   │                                            │ {event:"new-message",  │
   │                                            │  attachments:[...]}    │
   │                                            ▼                        │
   │                              ┌──────────────────────────┐           │
   │                              │  Alfred process          │           │
   │                              │                          │           │
   │                              │  aiohttp webhook         │           │
   │                              │   receiver               │           │
   │                              │   │                      │           │
   │                              │   ▼                      │           │
   │                              │  GET /api/v1/attachment/ │           │
   │                              │   <guid>/download ───────┼──┐        │
   │                              │                          │  │        │
   │                              │   bytes ◀────────────────┼──┘        │
   │                              │   │                      │           │
   │                              │   ▼                      │           │
   │                              │  WhisperClient ──────────┼─────┐     │
   │                              │   (OpenAI SDK against    │     │     │
   │                              │    WHISPER_BASE_URL)     │     │     │
   │                              │                          │     │     │
   │                              │   transcript ◀───────────┼──┐  │     │
   │                              │   │                      │  │  │     │
   │                              │   ▼                      │  │  │     │
   │                              │  Orchestrator.handle(    │  │  │     │
   │                              │    AgentRequest(         │  │  │     │
   │                              │     source=IMESSAGE,     │  │  │     │
   │                              │     user_message=txt))   │  │  │     │
   │                              │   │                      │  │  │     │
   │                              │   ▼                      │  │  │     │
   │                              │  NotificationService     │  │  │     │
   │                              │   → BlueBubblesClient.   │  │  │     │
   │                              │     send() ──────────────┼──┼──┼──┐  │
   │                              └──────────────────────────┘  │  │  │  │
   │                                            ▲               │  │  │  │
   │                                            └───────────────┼──┼──┘  │
   │                                              reply text    │  │     │
   └────────────────────────────────────────────────────────────┼──┼─────┘
                                                                │  │
                          home LAN (no WAN)                     │  │
                                                                │  │
   ┌────────────────────────────────────────────────────────────┼──┼─────┐
   │                       Homelab (RTX 4090)                   │  │     │
   │                                                            │  │     │
   │   ┌──────────────────────────────────────────────────┐     │  │     │
   │   │ Speaches container (alfred-whisper)              │     │  │     │
   │   │   POST /v1/audio/transcriptions ◀────────────────┼─────┘  │     │
   │   │   - faster-whisper-large-v3 (CUDA, fp16)         │        │     │
   │   │   - 5 min model TTL                              │        │     │
   │   │   transcript ─────────────────────────────────── ┼────────┘     │
   │   └──────────────────────────────────────────────────┘              │
   └─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Audio Format Handling

iMessage voice memos arrive in one of two container formats depending on the sender's iOS
version. Both are handled transparently by Whisper because Speaches/faster-whisper invokes
ffmpeg under the hood for any non-WAV input.

| Sender iOS  | Container | Codec     | Typical extension | Whisper handles natively? |
|-------------|-----------|-----------|-------------------|---------------------------|
| iOS 13 – 16 | CAF       | Opus/AAC  | `.caf`            | Yes (via ffmpeg)          |
| iOS 17+     | M4A       | AAC-LC    | `.m4a`            | Yes (via ffmpeg)          |
| Other       | WAV       | PCM       | `.wav`            | Yes (native)              |
| Other       | MP3       | MPEG-1 L3 | `.mp3`            | Yes (via ffmpeg)          |
| Other       | FLAC      | FLAC      | `.flac`           | Yes (via ffmpeg)          |
| Other       | OGG/Opus  | Opus      | `.ogg`            | Yes (via ffmpeg)          |

There is no conversion step in Alfred. We pass the raw bytes from BlueBubbles directly to
Speaches with the original filename so ffmpeg can sniff the container. If a future sender
ships an exotic codec the Speaches container's ffmpeg cannot decode, we surface the error
as a polite iMessage reply rather than attempting to fix it client-side.

---

## 4. Component Breakdown

### 4.1 `src/alfred/audio/whisper_client.py` (new)

A thin async wrapper around the OpenAI Python SDK pointed at `WHISPER_BASE_URL` (e.g.
`http://homelab.local:8000/v1`). Speaches exposes an OpenAI-compatible
`/v1/audio/transcriptions` endpoint, so we get the SDK's retry, streaming, and error
typing for free.

Rough shape:

```python
class WhisperClient:
    def __init__(self, base_url: str, model: str = "Systran/faster-distil-whisper-large-v3"):
        self._client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
        self._model = model

    async def transcribe(
        self,
        audio: bytes,
        filename: str,
        language: str | None = "en",
    ) -> str:
        resp = await self._client.audio.transcriptions.create(
            model=self._model,
            file=(filename, audio),
            language=language,
            response_format="text",
        )
        return resp.strip() if isinstance(resp, str) else resp.text.strip()

    async def health_check(self) -> bool: ...
    async def close(self) -> None: ...
```

Configuration lives in `config/settings.yaml` under a new `audio:` block; `WHISPER_BASE_URL`
already exists as an environment variable per the README.

### 4.2 Webhook receiver — choices

Two viable shapes for receiving BlueBubbles `new-message` events:

**(a) Native aiohttp/FastAPI endpoint inside the Alfred process** (recommended)

A small ASGI app (FastAPI for symmetry with the rest of the Python ecosystem, or aiohttp if
we want to avoid the dep) registered as part of `alfred.app` startup, listening on
`POST /webhooks/bluebubbles` on the Mac Mini's LAN interface. The handler:

1. Validates the BlueBubbles shared secret in the request header.
2. Inspects the payload for `attachments[*].mimeType` starting with `audio/`.
3. Pulls the attachment guid, downloads via `GET /api/v1/attachment/<guid>/download`.
4. Calls `WhisperClient.transcribe(...)`.
5. Constructs `AgentRequest(source=RequestSource.IMESSAGE, user_message=transcript,
   user_id=<sender handle>, metadata={"channel": "voice", "attachment_guid": ...})`.
6. Awaits `orchestrator.handle(...)`; reply is sent by the existing notification path.

**(b) n8n workflow that shells out to `alfred run-routine voice-message`**

n8n receives the BlueBubbles webhook, downloads the attachment, writes it to a tmp path,
and invokes the Alfred CLI via shell. This is the simpler-feeling option but loses on three
axes: added shell-out latency (~300–800 ms per cold Python start), poorer audit-log
fidelity (the orchestrator does not see the source as a structured webhook event), and a
second deployment surface to maintain (n8n credentials, workflow versioning).

**Decision: pick (a).** Lower steady-state latency, no subprocess fork per message, the
same audit log records the entire transaction, and the webhook handler can share the
already-constructed `Orchestrator`, `WhisperClient`, and `NotificationService` singletons
from `app.py`. We will still use n8n for cron and RSS triggers — voice just does not need
it.

The webhook code itself lives in a new package `src/alfred/webhooks/` with at minimum
`bluebubbles.py` (the handler) and `server.py` (ASGI app + lifespan hooks).

### 4.3 BlueBubbles attachment download

BlueBubbles exposes attachments by guid. The download endpoint is:

```
GET {base_url}/api/v1/attachment/{guid}/download?password={password}
```

Returns the raw bytes with a `Content-Type` matching the original attachment mime type and
a `Content-Disposition` carrying the original filename. We extend `BlueBubblesClient` (or
add a sibling `BlueBubblesAttachments` helper alongside it in
`src/alfred/notifications/channels/imessage.py` or a new `src/alfred/imessage/` module if
we want to separate inbound concerns from outbound) with:

```python
async def download_attachment(self, guid: str) -> tuple[bytes, str]:
    """Return (bytes, filename) for the given attachment guid."""
```

### 4.4 Reply path

Already implemented. `NotificationService` resolves the
`RequestSource.IMESSAGE`-originated request to `BlueBubblesClient.send(...)` via the
existing conditional-edge logic. The webhook handler does not call the notification service
directly — it lets the orchestrator graph do it, so voice messages go through the same
audit trail and budget gate as every other source.

**Implementation gap:** `BlueBubblesClient` currently sends to `notification.recipient` as
the chat guid. For voice replies the natural recipient is the chat the voice memo arrived
in (group or 1:1). The webhook handler must thread the inbound `chatGuid` through
`AgentRequest.metadata` and the notification service must read it when present, falling
back to the configured default recipient. Plumbing this through the existing
`Notification` model is a small additive change but is not in the code today.

---

## 5. Orchestrator Integration

Nothing in `src/alfred/orchestrator/` changes. The integration point is purely at the edge:

```python
request = AgentRequest(
    source=RequestSource.IMESSAGE,
    user_message=transcript,
    user_id=sender_handle,
    metadata={
        "channel": "voice",
        "attachment_guid": guid,
        "chat_guid": payload["data"]["chatGuid"],
        "audio_duration_s": duration_seconds,
    },
)
response = await orchestrator.handle(request)
```

`Orchestrator.handle` already sets `should_notify=True` whenever
`request.source in ("scheduler", "imessage", "webhook")` (see
`src/alfred/orchestrator/orchestrator.py:51`), so the conditional edge in the graph will
route the agent response back through `NotificationService` without any new branching.

`RequestSource.IMESSAGE` already exists in `src/alfred/core/constants.py` — no enum change
needed.

---

## 6. Latency Budget

Voice is interactive; perceived snappiness matters. Target steady-state under six seconds
from end-of-recording to first reply byte sent, with cold-start excursions tolerated.

| Stage                                    | Steady-state | Cold-start  | Notes                                              |
|------------------------------------------|--------------|-------------|----------------------------------------------------|
| BlueBubbles webhook delivery             | < 100 ms     | < 100 ms    | LAN, no auth round-trip beyond shared secret       |
| Attachment download                      | < 500 ms     | < 500 ms    | Bounded by audio size; voice memos are tens of KB  |
| Whisper transcription                    | 1 – 2 s      | 6 – 12 s    | Cold = 5 min TTL expired, model reload from disk   |
| Orchestrator intent + budget gate        | 50 – 200 ms  | 50 – 200 ms | Pure Python + a small LLM classify call            |
| Agent execution (LLM + tool calls)       | 1 – 3 s      | 1 – 3 s     | Depends on agent; calendar reads ~1.5 s typical    |
| BlueBubbles send                         | < 500 ms     | < 500 ms    | LAN POST to BlueBubbles REST                       |
| **Total**                                | **3 – 6 s**  | **8 – 16 s**| Cold start dominated by Whisper model reload       |

Mitigations if cold-start becomes painful in practice: lengthen `WHISPER__TTL` from 300 s
to e.g. 1800 s (trades VRAM for latency), or pin the model via TTL=0 (always resident).
Both are config-only changes in `docker-compose.yml`.

---

## 7. Security & Privacy Posture

- Audio bytes never leave the home LAN. The Mac Mini fetches the attachment from
  BlueBubbles (localhost) and POSTs it to Speaches on the homelab over the LAN. No cloud
  STT provider is in the loop.
- Speaches runs on the homelab GPU in a Docker container with no public port exposure
  (`8000` is bound on the LAN interface only; see `docker-compose.yml`).
- The webhook receiver validates a shared secret on every inbound request and rejects
  anything that does not originate from the configured BlueBubbles instance.
- Transcripts are written to the SQLite audit log under
  `AuditEventType.AGENT_DISPATCH` like any other request — searchable, retention-bound by
  the existing audit policy.
- Raw audio bytes are held in memory for the duration of the webhook handler and
  discarded. We do not persist audio to disk, the database, or anywhere else. This is an
  explicit code-level invariant, not just a convention: the webhook handler must not
  write bytes to disk and must not log them.
- LLM cost: only the agent that ultimately handles the transcribed request incurs cost,
  exactly as if the same text had arrived via CLI. Whisper itself is local-free.

---

## 8. Open Questions / Future Work

- **Streaming vs batch transcription.** Speaches supports both. Batch (single POST,
  single transcript response) is simpler, has lower operational surface, and is fine at
  the lengths voice memos typically run. Revisit streaming if we add a "push-to-talk in
  the kitchen" interface where partial transcripts let the agent start working earlier.
- **Speaker identification.** Out of scope for v1. The sender is taken from the
  BlueBubbles message metadata (`handle.address`). If the household ever wants per-voice
  personalization within a shared chat, speaker diarization is a separate problem.
- **Audio replies (TTS back to iMessage).** Out of scope for v1. The reply path is text.
  If we want spoken replies later, candidates are Piper or Coqui TTS on the same homelab
  GPU; iMessage will deliver M4A attachments fine, but the BlueBubbles send-attachment
  path is a different endpoint than the text-send path used today.
- **Long voice notes.** If a memo exceeds ~2 minutes we should chunk into ~60 s windows
  with a small overlap, transcribe in parallel, and concatenate. Not needed for v1; add
  when we see the first failure or first multi-minute memo.
- **Reply targeting.** As noted in the implementation gap above, the
  `Notification`/`NotificationService` pair needs to learn how to send back to a specific
  `chatGuid` (carried in `AgentRequest.metadata["chat_guid"]`) rather than the
  configured default recipient.

---

## 9. Dependencies on Other Phased Work

- **Requires Mac Mini deployment.** BlueBubbles only runs on macOS with a logged-in Apple
  ID. Until the Mac Mini is provisioned, the voice pipeline cannot exist end-to-end.
- **Requires Phase 1 (Calendar) to be useful.** A voice pipeline that can only answer
  "what time is it" is unmotivating. The first concrete user value is "what's on my
  schedule?" / "did the dentist confirm Tuesday?", so Google Calendar OAuth and the
  calendar agent's weekly digest are the prerequisites for shipping voice as a feature
  someone would actually use.
- **Does not require any new external services.** Whisper is already in
  `docker-compose.yml`. No new API keys, no new vendors, no new line items on the
  monthly bill.
- **Does not consume new budget.** Whisper inference is local and free. The only LLM
  cost is whatever the routed agent would have spent on the same request submitted as
  text — already covered by the existing per-agent budget caps.

---

## Implementation Gaps Found While Writing This Doc

1. `BlueBubblesClient` is send-only. There is no `download_attachment` method, no
   webhook receiver, and no inbound-message data model. All of this is net-new work.
2. `NotificationService` / `BlueBubblesClient.send` formats `chatGuid` as
   `f"iMessage;-;{notification.recipient}"`, which assumes 1:1 chats keyed by phone or
   email handle. Voice replies into group chats need the raw chat guid passed through
   unchanged — this is the reply-targeting gap called out in section 8.
3. There is no `src/alfred/audio/` package, no `src/alfred/webhooks/` package, and no
   ASGI server wired into `alfred.app`. The CLI today is the only entry point to the
   orchestrator.
4. `Orchestrator.handle` checks `request.source in ("scheduler", "imessage",
   "webhook")` as a literal-string tuple at `orchestrator.py:51` rather than against
   `RequestSource` enum members. This works today because `RequestSource` is a
   `StrEnum`, but it is a latent bug surface: there is no `RequestSource.WEBHOOK`, so
   the `"webhook"` entry is dead code. Worth cleaning up alongside the voice work even
   though it does not block the pipeline.
