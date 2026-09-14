# Architecture — the AI/ML system behind Seki

**Status:** draft · **Date:** 2026-09-14 · **Scope:** the whole ML system across
`robo-worker`, `robo-bridge`, `robo-be`, `proass`, `model-registry`, and the
planned `stt-tts-bench`. Per-repo detail lives in each repo; this document is
the one place that shows how they fit and where the ML lifecycle is complete,
partial, or missing.

Every section is labelled: **BUILT** (in code, verified this month), **LIVE**
(serving real sessions), **PLANNED** (designed, not built), **OPEN** (a decision
nobody has made yet). Claims about code cite the file they come from.

---

## 1. What the system is

Seki is a robot that teaches spoken English to Vietnamese children aged 5–8
(Pre-A1). A child speaks to the robot; the robot listens, decides what to say
as a tutor, and speaks back — in under about two seconds, in a mix of
Vietnamese and English, with pronunciation feedback. Parents follow along in the
Tuni app.

Every conversation turn is the same loop:

```
child speaks ─► STT ─► the brain (intent, lesson, memory, LLM) ─► TTS ─► robot speaks
                                     │
                                     └─► pronunciation assessment (proass), memory write
```

Four ML capabilities sit inside that loop — **STT**, **TTS**, **LLM**, and
**pronunciation assessment** — and the architecture question of this project is
where each runs, which model version, and how a better one gets in.

## 2. The repositories

| Repo | Runtime | Role | Status |
|---|---|---|---|
| `robo-worker` | Cloudflare Worker + Durable Objects (Agents SDK) | **The brain.** Sessions, intent, lessons, memory, prompt, LLM calls, device auth, parent notifications | LIVE |
| `robo-bridge` | Go on Cloud Run (asia-southeast1) | WebSocket↔gRPC shim so the Worker can reach Google STT, TTS **and the LLM**; voice-clone key handling | LIVE |
| `proass` | Cloud Run | Pronunciation assessment: scripted/unscripted scoring, word and phoneme scores 0–100, its own ASR transcript | LIVE |
| `robo-be` | Python on a single GPU host, NATS between containers | **Self-hosted STT/TTS.** LID-routed multi-model ASR, VoxCPM2 TTS. The original speech path; now the target of the migration back | BUILT, not wired |
| `model-registry` | Python library + MLflow server | Versioned models with movable aliases; every model robo-be runs, pinned to a commit | BUILT |
| `stt-tts-bench` | (new repo) | Tier 1 evaluation — raw model quality, open-source vs Google | PLANNED |
| `tuni-noti` | Cloudflare Worker | Parent notifications (queues from robo-worker) | LIVE, out of ML scope |

## 3. Runtime architecture — one session

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  xiaozhi robot                                                                   │
│  mic → Opus → WS      ◄─ Opus/PCM audio, mood → LED                              │
└────────────────────────────────────┬─────────────────────────────────────────────┘
                                     │ /xiaozhi/ws  (Worker-signed session token)
┌────────────────────────────────────▼─────────────────────────────────────────────┐
│  robo-worker  ·  Tutor Durable Object per child                          [LIVE]  │
│                                                                                  │
│   listen ─► bridge/stt.ts ──WSS──► robo-bridge /stt ──gRPC──► Google STT v2      │
│                                  (PCM frames, {type:"eos"}, finals only) Chirp 3 │
│   intent (LLM-classified) · lesson intents · assessment · praise-director        │
│   memory: extract ─► dedupe ─► consolidate ─► prune ─► retrieve                   │
│           Vectorize (child episodes) + DO SQLite (per child) + D1 (global)       │
│           embeddings: Workers AI                                                 │
│   prompt ─► llm/index.ts ──HTTPS──► robo-bridge ──► Gemini 3.7 Flash             │
│             streamText, structured output, "(mood)" markers split from stream    │
│   reply ─► bridge/tts.ts ──WSS──► robo-bridge /tts ──gRPC──► Google TTS          │
│             VI: Chirp 3 HD · EN: Gemini-TTS · ICV clones · R2 TTS cache          │
│             pause tags + silence injection (tts-pauses.ts, tts-silence.ts)       │
│   speech ─► services/ipa ──HTTPS──► proass  (word/phoneme scores, feedback)      │
│   course: R2 robo-course, CurriculumWorkflow, PersonalWorkflow                   │
│   parents: queues ─► tuni-noti                                                   │
│                                                                                  │
│   legacy: utils/gpu.ts GPUClient(ROBO_BE_URL) — TTS-only fallback used when      │
│           TTS_BRIDGE_URL is unset (dos/tutor.ts:3981). Not the live path.        │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│  robo-be  ·  self-hosted speech                                    [BUILT, not   │
│                                                                      wired]      │
│   /v1/ws/stt ─► NATS ─► stt-lid: Demucs ─► XLSR-53 phonemes ─► VoxLingua107 /    │
│                          MMS-LID ─► language spans ─► per-language ASR RPC       │
│                          cohere-asr (vi+en) · gipformer (vi) · qwen3 (fallback)  │
│   /v1/ws/tts ─► NATS ─► voxcpm2-tts (bilingual, voice cloning from ref wavs)     │
│   weights today: hardcoded HF repo ids; tomorrow: registry.resolve(name@alias)   │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│  model-registry                                                        [BUILT]   │
│   ModelRegistry facade ─► MLflow (aliases) ─► Postgres (metadata) + GCS (weights)│
│   models.yaml: 8 models pinned to commit SHAs · consumers: robo-be, bench        │
└──────────────────────────────────────────────────────────────────────────────────┘
```

Where things run, at the level that matters architecturally: the brain is at the
edge (Cloudflare), speech and LLM are Google managed APIs reached through one
proxy, pronunciation is a Cloud Run service, self-hosted speech is one GPU host.

## 4. The brain — `robo-worker`

*Source of truth for this section: `wrangler.jsonc` bindings and `src/` as of
2026-09-14. The April planning doc (`docs/system-architecture.md`, dated
06/04/2026) is superseded in several places noted below.*

**Session model.** One `Tutor` Durable Object per child (`src/dos/tutor.ts`),
built on the Agents SDK, holding the live WebSocket to the device. Sessions
reach it via the device-auth flow: device keypair → `/api/devices/register` →
claim to a child → `/xiaozhi/ota` verifies a chip-signed JWT (replay nonces in
KV) → session token → `/xiaozhi/ws`.

**What a turn does inside the DO** (`src/services/tutor/`): `listen` collects
the utterance; `intent` classifies it with the LLM against a registry of
lesson intents (`services/intent/`, structured output); `assessment` and
`course-speech` decide whether this was a pronunciation attempt or free
practice; `praise-director` and `feedback-policies` shape the pedagogical
response; `prompt` assembles persona + lesson + memory; `reply` streams the
LLM answer, splitting `(mood)` markers for the robot's expression; `voice`
picks the TTS voice per language.

**Memory.** MemPalace-inspired — *store everything, make it findable*
(`docs/mempalace-reference.md`). Pipeline in `services/memory/`:
`extract → dedupe → consolidate → prune → retrieve`. Per-child episodes are
embedded with Workers AI and stored in **Vectorize** (`VECTORIZE_CHILD_EPISODES`);
per-child structured state lives in **DO SQLite**; **D1** (`robo-d1`) holds
global relational data (children, devices). *This supersedes the April doc's
"D1 only, no Vectorize" decision.*

**Lessons and curriculum.** Lesson script bundles in R2 (`robo-course`), an
authoring pipeline (`src/course/authoring/`), and two Cloudflare Workflows —
`CurriculumWorkflow` and `PersonalWorkflow` — for curriculum alignment and
per-child personalisation. Games (`services/games/word-chain`) are lesson
activities.

**Speech seams.** Exactly three files know about external speech/LLM services:
`src/bridge/stt.ts`, `src/bridge/tts.ts`, `src/llm/index.ts`. Each reads one URL
(`STT_BRIDGE_URL`, `TTS_BRIDGE_URL`, `LLM_BRIDGE_URL`) and authenticates with a
WIF-minted ID token. **This is the migration's own alias-repoint**: swapping the
speech backend is changing two URLs, provided the thing behind them speaks the
same frames.

Two protocol facts that matter for the migration (from `stt.ts`):

- The Worker streams **PCM** frames and sends `{"type":"eos"}`; it treats
  interim results as noise — *"partial transcript (ignored here)"* — and folds
  each utterance into one recognition stream. **Interim support is not a
  requirement** for a replacement backend.
- One utterance per socket for STT; a persistent socket with per-chunk
  `tts_end` for TTS; a config frame `{voice, lang}` only when either changes.

**Pronunciation.** `services/ipa/client.ts` calls proass with the target
*text* (proass runs its own G2P) and receives `{mode, transcript,
scores: {utterance_score, words[{word, score, phonemes[{phoneme, score,
start, end}]}]}, feedback}` on a 0–100 scale. The legacy `pa-service` response
shape is still parsed so a rollback is a vars-only change — the same
config-as-deploy pattern as the bridge URLs.

**Other:** TTS output is cached in R2 (`robo-tts-cache`) with pause-tag-aware
cache keys; parent notifications go through two queues to `tuni-noti`.

## 5. Speech and language services — two paths

| | Path A — Google via robo-bridge | Path B — robo-be self-hosted |
|---|---|---|
| Status | **LIVE** | BUILT, not wired to the Worker |
| STT | Speech-to-Text v2, Chirp 3, `us` region (only streaming GA region) | LID-routed: Demucs denoise → XLSR-53 phonemes → VoxLingua107 / MMS-LID → Cohere Transcribe (vi+en), Gipformer (vi), Qwen3-ASR (fallback) |
| TTS | Gemini-TTS (EN), Chirp 3 HD (VI), Chirp 3 Instant Custom Voice clones (keys in a GCS bucket + `voices.json`) | VoxCPM2, bilingual, cloning from `vi_ref.wav` / `en_ref.wav`; ElevenLabs as an HTTP fallback profile |
| LLM | Gemini 3.7 Flash via the same bridge | — (the LLM stays on Path A regardless) |
| Audio in | PCM 16-bit | Opus packets |
| STT socket | one utterance per WS, finals only | persistent WS, server VAD, one event per segment |
| Quality (native VN, adult) | not measured against the same set | WER 10.2 % / CER 6.9 % (2026-04-28, denoiser on); target < 10 % |
| Model provenance | Google's | hardcoded HF repo ids, HEAD at first download — one already broken upstream (gipformer, 2026-08-21) |

**History.** Path B came first: `utils/gpu.ts` is the original GPU TTS client
and survives as a fallback. Google replaced it. The current plan is to move
speech back to self-hosted models — for cost, control, and the ability to
finetune on the actual population — while keeping the LLM on Gemini.

**What "wiring Path B" actually requires**, corrected against the Worker's
code: an adapter in front of robo-be that accepts PCM and `{"type":"eos"}` and
answers with the bridge's `final` frame and close semantics; interim results
are *not* needed. TTS is closer already (`tts_end`, config frame). Then
`STT_BRIDGE_URL`/`TTS_BRIDGE_URL` move.

**Model inventory** (the eight models robo-be runs; catalog in
`model-registry/models.yaml`, each pinned to a commit): `cohere-asr`,
`gipformer-asr-vi`, `qwen3-asr`, `funasr-nano`, `mms-lid-256`,
`voxlingua107-lid`, `xlsr53-phoneme`, `voxcpm2-tts`. Plus Demucs DNS64
(denoiser, not from the Hub) and the proass model, which is its own service.

## 6. Model management — `model-registry` (BUILT)

One versioned source of truth. A consumer calls
`ModelRegistry().resolve("cohere-asr", "production")` and gets a directory;
it never imports MLflow and never learns a version number. Promotion and
rollback are alias moves; every move is recorded (who, when, from what).
Weights live in GCS, metadata in Postgres, both behind an MLflow server that
proxies artifacts so consumers carry no storage credentials.

Why it exists in this system: robo-be's five workers name models as hardcoded
Hub repo ids and load HEAD; nobody could say what was live, and one upstream
rename already produced a latent outage. The registry makes "which model is
production" a query and "ship a better one" an alias move that reaches every
consumer — robo-be's sidecars and the bench — without a code change.

Design: `docs/superpowers/specs/2026-09-14-model-registry-design.md`.
Roadmap: `docs/superpowers/plans/2026-09-14-model-registry-extensions.md`.

## 7. Evaluation — two tiers (PLANNED, partially exists)

| | Tier 1 — model quality | Tier 2 — system performance |
|---|---|---|
| Question | which model? | is the deployed stack healthy? |
| Input | audio file → function call, offline | file → Opus → WS → NATS → worker |
| Adding a candidate | a ~30-line adapter | a Docker sidecar |
| GCP row | Speech v2 called directly | via robo-bridge |
| Status | **does not exist** | exists: `robo-be/benchmarks/{bench_wer,bench_stt,bench_tts}.py` |

Facts that shape this: **no GCP-vs-open-source comparison has ever been run**
(zero Chirp references in 26 bench reports); the existing harnesses only speak
robo-be's WebSocket, so a candidate must be Dockerized before it can be
measured; ground truth (`refs.csv`, `refs-mix.csv`) is on someone's machine,
not in a repo; and **there is no child speech in any evaluation set** — every
WER figure is on adult recordings or synthetic audio, for a product whose users
are five to eight years old.

Tier 1 design (plan Phase 2): a `Backend` protocol; a registry adapter and a
Chirp 3 adapter; **one** Vietnamese normalizer (NFC, case, punctuation,
digits→words) with a version stamped on every run; `jiwer`; a manifest with
`speaker ∈ {child, adult, synthetic}`, `lang`, `noise` slices; a report with
the Chirp 3 row always present and `$/min`; results logged to the same MLflow
as the registry so "best on the child slice" is a query. Success criteria are
to be written down *before* the first run (placeholder: child-slice WER ≤
Chirp 3 + 3 pp, code-switch preservation ≥ Chirp 3, cost below).

Pronunciation assessment has no evaluation set at all today.

## 8. The ML lifecycle — where the loop is complete and where it is broken

```
   ┌────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐   ┌────────────┐
   │  data  │──►│ training │──►│   eval   │──►│ registry │──►│ serving │──►│ monitoring │
   └────────┘   └──────────┘   └──────────┘   └──────────┘   └─────────┘   └──────┬─────┘
        ▲                                                                          │
        └───────────────────────── feedback: field signals → new data ─────────────┘
```

| Stage | Today | Gap |
|---|---|---|
| **Data** | adult recordings (t2xx), synthetic ElevenLabs clips, 15 phrases in `expected.json`; refs not in git | no child speech; no consent framework; no dataset versioning (`dataset_hash` is a tag convention with nothing behind it) |
| **Training** | none — all models pretrained | the registration path is ready (`register()` takes metrics/params); trainer helpers planned (Phase 4) |
| **Evaluation** | Tier 2 only, self-host vs self-host | Tier 1 missing; GCP baseline never run; no pronunciation eval set |
| **Registry** | BUILT; 8 models catalogued; 2 imported in a prototype | robo-be not yet a consumer (Phase 1) |
| **Serving** | Path A live; Path B built | protocol adapter for Path B; loaders take repo ids, not directories |
| **Monitoring** | `resolve_seconds`, `from_cache`, alias audit in the registry; Worker logs | no WER-in-the-field signal, no drift detection, no per-version latency in production |
| **Feedback** | — | audio is deliberately not stored ("*Audio không lưu*"), so today nothing flows back |

The loop is open at both ends: nothing feeds training, and nothing measures the
deployed model on the real population. Closing it is the substance of the next
year of ML work; the registry is the piece in the middle that the other pieces
attach to.

## 9. Cross-cutting decisions (decided)

1. **Config is the deploy mechanism, at every layer.** Bridge URLs in the
   Worker, alias names in the registry, `ROBO_*_REF` pins per robo-be worker
   (planned). Shipping a model or a backend is changing a pointer, never code.
2. **The LLM stays on Gemini** through the bridge; the migration is about
   speech.
3. **GCP compute stays** even as Google's managed speech APIs are replaced;
   leaving the APIs is not leaving the platform. The registry's weights live in
   GCS.
4. **Chirp 3 is the permanent yardstick.** Every Tier 1 report carries the
   Google row and a cost column; self-host quality is never reported alone.
5. **Consumers never see the model backend.** `mlflow-skinny` only; a
   directory in, engine unchanged.
6. **Aliases, not stages**, in the registry; **finals, not interims**, in STT;
   **one normalizer** in evaluation. Small fixed points that keep comparisons
   honest.
7. **Evaluate raw models before deploying them.** Tier 1 picks, Tier 2
   confirms — today the expensive step is being done for every candidate.

## 10. Open decisions (nobody has decided these)

1. **Privacy vs. learning from the population.** "*Audio không lưu*" is a
   design principle and the reason the feedback arrow above is missing. Any
   finetuning on real child speech requires storing it, with consent from
   parents and a retention policy. This decision gates Phase 4 entirely and
   should be made deliberately, not by whoever first writes a data collector.
2. **Child speech for evaluation.** Even 20–30 consented utterances would make
   every number meaningful. Who collects, under what consent, stored where.
3. **How Path B meets the Worker.** Adapter in front of robo-be speaking the
   bridge dialect (Worker untouched), or a second `bridge/` client in the Worker
   speaking robo-be's dialect. The adapter keeps the "two URLs" property.
4. **Which ASR wins for VI, EN, and code-switching** — unknowable until Tier 1
   runs with the Chirp 3 row present.
5. **Ownership of `promote`.** Anyone with the tracking URI can move
   `@production` today. CI-only promotion with `REGISTRY_ACTOR` set is the
   intended discipline; nothing enforces it yet.
6. **Where the Tier 1 bench lives** — a separate repo (recommended: it depends
   on neither robo-be's GPU stack nor the Worker) versus inside robo-be.
7. **Pronunciation assessment as a model in the registry.** proass is a
   service with its own model lifecycle outside this picture. Whether it joins
   the registry and the eval loop is undecided.

## 11. Reading map

| Want to understand… | Read |
|---|---|
| the brain, session flow, memory | `robo-worker/CLAUDE.md`, `src/dos/tutor.ts`, `docs/mempalace-reference.md`, `docs/device-auth-design.md` |
| the Google shim and its protocols | `robo-bridge/README.md`, `STT_BRIDGE_SPEC.md`, `TTS_BRIDGE_SPEC.md` |
| self-hosted speech internals | `robo-be/docs/STT-system-architecture.md`, `docs/websocket-api.md`, `docker-compose.yml` |
| model management | `model-registry/README.md`, `docs/superpowers/specs/2026-09-14-model-registry-design.md` |
| what happens next | `model-registry/docs/superpowers/plans/2026-09-14-model-registry-extensions.md` |
| the original April plan (historical) | `robo-worker/docs/system-architecture.md` — superseded where this document says so |
