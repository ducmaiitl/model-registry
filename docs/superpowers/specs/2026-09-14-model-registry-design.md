---
title: Model Registry — design
date: 2026-09-14
status: implemented (v0.1.0)
authors: ducmaiitl
---

# Model Registry — design

## Summary

One versioned source of truth for the STT/TTS models used across robo-be, the
benchmark harness, and future consumers. A consumer calls
`ModelRegistry().resolve("cohere-asr", "production")` and receives a local
directory containing the weights; it never learns that MLflow, Postgres, or GCS
exist, and never learns which concrete version it got. Promotion and rollback are
alias moves inside the registry, so shipping a model stops being a code change in
every consumer.

MLflow is the engine but is hidden entirely behind one class. Postgres holds
metadata (models, versions, aliases, params, metrics, tags); a GCS bucket holds
the weight files; the MLflow server proxies every artifact transfer, so consumers
carry no storage credentials and depend only on `mlflow-skinny`. Every model
robo-be runs is declared in `models.yaml`, pinned to an exact HuggingFace commit,
and imported idempotently.

## Goals

1. **Decouple the model lifecycle from consumers.** A consumer's code names an
   alias, not a version, not a repo id, not a file path. Nothing in a consumer
   changes when the model behind `@production` changes.
2. **Immutable versions, movable aliases.** Every registered version is frozen.
   Rollback is re-pointing an alias, and takes seconds.
3. **Provenance.** Every version is traceable to the run that produced it and,
   for imported models, to the exact upstream repository and commit SHA.
4. **No HuggingFace Hub at runtime.** Today robo-be downloads whatever is at HEAD
   on first start. After import, the registry owns the files; the Hub is a
   one-time source, not a dependency.
5. **Cheap consumer footprint.** `pip install` pulls `mlflow-skinny` only — no
   GCS/S3 libraries, no database drivers, no sqlalchemy.
6. **Observable by construction.** Resolve latency, cache hit/miss, and every
   alias move are recorded without the consumer doing anything.
7. **Finetune-ready without redesign.** The registration path already accepts
   metrics, params, and tags; a finetuning job uses the identical call.

## Non-goals

- **Inference.** The registry places files on disk. Loading and running them is
  the consumer's engine (transformers, sherpa-onnx, funasr, speechbrain, voxcpm).
- **Serving protocol.** WebSocket/NATS contracts between robo-be and its callers
  are untouched.
- **Training orchestration.** How a finetune job is scheduled or run is out of
  scope; only how its output is registered is in scope (see the extensions plan).
- **Dataset management.** `dataset_hash` is a tag convention today. Real dataset
  versioning is a planned extension, not part of v0.1.0.
- **Auth, RBAC, multi-tenancy, HA.** MLflow ships with none; hardening around it
  is outside this document.
- **Deployment topology.** Where the server runs is not described here. This
  document covers configuration (what the server needs), not hosting.

## Motivation

`robo-be` runs eight models across five GPU workers. Each worker names its model
as a hardcoded HF repo id and lets `from_pretrained()` (or `hf_hub_download`, or
funasr's `AutoModel`) fetch whatever is at HEAD into a shared `model_cache`
volume. The LID worker instead bakes ~2.5 GB of weights into its image at build
time and patches a SpeechBrain yaml in the Dockerfile.

Consequences observed:

1. **An upstream rename already broke production silently.** On 2026-08-21 the
   `g-group-ai-lab/gipformer-65M-rnnt` repo removed the `*-epoch-35-avg-6.onnx`
   filenames. `asr_gipformer.py` still requests them. The worker runs only where a
   pre-August cache survives; any fresh host cannot start the VI ASR sidecar.
2. **No rollback.** Reverting a model means editing a constant and rebuilding an
   image.
3. **Nobody can answer "what is live".** The truth is the union of five
   constants, a Docker build log, and whatever the cache happened to hold.
4. **The Hub is a runtime dependency**, including a gated repo that needs an
   `HF_TOKEN` inside two containers.
5. **Benchmarking a candidate requires Dockerizing it as a sidecar first**, so
   model selection is slow and only the deployed model ever gets measured.

## Design principles

- **Facade.** `ModelRegistry` is the only import a consumer makes. `import mlflow`
  appears in exactly one file (`src/registry/client.py`) and one place in the
  test fixture. Swapping the backend later cannot break a consumer.
- **Aliases, never stages.** MLflow deprecated stages in 2.9; the alias API is
  used everywhere (`set_registered_model_alias`, `get_model_version_by_alias`,
  `models:/<name>@<alias>`).
- **Metadata and files are separate**, and the file store is a configuration
  seam. The repo began on MinIO and moved to GCS by changing one server flag and
  one pip package; `client.py` and every consumer were untouched.
- **Version is always `str`.** MLflow returns `int` on several paths and
  `"v" + version` crashes at runtime. The cast happens once, inside the facade.
- **Fail loudly on anything that would ship a broken version.** A patch whose
  target is missing, an include pattern that selects zero files, a duplicated
  YAML key, a half-downloaded cache slot — all are errors, never silent fallbacks.
- **Idempotent import.** Re-running the catalog never duplicates a version and
  never moves an alias someone set by hand.
- **Only the facade knows about the storage.** Consumers receive a `Path`.

## 1. Topology

```
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │ robo-be      │  │ stt-tts-bench│  │ future       │   consumers
        │ GPU workers  │  │ (Tier 1)     │  │ consumers    │   import ModelRegistry only
        └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   dependency: mlflow-skinny
               │                 │                 │
               └─────────────────┼─────────────────┘
                                 │  resolve(name, ref) → Path
                        ┌────────▼─────────┐
                        │  ModelRegistry   │   src/registry/client.py — the facade
                        └────────┬─────────┘
                                 │  MLflow REST  (MLFLOW_TRACKING_URI)
                        ┌────────▼─────────┐
                        │  MLflow server   │   mlflow==2.16.2, --serve-artifacts
                        │  :5000           │   proxies every artifact byte
                        └────┬────────┬────┘
                             │        │
                ┌────────────▼──┐  ┌──▼──────────────┐
                │ PostgreSQL 16 │  │ GCS bucket      │
                │ models, vers, │  │ weight files    │
                │ aliases, tags │  │ gs://<bucket>/  │
                │ params,metrics│  │                 │
                └───────────────┘  └─────────────────┘
```

### Responsibilities

| Component | Owns | Never does |
|---|---|---|
| Consumer | naming an alias; loading files with its own engine | import mlflow; hold storage credentials; know a version number |
| `ModelRegistry` | alias→version→files; cache; audit tags; `str` casting | inference; storage access other than through the server |
| MLflow server | registry entities; artifact proxying | authentication (it has none) |
| Postgres | all metadata | any file content |
| GCS | all file content, under `gs://<bucket>/<experiment>/<run_id>/artifacts/model/` | any metadata |
| `models.yaml` | what exists upstream, at which commit, under which name and alias | anything about where the server runs |

### Invariants

1. A consumer process has `mlflow-skinny` and `MLFLOW_TRACKING_URI`, nothing
   else. Verified: a clean venv `pip install .` contains no sqlalchemy and no
   GCS library and resolves successfully.
2. `ResolvedModel.version` is `str` on every return path.
3. An alias download URI is `models:/<name>@<alias>`; the `/` form is parsed as
   a version number and fails.
4. `register()` creates the registered model **after** artifacts are uploaded. A
   failed or killed upload leaves no model entity (it may leave an orphan run;
   see §16).
5. A cache slot is trusted only if `<slot>/.registry-complete` exists. Downloads
   go to `<slot-parent>/<version>.partial-<pid>` and are renamed into place.
6. A catalog re-run sets an alias only on a fresh registration. On a skip it
   prints and does nothing, so a manual rollback survives re-runs.

## 2. Tech stack

| Component | Version | Role | Why this one |
|---|---|---|---|
| MLflow (server) | 2.16.2 | tracking server + model registry | mature registry with aliases; artifact proxying; the finetune tracking comes for free later |
| mlflow-skinny (client) | ≥2.16, <3 | what consumers install | client + artifact download without server/DB/storage extras |
| PostgreSQL | 16 | backend store | real database for metadata; sqlite is test-only |
| Google Cloud Storage | — | artifact store | user decision 2026-09-14; durability and metrics without a daemon to run |
| google-cloud-storage | 2.18.2 | server ↔ GCS | MLflow's `gs://` artifact repository dependency |
| psycopg2 | 2.9.9 | server ↔ Postgres | compiled in the server image (needs `libpq-dev`); never on a consumer |
| Docker Compose | v2 | two services: `postgres`, `mlflow` | one-command local/CI validation; healthchecks on both |
| Python | 3.11 | library and scripts | matches the server image |
| pytest | ≥7 | 27 tests against sqlite + local files | no services needed to run the suite |
| huggingface_hub | ≥0.24 | `scripts/register_from_hf.py` | pinned snapshots by commit SHA; only on the machine doing imports |
| PyYAML | (via mlflow) | `models.yaml` | loaded through a strict duplicate-key-rejecting loader |
| setuptools | ≥68 | packaging (`pyproject.toml`, src layout) | distribution `robo-model-registry`, import name `registry` |
| GitHub Actions | — | CI on push/PR | tests, strict catalog parse, compose validation |

## 3. Module layout

```
model-registry/
├── pyproject.toml                 distribution robo-model-registry; deps: mlflow-skinny; extras: dev, import
├── src/registry/
│   ├── __init__.py                exports ModelRegistry, ResolvedModel, __version__
│   └── client.py                  THE FACADE — the only file consumers depend on
├── scripts/
│   ├── register_pretrained.py     register any local directory as a version
│   ├── register_from_hf.py        import HF snapshots pinned to a commit; catalog mode
│   └── registry_cli.py            list / versions / promote / resolve
├── models.yaml                    catalog: 8 models robo-be runs, pinned SHAs, aliases, patches
├── examples/
│   ├── serving_fastapi.py         a consumer; deliberately imports no mlflow
│   └── fake_model/                two-file fixture used by make register-example
├── tests/
│   ├── test_registry.py           15 tests: contract, cache, audit, failure paths
│   └── test_register_from_hf.py   12 tests: file selection, patches, catalog, strict YAML
├── docker/mlflow.Dockerfile       python:3.11-slim + mlflow + psycopg2 + google-cloud-storage
├── docker-compose.yml             postgres + mlflow (artifacts to gs://$GCS_BUCKET)
├── Makefile                       up/down/logs/ps/test/register-*/catalog-*/clean; loads .env
├── requirements.txt               server image deps
├── requirements-dev.txt           `-e .[dev]`
├── .env.example                   every variable, documented
├── .github/workflows/ci.yml       tests + catalog parse + compose config
└── docs/
    ├── OBSERVABILITY.md           lineage, resolve timing, GCS/Postgres metrics, alerts, audit, tags
    └── superpowers/               this spec; the extensions plan
```

## 4. The API — `src/registry/client.py`

```python
@dataclass
class ResolvedModel:
    name: str
    version: str                 # always str
    alias: str                   # alias used, or "" when resolved by number
    local_path: Path             # directory of weights; the files sit directly in it
    run_id: Optional[str]
    metrics: dict[str, float]    # from the producing run (empty for pretrained)
    tags: dict[str, str]         # version tags, then run tags (version wins)
    resolve_seconds: float       # metadata + download (or cache hit) wall time
    from_cache: bool             # True when no download happened

class ModelRegistry:
    def __init__(self, tracking_uri: str = None)        # env MLFLOW_TRACKING_URI, default http://localhost:5000

    # write path
    def register(self, name, source_dir, flavor="artifact",
                 metrics=None, params=None, tags=None, description="") -> str
    def promote(self, name, version, alias="production", actor=None) -> dict

    # read path
    def resolve(self, name, ref="production", cache_dir=None) -> ResolvedModel
    def alias_info(self, name, alias="production") -> dict | None
    def list_versions(self, name) -> list[dict]        # version, aliases, run_id, status, created; oldest first
    def list_models(self) -> list[str]
    def find_version_by_tag(self, name, key, value) -> Optional[str]   # importer idempotency
```

Semantics worth stating:

- `register` is the single producer entrypoint for pretrained *and* finetuned
  models. Pretrained passes no metrics; a finetune passes real ones. Same call.
- `promote` lower-cases the alias, records the move (§6.3), returns the record.
- `resolve` accepts an alias or a numeric string. `"3"` pins; anything else is an
  alias. Alias and pin share one cache slot because they name the same version.
- `list_versions` sources aliases from `get_registered_model(...).aliases`
  because `search_model_versions` returns `aliases=[]` even for aliased versions.
- `find_version_by_tag` falls back to `get_model_version` per version because
  search may return empty tags on some backends.

## 5. Storage model

MLflow entities and how they are used:

| Entity | Used as | Notes |
|---|---|---|
| Registered model | one per model *name* (`cohere-asr`) | carries alias→version map and the `alias.<a>.*` audit tags |
| Model version | immutable snapshot, integer version cast to `str` | `source = runs:/<run_id>/model`; carries provenance + `promoted.<a>.*` tags |
| Run | the act of registering (or later, training) | params, metrics, run tags `registry.flavor`, `registry.model_name`; artifacts under `model/` |
| Alias | movable pointer | `production`, `fallback`, `candidate` in the catalog; freeform |

Artifact addressing: a version's `source` is `runs:/<id>/model`. With
`--serve-artifacts` that resolves to `mlflow-artifacts:/…`, which the server maps
to `gs://<bucket>/<experiment_id>/<run_id>/artifacts/model/`. Consumers never see
a `gs://` URI.

### Tag conventions

| Scope | Tag | Set by | Meaning |
|---|---|---|---|
| version | `source` | importer | `huggingface` |
| version | `hf_repo`, `hf_revision` | importer | upstream repo and **full** commit SHA |
| version | `base_model` | importer / finetune job | pretrained origin (repo id today; `name@version` for finetunes) |
| version | `framework`, `role`, `lang`, `license`, `consumer` | catalog | which engine loads it, what it does, who uses it |
| version | `registered_by`, `project` | catalog defaults | audit |
| version | `promoted.<alias>.at/by` | `promote()` | when/who last pointed that alias at this version |
| model | `alias.<alias>.version/previous_version/promoted_at/promoted_by` | `promote()` | latest move per alias; read via `alias_info` |

## 6. Data flows

### 6.1 Register a directory

```
caller ─ register(name, dir, metrics?, params?, tags?, description)
  │ FileNotFoundError if dir missing
  ├─ start_run(run_name="register-<name>", tags={registry.flavor, registry.model_name})
  │     log_params / log_metrics (if given)
  │     log_artifacts(dir, artifact_path="model")     ── proxied by server ──► GCS
  ├─ create_registered_model(name)   (idempotent; AFTER upload — invariant 4)
  ├─ create_model_version(name, source="runs:/<run>/model", run_id)
  ├─ set_model_version_tag × N, update_model_version(description)
  └─ return str(version)
```

### 6.2 Resolve

```
caller ─ resolve(name, ref, cache_dir?)
  ├─ ref.isdigit() ? get_model_version(name, ref) , uri="models:/name/ref", alias=""
  │                : get_model_version_by_alias(name, ref.lower()), uri="models:/name@alias"
  ├─ slot = <cache>/<name>/<version>          cache = cache_dir | $MODEL_REGISTRY_CACHE | ~/.cache/model-registry
  ├─ slot/.registry-complete exists?  yes → from_cache=True, no network for files
  │                                   no  → rmtree(slot if present)
  │                                         partial = <cache>/<name>/<version>.partial-<pid>
  │                                         download_artifacts(uri, dst=partial) ── server ──► GCS
  │                                         write partial/.registry-complete {name, version, alias, run_id, resolved_at}
  │                                         (another process finished first? use theirs)
  │                                         os.replace(partial, slot)
  ├─ metrics/tags from get_run(run_id)  (best effort; a deleted run is not fatal)
  └─ ResolvedModel(..., resolve_seconds, from_cache)
```

### 6.3 Promote / rollback

```
promote(name, version, alias, actor?)
  ├─ previous = get_model_version_by_alias(name, alias).version   (None if alias new)
  ├─ set_registered_model_alias(name, alias, version)              ← the deploy
  ├─ registered-model tags: alias.<alias>.{version, previous_version, promoted_at, promoted_by}
  ├─ version tags:          promoted.<alias>.{at, by}
  └─ return {alias, version, previous_version, promoted_at, promoted_by}
```

Rollback is the same call with the previous version. `promoted_by` is `actor`,
else `$REGISTRY_ACTOR`, else the OS user.

### 6.4 Import from HuggingFace (`register_from_hf.py`)

```
for each catalog entry (or --repo):
  ├─ sha   = model_info(repo, revision).sha           full SHA even if catalog gives a short one
  ├─ files = list_repo_files(repo, sha) filtered by include/exclude (fnmatch, same as snapshot_download)
  │           zero files → RuntimeError
  ├─ find_version_by_tag(name, "hf_revision", sha) → exists? print + skip (aliases untouched)
  ├─ --dry-run? list files, stop
  ├─ snapshot_download(repo, sha, allow/ignore patterns, local_dir=cache/hf-staging/<name>)
  ├─ strip .cache/ (hub bookkeeping)
  ├─ apply_patches: each {file, replace{old,new}}; old missing → RuntimeError
  ├─ register(name, staging, tags={source, hf_repo, hf_revision, base_model, **catalog tags}, description)
  ├─ alias? promote(name, version, alias)    ← only on fresh registration
  └─ rm staging (unless --keep)
failures are collected; the catalog continues; exit 1 at the end if any failed
```

## 7. The catalog — `models.yaml`

Schema:

```yaml
defaults:
  exclude: [glob, ...]           # applied to every entry (TF/Flax/Rust weights, PNGs)
  tags: {key: value, ...}        # merged under each entry's tags
models:
  - name: <registry name>        # required
    repo: <hf repo id>           # required
    revision: <sha|branch|tag>   # default HEAD; resolved to a full SHA at import
    alias: <alias>               # set only on fresh registration
    include: [glob, ...]         # whitelist; omitted = everything
    exclude: [glob, ...]         # appended to defaults
    patches:                     # text substitutions applied before registering
      - file: <path in snapshot>
        replace: {old: "...", new: "..."}
    tags: {key: value, ...}
    description: "..."
```

Loaded by `_strict_yaml_load`, which raises `DuplicateKeyError` on a repeated
mapping key (PyYAML's default silently keeps the last value).

Current entries:

| name | repo | alias | role | consumer | note |
|---|---|---|---|---|---|
| `cohere-asr` | CohereLabs/cohere-transcribe-03-2026 | production | asr vi+en | robo-infer-cohere | gated; licence accepted per account |
| `gipformer-asr-vi` | g-group-ai-lab/gipformer-65M-rnnt | production | asr vi | robo-infer-gipformer | **pinned to `ba00dad1`, behind HEAD on purpose** — last revision whose filenames match the loader |
| `qwen3-asr` | Qwen/Qwen3-ASR-1.7B | fallback | asr multi | robo-infer-qwen | |
| `funasr-nano` | FunAudioLLM/Fun-ASR-MLT-Nano-2512 | candidate | asr multi | (opt-in provider) | |
| `mms-lid-256` | facebook/mms-lid-256 | production | lid | robo-infer-stt-lid | `include` keeps safetensors only (repo ships .bin twice the size) |
| `voxlingua107-lid` | TalTechNLP/voxlingua107-xls-r-300m-wav2vec | production | lid | robo-infer-stt-lid | carries the SpeechBrain 1.0 yaml patch formerly in `Dockerfile.lid` |
| `xlsr53-phoneme` | facebook/wav2vec2-xlsr-53-espeak-cv-ft | production | phoneme-lid | robo-infer-stt-lid | |
| `voxcpm2-tts` | openbmb/VoxCPM2 | production | tts vi+en | robo-infer-tts-voxcpm | |

Deliberately absent: Demucs DNS64 (fetched by the `denoiser` package from a
non-Hub URL — register with `register_pretrained.py`), ElevenLabs (hosted API).

## 8. Resolve cache

- Layout `<cache>/<name>/<version>/`; alias and pin share the slot.
- Default root `~/.cache/model-registry`; override with `MODEL_REGISTRY_CACHE`
  or the `cache_dir` argument. In containers, mount it as a persistent volume or
  every restart re-downloads.
- Completion marker `.registry-complete` (JSON: name, version, alias, run_id,
  resolved_at) is written after the last file lands and before the atomic rename.
  A slot without it is discarded and re-fetched. A truncated checkpoint that
  still "loads" is the failure this exists to prevent.
- Per-process `.partial-<pid>` staging avoids two processes writing one
  directory; if another process completes the same version first, its slot wins
  and ours is removed.
- No eviction. Old versions accumulate until removed by hand (see the plan).
- `from_cache` on the result distinguishes a 15 ms hit from a 25 s miss when
  reading `resolve_seconds`.

## 9. Configuration

Environment variables (all in `.env.example`):

| Variable | Read by | Purpose |
|---|---|---|
| `PG_USER` / `PG_PASSWORD` / `PG_DB` | compose | Postgres credentials; Postgres has no host port mapping |
| `GOOGLE_CLOUD_PROJECT` | mlflow container | GCP project |
| `GCS_BUCKET` | compose → `--artifacts-destination gs://…` | **required**; compose fails fast with a message if unset |
| `GOOGLE_APPLICATION_CREDENTIALS` | mlflow container | empty when ADC is available; else a mounted key file |
| `MLFLOW_PORT` | compose | host port, default 5000. MLflow has no authentication; the port must not be public |
| `MLFLOW_TRACKING_URI` | `ModelRegistry`, scripts, Makefile | the only thing a consumer needs |
| `HF_TOKEN` | `register_from_hf.py` only | rate limits and gated repos at import time |
| `MODEL_REGISTRY_CACHE` | `resolve()` | cache root override |
| `REGISTRY_ACTOR` | `promote()` | who moved the alias (set it in CI) |

Compose: `postgres` (pg_isready healthcheck) and `mlflow` (`/health` healthcheck,
20 s start period, depends on postgres healthy). The Makefile `include`s `.env`
so CLI targets follow the same port the server uses.

## 10. Error handling

| Condition | Behaviour |
|---|---|
| `source_dir` does not exist | `FileNotFoundError` before any network call |
| artifact upload fails / process killed | no registered model, no version; an orphan run may remain (§16) |
| alias does not exist | `MlflowException` from `get_model_version_by_alias` propagates |
| run deleted after registration | `resolve` still returns files; metrics/tags empty |
| patch target string missing | `RuntimeError`; nothing registered |
| include/exclude select zero files | `RuntimeError` before download |
| duplicate key in `models.yaml` | `DuplicateKeyError` at load |
| gated repo without token / licence | `GatedRepoError`; that entry fails, catalog continues, exit 1 at end |
| download interrupted | slot has no marker; next resolve discards and re-fetches |
| `GCS_BUCKET` unset | compose refuses to start with a clear message |

## 11. Testing strategy

- **Unit (27 tests, no services).** Fixture creates `sqlite:///<tmp>/mlflow.db`
  with a local artifact dir, chdirs to a temp dir, and points
  `MODEL_REGISTRY_CACHE` at a temp dir. Anything that stubs a module-level
  function goes through `monkeypatch` — `reg._mlflow` *is* the global `mlflow`
  module, and a bare assignment leaked into later tests once.
- **What the client tests pin:** `register` returns `"1"` as `str`; alias repoint
  changes what an unchanged consumer resolves; rollback; numeric pin gives
  `alias == ""`; files land at `local_path`; `list_versions` reports aliases;
  `find_version_by_tag`; failed upload leaves no model; cache hit skips
  download; alias and pin share a slot; incomplete slot is refetched; default
  cache from env; promote records previous/actor; `alias_info` is `None` when
  never promoted.
- **What the importer tests pin:** include/exclude semantics match
  `snapshot_download`; patches rewrite and refuse when the target is missing;
  catalog defaults merge; empty catalog; duplicate keys rejected; `.cache/`
  stripped; `key=value` parsing.
- **CI (GitHub Actions):** `pip install -e .[dev]`, `pytest -q`, strict parse of
  `models.yaml`, `docker compose config -q`.
- **Not unit-tested, verified by hand:** network import (`--dry-run` against the
  Hub; two real imports incl. the VoxLingua patch, resolved back and inspected),
  proxied artifact round-trip through a live server, skinny-only consumer venv.

## 12. Observability

Built in (details and alert table in `docs/OBSERVABILITY.md`):

- `resolve_seconds` and `from_cache` on every resolve.
- Full lineage: alias → version → run → params/metrics/tags → artifacts.
- Alias audit: `alias_info()` answers who moved `@production`, from what, when.
- Provenance tags make "which upstream commit is v3" a lookup.

Not built in: metric emission (Prometheus/StatsD), alias-change notifications,
server request metrics. These are additive; see the extensions plan.

## 13. Consumer integration pattern

The registry replaces the *string* that names a model and the behaviour behind
it (HEAD + cache + token). It changes nothing about inference.

```python
# before                                       # after
_HF_REPO = "CohereLabs/cohere-transcribe-…"    files = ModelRegistry().resolve("cohere-asr", "production").local_path
AutoProcessor.from_pretrained(_HF_REPO)        AutoProcessor.from_pretrained(files)
Model.from_pretrained(_HF_REPO, ...)           Model.from_pretrained(files, ...)
# inference code identical
```

| Loader shape | Adaptation |
|---|---|
| transformers `from_pretrained(repo_id)` | pass `local_path` — it has always accepted a directory |
| sherpa-onnx via `hf_hub_download(repo, file)` ×4 | `local_path / "<file>"` — simpler than before |
| funasr `AutoModel(model=repo_id)` | `AutoModel(model=str(local_path))` |
| speechbrain `foreign_class(source=dir)` | already a directory; the bake step and yaml patch disappear |
| voxcpm `from_pretrained(repo_id)` | pass `local_path` |

Containers must mount `MODEL_REGISTRY_CACHE` as a persistent volume. `HF_TOKEN`
leaves every runtime container.

## 14. Extension points

Each is additive and small; sequencing and tasks are in
`docs/superpowers/plans/2026-09-14-model-registry-extensions.md`.

- **Finetuning.** `register()` is already the path; add `best_version(name,
  metric, lower_is_better)`, `promote_if_better(...)`, a training-run context so
  trainers stay behind the facade, dataset versioning (`mlflow.log_input` + a
  `datasets/` prefix), and `base_model = <name>@<version>` for finetunes.
- **Bench as consumer.** Tier 1 harness resolves candidates by alias or pin,
  scores them, logs runs to the same MLflow — "best on child speech" becomes a
  query.
- **Observability emission.** Optional metrics hook in the facade; alias-change
  notification; `REGISTRY_ACTOR` set by CI.
- **Alias strategies.** `canary`, `champion`/`challenger` are just more aliases;
  no code.
- **Hygiene.** `registry_cli gc` for orphan runs (`mlflow gc`), `cache prune` for
  old slots.

## 15. Decisions log

| # | Decision | Why | Consequence |
|---|---|---|---|
| 1 | Aliases, not stages | stages deprecated in MLflow 2.9 and inconsistent | `models:/name@alias` URIs; freeform alias names |
| 2 | Metadata in Postgres, files in object storage | weights are GB-scale; never in a DB or Git | two stores, one server |
| 3 | GCS as the artifact store (2026-09-14, user) | the registry runs on GCP; durability and metrics without a daemon | MinIO removed; swap was one flag + one package, zero client lines |
| 4 | Facade; consumers never import mlflow | backend can change without touching consumers | proven by decision 3 |
| 5 | `resolve(name, ref) → Path` as the single consumer call | the consumer only needs a directory | engines unchanged |
| 6 | Consumers depend on `mlflow-skinny` | server proxies artifacts | no storage libraries on consumers |
| 7 | `version` is always `str` | MLflow returns `int` on some paths; `"v"+version` crashed | cast inside the facade |
| 8 | Registered model created after upload | interrupted uploads left empty model shells (observed) | failed register leaves no entity |
| 9 | Marker-based cache with `.partial-<pid>` staging | a truncated checkpoint that loads is the worst failure | no eviction yet |
| 10 | gipformer pinned to `ba00dad1`, behind HEAD | upstream deleted the filenames the loader requests | bump together with the loader |
| 11 | Catalog re-run never moves aliases on skip | a re-run must not undo a manual rollback | aliases set only on fresh registration |
| 12 | Strict YAML loader | a duplicated key shipped once and PyYAML hid it | `DuplicateKeyError` |
| 13 | Deployment topology out of this document | separate concern | §9 covers configuration only |

## 16. Open questions

1. **Orphan runs.** A killed upload leaves a `RUNNING` run with partial
   artifacts in GCS. Policy and tooling for `mlflow gc` are not defined.
2. **Cache eviction.** Slots accumulate forever. A `prune` keeping the versions
   currently aliased plus N recent is the obvious shape.
3. **Concurrent resolves across hosts** sharing one cache volume are handled by
   the marker + per-pid staging, but not exhaustively tested.
4. **Who may `promote`.** Today anyone with the tracking URI. CI-only promotion
   with `REGISTRY_ACTOR` set is the intended discipline; nothing enforces it.
5. **Catalog as the source of truth vs the registry.** If someone registers by
   hand, the catalog no longer describes the registry. A `catalog verify`
   command that diffs the two would close this.
6. **Description/tag length limits** in MLflow for long model cards.

## 17. Glossary

- **Alias** — a movable name (`production`) pointing at exactly one version.
- **Version** — an immutable registered snapshot; a string like `"3"`.
- **Run** — the MLflow record of the act that produced a version; owns params,
  metrics, tags, artifacts.
- **Artifact proxying** — the server transfers files on the client's behalf, so
  the client needs no storage credentials.
- **Catalog** — `models.yaml`; the declared set of upstream models, pinned.
- **Slot** — `<cache>/<name>/<version>/`, trusted only with its completion marker.
- **Consumer** — anything that calls `resolve()`; a **producer** calls `register()`.
