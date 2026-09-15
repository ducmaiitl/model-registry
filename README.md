# Model Registry

One versioned source of truth for the STT/TTS models, shared by every consumer.

> **New to this codebase?** Start with the [code walkthrough book](docs/book/) —
> 17 chapters explaining every file, written for beginners.
>
> System picture (all repos): [`ARCHITECTURE.md`](ARCHITECTURE.md) ·
> Design spec: [`docs/superpowers/specs/2026-09-14-model-registry-design.md`](docs/superpowers/specs/2026-09-14-model-registry-design.md) ·
> Roadmap: [`docs/superpowers/plans/2026-09-14-model-registry-extensions.md`](docs/superpowers/plans/2026-09-14-model-registry-extensions.md)

## Why this exists

Without a registry, every consumer keeps its own copy of the model files. Changing
a model means editing each place, nobody can say which version is actually running
where, and there is no rollback. This repo replaces that with one registry that
holds versioned models, and a stable API every consumer pulls from.

The core idea is **decoupling**. A consumer asks:

> give me `whisper-stt` at `@production`

and gets back a path to weights already on disk. It does not know that MLflow,
GCS, or Postgres exist, and it does not know which concrete version it just got.
That means shipping a new model is a registry operation, not a code change in
every consumer.

## The one call consumers make

```python
from registry import ModelRegistry

reg = ModelRegistry()
model = reg.resolve("whisper-stt", "production")

model.local_path      # weights on disk, ready to load
model.version         # "3"  (always a str)
model.resolve_seconds # how long that took
```

That is the entire consumer-facing surface. Everything else in this repo exists to
make that call work.

## Install

```bash
pip install "git+https://github.com/ducmaiitl/model-registry@main"
```

Consumers get `mlflow-skinny` only — no GCS/S3 libraries, no database drivers —
because the server proxies artifacts. Set `MLFLOW_TRACKING_URI` and call
`resolve()`.

### Resolve cache

`resolve()` keeps each version at `<cache>/<name>/<version>/` and only downloads
when that slot is missing. The default cache is `~/.cache/model-registry`
(override with `MODEL_REGISTRY_CACHE` or the `cache_dir` argument). Mount it as a
persistent volume in containers, or every restart re-downloads the weights.

A slot counts as present only if it carries a `.registry-complete` marker, which
is written after the last file lands. A download that dies half-way leaves no
marker and is thrown away on the next resolve — a truncated checkpoint that
still "loads" is the failure this guards against. `ResolvedModel.from_cache`
tells you which path a call took, so `resolve_seconds` can be read correctly.

## Architecture

```
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │ FastAPI      │  │ bench repo   │  │ future       │   consumers
        │ gateway      │  │ stt-tts-bench│  │ consumers    │   (import ModelRegistry only)
        └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
               │                 │                 │
               └─────────────────┼─────────────────┘
                                 │  resolve(name, ref)
                        ┌────────▼─────────┐
                        │  ModelRegistry   │   facade — hides MLflow entirely
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  MLflow  :5000   │   tracking server + model registry
                        └────┬────────┬────┘
                             │        │
                ┌────────────▼──┐  ┌──▼──────────────┐
                │ PostgreSQL 16 │  │ GCS bucket      │
                │ metadata:     │  │ artifacts:      │
                │ models, vers, │  │ the actual      │
                │ aliases, tags │  │ weight files    │
                └───────────────┘  └─────────────────┘
```

Metadata and files are split deliberately: weight files run hundreds of MB to GB,
which belongs in object storage, not in a database or Git.

## Tech stack

| Component | Version | Role |
|---|---|---|
| MLflow | 2.16.2 | tracking server + model registry |
| PostgreSQL | 16 | backend store (metadata) |
| Google Cloud Storage | — | artifact store (weight files) |
| google-cloud-storage | 2.18.2 | MLflow ↔ GCS |
| psycopg2 | 2.9.9 | MLflow ↔ Postgres |
| Docker Compose | — | brings the cluster up |
| pytest | ≥7 | tests the client against sqlite, no docker |

Python 3.11.

Artifacts live in a **GCS bucket**; the registry itself runs on a GCP VM. The
artifact store is deliberately a configuration seam, not a code dependency: this
repo started on self-hosted MinIO and moved to GCS by changing one compose flag
(`--artifacts-destination`) and one pip package — `ModelRegistry` and every
consumer were untouched, because the server proxies artifacts. Moving again
(S3, or back to MinIO for a fully self-hosted deployment) is the same one-flag change.

## Quick start

```bash
cp .env.example .env
make up          # postgres + mlflow, one command
```

- MLflow UI → http://localhost:5000
- Artifacts land in `gs://$GCS_BUCKET` — create it once before the first `make up`:

  ```bash
  gcloud storage buckets create gs://$GCS_BUCKET --location=asia-southeast1 --uniform-bucket-level-access
  ```

  On a GCE VM the attached service account needs `roles/storage.objectAdmin` on the
  bucket and nothing else is required. Off-GCP, set `GOOGLE_APPLICATION_CREDENTIALS`
  in `.env` to a key file and uncomment the matching `volumes:` line in
  `docker-compose.yml`.

If port 5000 is already taken, `make up` fails with `port is already allocated`.
Override the host port in `.env` — the container-internal port never changes:

```bash
MLFLOW_PORT=5010
MLFLOW_TRACKING_URI=http://localhost:5010    # keep the client in sync
```

**MLflow has no authentication.** Never give this port a public IP; firewall it
to known consumers or put an authenticating reverse proxy in front.

The `Makefile` loads `.env`, so the CLI targets follow whatever you set here.

Register the bundled example model and resolve it back:

```bash
make register-example   # registers examples/fake_model, promotes it to @production
make resolve-example    # resolves @production, downloads weights to ./cache
```

Then run a real consumer against it:

```bash
python examples/serving_fastapi.py
```

## Aliases, not stages

MLflow deprecated stages (`Production`/`Staging`) in 2.9 and will remove them, and
they behave inconsistently in the meantime. This repo uses **model version
aliases** throughout — freeform names like `production`, `staging`, `canary`,
`champion`.

```bash
# Deploy v3: point the alias at it.
python scripts/registry_cli.py promote --name whisper-stt --version 3 --alias production

# v3 is bad: point it back. That is the entire rollback.
python scripts/registry_cli.py promote --name whisper-stt --version 2 --alias production
```

Versions are immutable; aliases move. No consumer restarts into a new model
because its code changed — it restarts into whatever the alias points at now.

Every move is recorded, because an alias move *is* a deploy: `promote()` returns
`{alias, version, previous_version, promoted_at, promoted_by}` and stores it on
the model, readable later with `reg.alias_info("whisper-stt", "production")`.
`promoted_by` is `$REGISTRY_ACTOR` if set (CI should set it), else the OS user.

Pinning still works when a consumer needs a specific version (benchmarks, mostly):

```python
reg.resolve("whisper-stt", "3")   # exact version; alias comes back as ""
```

## CLI

Register a pretrained model:

```bash
python scripts/register_pretrained.py \
    --name whisper-stt \
    --source-dir ./weights/whisper-small \
    --alias production \
    --desc "whisper-small, pretrained" \
    --tag base_model=openai/whisper-small \
    --tag framework=faster-whisper
```

Manage it:

```bash
python scripts/registry_cli.py list
python scripts/registry_cli.py versions --name whisper-stt
python scripts/registry_cli.py promote --name whisper-stt --version 2 --alias canary
python scripts/registry_cli.py resolve  --name whisper-stt --ref production
```

`--metric key=value` is accepted at registration but optional — pretrained models
have no training metrics to report.

## Importing from HuggingFace

robo-be's workers hardcode an HF repo id and load whatever is at HEAD. That
broke silently at least once: the gipformer repo renamed every weight file on
2026-08-21 and the loader still asks for the old names — it only runs where a
pre-August cache survives. `models.yaml` fixes this by pinning every model to
an exact commit and importing it into the registry, after which HF Hub is no
longer a runtime dependency.

```bash
make catalog-dry-run              # resolve pins, list the files each entry would store
make register-catalog             # import everything (idempotent)
make register-catalog ONLY=gipformer-asr-vi
```

Every imported version carries `source=huggingface`, `hf_repo`, `hf_revision`
(full commit SHA) and `base_model` tags, so a version is always traceable to
the exact upstream files. Re-running the catalog skips any entry whose
`hf_revision` is already registered — and deliberately does **not** touch
aliases on a skip, so a manual rollback of `@production` is never undone by
a re-run. Aliases are set only on a fresh registration.

Entries can `include`/`exclude` files (drop a 279 MB training checkpoint,
keep one of two duplicate weight formats) and declare `patches` — text
substitutions applied before registering. VoxLingua107 uses one: SpeechBrain
1.0 moved a class and the upstream yaml still points at the old path. That
patch used to live in `Dockerfile.lid`; now it lives with the artifact.

Gated repos (Cohere) need `HF_TOKEN` set or a cached `huggingface-cli login`.

## Extending to finetuning

Nothing here changes when finetuning starts. Registration already takes metrics,
params, and tags; pretrained models simply pass none. A finetuning job registers
through the exact same path:

```python
reg.register(
    name="whisper-stt",
    source_dir="./outputs/finetuned-v3",
    metrics={"wer": 0.081, "cer": 0.031},      # now populated
    params={"epochs": "3", "lr": "1e-5"},
    tags={"base_model": "openai/whisper-small",
          "dataset_hash": "sha256:...",
          "git_commit": "a1b2c3d"},
)
```

Then compare candidates on their metrics in the MLflow UI, promote the winner, and
every consumer picks it up. See [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) for
the tags worth standardizing on.

## Tests

The client tests run against a local MLflow (sqlite + local files). **No docker
needed:**

```bash
make test
```

They cover the behaviour that matters: that alias repointing changes what a
consumer resolves without the consumer changing, that rollback works, that exact
pinning works, and that versions always come back as strings.

For a dev environment:

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

The `[dev]` extra intentionally omits `psycopg2` and `google-cloud-storage`:
those are *server* dependencies — only the MLflow container talks to Postgres and
GCS, and it installs them inside its own image. Clients talk to the server, which
proxies artifacts, and the tests use sqlite, so requiring Postgres headers or GCP
libraries just to run `make test` would be friction for nothing.

## Repo layout

| Path | What it is |
|---|---|
| `docs/book/` | Book-length walkthrough of every file, for newcomers |
| `ARCHITECTURE.md` | The whole ML system — brain, speech paths, registry, evaluation, lifecycle gaps, open decisions |
| `src/registry/client.py` | The facade. The only file consumers depend on. |
| `pyproject.toml` | Installable package; consumers need only `mlflow-skinny` |
| `.github/workflows/ci.yml` | Tests, strict catalog parse, compose validation on every push |
| `scripts/register_pretrained.py` | Register a weights directory as a new version |
| `scripts/register_from_hf.py` | Import HF models pinned to a commit, with patches |
| `models.yaml` | Every model robo-be runs: repo, pinned SHA, alias, consumer |
| `scripts/registry_cli.py` | list / versions / promote / resolve |
| `examples/serving_fastapi.py` | Example consumer — note it never imports mlflow |
| `docker-compose.yml` | Postgres + MLflow (artifacts in GCS) |
| `docs/OBSERVABILITY.md` | Lineage, resolve timing, infra metrics, alerting |
| `docs/superpowers/specs/2026-09-14-model-registry-design.md` | Full design: tech stack, topology, API, storage model, data flows, decisions log |
| `docs/superpowers/plans/2026-09-14-model-registry-extensions.md` | What comes next, in phases: robo-be adoption, Tier 1 bench, observability, finetuning, hygiene |
