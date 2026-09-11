# Model Registry

One versioned source of truth for the STT/TTS models, shared by every consumer.

## Why this exists

Without a registry, every consumer keeps its own copy of the model files. Changing
a model means editing each place, nobody can say which version is actually running
where, and there is no rollback. This repo replaces that with one registry that
holds versioned models, and a stable API every consumer pulls from.

The core idea is **decoupling**. A consumer asks:

> give me `whisper-stt` at `@production`

and gets back a path to weights already on disk. It does not know that MLflow,
MinIO, or Postgres exist, and it does not know which concrete version it just got.
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
                │ PostgreSQL 16 │  │ MinIO  :9000    │
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
| MinIO | latest | artifact store (S3-compatible) |
| boto3 | 1.34.162 | MLflow ↔ MinIO |
| psycopg2 | 2.9.9 | MLflow ↔ Postgres |
| Docker Compose | — | brings the cluster up |
| pytest | ≥7 | tests the client against sqlite, no docker |

Python 3.11.

MinIO is used **because it is S3-compatible**. Self-hosting now matches the
move off Google Cloud, and switching to real S3/GCS later is an endpoint and
credentials change — no code change, no vendor lock-in.

## Quick start

```bash
cp .env.example .env
make up          # postgres + minio + mlflow, one command
```

- MLflow UI → http://localhost:5000
- MinIO console → http://localhost:9001 (default `minioadmin` / `minioadmin`)

If any of those ports is already taken (another MLflow stack, for instance),
`make up` fails with `port is already allocated`. Override the host ports in
`.env` — container-internal ports never change, so only these need adjusting:

```bash
MLFLOW_PORT=5010
MINIO_API_PORT=9010
MINIO_CONSOLE_PORT=9011
MLFLOW_TRACKING_URI=http://localhost:5010    # keep the client in sync
```

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
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

`requirements-dev.txt` intentionally omits `psycopg2`: only the MLflow *server*
talks to Postgres, and it installs the driver inside its own image (which has the
`libpq-dev` headers needed to build it). Client tests use sqlite, so requiring
Postgres headers just to run `make test` would be friction for nothing.

## Repo layout

| Path | What it is |
|---|---|
| `src/registry/client.py` | The facade. The only file consumers depend on. |
| `scripts/register_pretrained.py` | Register a weights directory as a new version |
| `scripts/registry_cli.py` | list / versions / promote / resolve |
| `examples/serving_fastapi.py` | Example consumer — note it never imports mlflow |
| `docker-compose.yml` | Postgres + MinIO + bucket setup + MLflow |
| `docs/OBSERVABILITY.md` | Lineage, resolve timing, infra metrics, alerting |
