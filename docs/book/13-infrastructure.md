# Chapter 13 — Infrastructure

[← Previous](12-tests.md) · [Contents](README.md) · [Next: End to end →](14-end-to-end.md)

---

**Files:** `docker-compose.yml`, `docker/mlflow.Dockerfile`, `Makefile`,
`pyproject.toml`, `.github/workflows/ci.yml`.

None of this ships to a consumer. It makes the project runnable, installable,
and self-checking.

## `pyproject.toml` — making it installable

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "robo-model-registry"
version = "0.1.0"
description = "Facade over MLflow: versioned STT/TTS models with movable aliases..."
requires-python = ">=3.11"
# Consumers only talk to the MLflow server, which proxies artifacts, so the
# skinny client is enough — no GCS/S3 libraries, no sqlalchemy.
dependencies = ["mlflow-skinny>=2.16,<3"]

[project.optional-dependencies]
# Tests run against a local sqlite backend, which needs the full mlflow.
dev = ["mlflow>=2.16,<3", "pytest>=7.0", "huggingface_hub>=0.24"]
import = ["huggingface_hub>=0.24"]

[tool.setuptools.packages.find]
where = ["src"]
```

`pyproject.toml` is the modern standard for describing a Python package. It
replaces the older `setup.py`.

### Distribution name versus import name

```toml
name = "robo-model-registry"        # what you pip install
```

```python
from registry import ModelRegistry  # what you import
```

These differ on purpose. `registry` is a short, pleasant import name but far too
generic to claim on a package index. `robo-model-registry` is specific. Python
allows the two to differ, and many packages do (you `pip install
beautifulsoup4`, then `import bs4`).

### The dependency decision

This single line is the most consequential in the file:

```toml
dependencies = ["mlflow-skinny>=2.16,<3"]
```

MLflow ships in two flavours:

| Package | Contains | Size |
|---|---|---|
| `mlflow` | server, web UI, database drivers, storage backends | large |
| `mlflow-skinny` | just the client | small |

Consumers never run a server or touch a database — they make HTTP calls to one.
So they need only the skinny client.

This was verified rather than assumed. Installing the package into a clean
environment shows:

```
mlflow-skinny       2.22.5
robo-model-registry 0.1.0
```

No sqlalchemy, no cloud storage libraries. For an ML service image where every
dependency is a potential conflict, that restraint matters.

`>=2.16,<3` is a **version constraint**: at least 2.16 (features we rely on),
below 3.0 (a major version may break compatibility).

### Optional dependencies

```toml
[project.optional-dependencies]
dev = ["mlflow>=2.16,<3", "pytest>=7.0", "huggingface_hub>=0.24"]
import = ["huggingface_hub>=0.24"]
```

Extras are installed by name:

```bash
pip install .              # consumer: mlflow-skinny only
pip install ".[dev]"       # developer: + full mlflow, pytest
pip install ".[import]"    # importing machine: + huggingface_hub
```

Note `dev` includes the **full** mlflow — because the tests run a SQLite backend
in-process (Chapter 12), which needs the server-side pieces. Consumers never do,
so they never carry them.

### `where = ["src"]`

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

This declares the **src layout**: code lives in `src/registry/` rather than
`registry/` at the top level.

The benefit is subtle but real. With a top-level layout, `import registry` finds
the local folder whether or not the package is correctly installed — so packaging
mistakes stay hidden until someone else installs it. With `src/`, the local
folder is not importable by accident, so tests run against the *installed*
package. Packaging bugs surface immediately.

## `docker-compose.yml` — the two services

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: ${PG_USER:-mlflow}
      ...
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${PG_USER:-mlflow} -d ${PG_DB:-mlflow}"]
      interval: 5s
      timeout: 5s
      retries: 10
    restart: unless-stopped
```

Docker Compose starts several containers described in one file. Two here:
Postgres and the MLflow server. (Storage is GCS, which is a cloud service, not a
container.)

### `${VAR:-default}`

Shell-style substitution: use the environment variable if set, otherwise the
default. The same argument → environment → default idea as the `or` chains in
Chapter 3, expressed in YAML.

### Volumes

```yaml
volumes:
  - pgdata:/var/lib/postgresql/data
```

Containers are disposable; their filesystems vanish when removed. A **volume**
is storage that outlives the container. Without this line, restarting Postgres
would erase the entire registry.

### Healthchecks

```yaml
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U ..."]
  interval: 5s
```

A command Docker runs repeatedly to decide whether a container is actually
working. "Running" and "ready" are different — Postgres takes seconds to accept
connections after its process starts.

That distinction is used directly:

```yaml
  mlflow:
    depends_on:
      postgres:
        condition: service_healthy
```

MLflow waits for Postgres to be **healthy**, not merely started. Without
`condition: service_healthy`, MLflow would launch immediately, fail to connect,
and exit.

### The server command

```yaml
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri postgresql://${PG_USER:-mlflow}:...@postgres:5432/${PG_DB:-mlflow}
      --artifacts-destination gs://${GCS_BUCKET:?set GCS_BUCKET in .env — the bucket that holds model weights}
      --serve-artifacts
```

Four flags that define the whole system:

| Flag | Effect |
|---|---|
| `--host 0.0.0.0` | accept connections from outside the container |
| `--backend-store-uri postgresql://...` | metadata → Postgres |
| `--artifacts-destination gs://...` | files → Google Cloud Storage |
| `--serve-artifacts` | **the server proxies file transfers** |

`--serve-artifacts` is what makes the skinny client possible. Without it, every
client would need cloud credentials and libraries to reach storage directly.
With it, clients upload and download **through the server**, which owns the
credentials. One flag, and consumers become infrastructure-free.

Note `postgres:5432` — inside a compose network, the service name is a hostname.

### Fail fast on missing configuration

```yaml
${GCS_BUCKET:?set GCS_BUCKET in .env — the bucket that holds model weights}
```

`:?message` means **error out with this message if unset**, rather than
substituting a default.

Without it, an unset variable yields `gs://` — and the server starts happily,
then fails on the first upload with something obscure. The colon-question-mark
turns a confusing runtime failure into a clear startup refusal.

### A security note in a comment

```yaml
    ports:
      # MLflow has NO authentication. Never expose this port publicly —
      # firewall it or put an authenticating reverse proxy in front.
      - "${MLFLOW_PORT:-5000}:5000"
```

MLflow ships with no authentication whatsoever. Anyone who can reach port 5000
can move `production` to any version — which, per Chapter 5, *is* deployment.

The comment sits where someone is most likely to be editing when they make the
mistake.

## `docker/mlflow.Dockerfile`

```dockerfile
FROM python:3.11-slim

# curl backs the compose healthcheck; build-essential + libpq-dev compile psycopg2.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# google-cloud-storage is what lets MLflow read/write gs:// artifact URIs.
RUN pip install --no-cache-dir \
        mlflow==2.16.2 psycopg2==2.9.9 google-cloud-storage==2.18.2

EXPOSE 5000
```

A recipe for an image. `python:3.11-slim` is a minimal base.

Two habits keep the image small — `rm -rf /var/lib/apt/lists/*` and
`--no-cache-dir` — because each `RUN` line becomes a permanent layer, so files
deleted in a *later* line still occupy space.

**Exact pins** (`mlflow==2.16.2`) rather than ranges: this image should build
identically in six months. Consumers get ranges; infrastructure gets pins.

`psycopg2` is the Postgres driver, and needs `libpq-dev` to compile — which is
exactly why it is a server dependency and not a consumer one (Chapter 12
touched on this: requiring Postgres headers to run unit tests would be absurd).

## `Makefile`

```makefile
ifneq (,$(wildcard .env))
include .env
export
endif

PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

test:  ## run client tests (sqlite, no docker needed)
	$(PYTHON) -m pytest tests/ -q
```

`make` is an old tool for running named commands. Here it is a command
shortcut: `make test` instead of remembering the full pytest invocation.

Two lines earn their place:

**`include .env`** — Docker Compose reads `.env` automatically; `make` does not.
Without this, changing `MLFLOW_PORT` in `.env` would move the server but leave
the CLI targets pointing at the old port. This was a real bug.

**The `PYTHON` line** picks `.venv/bin/python` if it exists, else `python3`. So
`make test` works without activating a virtual environment first.

> **Makefile gotcha:** recipe lines must be indented with a **tab**, not spaces.
> Spaces produce `Makefile:31: *** missing separator. Stop.`

## `.github/workflows/ci.yml`

```yaml
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install package with dev extras
        run: pip install -e ".[dev]"
      - name: Unit tests (sqlite backend, no services)
        run: pytest -q
      - name: Catalog parses under the strict loader
        run: ...
      - name: Compose file validates
        run: docker compose config -q
```

GitHub Actions runs this automatically on every push and pull request, on a
fresh machine.

That freshness is the point. Tests passing on your laptop may depend on
something you installed months ago and forgot. A clean machine proves the
project is genuinely self-contained — which is why the `pip install -e ".[dev]"`
step is itself a test.

Three checks, cheap and complementary:

1. **Tests** — the logic works.
2. **Catalogue parses** — `models.yaml` is valid under the strict loader, so a
   duplicate key is caught at review time rather than during an import.
3. **Compose validates** — `docker compose config -q` catches YAML and
   substitution errors without starting anything.

## How it fits together

```
 developer laptop                     GitHub                    server
 ────────────────                     ──────                    ──────
 pip install -e ".[dev]"        push → CI runs:                 docker compose up
 make test         (5s)                 install                   ├── postgres
 make catalog-dry-run                   pytest                    └── mlflow ──► GCS
                                        catalog parse
                                        compose config
 ────────────────────────────────────────────────────────────────────────────────
 consumer program:  pip install robo-model-registry  →  mlflow-skinny only
```

Three environments, three dependency sets, one codebase. The layering — skinny
for consumers, full for developers, pinned for the server image — is what keeps
a consumer's install from dragging in a database driver it will never use.

---

## What you now know

- `pyproject.toml` makes the library installable; distribution and import names
  may differ.
- `mlflow-skinny` for consumers is enabled by `--serve-artifacts` on the server.
- The **src layout** makes packaging mistakes visible early.
- Volumes persist data; healthchecks distinguish *running* from *ready*.
- `${VAR:?message}` turns a silent misconfiguration into a clear refusal.
- MLflow has no authentication — the port must never be public.
- CI on a clean machine proves the project is self-contained.

---

[← Previous](12-tests.md) · [Contents](README.md) · [Next: End to end →](14-end-to-end.md)
