# Chapter 2 — A tour of the repository

[← Previous](01-why-this-exists.md) · [Contents](README.md) · [Next: The facade →](03-the-facade.md)

---

Before reading any code closely, it helps to know what is where. This chapter is
the map. Nothing here is deep; every item gets its own chapter later.

## The shape of it

```
model-registry/
│
├── src/registry/              ★ THE LIBRARY — what other programs install
│   ├── __init__.py               4 lines   — what the package exposes
│   └── client.py               364 lines   — everything. The core.
│
├── scripts/                   ▸ COMMAND-LINE TOOLS — for humans
│   ├── registry_cli.py          69 lines   — list / promote / resolve
│   ├── register_pretrained.py   68 lines   — register a local folder
│   └── register_from_hf.py     281 lines   — import from HuggingFace
│
├── models.yaml                 120 lines   — the catalogue of 8 models
│
├── examples/
│   ├── serving_fastapi.py      111 lines   — how a consumer uses the library
│   └── fake_model/                         — two tiny files for testing
│
├── tests/                     ▸ 27 tests, no servers needed
│   ├── test_registry.py        222 lines
│   └── test_register_from_hf.py 145 lines
│
├── docker-compose.yml           57 lines   — starts Postgres + MLflow
├── docker/mlflow.Dockerfile                — builds the MLflow server image
├── Makefile                     53 lines   — shortcuts: make test, make up
├── pyproject.toml               28 lines   — makes this installable with pip
├── .github/workflows/ci.yml                — runs tests on every push
│
└── docs/
    ├── OBSERVABILITY.md                    — what to monitor
    ├── ARCHITECTURE.md (repo root)         — the wider system, all repos
    ├── superpowers/specs/                  — the formal design document
    ├── superpowers/plans/                  — the roadmap
    └── book/                               — you are here
```

About **1,500 lines of code** in total, and `client.py` is a quarter of it.
That ratio is intentional: the library is the product, and everything else is a
thin wrapper around it.

## Three kinds of file

Sorting the repository by *who consumes it* makes the structure click.

### 1. The library — for other programs

```
src/registry/client.py
src/registry/__init__.py
pyproject.toml
```

This is what gets installed elsewhere with `pip install`. When the speech
workers eventually adopt the registry, **this is all they import**. It must stay
small and dependency-light — and it does: a program installing it gets one
package, `mlflow-skinny`, and nothing else.

Covered in chapters 3–8.

### 2. The tools — for humans and automation

```
scripts/registry_cli.py           models.yaml
scripts/register_pretrained.py    Makefile
scripts/register_from_hf.py
```

You run these by hand or from a pipeline. They contain **no logic of their
own** — each one parses arguments and calls a method on `ModelRegistry`. That
is deliberate: if logic lived in the scripts, the Python API and the CLI would
slowly disagree with each other.

Covered in chapters 9–10.

### 3. The scaffolding — for the project

```
tests/                    docker-compose.yml
.github/workflows/ci.yml  docker/mlflow.Dockerfile
examples/                 docs/
```

None of this ships to a consumer. It keeps the project correct, runnable, and
understandable.

Covered in chapters 11–13.

## What each file actually does

### `src/registry/client.py` — the one that matters

Defines two things:

- **`ResolvedModel`** — a small container describing a model you asked for: the
  folder it lives in, its version, how long it took to fetch.
- **`ModelRegistry`** — the facade, with six public methods:

| Method | Direction | One-line job |
|---|---|---|
| `register()` | in | Upload a folder, create a new immutable version |
| `promote()` | in | Point an alias (`production`) at a version — **this is deployment** |
| `resolve()` | out | Alias or version → folder on disk. **The only call consumers make** |
| `list_versions()` | out | What versions exist, and which aliases they hold |
| `list_models()` | out | What model names exist |
| `find_version_by_tag()` | out | Find a version by a tag value (used by the importer) |

Plus `alias_info()` to read the audit trail, and one private helper,
`_materialize()`, which handles caching and is the trickiest code in the repo.

### `src/registry/__init__.py` — the front door

Four lines. Turns the folder into an importable package and declares the public
names, so users write `from registry import ModelRegistry` rather than
`from registry.client import ModelRegistry`.

### `scripts/registry_cli.py` — the management tool

```bash
python scripts/registry_cli.py list
python scripts/registry_cli.py versions --name cohere-asr
python scripts/registry_cli.py promote --name cohere-asr --version 3
python scripts/registry_cli.py resolve  --name cohere-asr --ref production
```

Four subcommands, each a handful of lines.

### `scripts/register_pretrained.py` — the simple way in

Point it at a folder, give it a name, it becomes a version. Used when you
already have model files locally.

### `scripts/register_from_hf.py` — the importer

The largest script, and the one that solves Problem 4 from Chapter 1. It
downloads a model from HuggingFace **pinned to an exact commit**, optionally
patches files inside it, and registers the result with provenance recorded.

Safe to re-run: if that exact commit is already registered, it skips.

### `models.yaml` — the catalogue

Not code — data. Declares all eight models: where each comes from, which commit,
which alias it should get, which files to keep and which to skip.

This file is also the answer to *"what models does this system use?"* — a
question that previously required reading five source files.

### `examples/serving_fastapi.py` — the payoff

A pretend speech service that loads a model from the registry. Read it for what
it **lacks**: no MLflow import, no bucket name, no version number. That absence
is the entire point of the project.

### `tests/` — executable documentation

27 tests that run against SQLite (a database in a single file) instead of
Postgres, so they need **no servers, no Docker, no GPU**. `make test` takes
about five seconds.

Tests are often the best way into an unfamiliar codebase: each one is a small,
complete, working example.

### The infrastructure files

| File | Job |
|---|---|
| `docker-compose.yml` | Starts two containers: Postgres and the MLflow server |
| `docker/mlflow.Dockerfile` | Recipe for the MLflow server image |
| `Makefile` | Shortcuts. `make test`, `make up`, `make register-catalog` |
| `pyproject.toml` | Makes the project installable by pip; declares dependencies |
| `.github/workflows/ci.yml` | Runs the tests automatically on every push to GitHub |

## How data flows

Two paths through the system. Everything else is detail.

**Putting a model in (rare — done by an engineer or a pipeline):**

```
a folder of files
    │
    │  register_from_hf.py  or  register_pretrained.py
    ▼
ModelRegistry.register()
    │
    ├── uploads the files ──────────────► GCS       (gigabytes)
    └── creates a version ──────────────► Postgres  (a few rows)
    │
    │  ModelRegistry.promote()
    ▼
alias "production" now points at that version ────► Postgres
```

**Getting a model out (constant — every program, every start):**

```
ModelRegistry.resolve("cohere-asr", "production")
    │
    ├── asks: what version is production? ────────► Postgres → "3"
    ├── already in the local cache? ──── yes ─────► done, ~0.015 seconds
    │                               └── no  ─────► download from GCS, ~25 seconds
    ▼
ResolvedModel(version="3", local_path=/some/folder, from_cache=True)
    │
    ▼
your program loads the files from that folder
```

Notice the asymmetry: **writing is rare and slow, reading is constant and must
be fast.** That is why an entire chapter (7) is devoted to the cache.

## Suggested reading order

If you plan to read the code rather than only this book:

1. `examples/serving_fastapi.py` — 111 lines, shows the goal
2. `tests/test_registry.py:52` — the test called `test_alias_decoupling`; six
   lines that demonstrate the core idea
3. `src/registry/client.py` — `resolve()` first (line 198), then `promote()`
   (138), then `register()` (83). Leave `_materialize()` (255) until last.
4. `scripts/registry_cli.py` — the class driven from a terminal
5. `models.yaml` — real data, real comments
6. `scripts/register_from_hf.py` — the biggest script, most rewarding last

The next chapter starts on `client.py`.

---

[← Previous](01-why-this-exists.md) · [Contents](README.md) · [Next: The facade →](03-the-facade.md)
