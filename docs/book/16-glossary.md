# Chapter 16 — Glossary

[← Previous](15-exercises.md) · [Contents](README.md) · [Appendix A →](A-python-concepts.md)

---

Every term this book uses, one line each. Chapter references point at where the
term is explained properly.

## Registry concepts

**Alias** — a movable label pointing at exactly one version, e.g. `production`.
Moving it is how you deploy. Like a git branch. *(Ch. 1, 5)*

**Artifact** — MLflow's word for a file stored with a run. Here, the model
weights. *(Ch. 4)*

**Artifact proxying** — the server transfers files on the client's behalf, so
clients need no storage credentials. Enabled by `--serve-artifacts`. *(Ch. 13)*

**Cache slot** — one folder holding one version: `<cache>/<name>/<version>/`.
*(Ch. 7)*

**Catalogue** — `models.yaml`. Declares every model, pinned to a commit. *(Ch. 10)*

**Completion marker** — `.registry-complete`, written after the last file lands.
Its presence is the only proof a download finished. *(Ch. 7)*

**Consumer** — any program that calls `resolve()`. Contrast **producer**, which
calls `register()`. *(Ch. 11)*

**Facade** — a small interface hiding a larger one. `ModelRegistry` hides MLflow,
so the backend can change without consumers changing. *(Ch. 1, 3)*

**Idempotent** — running it twice has the same effect as running it once. The
importer is idempotent via `find_version_by_tag`. *(Ch. 10)*

**Lineage** — the trail from a version back to the run, commit, and data that
produced it. *(Ch. 4)*

**Model version** — an immutable numbered snapshot, e.g. `"3"`. Never changes.
Like a git commit. *(Ch. 1, 4)*

**Pin** — resolve by exact version number rather than alias, for
reproducibility. *(Ch. 6)*

**Promote** — point an alias at a version. This is deployment. *(Ch. 5)*

**Provenance** — where something came from: `hf_repo`, `hf_revision`,
`base_model`. *(Ch. 10)*

**Registered model** — the named container that versions belong to, e.g.
`cohere-asr`. *(Ch. 4)*

**Resolve** — turn a name plus alias-or-version into a folder on disk. The only
call consumers make. *(Ch. 6)*

**Roll back** — promote the previous version. Same mechanism, opposite
direction. *(Ch. 5)*

**Run** — MLflow's record of an act (a registration, later a training job),
owning params, metrics, tags, artifacts. *(Ch. 4)*

**Staging folder** — where a download lands before being renamed into place;
suffixed with the process id. *(Ch. 7)*

**Tag** — a key/value label on a version or model. How provenance and audit
data are stored. *(Ch. 4, 5)*

## Infrastructure

**Docker Compose** — runs several containers described in one YAML file. *(Ch. 13)*

**Container** — an isolated process with its own filesystem. Disposable. *(Ch. 13)*

**GCS (Google Cloud Storage)** — object storage; holds the weight files. *(Ch. 1)*

**Healthcheck** — a command Docker runs to decide whether a container is
*ready*, as opposed to merely running. *(Ch. 13)*

**MLflow** — the open-source tool underneath this registry. Not ours. *(Ch. 1,
App. B)*

**Backend store** — where MLflow keeps metadata: a database. Ours is Postgres.
*(App. B)*

**Artifact store** — where MLflow keeps files. Ours is a GCS bucket. *(App. B)*

**Experiment** — MLflow's folder for runs. We use the default, id `0`. *(App. B)*

**Flavor / pyfunc** — MLflow's model packaging format, which this project
deliberately does not use. *(App. B)*

**mlflow-skinny** — the client-only MLflow package; what consumers install.
*(Ch. 13)*

**Object storage** — storage for large files addressed by key. Good at
gigabytes, bad at queries. *(Ch. 1)*

**PostgreSQL** — the database holding metadata: names, versions, aliases, tags.
*(Ch. 1)*

**SQLite** — a database in a single file, no server. Used by the tests. *(Ch. 12)*

**Tracking URI** — where the MLflow server (or SQLite file) lives;
`MLFLOW_TRACKING_URI`. *(Ch. 3)*

**Volume** — storage that survives a container being removed. *(Ch. 13)*

## URIs and addresses

**`runs:/<run_id>/model`** — MLflow address for artifacts inside a run. Used as
a version's `source`. *(Ch. 4)*

**`models:/<name>@<alias>`** — address for "whatever this alias points at". Note
the **`@`**. *(Ch. 6)*

**`models:/<name>/<version>`** — address for an exact version. Note the **`/`**.
*(Ch. 6)*

**HF repo id** — a HuggingFace address like `CohereLabs/cohere-transcribe-03-2026`.
Points at a repository whose contents can change. *(Ch. 10)*

**Commit SHA** — a 40-character hash identifying one exact state of a
repository. Cannot change. *(Ch. 10)*

## Python and tooling

**argparse** — the standard library's command-line argument parser. *(Ch. 9)*

**Atomic operation** — happens completely or not at all; no observable
in-between. `os.replace` is atomic. *(Ch. 7)*

**Context manager** — the `with` statement; guarantees cleanup even on error.
*(Ch. 4)*

**Dataclass** — `@dataclass`; generates `__init__`, `__repr__`, `__eq__` from
type annotations. *(Ch. 3)*

**Dict comprehension** — `{k: v for ...}`; builds a dict in one expression.
*(Ch. 5)*

**Fixture** — pytest setup injected into a test by naming it as a parameter.
*(Ch. 12)*

**fnmatch** — glob matching: `*.onnx` matches `encoder.onnx`. *(Ch. 10)*

**f-string** — `f"v{version}"`; string interpolation. `!r` uses `repr()`,
`:.2f` formats decimals. *(Ch. 5, 9)*

**Lambda** — a small unnamed function, e.g. a `sort` key. *(Ch. 8)*

**Lazy import** — importing inside a function instead of at module top, to defer
the cost. *(Ch. 3)*

**List comprehension** — `[x.name for x in items]`; builds a list in one
expression. *(Ch. 8)*

**monkeypatch** — pytest tool that changes something and undoes it after the
test. *(Ch. 12)*

**Regression test** — a test written after a bug, to stop it returning. *(Ch. 12)*

**Shebang** — `#!/usr/bin/env python3`; tells Unix how to run the file. *(Ch. 9)*

**src layout** — package code under `src/`, so tests exercise the installed
package and packaging errors surface early. *(Ch. 13)*

**Type hint** — `name: str`. Documentation and tooling; not enforced at runtime.
*(Ch. 3)*

**Unpacking** — `a, b = f()` splits a returned tuple; `{**a, **b}` merges dicts.
*(Ch. 6, 10)*

## Project-specific

**robo-be** — the self-hosted speech service that will consume this registry.

**Tier 1 / Tier 2** — planned evaluation: raw model quality versus the deployed
system. See `ARCHITECTURE.md`.

**superpowers** — the docs convention used across these repositories:
`specs/` (design), `plans/` (roadmap).

---

[← Previous](15-exercises.md) · [Contents](README.md) · [Appendix A →](A-python-concepts.md)
