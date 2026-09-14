# Observability

Three things are worth watching: **what is deployed** (lineage), **how fast
consumers get it** (resolve performance), and **whether the storage under it is
healthy** (infra).

## 1. Built-in lineage & metadata

Every version is traceable back to the run that produced it, and every run
records its params, metrics, and tags. The chain is:

```
alias (@production) -> model version -> run -> params / metrics / tags -> artifacts in GCS
```

Read it from the CLI:

```bash
python scripts/registry_cli.py versions --name whisper-stt
```

Each entry carries `version`, `aliases`, `run_id`, `status`, and `created`. The
`run_id` is the join key into the MLflow UI at `:5000`, where the full param and
metric history lives.

Because aliases are stored on the version, `list_versions` also answers "what is
live right now" — the entry carrying `production` in its `aliases` list.

## 2. Resolve performance

Every `resolve()` returns `resolve_seconds`, covering metadata lookup **plus**
artifact download. This is the number a consumer feels at startup, so it is the
one to track.

```python
resolved = reg.resolve("whisper-stt", "production")
logger.info("model_resolve", extra={
    "model": resolved.name,
    "version": resolved.version,
    "seconds": resolved.resolve_seconds,
})
```

Emit it as a histogram from each consumer. What the shape tells you:

- **First resolve is slow, later ones fast** — normal; check `from_cache` on the result.
- **Every resolve is slow** — the cache directory is not persisting between
  restarts (common in containers with no mounted volume).
- **Sudden jump across all consumers** — look at GCS request errors, not the consumers.

## 3. Infra metrics

**GCS** reports into Cloud Monitoring with nothing to install. The metrics that
matter for a weights bucket:

- `storage.googleapis.com/storage/total_bytes` — bucket size; weight files grow it fast
- `storage.googleapis.com/api/request_count` filtered by `response_code` — 403s mean
  the server's service account lost bucket access; 5xx is a GCS incident
- `storage.googleapis.com/network/sent_bytes_count` — egress, i.e. resolve traffic
  and cost. A consumer re-downloading on every restart shows up here first.

Durability and availability are GCS's problem, not yours — there is no disk to
fill and no storage daemon to keep alive. What remains yours is permissions
(the 403 signal above) and cost (egress).

**Postgres** needs `postgres_exporter` alongside it to expose metrics. The
registry's own load is light — the interesting numbers are connection count,
transaction rate, and database size.

**MLflow** has no metrics endpoint of its own. Treat `GET /health` on `:5000` as
liveness and put request-level monitoring in a reverse proxy if you need it.

## 4. What to alert on

| Signal | Condition | Why it matters |
|---|---|---|
| MLflow `/health` | fails 2 checks in a row | No consumer can resolve; new deploys are blocked |
| GCS `request_count` 403s | any | Server lost bucket permission — registration and resolve both fail |
| GCS egress | > 3x weekly baseline | A consumer is re-downloading instead of caching |
| Postgres `pg_isready` | fails | Total registry outage — MLflow cannot serve anything |
| `resolve_seconds` p95 | > 30s, or 3x its baseline | Consumer startup and autoscaling get slow |
| Resolve error rate | any sustained non-zero | Usually a missing alias or a deleted version |
| Alias repoint | any change to `@production` | Deploy audit — should match an intended release |

The last one is a *notification*, not a page: an alias moving is the deploy
event, so it belongs in whatever channel tracks releases.

## 5. Audit: who deployed what

MLflow itself does not record who moved an alias, so the facade does. Every
`promote()` writes `alias.<alias>.{version,previous_version,promoted_at,promoted_by}`
on the registered model and `promoted.<alias>.{at,by}` on the version:

```python
reg.alias_info("whisper-stt", "production")
# {'alias': 'production', 'version': '3', 'previous_version': '2',
#  'promoted_at': '2026-09-14T09:12:41+00:00', 'promoted_by': 'ci'}
```

`promoted_by` comes from `$REGISTRY_ACTOR`, falling back to the OS user — set it
explicitly in CI so the record names the pipeline, not `runner`. The model-level
tags hold the *latest* move per alias; the per-version tags give you when each
version last held it, which together reconstruct the deploy history.

For stronger guarantees put alias changes behind CI rather than laptops: the CI
run becomes the audit log and can stamp `git_commit` on the version as well.

## 6. Tags to standardize

Tags are free-form, which means they are only useful if applied consistently.
Standardize on these:

| Tag | Example | Why |
|---|---|---|
| `git_commit` | `a1b2c3d` | Ties the version to the code that built it |
| `base_model` | `openai/whisper-small` | Which pretrained model this came from |
| `dataset_hash` | `sha256:...` | Which data a finetune used — the key reproducibility fact |
| `framework` | `faster-whisper` | Tells a consumer which engine can load it |
| `registered_by` | `ci` / `mike` | Audit trail |

Today, pretrained registrations set `base_model` and `framework`; `dataset_hash`
becomes meaningful once finetuning starts.
