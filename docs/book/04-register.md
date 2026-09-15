# Chapter 4 — Putting a model in: `register()`

[← Previous](03-the-facade.md) · [Contents](README.md) · [Next: Deploying →](05-promote.md)

---

**File:** `src/registry/client.py`, lines 83–136.

**Job:** take a folder of files, store them, and return a new version number.

```python
version = reg.register("cohere-asr", "/path/to/weights")
# version == "1"
```

This is the **write path**. It runs rarely — when a new model arrives or a
training job finishes — and it is allowed to be slow, because it moves
gigabytes.

## The signature

```python
def register(
    self,
    name: str,
    source_dir: str | Path,
    flavor: str = "artifact",
    metrics: dict[str, float] | None = None,
    params: dict[str, str] | None = None,
    tags: dict[str, str] | None = None,
    description: str = "",
) -> str:
```

| Parameter | Required | Purpose |
|---|---|---|
| `name` | yes | registry name, e.g. `"cohere-asr"` |
| `source_dir` | yes | folder whose contents become the version |
| `flavor` | no | what kind of thing this is; recorded as a tag |
| `metrics` | no | numbers, e.g. `{"wer": 0.081}` |
| `params` | no | settings used, e.g. `{"epochs": "3"}` |
| `tags` | no | labels, e.g. `{"hf_repo": "..."}` |
| `description` | no | free text |

`str | Path` means "either a plain string or a `Path` object" — the `|` is
modern type-hint syntax for "or". `-> str` promises a string comes back.

### Why the optional parameters exist before they are needed

Today every model here is **pretrained** — downloaded from the internet,
untrained by us. Pretrained models have no training metrics, so `metrics` is
almost always `None`.

They are in the signature anyway, because when finetuning starts, a training job
registers its output **through this same method**:

```python
reg.register(
    name="cohere-asr",
    source_dir="./outputs/finetuned-v3",
    metrics={"wer": 0.081, "cer": 0.031},        # now filled in
    params={"epochs": "3", "lr": "1e-5"},
    tags={"base_model": "cohere-asr@2", "dataset_hash": "sha256:..."},
)
```

No new code path, no second method. The design decision is: *one way in, whether
the model was downloaded or trained*.

## Step 1 — check the input (lines 99–101)

```python
source = Path(source_dir)
if not source.exists():
    raise FileNotFoundError(f"source_dir does not exist: {source}")
```

`Path(source_dir)` accepts either a string or a `Path` and normalises it.

The check exists so that a typo fails **immediately with a clear message**,
rather than thirty seconds into an upload with something obscure. This is
*fail-fast*: validate what you can before doing expensive or irreversible work.

## Step 2 — create a run and upload (lines 103–110)

```python
run_tags = {"registry.flavor": flavor, "registry.model_name": name}
with self._mlflow.start_run(run_name=f"register-{name}", tags=run_tags) as run:
    run_id = run.info.run_id
    if params:
        self._mlflow.log_params(params)
    if metrics:
        self._mlflow.log_metrics(metrics)
    self._mlflow.log_artifacts(str(source), artifact_path=_ARTIFACT_PATH)
```

### What a "run" is

In MLflow, a **run** records that something happened. Normally it is a training
job: these parameters, those metrics, this output. Here it records "we
registered a model", which is a modest use of a powerful idea — but it means a
finetuning job later fits naturally, because that is what runs are designed for.

Each run gets a `run_id`, a long random string. That id is the thread that ties
a version back to its origin.

### The `with` block

```python
with self._mlflow.start_run(...) as run:
    ...
```

This is a **context manager**. It guarantees that whatever needs to happen at
the end happens — here, closing the run — *even if an error is raised inside the
block*. You have met the pattern before:

```python
with open("file.txt") as f:      # file is closed no matter what
    data = f.read()
```

Without it, a crash mid-upload would leave a run open forever.

### The `if params:` guard

```python
if params:
    self._mlflow.log_params(params)
```

In Python an empty dict is *falsy*, so this means "only if there is something to
log". It avoids a pointless call when `params` is `None` or `{}`.

### The line that moves the gigabytes

```python
self._mlflow.log_artifacts(str(source), artifact_path=_ARTIFACT_PATH)
```

`log_artifacts` uploads **every file in the folder**. This is the slow line —
minutes for a multi-gigabyte model.

`artifact_path="model"` puts them in a subfolder rather than at the run's root.
That matters because a run can hold other things (logs, plots, sample outputs),
and the model needs a stable address inside the run. It is why the version will
point at `runs:/<run_id>/model` rather than `runs:/<run_id>`.

Note the files go to **GCS**, but this code never mentions GCS. It talks to the
MLflow server, which is configured to store artifacts there and proxies the
transfer. That is the facade holding: change the storage, this line is untouched.

## Step 3 — create the registry entry, *after* the upload (lines 112–119)

```python
# Only now, after the upload succeeded, touch the registry. Doing this
# first (as an earlier version did) meant an interrupted multi-GB
# upload left an empty registered model behind with no versions.
try:
    self.client.create_registered_model(name)
except Exception:
    # Already registered; new versions just get appended to it.
    pass
```

**This ordering is a bug fix, and it is the most instructive part of the
method.**

The original code created the registry entry *first*, which reads more
naturally — make the container, then fill it. Then two imports were interrupted
part-way through their uploads. The result: model entries existing in the
registry with **zero versions**. Useless rows that had to be cleaned up by hand,
and which would confuse anyone listing the registry.

Reordering fixes it at the root. If the upload fails, execution never reaches
line 116, and the registry stays clean. There is nothing to clean up because
nothing was created.

> **The general lesson:** do the risky, slow, failable work *first*; record it
> only once it has succeeded. A record of something that did not happen is
> worse than no record.

A test now enforces this, so it cannot regress
(`tests/test_registry.py:129`, covered in Chapter 12).

### The bare `try`/`except: pass`

Swallowing all exceptions is usually a bad habit — it hides real errors. Here it
is deliberate and narrow: MLflow raises when the name already exists, and
"already exists" is the *normal* case for versions 2, 3, 4 and onward. There is
no cheap "create if missing" call, so this is the idiomatic shape.

The cost of being sloppy: a genuine failure (say, the server being down) is also
swallowed here — but it will immediately resurface on the next line, which
cannot succeed either. The error surfaces, just one step later.

## Step 4 — create the version (lines 121–125)

```python
mv = self.client.create_model_version(
    name=name,
    source=f"runs:/{run_id}/{_ARTIFACT_PATH}",
    run_id=run_id,
)
```

This creates the **version** — the immutable numbered snapshot — and points it
at the files uploaded in step 2.

`source` is an **MLflow URI**, its own address scheme:

```
runs:/2b7f1e9a4c3d.../model
└─┬─┘ └──────┬─────┘ └─┬─┘
scheme    run id    subfolder
```

Read as: *"the `model` folder inside run 2b7f1e9a…"*. Like `https://` for the
web, `runs:/` is meaningful to MLflow and nothing else. In Chapter 6 you will
meet its sibling, `models:/`.

Passing `run_id` as well links version → run, which is what makes lineage
queries possible later.

`mv` is short for "model version" — the object MLflow returns, carrying
`mv.version`, the new number.

## Step 5 — attach tags and description (lines 127–134)

```python
if tags:
    for key, value in tags.items():
        self.client.set_model_version_tag(name, str(mv.version), key, str(value))

if description:
    self.client.update_model_version(
        name=name, version=str(mv.version), description=description
    )
```

Tags are set one call at a time — MLflow has no bulk API for this. For the
handful of tags used here that is fine.

Note `str(mv.version)` and `str(value)`: version numbers stringified for the
reason from Chapter 3, and tag values stringified because MLflow stores tags as
text and a passed-in integer would be rejected.

**Tags are how everything interesting is recorded later.** The importer
attaches `hf_repo` and `hf_revision`; `promote()` attaches who deployed what.
They turn the registry from a file store into something you can ask questions of.

## Step 6 — return (line 136)

```python
return str(mv.version)
```

`str()` one final time. `"1"`, not `1`.

## The whole method in one picture

```
register("cohere-asr", "/path/to/weights", tags={...})
    │
    ├─ 1. does the folder exist?  no → FileNotFoundError        [instant]
    │
    ├─ 2. start_run()                                            ──► Postgres
    │        log_params / log_metrics                            ──► Postgres
    │        log_artifacts()  ← THE SLOW PART, gigabytes         ──► GCS
    │     (with-block closes the run, even on error)
    │
    ├─ 3. create_registered_model("cohere-asr")   ← only now     ──► Postgres
    │        already exists? fine, carry on
    │
    ├─ 4. create_model_version(source="runs:/<id>/model")        ──► Postgres
    │
    ├─ 5. set_model_version_tag() × N                            ──► Postgres
    │     update_model_version(description)                      ──► Postgres
    │
    └─ 6. return "1"
```

One slow step, five fast ones. The slow one happens before anything is recorded.

## What this method does *not* do

It does not set an alias. A freshly registered version is stored but **not
deployed** — nothing resolving `production` will get it until someone promotes
it.

That separation is deliberate: registering is safe and reversible, deploying is
the decision. They are different actions, so they are different methods.

That method is next.

---

## What you now know

- `register()` turns a folder into an immutable numbered version.
- A **run** records the act; the version points into it via `runs:/<id>/model`.
- `with` guarantees the run closes even on error.
- **Upload first, record second** — because interrupted uploads used to leave
  empty registry entries behind.
- Optional `metrics`/`params` exist so finetuning uses this same method later.
- Registering does not deploy.

---

[← Previous](03-the-facade.md) · [Contents](README.md) · [Next: Deploying →](05-promote.md)
