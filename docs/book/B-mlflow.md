# Appendix B — MLflow in twenty minutes

[← Appendix A](A-python-concepts.md) · [Contents](README.md)

---

The main chapters explain **our** code and mention MLflow only where the two
touch. This appendix explains MLflow itself: what it is, what it stores, what we
use, and — just as important — what we deliberately ignore.

Read it before Chapter 3 if MLflow is new to you. Everything here applies to
**MLflow 2.16**, the version this project pins.

## 1. What MLflow is

MLflow is an open-source tool for managing machine-learning work. We did not
write it; we depend on it. It has several components, and **this project uses
two**:

| Component | What it does | Do we use it? |
|---|---|---|
| **Tracking** | records runs: parameters, metrics, files | ✅ lightly |
| **Model Registry** | named models, versions, aliases | ✅ this is the one |
| Models | a packaging format (`pyfunc`, "flavors") | ❌ |
| Projects | a way to package runnable code | ❌ |
| Evaluate / Serving / Prompts | benchmarking, model servers, prompt tools | ❌ |

That ratio is worth internalising: **we use maybe 20% of MLflow**, and the rest
is noise when you read its documentation. A tutorial that starts with
`mlflow.sklearn.log_model(...)` is showing you a part we deliberately avoid
(section 7 explains why).

## 2. The architecture

MLflow splits into a client and a server, and the server splits its storage in
two:

```
   your Python process
   ┌───────────────────────────┐
   │  mlflow / MlflowClient    │   the CLIENT library
   └─────────────┬─────────────┘
                 │  HTTP  (MLFLOW_TRACKING_URI)
                 ▼
   ┌───────────────────────────┐
   │  mlflow server  :5000     │   the TRACKING SERVER
   │  + a web UI               │
   └────┬─────────────────┬────┘
        │                 │
        ▼                 ▼
 ┌──────────────┐  ┌──────────────┐
 │ BACKEND      │  │ ARTIFACT     │
 │ STORE        │  │ STORE        │
 │              │  │              │
 │ metadata:    │  │ files:       │
 │ runs,        │  │ weights,     │
 │ metrics,     │  │ plots,       │
 │ versions,    │  │ anything     │
 │ aliases      │  │              │
 │              │  │              │
 │ Postgres     │  │ GCS bucket   │
 │ (or SQLite)  │  │ (or a folder)│
 └──────────────┘  └──────────────┘
```

Three terms you will meet constantly in MLflow's documentation:

**Tracking URI** — where the client sends everything. Set via
`MLFLOW_TRACKING_URI` or `mlflow.set_tracking_uri()`. It can be:

| Value | Meaning |
|---|---|
| `http://localhost:5000` | talk to a server — **what production uses** |
| `sqlite:///path/to.db` | no server; use a local database file — **what our tests use** |
| `./mlruns` | no server; plain files in a folder — MLflow's default |

**Backend store** — where metadata goes. A database. Ours is Postgres; the tests
use SQLite. Set by the server's `--backend-store-uri`.

**Artifact store** — where files go. Ours is a GCS bucket. Set by the server's
`--artifacts-destination`.

That the same code works against a server *and* against a SQLite file is what
lets our 27 tests run with no infrastructure (Chapter 12). You are not testing a
mock — you are testing real MLflow, with a smaller backend.

### `--serve-artifacts`, and why it matters here

Normally an MLflow client uploads files **directly** to the artifact store,
which means every client needs cloud credentials and storage libraries.

Our server runs with `--serve-artifacts`, which makes it **proxy** file
transfers. Clients send files to the server; the server writes to GCS.

That single flag is why a consumer of this registry installs `mlflow-skinny` and
nothing else — no GCS library, no credentials (Chapter 13).

## 3. The data model

MLflow has two halves that beginners often conflate. They are connected, but
distinct.

### Half 1 — Tracking: experiments and runs

```
Experiment  ("Default", id 0)
   └── Run   (id: 3737fd216d95...)
         ├── params    {"epochs": "3"}          set once, immutable
         ├── metrics   {"wer": 0.081}           can have a history over steps
         ├── tags      {"registry.flavor": "artifact"}   mutable labels
         └── artifacts  any files
```

**Experiment** — a folder for runs. Every run belongs to exactly one. If you
never choose one, MLflow uses `Default`, which has id `0`.

> You have already seen that `0`. Artifacts land at
> `gs://<bucket>/0/<run_id>/artifacts/model/` — the `0` is the default
> experiment's id.

**Run** — a record that something happened. Designed for "I trained a model",
which is why it holds params, metrics and files. We use it more modestly: one
run per registration (Chapter 4).

The difference between the three data types matters:

| | Set when | Changeable | Typical use |
|---|---|---|---|
| **param** | once, at the start | no | epochs, learning rate |
| **metric** | any time, with a `step` | appends history | loss per epoch, final WER |
| **tag** | any time | yes, overwritten | labels, provenance, audit |

Params being immutable is why our provenance data (`hf_revision`, `promoted_by`)
is stored as **tags** — tags can be added later and changed.

### Half 2 — Registry: models, versions, aliases

```
Registered Model  ("cohere-asr")
   ├── tags        {"alias.production.version": "3", ...}
   ├── aliases     {"production": 3, "canary": 4}
   │
   ├── Version 1   ── source: runs:/abc.../model ──► a Run
   │     └── tags  {"hf_revision": "ba00dad1...", "framework": "sherpa-onnx"}
   ├── Version 2
   └── Version 3
```

**Registered Model** — a name, e.g. `cohere-asr`. A container.

**Model Version** — an immutable numbered snapshot, `1`, `2`, `3`. Points at
files via a `source` URI, and usually links to the run that produced it.

**Alias** — a movable label on the model, pointing at one version. `production`,
`canary`, anything you like.

### How the halves join

```
Model Version 3  ──source: "runs:/3737fd21.../model"──►  Run 3737fd21...
                                                             │
                                                             └── artifacts/model/*
```

A version does not store files itself. It **points** at a run's artifacts. That
is why `register()` (Chapter 4) creates a run, uploads into it, and only then
creates a version pointing back at it.

## 4. How this repository maps onto MLflow

| Our concept | MLflow entity | Where |
|---|---|---|
| model name (`cohere-asr`) | Registered Model | `register()` |
| an immutable version | Model Version | `register()` |
| `@production` | Alias | `promote()` |
| the act of registering | Run | `register()` |
| provenance (`hf_revision`) | Model Version tag | importer, Ch. 10 |
| audit (`promoted_by`) | Registered Model tag + version tag | `promote()`, Ch. 5 |
| WER of a finetune (future) | Run metric | `register(metrics=...)` |
| the weight files | Artifacts under `model/` | `log_artifacts()` |

One design note: MLflow gives us **experiments** and we barely use them —
everything lands in `Default`. With a handful of models that is fine. If this
grew to many teams, one experiment per project would be the natural next step.

## 5. The entire API surface we use

Nineteen calls. That is all of MLflow this repository touches — a useful thing
to know when MLflow's documentation feels overwhelming.

### Module-level functions (`import mlflow`)

| Call | Purpose | Chapter |
|---|---|---|
| `mlflow.set_tracking_uri(uri)` | point the client at a server or file | 3 |
| `mlflow.set_experiment(name)` | choose the experiment (tests only) | 12 |
| `mlflow.start_run(run_name=, tags=)` | open a run; used as a `with` block | 4 |
| `mlflow.log_params(dict)` | record settings | 4 |
| `mlflow.log_metrics(dict)` | record numbers | 4 |
| `mlflow.log_artifacts(dir, artifact_path=)` | **upload files** | 4 |
| `mlflow.artifacts.download_artifacts(artifact_uri=, dst_path=)` | **download files** | 7 |

### `MlflowClient` methods

| Call | Purpose | Chapter |
|---|---|---|
| `create_registered_model(name)` | create the named container | 4 |
| `create_model_version(name, source, run_id)` | create an immutable version | 4 |
| `get_model_version(name, version)` | fetch one version by number | 6 |
| `get_model_version_by_alias(name, alias)` | **fetch what an alias points at** | 6 |
| `set_registered_model_alias(name, alias, version)` | **the deploy** | 5 |
| `get_registered_model(name)` | fetch the model, incl. its alias map | 5, 8 |
| `set_registered_model_tag(name, key, value)` | tag the model (audit) | 5 |
| `set_model_version_tag(name, version, key, value)` | tag a version (provenance) | 4, 5 |
| `update_model_version(name, version, description)` | set a description | 4 |
| `search_model_versions(filter)` | list versions of a model | 8 |
| `search_registered_models()` | list model names | 8 |
| `get_run(run_id)` | fetch a run's params/metrics/tags | 6 |

Two ways to reach MLflow exist because the module-level functions carry hidden
state (the "active run"), while `MlflowClient` is explicit. Our code uses the
module for file operations and the client for everything else (Chapter 3).

## 6. The web UI

The MLflow server serves a UI at the same port as its API:

```
http://localhost:5000
```

Two sections, matching the two halves of the data model:

**Experiments** — a table of runs. Click one to see its params, metrics, tags,
and an artifact browser. This is where you would compare finetuning runs by WER.

**Models** — the registry. Each registered model lists its versions, their
aliases, tags, and descriptions.

Useful things you can do there that our CLI does not expose:

- browse artifacts to confirm what was actually uploaded
- compare several runs' metrics side by side
- read a version's description and full tag list
- see timestamps

⚠️ **MLflow has no authentication.** Anyone who can open that page can move
`production` to any version — which *is* deployment (Chapter 5). This is why the
compose file carries a warning against exposing the port (Chapter 13).

To explore without a server, you can also run `mlflow ui --backend-store-uri
sqlite:///path/to.db` against a local database file — which is exactly what
section 9 suggests.

## 7. What MLflow offers that we do not use

Knowing what you are *not* using saves confusion when reading MLflow tutorials.

### Flavors and `pyfunc` — not used

MLflow's headline feature is packaging a model so MLflow itself can load and run
it:

```python
mlflow.sklearn.log_model(model, "model")          # we never do this
loaded = mlflow.pyfunc.load_model("models:/name@production")
prediction = loaded.predict(data)                 # MLflow runs the model
```

We store **plain files** with `log_artifacts()` and return a folder path. Why:

1. Our models load through five different engines — `transformers`,
   `sherpa-onnx`, `funasr`, `speechbrain`, `voxcpm`. Some have no MLflow flavor.
2. A flavor would tie the stored format to an MLflow version.
3. The consumer already knows how to load its own model; it only lacked the
   files.

The cost: MLflow cannot serve our models itself. We never wanted it to. This is
the boundary Chapter 1 drew — **the registry supplies files, the engine runs
them**.

### `autolog()` — not used

```python
mlflow.autolog()     # automatically captures params/metrics from popular libraries
```

Useful in training. We have no training yet; when finetuning arrives it becomes
relevant.

### Stages — deprecated, never used

Old MLflow had fixed stages: `Staging`, `Production`, `Archived`.

```python
client.transition_model_version_stage(...)    # ✗ deprecated in 2.9
```

**If a tutorial uses this, it is out of date.** Aliases replaced stages and are
strictly better — you can invent `canary`, `champion`, `shadow` without asking
anyone. This repository never calls the stage API (Chapter 5).

### Projects, Recipes, Model Serving — not used

Ways to package runnable code and to serve models over HTTP. Out of scope; our
serving is done by the speech workers.

## 8. Quirks worth knowing

Real behaviours that cost debugging time in this project.

### `search_model_versions` does not populate aliases

```python
[(mv.version, mv.aliases) for mv in client.search_model_versions("name='m'")]
# → [(2, []), (1, [])]      ← empty, even though version 1 holds "production"

client.get_model_version("m", "1").aliases      # → ['production']
client.get_registered_model("m").aliases        # → {'production': 1}
```

No error; the field is simply empty. `list_versions()` works around it by
reading the alias map from the registered model (Chapter 8). Tags can behave the
same way, which is why `find_version_by_tag` re-fetches.

### Versions come back as `int` sometimes

`mv.version` may be `1` rather than `"1"`. Concatenating it crashes. Our facade
casts to `str` everywhere — the rule introduced in Chapter 3.

### `@` versus `/` in model URIs

```
models:/name@production     ✓ alias
models:/name/3              ✓ version number
models:/name/production     ✗ fails — "production" is parsed as a version
```

### `runs:/` versus `models:/`

| URI | Means |
|---|---|
| `runs:/<run_id>/model` | artifacts inside a run — used as a version's `source` |
| `models:/<name>@<alias>` | resolve through the registry |

### Tag values must be strings

Passing an `int` is rejected. Hence `str(value)` in `register()`, and the
`record[key] or ""` in `promote()` that converts `None` to `""` (Chapter 5).

### An interrupted upload leaves the run behind

Killing a registration mid-upload leaves a `RUNNING` run with partial artifacts.
Our ordering fix (Chapter 4) keeps the *registry* clean, but the run remains.
Cleaning those up is a planned task.

## 9. Explore MLflow safely

The fastest way to understand MLflow is to poke at it with no server and nothing
to break:

```bash
cd /tmp && mkdir -p mlflow-play && cd mlflow-play
```

```python
# play_mlflow.py — raw MLflow, no registry facade
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///play.db")
mlflow.set_experiment("learning")
client = MlflowClient("sqlite:///play.db")

# --- tracking half ---
with mlflow.start_run(run_name="my-first-run") as run:
    run_id = run.info.run_id
    mlflow.log_params({"epochs": "3", "lr": "1e-5"})
    mlflow.log_metrics({"wer": 0.12})
    mlflow.log_metrics({"wer": 0.09})          # same key again → history
    with open("weights.txt", "w") as f:
        f.write("pretend model")
    mlflow.log_artifact("weights.txt", artifact_path="model")

print("run:", run_id)
print("metrics:", client.get_run(run_id).data.metrics)

# --- registry half ---
client.create_registered_model("demo")
mv = client.create_model_version("demo", f"runs:/{run_id}/model", run_id=run_id)
print("version:", mv.version, type(mv.version))       # note the type!

client.set_registered_model_alias("demo", "production", mv.version)
print("alias →", client.get_model_version_by_alias("demo", "production").version)

# the quirk, live:
print("search says aliases are:", [m.aliases for m in client.search_model_versions("name='demo'")])
print("but the model says:     ", client.get_registered_model("demo").aliases)
```

```bash
/home/mike/work/model-registry/.venv/bin/python play_mlflow.py
```

Then look at it in the UI:

```bash
/home/mike/work/model-registry/.venv/bin/mlflow ui --backend-store-uri sqlite:///play.db
# open http://localhost:5000
```

Two things in that output are worth pausing on, because they are the quirks
from section 8 happening live:

- `version: 1 <class 'int'>` — MLflow really does hand back an integer. Our
  facade casts it to `str` so no caller ever meets it.
- `search says aliases are: [[]]` while `the model says: {'production': 1}` —
  the same fact, one source empty and one correct.

You will also notice the files landed in `./mlruns/1/<run_id>/artifacts/model/`,
not `0/`. That `1` is the id of the `learning` experiment the script created;
`0` is the `Default` experiment, which is what our real registrations use since
they never call `set_experiment`. With a SQLite tracking URI and no artifact
location configured, MLflow stores files in a local `mlruns/` folder — the
third storage option from section 2.

Things to try:

- Log metrics several times with `step=` and watch the UI draw a chart.
- Create a second version and move the alias; watch the Models page change.
- Find the artifacts on disk: `find . -name "weights.txt"`.
- Compare `mv.version`'s type with what our facade returns.

**Then read Chapter 3** and notice how little of this the facade exposes — and
how much confusion that hiding saves.

## 10. Where to read more

- **MLflow docs:** <https://mlflow.org/docs/latest/index.html> — the Model
  Registry and Tracking sections are the relevant ones. Ignore Projects,
  Recipes, and flavor-specific guides.
- **This repo's spec:** `docs/superpowers/specs/2026-09-14-model-registry-design.md`
  §5 documents exactly which MLflow entities we use and how.
- **Version warning:** we pin **MLflow 2.16**. MLflow 3.x reorganised parts of
  the model API. If documentation mentions something absent here, check the
  version it targets.

---

## What you now know

- MLflow = client + tracking server + backend store (metadata) + artifact store
  (files). We use Tracking and the Model Registry; the rest we ignore.
- Experiment → Run → params/metrics/tags/artifacts; Registered Model → Version →
  alias. A version *points* at a run's artifacts.
- `--serve-artifacts` is why consumers need no cloud credentials.
- Nineteen calls is the whole surface this project uses.
- Flavors/`pyfunc` are skipped deliberately: we ship files, not runnable models.
- Stages are deprecated; aliases replace them.
- `search_model_versions` silently omits aliases, versions can be `int`, and
  `@` versus `/` matters in model URIs.

---

[← Appendix A](A-python-concepts.md) · [Contents](README.md)
