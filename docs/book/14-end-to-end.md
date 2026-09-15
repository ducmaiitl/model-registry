# Chapter 14 — End to end

[← Previous](13-infrastructure.md) · [Contents](README.md) · [Next: Exercises →](15-exercises.md)

---

Two complete journeys, from a typed command to bytes on disk. Everything in this
chapter has already been explained; the value is seeing the pieces connect.

## Journey 1 — a model enters the registry

You type:

```bash
python scripts/register_from_hf.py --catalog models.yaml --only gipformer-asr-vi
```

### Step 1 — the script starts

`scripts/register_from_hf.py` adds `src/` to `sys.path`, imports
`ModelRegistry`, and argparse produces `args.catalog` and `args.only`.

### Step 2 — read the catalogue

`load_catalog()` parses `models.yaml` through `_strict_yaml_load` — which would
raise on a duplicate key. Defaults are merged into each entry: excludes
concatenated, tags overridden. `--only` filters to one entry:

```python
{
  "name": "gipformer-asr-vi",
  "repo": "g-group-ai-lab/gipformer-65M-rnnt",
  "revision": "ba00dad1f33ce28ceec5677fb5890b9ade0d9d50",
  "alias": "production",
  "include": ["*.onnx", "tokens.txt", "bpe.model", "config.json"],
  "exclude": ["*.h5", "*.msgpack", "*.ot", "*.tflite", "*.png"],
  "tags": {"registered_by": "catalog", "framework": "sherpa-onnx", ...},
}
```

### Step 3 — ask HuggingFace what exists

```
→ HuggingFace API: model_info(repo, revision="ba00dad1f33c...")
← sha = "ba00dad1f33ce28ceec5677fb5890b9ade0d9d50"

→ HuggingFace API: list_repo_files(repo, revision=sha)
← 13 filenames
```

`selected()` filters them with the include/exclude rules — 13 files become 9.
The 279 MB training checkpoint is dropped; only inference files remain.

### Step 4 — have we done this before?

```python
existing = reg.find_version_by_tag("gipformer-asr-vi", "hf_revision", sha)
```

```
→ MLflow server: search_model_versions(name='gipformer-asr-vi')
← []            (nothing registered yet)
```

`None` comes back, so the import proceeds. On a **second** run this returns
`"1"`, the script prints *"already registered as version 1 — skipping"*, and —
importantly — **does not touch the alias**.

### Step 5 — download

```
   downloading 9 files -> cache/hf-staging/gipformer-asr-vi
→ HuggingFace CDN: 9 files, 329 MB
```

Into a staging folder, pinned to that exact commit. Nothing that upstream does
afterwards can change what was fetched.

### Step 6 — clean and patch

`strip_download_metadata()` removes the `.cache/` bookkeeping folder.
`apply_patches()` does nothing here — this entry declares no patches. (For
`voxlingua107-lid`, this is where the SpeechBrain class path is rewritten, and
where a missing target would abort the import.)

### Step 7 — register

```python
reg.register(name="gipformer-asr-vi", source_dir=dest, tags={...}, description=...)
```

Inside `register()` (Chapter 4):

```
→ MLflow server: start_run(run_name="register-gipformer-asr-vi")
←                run_id = "3737fd216d9549a8..."          [Postgres: 1 row]

→ MLflow server: log_artifacts(dest, artifact_path="model")
                   server streams 329 MB ──────────────► GCS bucket
                   gs://<bucket>/0/3737fd21.../artifacts/model/

   (the with-block closes the run)

→ MLflow server: create_registered_model("gipformer-asr-vi")   ← only now
←                created                                  [Postgres: 1 row]

→ MLflow server: create_model_version(source="runs:/3737fd21.../model")
←                version = 1                              [Postgres: 1 row]

→ MLflow server: set_model_version_tag × 8
                   hf_repo, hf_revision, source, base_model,
                   framework, role, lang, license           [Postgres: 8 rows]
```

Returns `"1"` — a string.

Had the upload failed at any point, execution would never have reached
`create_registered_model`, and the registry would be untouched.

### Step 8 — promote

```python
reg.promote("gipformer-asr-vi", "1", "production")
```

```
→ get_model_version_by_alias("gipformer-asr-vi", "production")
← error → previous = None                        (first promotion)

→ set_registered_model_alias(name, "production", "1")    ← THE DEPLOY
→ set_registered_model_tag × 4     alias.production.*
→ set_model_version_tag  × 2       promoted.production.*
```

### Step 9 — clean up

The staging folder is deleted. Output:

```
== gipformer-asr-vi  <-  g-group-ai-lab/gipformer-65M-rnnt @ ba00dad1
   + bpe.model
   + config.json
   + decoder-epoch-35-avg-6.onnx
   ... 9 files
   downloading 9 files -> cache/hf-staging/gipformer-asr-vi
   registered gipformer-asr-vi version 1
   promoted v1 -> @production
```

### Final state

| Where | What |
|---|---|
| **Postgres** | 1 model, 1 version, 1 alias, 8 tags, 1 run |
| **GCS** | 329 MB of ONNX files under the run's artifact path |
| **Local disk** | nothing — staging was deleted |

## Journey 2 — a program uses the model

A speech worker container starts.

### Step 1 — construct

```python
reg = ModelRegistry()
```

`MLFLOW_TRACKING_URI` is read from the environment, mlflow is imported lazily,
two handles are stored. **No network yet.**

### Step 2 — resolve

```python
resolved = reg.resolve("gipformer-asr-vi", "production")
```

```
started = time.perf_counter()

"production".isdigit() → False → alias branch

→ MLflow server: get_model_version_by_alias("gipformer-asr-vi", "production")
← version = 1, run_id = "3737fd21..."                       [~20 ms]

uri = "models:/gipformer-asr-vi@production"      ← @ not /
```

### Step 3 — materialise (first run — cold)

```
root  = ~/.cache/model-registry          (no MODEL_REGISTRY_CACHE set)
final = ~/.cache/model-registry/gipformer-asr-vi/1

final/.registry-complete exists? NO

final exists? no
partial = ~/.cache/model-registry/gipformer-asr-vi/1.partial-4821

→ MLflow server: download_artifacts("models:/gipformer-asr-vi@production")
                   server reads from GCS ──────────────► streams to us
←                329 MB into partial/                        [~7 s]

write partial/.registry-complete   {name, version, alias, run_id, resolved_at}

final/.registry-complete exists now? no (nobody raced us)
os.replace(partial → final)                     ← atomic

return (final, from_cache=False)
```

### Step 4 — gather extras, return

```
→ MLflow server: get_run("3737fd21...")
← metrics {}, tags {...}                                     [~15 ms]

ResolvedModel(
    name="gipformer-asr-vi", version="1", alias="production",
    local_path=~/.cache/model-registry/gipformer-asr-vi/1,
    resolve_seconds=7.4, from_cache=False,
    tags={"hf_revision": "ba00dad1...", "framework": "sherpa-onnx", ...},
)
```

### Step 5 — the program loads the model

```python
recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=str(resolved.local_path / "encoder-epoch-35-avg-6.onnx"),
    decoder=str(resolved.local_path / "decoder-epoch-35-avg-6.onnx"),
    joiner=str(resolved.local_path / "joiner-epoch-35-avg-6.onnx"),
    tokens=str(resolved.local_path / "tokens.txt"),
)
```

**This is not registry code.** The registry's involvement ended when it returned
a path. Loading and inference belong to the engine, unchanged.

### Step 6 — the container restarts (warm)

```
→ get_model_version_by_alias(...) → version 1               [~20 ms]

final/.registry-complete exists? YES
return (final, from_cache=True)                             [~0 ms]

resolve_seconds = 0.035, from_cache = True
```

**7.4 seconds → 0.035 seconds.** No download, no GCS traffic. That is Chapter 7
earning its complexity.

## Journey 3 — deploying a better model

The short one, and the reason for everything above.

A finetuned model scores better. You register it:

```bash
python scripts/register_pretrained.py \
    --name gipformer-asr-vi --source-dir ./outputs/finetuned \
    --metric wer=0.081
```

It becomes **version 2**. Nothing changes for anyone — `production` still points
at version 1.

You test it by pinning:

```bash
MODEL_STAGE=2 python examples/serving_fastapi.py
```

Happy, you deploy:

```bash
python scripts/registry_cli.py promote --name gipformer-asr-vi --version 2
```

```
gipformer-asr-vi @production: v1 -> v2  (by mike at 2026-09-15T10:31:02+00:00)
```

Then restart the workers. Each one calls the same unchanged line of code:

```python
reg.resolve("gipformer-asr-vi", "production")
```

and receives version 2.

**What did not happen:** no source file was edited. No container image was
rebuilt. No configuration was redeployed. No worker learned a version number.

And if version 2 is worse in production:

```bash
python scripts/registry_cli.py promote --name gipformer-asr-vi --version 1
```

Restart. Back to version 1 — instantly, because its cache folder was never
deleted.

## The one-page summary

```
  IMPORT                          STORE                        USE
  ──────                          ─────                        ───
  models.yaml                                                  worker starts
      │ pinned commit                                               │
      ▼                                                             │
  register_from_hf.py                                               │
      │ download 329 MB                                             │
      ▼                                                             │
  ModelRegistry.register() ──► MLflow ──┬──► Postgres               │
      │                                 │    version 1, tags        │
      │                                 └──► GCS                    │
      │                                      329 MB of files        │
      ▼                                                             │
  ModelRegistry.promote()  ──► Postgres                             │
      │                        @production → v1                     │
      │                                                             ▼
      └──────────────────────────────────────► ModelRegistry.resolve("...", "production")
                                                     │  which version? → 1
                                                     │  cached? no → download
                                                     ▼
                                               local_path = ~/.cache/.../1
                                                     │
                                                     ▼
                                               engine.load(local_path)
```

---

## What you now know

- How a model travels from an upstream commit to a running program, in full.
- The cold path costs seconds; the warm path costs milliseconds.
- Deploying is one command and a restart; nothing is rebuilt.
- Rolling back is instant because old versions stay cached.

---

[← Previous](13-infrastructure.md) · [Contents](README.md) · [Next: Exercises →](15-exercises.md)
