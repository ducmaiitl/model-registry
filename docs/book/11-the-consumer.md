# Chapter 11 — A consumer

[← Previous](10-the-importer.md) · [Contents](README.md) · [Next: The tests →](12-tests.md)

---

**File:** `examples/serving_fastapi.py` (111 lines).

Everything so far has been the registry looking at itself. This chapter is the
other side: a program that *uses* it. This file is a pretend speech service, and
it is the shortest explanation of why the project exists.

## Read it for what is missing

```python
"""Example consumer: a serving process that pulls its model from the registry.

The point of this file is what it does NOT contain. There is no `import mlflow`,
no GCS bucket name, no Postgres connection string, and no hardcoded version
number. It asks for "whisper-stt at @production" and gets back a path.

That means shipping a new model is an alias repoint in the registry — this
service picks it up on its next restart, with no code change, no rebuild, and no
redeploy. Rolling back is the same move in reverse.
"""
```

Four absences, each deliberate:

| Not present | Why it matters |
|---|---|
| `import mlflow` | The backend can be replaced without touching this file |
| a bucket name | No storage credentials, no cloud coupling |
| a database connection | No infrastructure knowledge at all |
| a version number | **This is the one that enables deployment without code changes** |

A file defined by what it lacks is unusual. But that absence *is* the product:
every one of those four things used to be scattered across five programs.

## Configuration (lines 20–21)

```python
MODEL_NAME = os.getenv("MODEL_NAME", "whisper-stt")
MODEL_STAGE = os.getenv("MODEL_STAGE", "production")
```

Two environment variables with defaults. `os.getenv(key, default)` returns the
variable if set, else the default.

This gives you a knob without a code change:

```bash
MODEL_STAGE=canary python examples/serving_fastapi.py     # test a candidate
MODEL_STAGE=3       python examples/serving_fastapi.py     # pin a version
```

Because `resolve()` accepts an alias *or* a number (Chapter 6), that second
command works with no extra code. Useful for debugging: pin the exact version
that misbehaved.

## `ModelHandle` (lines 24–51)

```python
class ModelHandle:
    """Wraps a resolved model plus whatever engine actually loads the weights."""

    def __init__(self, resolved: ResolvedModel):
        self.resolved = resolved
        # resolve() downloads the version's artifact directory itself, so
        # local_path already *is* the weights dir — model.bin and config.json
        # sit directly in it. Do not append "model" here.
        self.weights_path = resolved.local_path
        self.engine = self._load_engine()
```

Holds two things: the `ResolvedModel` (metadata — what you got) and the engine
(the loaded model — what runs). Keeping the metadata is what lets the service
report which version it is serving.

### A bug preserved in a comment

That three-line comment marks a mistake that was made and fixed.

The original specification said `weights_path = resolved.local_path / "model"`,
reasoning that `register()` uploads into a subfolder called `model` (Chapter 4),
so the files must be one level down.

Plausible, and wrong. `resolve()` downloads *the version's artifact directory
itself* — so `local_path` already points at the folder containing the weights.
Appending `"model"` gives a path that does not exist.

Why it was not obvious: with the placeholder engine below, nothing reads the
path, so nothing failed. The bug would have appeared the day someone connected a
real engine, as a confusing "file not found" pointing at a model library rather
than at this line.

It was caught by listing the resolved folder and looking at what was actually
in it.

> **The habit worth taking:** when code computes a path, print it and look. Do
> not reason about what should be there.

### The placeholder engine

```python
def _load_engine(self):
    """Load the real inference engine. Placeholder for now.

    With faster-whisper this would be:

        from faster_whisper import WhisperModel
        return WhisperModel(str(self.weights_path), device="cuda")

    The registry does not care which engine this is — it only supplies files.
    """
    return None
```

Returns `None` because this is an example and loading a real model needs a GPU.
The docstring shows the real line.

The last sentence is the boundary of the whole project:

> **The registry does not care which engine this is — it only supplies files.**

Different models here use completely different engines — `transformers`,
`sherpa-onnx`, `funasr`, `speechbrain`. The registry treats all of them
identically, because all it does is put a folder on disk.

### Degrading instead of crashing

```python
def transcribe(self, audio_path: str) -> str:
    if self.engine is None:
        return f"[placeholder] would transcribe {audio_path}"
    segments, _ = self.engine.transcribe(audio_path)
    return " ".join(segment.text for segment in segments)
```

With no engine, it returns a marked string rather than crashing — so the file
runs end-to-end without a GPU and still demonstrates the flow.

`segments, _ = ...` unpacks two return values and **discards the second** by
convention: `_` means "I know something is here and I do not need it".

## `load_model()` (lines 54–65)

```python
def load_model() -> ModelHandle:
    """The one call a consumer makes."""
    reg = ModelRegistry()
    resolved = reg.resolve(MODEL_NAME, MODEL_STAGE)
    print(
        f"loaded {resolved.name} v{resolved.version} "
        f"(@{resolved.alias or 'pinned'}) in {resolved.resolve_seconds:.2f}s"
    )
    print(f"  weights: {resolved.local_path}")
    if resolved.metrics:
        print(f"  metrics: {resolved.metrics}")
    return ModelHandle(resolved)
```

The docstring is the point: **one call**.

`resolved.alias or 'pinned'` handles the empty-string case from Chapter 6 — when
you resolve by number, `alias` is `""`, which is falsy, so it prints `pinned`.

`{resolved.resolve_seconds:.2f}` is f-string formatting: two decimal places.

Printing what was loaded is a small habit with large payoff. When output changes
unexpectedly, the first question is "which version is running?" — and this line
already answered it, in the logs, at startup.

## The commented-out web service (lines 68–105)

```python
# from contextlib import asynccontextmanager
# from fastapi import FastAPI
#
# STATE: dict[str, ModelHandle] = {}
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Resolve once at startup, not per request.
#     STATE["model"] = load_model()
#     yield
#     STATE.clear()
#
# app = FastAPI(title="STT gateway", lifespan=lifespan)
#
# @app.get("/health")
# def health():
#     return {"status": "ok", "model_loaded": "model" in STATE}
#
# @app.get("/model-info")
# def model_info():
#     r = STATE["model"].resolved
#     return {"name": r.name, "version": r.version, "alias": r.alias, ...}
```

Commented out so the file runs without installing FastAPI. Three ideas are worth
extracting even in comment form.

### Load once, at startup

```python
STATE["model"] = load_model()
yield
STATE.clear()
```

The `lifespan` function runs at startup, yields while the service serves
requests, then runs cleanup at shutdown. The model is resolved **once**.

Resolving per request would be a serious bug: even a cache hit costs ~15ms and
a filesystem check, and loading a model into GPU memory takes seconds. A model
is loaded at startup and used for the process's life.

This is also why "pick up a new model" means "restart the service" — the alias
is read once, at boot.

### `/model-info` — the endpoint worth copying

```python
@app.get("/model-info")
def model_info():
    r = STATE["model"].resolved
    return {"name": r.name, "version": r.version, "alias": r.alias,
            "resolve_seconds": r.resolve_seconds, "metrics": r.metrics}
```

An HTTP endpoint that answers *"which model is this instance actually
serving?"* — from the running process, not from a database someone might have
changed since.

With ten instances behind a load balancer, this is how you find the one still
running the old version after a promotion. Cheap to add, invaluable during an
incident.

### Health versus readiness

```python
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE}
```

Reporting *both* "I am alive" and "my model is loaded" lets a load balancer hold
traffic until the model is ready, instead of sending requests to a process still
downloading 4 GB.

## The smoke test (lines 108–111)

```python
if __name__ == "__main__":
    # Smoke test: resolve and print, no server required.
    handle = load_model()
    print(f"  engine: {handle.engine or 'placeholder (no engine wired up)'}")
```

Running the file directly resolves a model and prints what it got:

```
loaded whisper-stt v2 (@production) in 0.34s
  weights: /home/you/.cache/model-registry/whisper-stt/2
  engine: placeholder (no engine wired up)
```

A **smoke test** is the smallest check that a system is wired up at all. This one
proves: the registry is reachable, the alias resolves, the files download, and
the path exists. Seconds to run, and it fails loudly when configuration is wrong.

## What the real integration looks like

The speech workers today contain:

```python
_HF_REPO = "CohereLabs/cohere-transcribe-03-2026"

processor = AutoProcessor.from_pretrained(_HF_REPO)
model = CohereAsrForConditionalGeneration.from_pretrained(
    _HF_REPO, torch_dtype=torch.bfloat16, device_map="auto")
```

Adopting the registry changes one thing:

```python
files = ModelRegistry().resolve("cohere-asr", "production").local_path

processor = AutoProcessor.from_pretrained(files)
model = CohereAsrForConditionalGeneration.from_pretrained(
    files, torch_dtype=torch.bfloat16, device_map="auto")
```

`from_pretrained` has always accepted a local directory as well as a repository
id — it simply skips the download. **The inference code below is untouched.**

Other engines adapt the same way, and two get simpler:

| Engine | Before | After |
|---|---|---|
| transformers | `from_pretrained(repo_id)` | `from_pretrained(local_path)` |
| sherpa-onnx | four `hf_hub_download` calls | `local_path / "encoder.onnx"` |
| funasr | `AutoModel(model=repo_id)` | `AutoModel(model=str(local_path))` |
| speechbrain | download in Dockerfile, then load folder | `foreign_class(source=local_path)` |

The speechbrain row is the best case: an entire Dockerfile step — downloading
2.5 GB at image build time and patching a config file — disappears, because the
registry already holds the patched files.

---

## What you now know

- A consumer needs **one call** and imports **one class**.
- What the file lacks — MLflow, buckets, versions — is the product.
- `local_path` *is* the weights directory; do not append anything.
- Resolve once at startup, not per request.
- Expose which version you loaded, in logs and ideally an endpoint.
- Adopting the registry is a one-line change; inference code is untouched.

---

[← Previous](10-the-importer.md) · [Contents](README.md) · [Next: The tests →](12-tests.md)
