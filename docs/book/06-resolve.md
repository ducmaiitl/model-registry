# Chapter 6 — Getting a model out: `resolve()`

[← Previous](05-promote.md) · [Contents](README.md) · [Next: The cache →](07-the-cache.md)

---

**File:** `src/registry/client.py`, lines 198–253.

**Job:** turn a name and an alias into a folder on disk.

This is **the method the whole repository exists for**. Every other method
serves engineers or pipelines. This one serves the running system, and it is the
only call a consuming program ever makes:

```python
model = ModelRegistry().resolve("cohere-asr", "production")
model.local_path     # → Path("/home/you/.cache/model-registry/cohere-asr/3")
```

## The signature

```python
def resolve(
    self,
    name: str,
    ref: str = "production",
    cache_dir: str | Path | None = None,
) -> ResolvedModel:
```

The middle parameter is called `ref`, not `alias`, because it accepts **either**
an alias or an exact version number. "Ref" is borrowed from git, where it also
means "a name or a hash".

```python
reg.resolve("cohere-asr")                 # alias "production" (the default)
reg.resolve("cohere-asr", "canary")       # a different alias
reg.resolve("cohere-asr", "3")            # pinned to exactly version 3
```

Why support both? Different callers want different things:

- A **production service** wants the alias. It should get whatever is current
  without knowing or caring which version that is.
- A **benchmark** wants an exact version. If you are comparing model A against
  model B, the numbers are worthless if the model silently changed halfway.

## Step 1 — start the clock (line 210)

```python
started = time.perf_counter()
```

`perf_counter()` is a **monotonic** clock: it only ever moves forward, and is
designed for measuring elapsed time. `time.time()` gives wall-clock time, which
can jump backwards when the system clock syncs — producing negative durations.

Rule of thumb: `time.time()` for *when something happened*,
`time.perf_counter()` for *how long it took*.

The result lands in `resolve_seconds`, which matters more than it looks. Chapter
1 said model loading was previously invisible; this number is the first thing
that makes it measurable.

## Step 2 — alias or version? (lines 212–223)

```python
if str(ref).isdigit():
    version = str(ref)
    alias = ""
    mv = self.client.get_model_version(name, version)
    uri = f"models:/{name}/{version}"
else:
    alias = str(ref).lower()
    mv = self.client.get_model_version_by_alias(name, alias)
    version = str(mv.version)
    # Alias downloads use '@', not '/'. The '/' form is read as a
    # version number and fails.
    uri = f"models:/{name}@{alias}"
```

One branch, decided by `isdigit()` — a string method returning `True` when every
character is a digit. `"3"` → `True`, `"production"` → `False`.

### The pinned branch

`alias = ""` — the empty string records *"you did not use an alias"*. A consumer
inspecting `resolved.alias` can tell the difference between "I followed
production" and "I was pinned".

### The alias branch

`get_model_version_by_alias` asks the server: *what version does `production`
point at right now?* The answer is exactly what the last `promote()` wrote.

This is the moment the indirection pays off. The program said `"production"`;
the server says `3`; the program will never learn that number unless it looks.

### The URI, and a gotcha worth memorising

```python
uri = f"models:/{name}@{alias}"      # ✓ correct
uri = f"models:/{name}/{alias}"      # ✗ silently wrong
```

MLflow has two URI schemes. You met `runs:/` in Chapter 4; this is `models:/`:

```
models:/cohere-asr@production     "whatever production points at"
models:/cohere-asr/3              "exactly version 3"
```

The separator carries the meaning. `@` means alias, `/` means version number.
Write `models:/cohere-asr/production` and MLflow tries to parse `production` as
a version number and fails with a confusing error.

The comment on lines 221–222 exists because this cost real debugging time. Small
comments that record a trap are some of the most valuable in a codebase.

## Step 3 — get the files (lines 225–227)

```python
local_path, from_cache = self._materialize(
    name, version, uri, cache_dir, run_id=mv.run_id, alias=alias
)
```

All the downloading and caching lives in its own method, and Chapter 7 is
devoted to it. What matters here is the shape of the answer.

**It returns two values.** Python does this with a tuple, and the two names on
the left unpack it:

```python
local_path, from_cache = some_function()
```

Equivalent to `result = some_function()` then `local_path = result[0]`,
`from_cache = result[1]` — just far more readable.

`from_cache` is `True` when the files were already on disk and nothing was
downloaded. Without it, `resolve_seconds` would be uninterpretable: 0.015
seconds and 25 seconds are both normal, and you cannot tell which to worry about
unless you know whether a download happened.

## Step 4 — gather the extras (lines 229–241)

```python
run_id = mv.run_id or None
metrics: dict[str, float] = {}
tags: dict[str, str] = dict(mv.tags or {})
if run_id:
    try:
        run = self.client.get_run(run_id)
        metrics = dict(run.data.metrics or {})
        for key, value in (run.data.tags or {}).items():
            tags.setdefault(key, value)
    except Exception:
        # Run may have been deleted; the version and its files are
        # still perfectly usable, so this is not fatal.
        pass
```

Now that the files are secured, collect the information *about* them.

### `or {}` everywhere

`dict(mv.tags or {})` guards against `None`. MLflow may return `None` rather
than an empty dict, and `dict(None)` raises. The `or {}` substitutes an empty
dict, and `dict(...)` copies it so that modifying our copy cannot affect
MLflow's internal object.

### `setdefault` — who wins a conflict

```python
for key, value in (run.data.tags or {}).items():
    tags.setdefault(key, value)
```

There are two sources of tags: the **version** and the **run**. They can use the
same key.

`setdefault(key, value)` writes only if the key is absent. Since `tags` was
seeded from the version's tags, **version tags win**. That is the right
precedence: a version tag was set deliberately for that version, while a run tag
describes the broader act.

### The `try`/`except` that is genuinely correct

This is the good kind of broad exception handling, and the comment explains why:

> Run may have been deleted; the version and its files are still perfectly
> usable, so this is not fatal.

The caller asked for **model files**. Those are already on disk from step 3.
Metrics and tags are a bonus. If the run was cleaned up, failing the whole call
would deny the caller something it has, because something optional is missing.

Compare with step 1: a missing folder in `register()` raises immediately,
because without it the method cannot do its job at all. Same construct,
opposite decision, both right.

> **The question to ask** when writing `except`: *can the caller still get what
> it actually asked for?* If yes, degrade. If no, raise.

## Step 5 — build the answer (lines 243–253)

```python
return ResolvedModel(
    name=name,
    version=version,
    alias=alias,
    local_path=local_path,
    run_id=run_id,
    metrics=metrics,
    tags=tags,
    resolve_seconds=time.perf_counter() - started,
    from_cache=from_cache,
)
```

`time.perf_counter() - started` closes the timer opened in step 1, covering
metadata lookup *and* download — the number a consumer actually experiences at
startup.

Using keyword arguments (`name=name`) rather than positional ones is worth
copying: nine positional values would be unreadable and one transposed pair
would be a silent bug.

## The full flow

```
resolve("cohere-asr", "production")
    │
    ├─ start timer
    │
    ├─ "production".isdigit()? no → alias branch
    │     ask server: what version is @production?      ──► Postgres → "3"
    │     uri = "models:/cohere-asr@production"
    │
    ├─ _materialize(...)                                      [Chapter 7]
    │     cache hit?  yes → return folder, from_cache=True     0.015 s
    │                 no  → download from GCS                  25 s
    │
    ├─ fetch metrics + tags (optional; failure tolerated) ──► Postgres
    │
    └─ ResolvedModel(version="3", local_path=..., resolve_seconds=..., from_cache=...)
```

## Using it

The three lines a consumer writes:

```python
from registry import ModelRegistry

resolved = ModelRegistry().resolve("cohere-asr", "production")
model = AutoModel.from_pretrained(resolved.local_path)
```

And a version that logs what it loaded — worth doing in any real service,
because it turns "why did output change?" into a question with an answer:

```python
resolved = reg.resolve("cohere-asr", "production")
logger.info(
    "model loaded",
    extra={
        "model": resolved.name,
        "version": resolved.version,          # ← what you actually got
        "cached": resolved.from_cache,
        "seconds": round(resolved.resolve_seconds, 2),
        "upstream": resolved.tags.get("hf_revision"),
    },
)
```

## What `resolve()` deliberately does not do

- **It does not load the model.** It returns a folder. Loading is your engine's
  job — `transformers`, `sherpa-onnx`, whatever you use.
- **It does not validate the files.** It checks a download completed (Chapter 7),
  not that the weights are correct. That would require loading them.
- **It does not refresh.** Called once at startup, it gives what the alias meant
  at that moment. To pick up a promotion, the program restarts. This is a
  deliberate simplification: hot-swapping a multi-gigabyte model in a live
  process is a much harder problem, and restarts are cheap.

---

## What you now know

- `resolve()` is the only call consumers make; `ref` is an alias *or* a version.
- `models:/name@alias` uses `@`; `models:/name/3` uses `/`. Mixing them fails.
- `from_cache` is what makes `resolve_seconds` interpretable.
- Version tags beat run tags, via `setdefault`.
- Optional extras degrade; essential inputs raise.

Next: the caching logic hiding behind `_materialize()` — the trickiest code
here, and the one with the most interesting failure mode.

---

[← Previous](05-promote.md) · [Contents](README.md) · [Next: The cache →](07-the-cache.md)
