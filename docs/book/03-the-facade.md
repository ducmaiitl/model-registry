# Chapter 3 — The facade

[← Previous](02-repo-tour.md) · [Contents](README.md) · [Next: Putting a model in →](04-register.md)

---

**File:** `src/registry/client.py`, lines 1–79.

This chapter covers the top of the file: what it declares before any real work
happens. Four short sections — the docstring, the constants, the `ResolvedModel`
container, and the constructor.

## The docstring (lines 1–10)

```python
"""Facade over MLflow for the model registry.

Consumers import ``ModelRegistry`` from this package and nothing else. They never
import mlflow, never talk to GCS or Postgres, and never learn which concrete
version is serving traffic. That indirection is the whole point: the model
lifecycle moves independently of the code that consumes models.

The single consumer entrypoint is :meth:`ModelRegistry.resolve`, which returns a
:class:`ResolvedModel` whose ``local_path`` already has the weights on disk.
"""
```

A string as the very first thing in a file is a **module docstring**. Python
stores it as `__doc__`, and tools display it as documentation. It is not a
comment — it is part of the module.

This one earns its place by stating the *rule* the file exists to enforce:
consumers never import mlflow, never learn a version. When you later wonder
"should I expose this MLflow object publicly?", the answer is written here.

> **Habit worth copying:** a docstring that says *why* is more useful than one
> that says *what*. "Facade over MLflow" is what. "They never learn which
> concrete version is serving traffic" is why.

## The imports (lines 12–22)

```python
from __future__ import annotations

import getpass
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
```

`from __future__ import annotations` is the odd one. It changes how Python reads
type hints — they become text rather than being evaluated immediately. The
practical effect is that modern syntax like `dict[str, float] | None` works on
older Python versions. Harmless, and common at the top of typed modules.

Everything else is the standard library — nothing installed, nothing exotic:

| Import | Used for |
|---|---|
| `getpass` | the OS username, for the audit trail |
| `json` | writing the cache marker file |
| `os` | environment variables, atomic rename, process id |
| `shutil` | deleting folder trees |
| `time` | measuring how long a resolve took |
| `dataclass`, `field` | building `ResolvedModel` without boilerplate |
| `datetime`, `timezone` | timestamps for promotions |
| `Path` | file paths as objects instead of strings |
| `Optional` | type hints for "this may be None" |

**Notice what is missing: `import mlflow`.** It is not at the top of the file.
Why is explained at the end of this chapter.

## The constants (lines 24–36)

```python
DEFAULT_TRACKING_URI = "http://localhost:5000"

# Artifacts are logged under this subdirectory of the run, so a model version
# points at ``runs:/<run_id>/model`` rather than the run root.
_ARTIFACT_PATH = "model"

# Where resolve() keeps downloaded versions when no cache_dir is given.
# Overridable with MODEL_REGISTRY_CACHE. Layout: <cache>/<name>/<version>/.
DEFAULT_CACHE_DIR = "~/.cache/model-registry"

# Written into a version's cache slot only after every file has landed. Its
# absence means a previous download died half-way and the slot is untrusted.
_COMPLETE_MARKER = ".registry-complete"
```

Four values that would otherwise be scattered through the file as literals.
Naming them means changing one line instead of hunting for every copy.

**The leading underscore** on `_ARTIFACT_PATH` and `_COMPLETE_MARKER` is a
Python convention meaning *internal — not part of the public API*. Nothing in
the language enforces it. It is a message to the next human: the two without
underscores are safe to reference from outside; the two with are implementation
details that might change.

Two of these deserve a note now and a full explanation later:

- `_ARTIFACT_PATH = "model"` — when files are uploaded, they go into a
  subfolder called `model` rather than the top level. Chapter 4 explains why.
- `_COMPLETE_MARKER` — the file whose presence means "this download finished".
  Chapter 7 is entirely about it.

## `ResolvedModel` (lines 39–56)

```python
@dataclass
class ResolvedModel:
    """What a consumer gets back from :meth:`ModelRegistry.resolve`.

    ``version`` is always a string. MLflow hands back an int on some paths, and
    string-concatenating it (``"v" + version``) blows up at runtime, so the
    cast happens here once instead of at every call site.
    """

    name: str
    version: str
    alias: str
    local_path: Path
    run_id: Optional[str] = None
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    resolve_seconds: float = 0.0
    from_cache: bool = False  # True when no download happened
```

### What `@dataclass` does

Without it, you would write this by hand:

```python
class ResolvedModel:
    def __init__(self, name, version, alias, local_path, run_id=None,
                 metrics=None, tags=None, resolve_seconds=0.0, from_cache=False):
        self.name = name
        self.version = version
        self.alias = alias
        # ... five more lines of self.x = x
```

`@dataclass` is a **decorator** — a function that modifies the class beneath it.
It reads the type annotations and generates `__init__` for you, plus `__repr__`
(so printing the object shows its contents instead of `<object at 0x7f...>`) and
`__eq__` (so two objects with equal fields compare equal).

### Why `field(default_factory=dict)` and not `= {}`

This is a real Python trap, and worth understanding properly.

```python
metrics: dict[str, float] = {}                      # ✗ DANGEROUS
metrics: dict[str, float] = field(default_factory=dict)   # ✓ correct
```

A default value in Python is evaluated **once**, when the class is defined — not
each time an object is created. With `= {}`, every `ResolvedModel` ever created
would share *the same dictionary*. Add a metric to one, and it appears in all of
them. `default_factory=dict` says "call `dict()` fresh for each new object".

You only hit this with mutable defaults: lists, dicts, sets. `resolve_seconds:
float = 0.0` is fine, because numbers cannot be modified in place.

### The fields

| Field | Meaning |
|---|---|
| `name` | what you asked for, e.g. `"cohere-asr"` |
| `version` | which concrete version you got, e.g. `"3"` |
| `alias` | which alias you used, or `""` if you asked by number |
| `local_path` | **the folder containing the weights — the payload** |
| `run_id` | link back to the record of how this version was created |
| `metrics` | numbers attached to that record, e.g. `{"wer": 0.081}` |
| `tags` | labels, e.g. `{"hf_revision": "b1eacc26...", "framework": "transformers"}` |
| `resolve_seconds` | how long this call took, for monitoring |
| `from_cache` | `True` if no download happened |

Most programs use only `local_path`. The rest exists so that a program *can*
log what it loaded — which matters when you are debugging why yesterday's
results differ from today's.

### The `str` rule

The docstring flags it, and it is worth taking seriously:

> `version` is always a string.

MLflow returns version numbers as integers in some places and strings in others.
Write `"v" + version` against an integer and Python raises `TypeError`. Rather
than making every caller remember, the conversion happens once — here. You will
see `str(...)` repeatedly through this file for exactly this reason.

It looks like paranoia. It is a bug that had already happened.

## The constructor (lines 59–79)

```python
class ModelRegistry:
    """Versioned model storage with movable aliases.

    Aliases, not stages: MLflow deprecated the stage API in 2.9 and it behaves
    inconsistently, so promotion is always "point an alias at a version".
    """

    def __init__(self, tracking_uri: str = None):
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI
        )

        # Imported lazily so that merely importing this module does not pay the
        # (substantial) mlflow import cost, and so consumers can depend on the
        # package without mlflow resolved at import time.
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(self.tracking_uri)
        self._mlflow = mlflow
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
```

### The `or` chain

```python
self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
```

A common Python idiom meaning **"first thing that isn't empty wins"**. In
Python, `or` returns the first *truthy* operand rather than `True`/`False`, so
this reads as a priority list:

1. the argument, if the caller passed one
2. else the `MLFLOW_TRACKING_URI` environment variable
3. else `http://localhost:5000`

This ordering is what lets the same code run in tests (argument), in production
(environment variable), and on a laptop (default) with no branching.

### Why the import is inside the function

Unusual enough that it carries a three-line comment. Two reasons:

**Speed.** `import mlflow` takes roughly a second and pulls in a large
dependency tree. At the top of the file, *anyone* importing this module pays
that cost — including tools that only want to read `ResolvedModel`. Inside the
constructor, the cost is paid only when someone actually creates a registry.

**Decoupling.** `from registry import ModelRegistry` succeeds even in an
environment where mlflow is not installed. The failure happens later, at the
point of use, with a clearer message.

This is called a **lazy import**. Use it sparingly — it hides dependencies from
readers — but for one heavy library behind a facade, it earns its keep.

### The two handles

```python
self._mlflow = mlflow      # the module    — uploads, downloads
self.client = MlflowClient(...)   # the API object — registry operations
```

MLflow has two interfaces and this class uses both:

- `self._mlflow` for file operations: `log_artifacts`, `download_artifacts`.
- `self.client` for metadata operations: `create_model_version`,
  `set_registered_model_alias`.

`self._mlflow` has an underscore (internal); `self.client` does not, which is a
small inconsistency — it is public mostly because tests reach for it.

---

## What you now know

- A **facade** hides MLflow so it can be replaced without touching consumers.
- `@dataclass` removes constructor boilerplate; `field(default_factory=dict)`
  avoids the shared-mutable-default trap.
- `version` is always a string, on purpose, because of a real crash.
- The `or` chain gives argument → environment → default precedence.
- The lazy `import mlflow` keeps the module cheap to import.

Next: the first real method — how a folder of files becomes a version.

---

[← Previous](02-repo-tour.md) · [Contents](README.md) · [Next: Putting a model in →](04-register.md)
