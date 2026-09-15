# Appendix A — Python concepts used here

[← Previous](16-glossary.md) · [Contents](README.md) · [Appendix B →](B-mlflow.md)

---

Language features this codebase uses that a newcomer may not have met. Each is
shown as it actually appears, with the reason it is there.

## `@dataclass`

```python
from dataclasses import dataclass, field

@dataclass
class ResolvedModel:
    name: str
    version: str
    metrics: dict[str, float] = field(default_factory=dict)
```

Generates `__init__`, `__repr__` and `__eq__` from the annotations. Without it
you would hand-write nine `self.x = x` lines.

**The trap it protects you from:**

```python
metrics: dict = {}                         # ✗ every instance SHARES one dict
metrics: dict = field(default_factory=dict) # ✓ a new dict per instance
```

Defaults are evaluated once at class-definition time. For mutable types — list,
dict, set — that means all instances share the object. `default_factory` calls
the function fresh for each instance.

*Used in:* `client.py:39` · *Chapter 3*

## The `or` chain

```python
self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT)
```

In Python, `or` returns the first **truthy** operand, not `True`/`False`. So this
reads as a priority list: argument, then environment, then default.

Falsy values: `None`, `False`, `0`, `""`, `[]`, `{}`.

**Watch out:** `port or 8080` turns an explicit `0` into `8080`. When zero or
empty string are legitimate values, test with `is None` instead.

*Used in:* `client.py:67`, `client.py:169` · *Chapters 3, 5*

## Context managers (`with`)

```python
with self._mlflow.start_run(...) as run:
    ...
```

Guarantees cleanup when the block exits — normally or by exception. The same
mechanism as `with open(f) as file:`.

*Used in:* `client.py:104` · *Chapter 4*

## `Path` instead of strings

```python
from pathlib import Path

root = Path("~/.cache/model-registry").expanduser()
final = root / name / version
(final / ".registry-complete").is_file()
```

`Path` overloads `/` to join segments — not division, but a meaning the type
defines for itself. Works on Windows too.

Useful methods here: `.exists()`, `.is_file()`, `.mkdir(parents=True)`,
`.read_text()`, `.write_text()`, `.expanduser()`, `.resolve()`, `.parents[n]`.

**Note:** `json.dumps` cannot serialise a `Path`; convert with `str()` first.

*Used throughout* · *Chapters 7, 9*

## Comprehensions

```python
[rm.name for rm in self.client.search_registered_models()]              # list
{k[len(prefix):]: v for k, v in tags.items() if k.startswith(prefix)}   # dict
```

Build a collection in one expression. The `if` at the end filters.

Readable up to about one line. Beyond that, a loop is clearer.

*Used in:* `client.py:332`, `client.py:185` · *Chapters 5, 8*

## Unpacking

```python
local_path, from_cache = self._materialize(...)    # tuple → two names
segments, _ = self.engine.transcribe(path)         # _ = "don't care"
key, value = item.split("=", 1)                    # two-part split
{**defaults, **entry}                              # merge dicts, later wins
```

`_` is a convention, not a keyword — it signals "deliberately ignored".

*Used in:* `client.py:225`, `register_from_hf.py:121` (dict merge), `register_pretrained.py:28` (split) · *Chapters 6, 10*

## `dict.get()` and `dict.setdefault()`

```python
tags.get(key)                       # None if missing, instead of KeyError
tags.get(key, default)              # your own fallback
tags.setdefault(key, value)         # write only if absent
aliases.setdefault(v, []).append(a) # get-or-create, then append
```

`setdefault` decides precedence: seeding a dict then calling `setdefault` means
**the first writer wins**. That is how version tags beat run tags in
`resolve()`.

*Used in:* `client.py:237`, `client.py:313` · *Chapters 6, 8*

## f-strings

```python
f"models:/{name}@{alias}"                    # interpolation
f"expected key=value, got: {item!r}"         # !r → repr(), shows quotes
f"{resolved.resolve_seconds:.2f}s"           # 2 decimal places
f"{version}.partial-{os.getpid()}"
```

`!r` makes whitespace and type visible: `got: 'abc '` rather than `got: abc`.

*Used throughout* · *Chapters 5, 9*

## Conditional expressions

```python
return str(max(matches)) if matches else None
prev = f"v{rec['previous_version']}" if rec["previous_version"] else "none"
```

Python's one-line if/else. Reads as *value-if-true* `if` *condition* `else`
*value-if-false*.

*Used in:* `client.py:357` · *Chapters 8, 9*

## `sort(key=lambda ...)`

```python
out.sort(key=lambda item: int(item["version"]))
```

`lambda` is a small unnamed function; `key=` says what to sort by.

The `int()` matters: as strings, `"10"` sorts before `"2"`.

*Used in:* `client.py:327` · *Chapter 8*

## Exceptions: when to swallow, when to raise

```python
# raise — the caller cannot get what it asked for
if not source.exists():
    raise FileNotFoundError(f"source_dir does not exist: {source}")

# swallow — this is the normal case
try:
    self.client.create_registered_model(name)
except Exception:
    pass                      # already exists

# swallow — optional extra; the caller already has the real payload
try:
    run = self.client.get_run(run_id)
    ...
except Exception:
    pass                      # run deleted; files still fine
```

**The question:** can the caller still get what it actually asked for? If yes,
degrade. If no, raise.

`raise SystemExit("message")` exits a CLI cleanly with no traceback — the right
failure for user error.

*Used in:* `client.py:100`, `115`, `238` · *Chapters 4, 6*

## Lazy imports

```python
def __init__(self, ...):
    import mlflow          # inside the function, not at module top
```

Defers a slow import until it is actually needed, and lets the module be
imported in environments where the dependency is absent.

Use sparingly — it hides dependencies from readers.

*Used in:* `client.py:74` · *Chapter 3*

## `if __name__ == "__main__":`

```python
if __name__ == "__main__":
    main()
```

`__name__` is `"__main__"` when the file is run directly, and the module name
when imported. So the CLI runs on execution but not on import — which is what
lets tests import the script and test its helpers.

*Used in:* every script · *Chapters 9, 12*

## Functions as values

```python
def _kv(pairs, cast=str):
    ...
    out[key] = cast(value)

tags = _kv(args.tag)                    # values stay strings
metrics = _kv(args.metric, cast=float)  # values become floats
```

`cast` holds a function and calls it. One helper serves both cases.

*Used in:* `register_pretrained.py:22` · *Chapter 9*

## `*args, **kwargs`

```python
def boom(*args, **kwargs):
    raise OSError("simulated upload failure")
```

"Accept any arguments." Essential for test stubs that must stand in for a
function of unknown signature.

*Used in:* `tests/test_registry.py:132` · *Chapter 12*

## Type hints

```python
def resolve(self, name: str, ref: str = "production",
            cache_dir: str | Path | None = None) -> ResolvedModel:
```

Documentation for humans and tools. **Not enforced at runtime** — passing the
wrong type raises nothing by itself.

`str | Path | None` means "any of these". `Optional[str]` is the older spelling
of `str | None`.

`from __future__ import annotations` at the top of a file makes modern syntax
work on older Python versions.

*Used throughout* · *Chapter 3*

## Subclassing to change library behaviour

```python
class Loader(yaml.SafeLoader):
    pass

def construct_mapping(loader, node, deep=False):
    ...raise on duplicate keys...

Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
return yaml.load(text, Loader=Loader)
```

The most advanced code in the repository. A subclass is created rather than
modifying `SafeLoader` directly, so the stricter behaviour applies only to our
parsing — not to every other YAML user in the process.

**The pattern:** when a library's default is silently wrong for your purpose,
override it locally rather than relying on everyone remembering.

*Used in:* `register_from_hf.py:89` · *Chapter 10*

## Filesystem operations

```python
shutil.rmtree(path)                      # delete folder + contents (rm -rf)
shutil.rmtree(path, ignore_errors=True)  # ...and don't complain if absent
path.mkdir(parents=True)                 # create, with parents (mkdir -p)
os.replace(src, dst)                     # ATOMIC rename
os.getpid()                              # this process's id
```

`os.replace` is the important one: atomic at the OS level, which is what makes
write-then-rename safe.

*Used in:* `client.py:280–299` · *Chapter 7*

## Timestamps

```python
datetime.now(timezone.utc).isoformat(timespec="seconds")
# → "2026-09-14T09:12:41+00:00"

time.perf_counter()    # monotonic; for measuring durations
time.time()            # wall clock; for recording when
```

Always attach a timezone. `perf_counter` never goes backwards, so it cannot
produce a negative duration when the system clock adjusts.

*Used in:* `client.py:168`, `client.py:210` · *Chapters 5, 6*

---

[← Previous](16-glossary.md) · [Contents](README.md) · [Appendix B →](B-mlflow.md)
