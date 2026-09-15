# Chapter 9 — Scripts and the CLI

[← Previous](08-queries.md) · [Contents](README.md) · [Next: The importer →](10-the-importer.md)

---

**Files:** `scripts/registry_cli.py` (69 lines), `scripts/register_pretrained.py` (68 lines).

`client.py` is a Python library — useful from Python. But an engineer promoting
a model at 2am wants a terminal command, not an interpreter. These two scripts
provide that.

The design rule for both:

> **Scripts contain no logic. They parse arguments and call a method.**

If logic lived here, the CLI and the Python API would slowly diverge, and you
would have two behaviours to keep in your head. Keeping scripts thin means the
CLI is always exactly the library.

## The shared shape

Both scripts open the same way:

```python
#!/usr/bin/env python3
"""Management CLI for the model registry. ..."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from registry import ModelRegistry  # noqa: E402
```

### The shebang

```python
#!/usr/bin/env python3
```

On Unix, if a file starts with `#!` and is executable, the system runs it with
the named program. `/usr/bin/env python3` means "whichever `python3` is first on
the PATH" — more portable than hardcoding a path, because it respects virtual
environments.

### The `sys.path` line

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
```

This is the one piece of genuine awkwardness in the scripts, so it is worth
understanding rather than copying blindly.

`sys.path` is the list of folders Python searches for imports. The library lives
in `src/registry/`, which is not one of them by default, so `from registry
import ModelRegistry` would fail.

Reading the expression from the inside out:

| Fragment | Result |
|---|---|
| `__file__` | this script's path, e.g. `scripts/registry_cli.py` |
| `.resolve()` | make it absolute: `/home/you/model-registry/scripts/registry_cli.py` |
| `.parents[1]` | go up two levels: `/home/you/model-registry` |
| `/ "src"` | append: `/home/you/model-registry/src` |

`insert(0, ...)` puts it **first**, so this copy wins over any installed copy.

`# noqa: E402` tells linters to allow an import that is not at the top of the
file — it *has* to come after the path change, or it would fail.

> **Is this good practice?** For scripts inside a repository, it is acceptable
> and common. It exists because the scripts predate `pyproject.toml`
> (Chapter 13). Now that the package is installable, `pip install -e .` makes
> these lines unnecessary — but they are harmless and keep the scripts runnable
> in a fresh clone with nothing installed.

## `registry_cli.py` — the management tool

Four subcommands:

```bash
python scripts/registry_cli.py list
python scripts/registry_cli.py versions --name cohere-asr
python scripts/registry_cli.py promote  --name cohere-asr --version 3
python scripts/registry_cli.py resolve  --name cohere-asr --ref production
```

### Subcommands with argparse

```python
ap = argparse.ArgumentParser(description="Model registry management CLI.")
sub = ap.add_subparsers(dest="cmd", required=True)

sub.add_parser("list", help="list registered model names")

p_versions = sub.add_parser("versions", help="list versions of a model")
p_versions.add_argument("--name", required=True)

p_promote = sub.add_parser("promote", help="point an alias at a version")
p_promote.add_argument("--name", required=True)
p_promote.add_argument("--version", required=True)
p_promote.add_argument("--alias", default="production")

p_resolve = sub.add_parser("resolve", help="resolve a model and download weights")
p_resolve.add_argument("--name", required=True)
p_resolve.add_argument("--ref", default="production", help="alias or exact version")
p_resolve.add_argument("--cache-dir", default=None)
```

`argparse` is Python's standard argument parser. `add_subparsers` creates the
git-style pattern where the first word selects a mode, each with its own flags.

You get a lot for free:

- `--help` on the main command and on every subcommand
- errors for missing required arguments, before any code runs
- `--cache-dir` on the command line becomes `args.cache_dir` in Python
  (argparse converts the dash)

`dest="cmd"` stores which subcommand was chosen in `args.cmd`.
`required=True` means running the script bare prints help and exits non-zero,
rather than doing nothing.

### The dispatch

```python
args = ap.parse_args()
reg = ModelRegistry()

if args.cmd == "list":
    for name in reg.list_models():
        print(name)

elif args.cmd == "versions":
    print(json.dumps(reg.list_versions(args.name), indent=2))

elif args.cmd == "promote":
    rec = reg.promote(args.name, args.version, args.alias)
    prev = f"v{rec['previous_version']}" if rec["previous_version"] else "none"
    print(f"{args.name} @{rec['alias']}: {prev} -> v{rec['version']}  "
          f"(by {rec['promoted_by']} at {rec['promoted_at']})")

elif args.cmd == "resolve":
    r = reg.resolve(args.name, args.ref, cache_dir=args.cache_dir)
    print(json.dumps({
        "name": r.name,
        "version": r.version,
        "alias": r.alias,
        "local_path": str(r.local_path),
        "metrics": r.metrics,
        "resolve_seconds": round(r.resolve_seconds, 4),
        "from_cache": r.from_cache,
    }, indent=2))
```

Each branch is one library call plus printing. That is the whole script.

### Two output styles, chosen deliberately

**`list` prints bare lines** because that composes with other tools:

```bash
python scripts/registry_cli.py list | grep asr
```

**`versions` and `resolve` print JSON** because the data is structured, and JSON
can be piped into `jq` or parsed by another program:

```bash
python scripts/registry_cli.py resolve --name cohere-asr | jq -r .local_path
```

**`promote` prints a sentence**, because a human is reading it and the important
information is the change:

```
cohere-asr @production: v2 -> v3  (by ci at 2026-09-14T09:12:41+00:00)
```

That line only exists because `promote()` returns a record (Chapter 5). The
`if/else` handles a first promotion, where there is no previous version:

```python
prev = f"v{rec['previous_version']}" if rec["previous_version"] else "none"
```

### `str(r.local_path)` and `round(...)`

```python
"local_path": str(r.local_path),
"resolve_seconds": round(r.resolve_seconds, 4),
```

`json.dumps` cannot serialise a `Path` object — it raises `TypeError`. Hence
`str()`.

`round(..., 4)` because `0.01528596878051758` is noise; `0.0153` is the fact.

## `register_pretrained.py` — the simple way in

Registers a folder you already have:

```bash
python scripts/register_pretrained.py \
    --name whisper-stt \
    --source-dir ./weights/whisper-small \
    --alias production \
    --tag base_model=openai/whisper-small \
    --tag framework=faster-whisper
```

"Pretrained" means a model downloaded from someone else rather than trained by
us — which today is all of them.

### Repeatable flags

```python
ap.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE",
                help="version tag (repeatable)")
```

`action="append"` lets a flag appear several times, collecting values into a
list:

```bash
--tag framework=whisper --tag license=MIT
# args.tag == ["framework=whisper", "license=MIT"]
```

`metavar="KEY=VALUE"` controls how the flag appears in `--help`, showing the
expected shape rather than a generic placeholder.

### Parsing `key=value`

```python
def _kv(pairs, cast=str):
    """Parse repeated key=value flags into a dict."""
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"expected key=value, got: {item!r}")
        key, value = item.split("=", 1)
        out[key] = cast(value)
    return out
```

Three things worth noticing.

**`split("=", 1)`** — the `1` limits the split to the **first** `=` only. Without
it, `--tag note=a=b` would produce three fragments and crash. With it, the key is
`note` and the value is `a=b`.

**`raise SystemExit(...)`** — exits with an error message and a non-zero status,
without a traceback. For a CLI that is the right failure: a user who typed a
flag wrongly does not need a stack trace.

**`{item!r}`** — the `!r` in an f-string uses `repr()` instead of `str()`, so the
value appears with quotes: `got: 'framework'`. That makes stray whitespace
visible, which plain printing would hide.

**`cast=str`** is a parameter holding a *function*. It is called on each value,
so the same helper parses both tags and metrics:

```python
tags = _kv(args.tag)                    # values stay strings
metrics = _kv(args.metric, cast=float)  # values become numbers
```

Functions being passable as values is what makes this work — and the reason
`float` can be substituted for `str` without changing `_kv` at all.

### Turning a crash into a message

```python
try:
    metrics = _kv(args.metric, cast=float)
except ValueError:
    raise SystemExit("--metric values must be numbers")
```

`float("abc")` raises `ValueError`. Uncaught, the user sees a traceback ending
in `ValueError: could not convert string to float: 'abc'` — accurate but
unfriendly. Catching it produces one clear line.

### The body

```python
reg = ModelRegistry()
version = reg.register(
    name=args.name,
    source_dir=args.source_dir,
    flavor="artifact",
    metrics=metrics or None,
    tags=tags or None,
    description=args.desc,
)
print(f"registered {args.name} version {version}")

if args.alias:
    reg.promote(args.name, version, args.alias)
    print(f"promoted {args.name} v{version} -> @{args.alias.lower()}")
```

`metrics or None` converts an empty dict to `None`, matching what `register()`
expects for "nothing here" (Chapter 4's `if metrics:` guard treats both the
same, so this is tidiness rather than necessity).

The promotion is **conditional**. Without `--alias`, the model is registered but
not deployed — the Chapter 4 separation, surfaced at the command line: you can
store something without making it live.

## `if __name__ == "__main__":`

Both scripts end with:

```python
if __name__ == "__main__":
    main()
```

`__name__` is `"__main__"` when a file is run directly, and the module's name
when it is imported. So `main()` runs when you execute the script, and does not
run when something imports it.

Chapter 12 relies on this: the tests import `register_from_hf` to test its
helper functions, and this guard is why importing does not trigger a CLI run.

---

## What you now know

- Scripts stay thin so the CLI and the library cannot drift apart.
- `argparse` subcommands give help, validation, and git-style usage for free.
- Output format follows the audience: lines to pipe, JSON to parse, sentences
  to read.
- `split("=", 1)`, `SystemExit`, and `!r` are small CLI-writing habits worth
  keeping.
- Passing a function (`cast=float`) lets one helper serve two purposes.
- `if __name__ == "__main__":` is what makes a script importable by tests.

---

[← Previous](08-queries.md) · [Contents](README.md) · [Next: The importer →](10-the-importer.md)
