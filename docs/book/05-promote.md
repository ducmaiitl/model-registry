# Chapter 5 — Deploying: `promote()`

[← Previous](04-register.md) · [Contents](README.md) · [Next: Getting a model out →](06-resolve.md)

---

**File:** `src/registry/client.py`, lines 138–194.

**Job:** point an alias at a version.

This is the shortest important method in the repository. It is also,
conceptually, the most important — because **this is what "deploying a model"
means here**. There is no other mechanism.

## The idea, before the code

After Chapter 4 you have versions 1, 2, 3 sitting in the registry. None of them
is "live" — they are just stored. Meanwhile programs are running and calling:

```python
reg.resolve("cohere-asr", "production")
```

For that to work, something must decide what `production` means. That is this
method:

```python
reg.promote("cohere-asr", "3", "production")
```

Now `production` means version 3. Every program that restarts gets version 3.
No program changed. No image was rebuilt. Nothing was redeployed.

And if version 3 turns out to be bad:

```python
reg.promote("cohere-asr", "2", "production")
```

That is the rollback. Same call, older number, seconds to execute.

## The signature

```python
def promote(
    self, name: str, version: str, alias: str = "production", actor: str | None = None
) -> dict:
```

`alias` defaults to `"production"` because that is the common case. `actor` says
who is doing this, and is usually left out — the method works it out.

It returns a `dict` describing what happened. An earlier version returned
`None`; returning the record means the caller can print or log it, which the CLI
does.

## Step 1 — normalise (lines 154–155)

```python
alias = alias.lower()
version = str(version)
```

Two small defensive lines with real consequences.

`alias.lower()` means `"Production"`, `"PRODUCTION"` and `"production"` are the
same alias. Without it you could create two aliases differing only in case —
one of which nothing resolves, with no error to tell you.

`str(version)` accepts an integer from a caller and normalises it. The `str`
rule again.

## Step 2 — remember what it pointed at (lines 157–160)

```python
try:
    previous = str(self.client.get_model_version_by_alias(name, alias).version)
except Exception:
    previous = None  # alias did not exist yet
```

Before moving the alias, read where it currently points.

**The order is essential.** After line 162 that information is gone — an alias
holds one version and no history. Read it now or lose it.

The `except` handles the first promotion of a new alias: nothing to look up, so
`previous` is `None`. Using `None` rather than `""` keeps "there was no previous
version" distinct from "the previous version was empty string".

## Step 3 — the deploy (line 162)

```python
self.client.set_registered_model_alias(name, alias, version)
```

**This single line is the deployment.**

Before it, `production` → version 2. After it, `production` → version 3. It
writes one row in Postgres and takes milliseconds.

Everything else in this method is bookkeeping around it. Everything else in the
repository exists to make this line safe, recorded, and reversible.

It is worth pausing on how little happens here. No files move. No service
restarts. No container is rebuilt. A pointer changes, and the next program to
ask gets a different answer. That is the entire benefit of the indirection built
in Chapters 3 and 4.

## Step 4 — record who did it (lines 164–176)

```python
record = {
    "alias": alias,
    "version": version,
    "previous_version": previous,
    "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "promoted_by": actor or os.getenv("REGISTRY_ACTOR") or _os_user(),
}
for key in ("version", "previous_version", "promoted_at", "promoted_by"):
    self.client.set_registered_model_tag(
        name, f"alias.{alias}.{key}", record[key] or ""
    )
self.client.set_model_version_tag(name, version, f"promoted.{alias}.at", record["promoted_at"])
self.client.set_model_version_tag(name, version, f"promoted.{alias}.by", record["promoted_by"])
return record
```

### Why record anything?

MLflow stores *where* an alias points. It does not store **who moved it, when,
or from what**. Six months later, "why is production on version 3, and what was
it before?" has no answer.

Since an alias move is a deploy, and deploys are exactly the events you want an
audit trail for, the facade adds one.

### The timestamp

```python
datetime.now(timezone.utc).isoformat(timespec="seconds")
# → "2026-09-14T09:12:41+00:00"
```

Three deliberate choices:

- `timezone.utc` — a timestamp without a timezone is ambiguous the moment two
  machines in different places write one. UTC everywhere, convert for display.
- `.isoformat()` — a text format that sorts correctly as text, which matters
  because tags are stored as text.
- `timespec="seconds"` — microseconds are noise for a deploy log.

### The actor chain

```python
"promoted_by": actor or os.getenv("REGISTRY_ACTOR") or _os_user(),
```

The `or` chain again (Chapter 3), with three levels:

1. `actor` — the caller was explicit
2. `REGISTRY_ACTOR` — an environment variable, meant for CI to set so the record
   names the pipeline rather than a machine account
3. `_os_user()` — the logged-in username, for someone running it by hand

The helper at the bottom of the file (line 360) is small but careful:

```python
def _os_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # no passwd entry inside some containers
        return "unknown"
```

`getpass.getuser()` can genuinely raise inside a minimal container that has no
user database. Returning `"unknown"` means an audit record with a vague name —
far better than a promotion that crashes because nobody could be identified.

### Where the record is stored

Two places, for two different questions.

**On the registered model** — "what is the latest move of this alias?"

```
alias.production.version           = "3"
alias.production.previous_version  = "2"
alias.production.promoted_at       = "2026-09-14T09:12:41+00:00"
alias.production.promoted_by       = "ci"
```

Composing the key from the alias name (`f"alias.{alias}.{key}"`) means
`production`, `staging` and `canary` each keep their own record without any
extra code.

**On the version** — "when did *this* version hold that alias?"

```
promoted.production.at = "2026-09-14T09:12:41+00:00"
promoted.production.by = "ci"
```

Together they let you reconstruct a history: each version remembers when it was
promoted, and the model remembers the most recent move.

### One wrinkle: `record[key] or ""`

```python
self.client.set_registered_model_tag(name, f"alias.{alias}.{key}", record[key] or "")
```

`previous_version` may be `None` on a first promotion, and MLflow tag values
must be strings. The `or ""` converts `None` to an empty string.

This is why `alias_info()` has to convert it back, which we see next.

## Reading it back: `alias_info()` (lines 179–194)

```python
def alias_info(self, name: str, alias: str = "production") -> dict | None:
    alias = alias.lower()
    tags = dict(self.client.get_registered_model(name).tags or {})
    prefix = f"alias.{alias}."
    found = {k[len(prefix):]: v for k, v in tags.items() if k.startswith(prefix)}
    if "version" not in found:
        return None
    return {
        "alias": alias,
        "version": found["version"],
        "previous_version": found.get("previous_version") or None,
        "promoted_at": found.get("promoted_at"),
        "promoted_by": found.get("promoted_by"),
    }
```

Usage:

```python
reg.alias_info("cohere-asr", "production")
# {'alias': 'production', 'version': '3', 'previous_version': '2',
#  'promoted_at': '2026-09-14T09:12:41+00:00', 'promoted_by': 'ci'}
```

### The dict comprehension

```python
found = {k[len(prefix):]: v for k, v in tags.items() if k.startswith(prefix)}
```

Reading it right to left: take every tag; keep only those starting with
`alias.production.`; strip that prefix from the key; build a new dict.

So `{"alias.production.version": "3"}` becomes `{"version": "3"}`. The slice
`k[len(prefix):]` means "everything from position `len(prefix)` onward" —
standard Python string slicing.

### The `None` cases

Two different "nothings", handled separately:

```python
if "version" not in found:
    return None                                    # never promoted at all
...
"previous_version": found.get("previous_version") or None,   # "" → None
```

The first returns `None` for the whole record: this alias has never been
promoted through this facade. The second turns the empty string (stored because
MLflow needed a string) back into `None`, restoring the distinction the storage
layer flattened.

`dict.get(key)` returns `None` instead of raising when a key is missing — the
right tool when tags may be incomplete, for instance if they were written by an
older version of the code.

## A limitation worth knowing

The model-level tags hold only the **latest** move per alias. Promote three
times and the first two records are overwritten.

Full history would need either an append-only log or a row per event — more
machinery than the current need justifies. The per-version tags partly
compensate: each version remembers when it last held the alias, so you can
reconstruct a rough sequence. If a complete audit trail becomes a requirement,
this is the place to extend, and the plan in
`docs/superpowers/plans/` notes it.

## Why "alias" and not "stage"

The class docstring (line 62) answers a question you might have if you read
MLflow tutorials:

> Aliases, not stages: MLflow deprecated the stage API in 2.9 and it behaves
> inconsistently, so promotion is always "point an alias at a version".

Older MLflow had fixed **stages**: `Staging`, `Production`, `Archived`. They
were deprecated in MLflow 2.9 and will be removed. Aliases replace them and are
better anyway — you can invent `canary`, `champion`, `challenger`,
`shadow` without asking anyone's permission.

If you find a tutorial using `transition_model_version_stage`, it is out of
date. This repository never uses it.

---

## What you now know

- `promote()` is deployment; one line (162) does it.
- Read the previous version **before** moving the alias, or lose it.
- The audit trail is ours, not MLflow's: who, when, from what.
- Tags are stored under composed keys so each alias keeps its own record.
- `None` versus `""` matters, and gets converted at both ends.
- Aliases replaced stages; stages are deprecated.

Next: the other side — how a program gets the model out.

---

[← Previous](04-register.md) · [Contents](README.md) · [Next: Getting a model out →](06-resolve.md)
