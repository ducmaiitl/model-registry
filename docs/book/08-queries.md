# Chapter 8 — Questions you can ask

[← Previous](07-the-cache.md) · [Contents](README.md) · [Next: Scripts and the CLI →](09-scripts-and-cli.md)

---

**File:** `src/registry/client.py`, lines 302–357.

Three read-only methods that answer questions about what is in the registry.
None is complicated, but one contains an MLflow quirk that would have made the
registry quietly useless.

## `list_models()` (lines 330–332)

```python
def list_models(self) -> list[str]:
    """Names of every registered model."""
    return [rm.name for rm in self.client.search_registered_models()]
```

The simplest method in the file. A **list comprehension** — Python's compact
form of "build a list by looping" — turns MLflow's objects into plain names:

```python
[rm.name for rm in self.client.search_registered_models()]
```

is equivalent to:

```python
result = []
for rm in self.client.search_registered_models():
    result.append(rm.name)
return result
```

Returning strings rather than MLflow objects is the facade doing its job. If
this returned `RegisteredModel` objects, every caller would be coupled to
MLflow's types — and swapping the backend would break them all.

```python
>>> reg.list_models()
['cohere-asr', 'gipformer-asr-vi', 'voxlingua107-lid', 'voxcpm2-tts']
```

## `list_versions()` (lines 302–328)

```python
reg.list_versions("cohere-asr")
# [{'version': '1', 'aliases': [], 'run_id': 'abc...', 'status': 'READY', 'created': 1789...},
#  {'version': '2', 'aliases': ['production'], 'run_id': 'def...', ...}]
```

Everything about one model: which versions exist, which aliases each holds.

### The bug hiding in the obvious approach

The natural implementation is one call:

```python
versions = self.client.search_model_versions(f"name='{name}'")
# then read mv.aliases for each one
```

It returns objects that *have* an `aliases` attribute. It looks finished. It is
wrong:

```python
# what search_model_versions actually returns
[(2, []), (1, [])]        # aliases empty — for BOTH versions

# what the truth is
get_model_version("cohere-asr", "1").aliases      → ['production']
get_registered_model("cohere-asr").aliases        → {'production': 1}
```

**`search_model_versions` does not populate `aliases`.** It returns an empty
list even for a version that definitely holds one.

Nothing raises. The field exists, has the right type, and is empty. The result:
`list_versions()` would report that no version holds any alias — so the question
*"which version is production?"* would answer "none", forever.

This was caught by running the code against a real registry and noticing the
output was implausible, not by reading documentation.

### The fix (lines 306–315)

```python
# search_model_versions does NOT populate mv.aliases (it comes back
# empty even for aliased versions), so build the mapping from the
# registered model, which stores it as {alias: version}.
aliases_by_version: dict[str, list[str]] = {}
try:
    rm = self.client.get_registered_model(name)
    for alias, version in (rm.aliases or {}).items():
        aliases_by_version.setdefault(str(version), []).append(alias)
except Exception:
    pass
```

The registered model stores the mapping the other way round:

```python
{'production': 1, 'canary': 2}        # alias → version
```

We need version → aliases, so the loop **inverts** it:

```python
{'1': ['production'], '2': ['canary']}
```

`setdefault(str(version), [])` returns the existing list for that version, or
creates an empty one first — so several aliases pointing at the same version
accumulate rather than overwrite:

```python
{'3': ['production', 'canary']}        # both point at version 3
```

`str(version)` because MLflow gives an integer here. The `str` rule, again.

### Assembling the result (lines 317–327)

```python
out = [
    {
        "version": str(mv.version),
        "aliases": sorted(aliases_by_version.get(str(mv.version), [])),
        "run_id": mv.run_id,
        "status": mv.status,
        "created": mv.creation_timestamp,
    }
    for mv in versions
]
out.sort(key=lambda item: int(item["version"]))
return out
```

Three details worth noticing:

**`.get(key, [])`** returns an empty list when a version holds no alias, instead
of raising `KeyError`. Most versions hold none.

**`sorted(...)`** makes alias order deterministic. Without it, order depends on
dictionary iteration and could differ between runs — which makes output diffs
and tests unstable for no reason.

**The sort key** is the interesting one:

```python
out.sort(key=lambda item: int(item["version"]))
```

A `lambda` is a small unnamed function; `key=` tells `sort` what to order by.
The `int()` is essential. Sorting these as **strings** gives:

```
"1", "10", "11", "2", "3"        ← alphabetical: wrong
```

because `"10"` precedes `"2"` in text. Converting to integers gives:

```
1, 2, 3, 10, 11                  ← numerical: right
```

Here is the tension in the design, stated plainly: versions are strings
everywhere (Chapter 3) because string concatenation crashed on integers — but
*sorting* needs numbers. So they are stored and returned as strings, and
converted to integers only where numeric ordering is required. Both bugs
avoided, each in its place.

## `find_version_by_tag()` (lines 334–357)

```python
def find_version_by_tag(self, name: str, key: str, value: str) -> Optional[str]:
    """Latest version of ``name`` whose tag ``key`` equals ``value``, else None."""
```

Answers: *"is there already a version with this tag value?"*

Its purpose is **idempotency** for the importer (Chapter 10). Before downloading
4 GB from HuggingFace, the importer asks:

```python
existing = reg.find_version_by_tag("cohere-asr", "hf_revision", "b1eacc26...")
if existing:
    print(f"already registered as version {existing} — skipping")
    return
```

Without this, re-running the import would create a duplicate version of every
model, every time.

### The implementation

```python
try:
    versions = self.client.search_model_versions(f"name='{name}'")
except Exception:
    return None
```

A model that does not exist yet is not an error — it means "no match", which is
exactly `None`. The first import of a model hits this path.

```python
matches = []
for mv in versions:
    tags = dict(mv.tags or {})
    if not tags:
        # Like aliases, tags may come back empty from search on some
        # backends; a direct fetch is authoritative.
        try:
            tags = dict(self.client.get_model_version(name, str(mv.version)).tags or {})
        except Exception:
            continue
    if tags.get(key) == value:
        matches.append(int(mv.version))
return str(max(matches)) if matches else None
```

The comment points at the same class of quirk as the aliases problem: search
results may carry empty tags. So when tags look empty, the code **re-fetches
that version directly**, where the data is reliable.

This costs an extra call per version, which is fine for tens of versions and
would need rethinking for thousands.

`continue` skips to the next iteration — used here when a version cannot be
fetched (deleted mid-loop, say) rather than failing the whole search.

`max(matches)` picks the **newest** matching version — relevant if the same
upstream commit was registered twice, which can happen if someone imported by
hand before the catalogue existed.

The final line is a **conditional expression**, Python's one-line if/else:

```python
return str(max(matches)) if matches else None
```

Read as: *"give `str(max(matches))` if `matches` is non-empty, otherwise
`None`"*. And `str(...)` once more, so the return type is consistently a string.

## What these three have in common

All read-only. All return **plain Python types** — strings, lists, dicts — never
MLflow objects. That is what lets `scripts/registry_cli.py` do this:

```python
print(json.dumps(reg.list_versions(args.name), indent=2))
```

`json.dumps` serialises plain types and would fail on an MLflow object. The
facade's habit of converting at the boundary is what makes the layers above it
simple.

## The pattern behind two bugs

Both quirks in this chapter share a shape:

> **An API returned something that looked right and was empty.**

No exception, correct type, plausible value. The only way to catch this is to
run the code against a real system and ask *"is this answer believable?"* —
which is why, alongside 27 unit tests, the real imports in Chapter 10 were
inspected by hand.

Unit tests would not have caught either. Both were found by looking at output
and thinking it looked wrong.

---

## What you now know

- `list_models()` and `list_versions()` return plain types, never MLflow objects.
- `search_model_versions` silently omits aliases and sometimes tags; both are
  worked around by fetching from an authoritative source.
- Versions sort by `int()` even though they are stored as `str`.
- `find_version_by_tag()` is what makes re-running the importer safe.
- Some bugs are only visible by running code and judging the output.

That completes `client.py`. Next: the scripts that drive it.

---

[← Previous](07-the-cache.md) · [Contents](README.md) · [Next: Scripts and the CLI →](09-scripts-and-cli.md)
