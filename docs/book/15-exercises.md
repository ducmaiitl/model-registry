# Chapter 15 — Exercises

[← Previous](14-end-to-end.md) · [Contents](README.md) · [Next: Glossary →](16-glossary.md)

---

Reading gets you halfway. These exercises get you the rest, in rough order of
difficulty.

**Everything here is safe.** Nothing downloads a model, needs a GPU, or touches
production. The tests use a throwaway database in a temporary folder, so you
cannot break anything by experimenting.

Set up once:

```bash
cd ~/work/model-registry
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
make test          # should print: 27 passed
```

---

## 1. Watch a test run (2 minutes)

```bash
.venv/bin/python -m pytest tests/test_registry.py::test_alias_decoupling -v
```

Then open `tests/test_registry.py:52` and read the six lines that just ran.

**Ask yourself:** the same function call appears twice with different results.
What changed between them?

---

## 2. Make a test fail on purpose (5 minutes)

Break the code and watch a test catch it. In `src/registry/client.py:136`:

```python
return str(mv.version)        # change to:  return mv.version
```

Run `make test`.

**Expect:** `test_register_returns_version` fails, because `assert
isinstance(version, str)` no longer holds.

**The point:** that `str()` looks like pointless defensiveness until a test
proves something depends on it. Now undo the change.

---

## 3. Print your way through `resolve()` (10 minutes)

The fastest way to confirm you understand a flow. Add prints to
`src/registry/client.py`:

```python
def resolve(self, name, ref="production", cache_dir=None):
    print(f"[resolve] asked for {name!r} at {ref!r}")
    started = time.perf_counter()

    if str(ref).isdigit():
        print("[resolve] numeric → pinning")
        ...
    else:
        print("[resolve] not numeric → alias branch")
        ...
    print(f"[resolve] uri = {uri}")
```

And in `_materialize`:

```python
if (final / _COMPLETE_MARKER).is_file():
    print(f"[cache] HIT  {final}")
    return final, True
print(f"[cache] MISS {final}")
```

Then:

```bash
.venv/bin/python -m pytest tests/test_registry.py -k cache -s
```

`-s` lets prints reach your terminal — without it pytest captures them.

**Watch for:** `MISS` then `HIT` in `test_second_resolve_is_served_from_cache`,
and both resolves producing the *same* path in
`test_alias_and_pinned_version_share_one_cache_slot` even though one asked for
an alias and the other for a number.

Remove the prints when done.

---

## 4. Explore the registry from Python (10 minutes)

No server needed — point at a SQLite file, exactly as the tests do:

```bash
cd /tmp && mkdir -p play && cd play
```

```python
# play.py
import sys
sys.path.insert(0, "/home/mike/work/model-registry/src")

import mlflow
from registry import ModelRegistry

mlflow.set_tracking_uri("sqlite:///play.db")
mlflow.set_experiment("play")
reg = ModelRegistry("sqlite:///play.db")

# make a fake model
from pathlib import Path
d = Path("fake"); d.mkdir(exist_ok=True)
(d / "model.bin").write_bytes(b"version one")

v1 = reg.register("my-model", d, tags={"note": "first try"})
print("registered", v1)

(d / "model.bin").write_bytes(b"version two")
v2 = reg.register("my-model", d, tags={"note": "second try"})
print("registered", v2)

print(reg.promote("my-model", v1, "production"))
print("production is:", reg.resolve("my-model", "production").version)

print(reg.promote("my-model", v2, "production"))
print("production is now:", reg.resolve("my-model", "production").version)

print("alias info:", reg.alias_info("my-model"))
for v in reg.list_versions("my-model"):
    print(v)
```

```bash
/home/mike/work/model-registry/.venv/bin/python play.py
```

**Then try:**
- Read the cache: `find ~/.cache/model-registry/my-model -type f`
- Open a marker: `cat ~/.cache/model-registry/my-model/1/.registry-complete`
- Confirm the files differ between versions — proof that versions are immutable
  even though the source folder was overwritten.

---

## 5. Break the cache and watch it recover (10 minutes)

Continuing from exercise 4:

```bash
# corrupt version 1's cache
echo "junk" > ~/.cache/model-registry/my-model/1/model.bin
rm ~/.cache/model-registry/my-model/1/.registry-complete
```

Then resolve version 1 again in Python:

```python
r = reg.resolve("my-model", "1")
print("from_cache:", r.from_cache)
print("content:", (r.local_path / "model.bin").read_bytes())
```

**Expect:** `from_cache: False` and `b'version one'` — the corruption was thrown
away and the real file refetched.

Now try it the other way: corrupt the file but **leave the marker**.

**Expect:** `from_cache: True` and the junk returned. That is the honest limit of
the design (Chapter 7): it proves a download *finished*, not that the bytes are
correct.

---

## 6. Add a method (30 minutes)

A real contribution. Add `count_versions(name)`:

```python
def count_versions(self, name: str) -> int:
    """How many versions this model has."""
    return len(self.list_versions(name))
```

Then write its test in `tests/test_registry.py`:

```python
def test_count_versions(reg, model_dir):
    assert reg.count_versions("m") == 0
    reg.register("m", model_dir)
    assert reg.count_versions("m") == 1
    reg.register("m", model_dir)
    assert reg.count_versions("m") == 2
```

```bash
make test
```

**Question to sit with:** what *should* happen for a model that does not exist?
Zero, or an error? Look at how `list_versions` behaves, then decide — and make
your test assert whichever you chose. There is no single right answer; the point
is that the test records the decision.

---

## 7. Read a dry run (10 minutes)

This one touches the network, but downloads nothing:

```bash
cd ~/work/model-registry
make catalog-dry-run ONLY=gipformer-asr-vi
```

Then open `models.yaml`, find that entry's `include:` line, and compare it with
the printed file list.

**Try:** temporarily add `"*.onnx"` to that entry's `exclude:` and re-run.
Watch the list shrink. Undo it.

**Understand:** `selected()` (Chapter 10) implements exactly the rules the real
downloader uses — which is why a dry run can be trusted.

---

## 8. Find a bug I left in (harder)

Every codebase has rough edges. Some known ones here:

- **Nothing evicts the cache.** Versions accumulate forever. Where would a
  `prune()` go, and how would it decide what to keep? (Hint: aliased versions
  must survive, and the marker records `resolved_at`.)
- **`promote()` keeps only the latest move per alias.** Promote three times and
  the first two records are overwritten. What would full history require?
- **An interrupted upload leaves an orphan run.** The model entry is clean
  (Chapter 4), but the run and its partial files remain. How would you find
  them? (Hint: a run whose id no model version references.)

Pick one, write down how you would fix it, then compare with
`docs/superpowers/plans/2026-09-14-model-registry-extensions.md` — all three are
tasks in Phase 5.

---

## 9. Trace a bug backwards (harder)

Open `tests/test_registry.py:129`:

```python
def test_failed_upload_leaves_no_model_shell(reg, model_dir, monkeypatch):
```

**Questions:**
1. Which lines in `client.py` does this test protect? (Chapter 4 names them.)
2. Reorder `register()` so the registered model is created *before* the upload,
   and run the test. What exactly does it report?
3. Why did this bug only appear when uploads were **interrupted**, and not in
   normal use?

Question 3 is the interesting one: the bug was invisible until something was
killed mid-operation. Most failure-path bugs are like that.

---

## 10. Explain it to someone else

The real test of understanding. Try to explain, without looking:

- Why does `resolve()` return a folder instead of a loaded model?
- What is the difference between a version and an alias?
- Why is `version` always a string?
- Why does the cache need a marker file rather than just checking the folder?
- What does `--serve-artifacts` allow?

If any answer feels shaky, the relevant chapter is 6, 1, 3, 7, 13 respectively.

---

## Where to go next

**To use the registry:** Chapter 11 and `examples/serving_fastapi.py`.

**To extend it:** `docs/superpowers/plans/2026-09-14-model-registry-extensions.md` —
24 concrete tasks with files and tests specified.

**To understand the wider system:** `ARCHITECTURE.md` in the repository root —
how this fits with the speech services, the brain, and the evaluation work.

**To go deeper on design:** `docs/superpowers/specs/2026-09-14-model-registry-design.md` —
the formal spec, including a decisions log with the reasoning behind each choice.

---

[← Previous](14-end-to-end.md) · [Contents](README.md) · [Next: Glossary →](16-glossary.md)
