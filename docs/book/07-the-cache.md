# Chapter 7 — The cache

[← Previous](06-resolve.md) · [Contents](README.md) · [Next: Questions you can ask →](08-queries.md)

---

**File:** `src/registry/client.py`, lines 255–300.

**Job:** return the folder for a version, downloading only if necessary.

This is the trickiest code in the repository — 45 lines that took more thought
than the other 300. Not because caching is hard in principle, but because of
what happens when it goes *wrong*.

## Two problems, and one of them is nasty

### Problem 1 — re-downloading is wasteful

Model files are 0.3–5 GB. A container restarting every day would re-download
gigabytes daily: slow startups, wasted money, unnecessary load. Obvious problem,
obvious fix — keep files after the first download.

### Problem 2 — a *partial* download is worse than none

This is the one that shapes the design.

Say a download dies at 80%: network drop, container killed, disk full. Now there
is a folder with most of a model in it. The naive cache check —

```python
if folder.exists():
    return folder       # ✗ DANGEROUS
```

— sees a folder and hands it over. Then one of three things happens:

1. The engine fails to load it. **Best case**, because it is loud.
2. The engine loads it, and the model produces subtly wrong output forever.
3. It works on your machine (complete cache) and fails in production (partial).

Case 2 is genuinely bad. A model quietly producing worse transcriptions, with no
error anywhere, is the kind of bug that survives for months and gets blamed on
the model rather than the download.

So the cache is designed around one rule:

> **A folder is trustworthy only if we can prove the download finished.**

Existence is not proof. We need a signal written *after* the last byte lands.

## The completion marker

```python
_COMPLETE_MARKER = ".registry-complete"
```

A small JSON file written into the folder **only after** every model file has
arrived. Present → the download finished. Absent → it did not, whatever else is
in there.

The leading dot is the Unix convention for a hidden file, keeping it out of the
way of directory listings.

## The layout

```
~/.cache/model-registry/
├── cohere-asr/
│   ├── 2/                      ← old version, still here
│   │   ├── .registry-complete
│   │   └── model.safetensors
│   └── 3/                      ← current
│       ├── .registry-complete
│       ├── config.json
│       └── model.safetensors
└── gipformer-asr-vi/
    └── 1/
        ├── .registry-complete
        └── encoder.onnx
```

One folder per **version**, not per alias. That matters: when `production` and
`canary` both point at version 3, they share one folder. And when `production`
moves from 3 to 4, version 3's folder survives — so rolling back is instant,
with no download.

## Reading the code

### Step 1 — work out where the folder should be (lines 274–275)

```python
root = Path(cache_dir or os.getenv("MODEL_REGISTRY_CACHE", DEFAULT_CACHE_DIR)).expanduser()
final = root / name / version
```

The `or` chain again: explicit argument → environment variable → default.

`.expanduser()` turns `~/.cache/...` into `/home/you/.cache/...`. Python does
not expand `~` automatically; the shell normally does it, and code that forgets
ends up creating a literal folder named `~`.

`root / name / version` — `Path` overloads the `/` operator to join path
segments. It is not division; `Path` defines what `/` means for its own type.
This works on Windows too, where the separator is `\`.

### Step 2 — the fast path (lines 276–277)

```python
if (final / _COMPLETE_MARKER).is_file():
    return final, True
```

The whole point of the cache, in two lines. Marker present → return
immediately, `from_cache=True`. Typically 15 milliseconds versus 25 seconds.

`.is_file()` rather than `.exists()`, because `.exists()` is also true for a
directory of that name. Being specific costs nothing.

### Step 3 — clear out anything untrustworthy (lines 279–283)

```python
if final.exists():
    shutil.rmtree(final)
partial = root / name / f"{version}.partial-{os.getpid()}"
shutil.rmtree(partial, ignore_errors=True)
partial.mkdir(parents=True)
```

Reaching here means the marker was absent. If the folder exists anyway, it is a
**failed previous attempt** — delete it. This is the line that makes problem 2
impossible: junk is removed rather than served.

`shutil.rmtree` deletes a folder and everything in it (`rm -rf`).

Then a staging folder is created, named with the **process id**:

```
cohere-asr/3.partial-4821
```

`os.getpid()` returns the current process id, unique among running processes.
So two programs downloading the same version at the same moment write to
different folders and cannot corrupt each other.

`ignore_errors=True` means "delete it if it exists, say nothing if it does
not" — avoiding a try/except around a best-effort cleanup.

`mkdir(parents=True)` creates intermediate directories as needed (`mkdir -p`).

### Step 4 — download, then mark (lines 285–292)

```python
got = Path(self._mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(partial)))
(got / _COMPLETE_MARKER).write_text(json.dumps({
    "name": name,
    "version": version,
    "alias": alias,
    "run_id": run_id,
    "resolved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}))
```

The download goes into the staging folder. `download_artifacts` returns the path
it actually wrote to — which may be nested one level deeper than `dst_path`,
hence capturing it as `got` rather than assuming.

Then the marker is written. **The order is the entire safety property.** Files
first, marker last. If the process dies at any point during the download, the
marker is never written, and step 3 on the next run will delete the remains.

The marker's contents are small but useful when debugging: which version this
is, which alias asked for it, which run produced it, when it was fetched. A
future cache-cleanup command will read `resolved_at` to decide what to evict.

### Step 5 — move it into place (lines 294–300)

```python
if (final / _COMPLETE_MARKER).is_file():
    # Another process finished the same version while we downloaded.
    shutil.rmtree(partial, ignore_errors=True)
    return final, True
os.replace(got, final)
shutil.rmtree(partial, ignore_errors=True)
return final, False
```

The check is repeated because **25 seconds have passed**. Another process may
have finished the same version meanwhile. If so, discard our copy and use theirs
— identical files, and no reason to fight over the destination.

Then:

```python
os.replace(got, final)
```

`os.replace` renames a path. It is **atomic** at the operating-system level:
either it fully happens or it does not. There is no instant at which `final`
exists in a half-built state.

This matters because another process may check `final` at any moment. With a
copy-then-delete approach, it could observe a partly written folder. With an
atomic rename, `final` goes from "absent" to "complete" with nothing in between.

> **The write-then-rename pattern** — build in a temporary location, then rename
> into place — is the standard way to update a file or folder safely. It works
> because rename within a filesystem is atomic, while writing is not.

## Why not use a lock?

A file lock would also prevent two processes downloading simultaneously. The
marker-plus-rename approach was chosen because:

- **Locks need cleanup.** A process killed while holding one leaves a stale lock
  that blocks everyone until removed by hand.
- **Duplicate work is cheap here.** Two processes downloading the same version
  wastes bandwidth once. Compared with a stuck service, that is a good trade.
- **No coordination is required.** Nothing has to agree on lock file locations
  or timeouts. The filesystem provides the guarantee directly.

The cost is honest: occasionally, two processes download the same 4 GB. Rare, and
recoverable.

## What is missing

**Nothing is ever deleted.** Promote through versions 1 to 10 and all ten
folders remain — potentially 40 GB.

That is a known gap, not an oversight. A `prune` command (keep aliased versions
plus the two most recent) is written up as a task in
`docs/superpowers/plans/`. Until then, cache growth is manual housekeeping.

There is also no verification beyond completion. A download that finished but
was corrupted in transit would pass. Checksums would catch it; MLflow does not
expose per-file hashes conveniently, and completion covers the failure mode that
actually occurred.

## The tests that hold this together

Four tests in `tests/test_registry.py` cover this logic. The important one:

```python
def test_incomplete_slot_is_discarded_and_refetched(reg, model_dir, tmp_path):
    """A slot without the completion marker is a half-finished download."""
    v1 = reg.register("m", model_dir)
    reg.promote("m", v1, "production")
    slot = tmp_path / "cache" / "m" / v1
    slot.mkdir(parents=True)
    (slot / "model.bin").write_bytes(b"trunc")      # partial file, no marker
    (slot / "garbage.tmp").write_bytes(b"x")

    resolved = reg.resolve("m", "production", cache_dir=tmp_path / "cache")
    assert resolved.from_cache is False             # it refetched
    assert (slot / "model.bin").read_bytes() == b"weights"   # real content now
    assert not (slot / "garbage.tmp").exists()      # junk gone
```

It fabricates exactly the dangerous situation — a folder with a truncated file
and no marker — and proves the code throws it away rather than serving it.

Another proves a cache hit performs **no download at all**, by replacing the
download function with one that fails if called:

```python
def no_download(*a, **k):
    raise AssertionError("download_artifacts must not be called on a cache hit")

monkeypatch.setattr(reg._mlflow.artifacts, "download_artifacts", no_download)
second = reg.resolve("m", "production", cache_dir=tmp_path / "cache")
assert second.from_cache is True
```

Asserting on *behaviour* (nothing was downloaded) rather than on timing is what
makes this test reliable rather than flaky.

---

## What you now know

- A partial download that still "loads" is the failure mode worth designing
  against.
- Trust comes from a **marker written after** the files, never from existence.
- Layout is per-version, so rollback is free and aliases share folders.
- Staging is per-process (`os.getpid()`); the move is atomic (`os.replace`).
- Write-then-rename is the general safe-update pattern.
- Nothing is evicted yet — a known, documented gap.

---

[← Previous](06-resolve.md) · [Contents](README.md) · [Next: Questions you can ask →](08-queries.md)
