# Chapter 12 — The tests

[← Previous](11-the-consumer.md) · [Contents](README.md) · [Next: Infrastructure →](13-infrastructure.md)

---

**Files:** `tests/test_registry.py` (222 lines, 15 tests),
`tests/test_register_from_hf.py` (145 lines, 12 tests).

27 tests, about five seconds, **no servers, no Docker, no GPU, no network**.

```bash
$ make test
...........................                                              [100%]
27 passed in 4.89s
```

That "no servers" claim is the interesting part. This project needs Postgres and
cloud storage to work — yet its tests need neither. This chapter explains how,
and why the tests are also the best documentation in the repository.

## Tests as documentation

An unfamiliar codebase is often easier to enter through its tests than its
source. Each test is small, complete, and *runs* — so it cannot drift out of
date the way a comment can.

If you read one thing in the test suite, read this:

```python
def test_alias_decoupling(reg, model_dir):
    """The consumer call is identical before and after promotion."""
    v1 = reg.register("m", model_dir)
    v2 = reg.register("m", model_dir)

    reg.promote("m", v1, "production")
    assert reg.resolve("m", "production").version == v1

    # Deploy a new model: repoint the alias. No consumer code changes.
    reg.promote("m", v2, "production")
    assert reg.resolve("m", "production").version == v2
```

Six lines containing the entire thesis of the project. The consumer call —
`resolve("m", "production")` — is **character-for-character identical** on both
sides. Only the answer changed. That is the decoupling, demonstrated rather than
asserted in prose.

## Fixtures: the setup that makes it possible

```python
@pytest.fixture
def reg(tmp_path, monkeypatch):
    """A registry backed by a throwaway sqlite db + artifact dir."""
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    # Keep mlruns/ out of the repo, and the resolve cache out of ~/.cache.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_REGISTRY_CACHE", str(tmp_path / "default-cache"))

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    registry = ModelRegistry(tracking_uri)
    mlflow.set_experiment("test")
    return registry
```

A **fixture** is setup code pytest runs before a test. A test asks for one by
naming it as a parameter:

```python
def test_rollback(reg, model_dir):    # pytest supplies both
```

That is dependency injection: the test declares what it needs and pytest builds
it. No setup code inside the test, no shared global state.

### `sqlite:///` — the trick that removes the servers

```python
tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
```

In production, MLflow stores metadata in **Postgres** and files in **GCS**. Here
it stores metadata in **SQLite** — a complete database inside a single file, with
no server process — and files in a local folder.

MLflow supports both, and our code never knows the difference. It talks to a
tracking URI; whether that points at a server or a file is MLflow's problem.

This is the facade paying off a third time. Chapter 3 said hiding MLflow means
the backend can change; the storage swap proved it in production; and the tests
prove it again — an entirely different storage engine, same code under test.

### `tmp_path` — a fresh folder per test

`tmp_path` is a pytest built-in: a unique empty directory for **each test**.
Combined with the SQLite URI, every test gets its own private database.

Consequences worth appreciating:

- Tests cannot affect each other, no matter what they write.
- Tests can run in parallel.
- No cleanup code — the operating system reclaims the folder.
- A failing test leaves its folder behind for inspection.

### `monkeypatch` — changes that undo themselves

```python
monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
monkeypatch.chdir(tmp_path)
monkeypatch.setenv("MODEL_REGISTRY_CACHE", str(tmp_path / "default-cache"))
```

`monkeypatch` modifies environment variables, attributes, or the working
directory, and **reverses every change when the test ends**.

Doing this by hand would mean saving the old value and restoring it in a
`finally` block — for each variable, in each test. Forget once and later tests
inherit the change, producing failures that depend on test order. Those are
among the most frustrating bugs to diagnose.

The two comments explain what each line prevents:

- `chdir(tmp_path)` — MLflow writes an `mlruns/` folder into the current
  directory. Without this, running tests would litter the repository.
- `MODEL_REGISTRY_CACHE` — otherwise `resolve()` would use the real
  `~/.cache/model-registry`, mixing test data into your actual cache.

### The second fixture

```python
@pytest.fixture
def model_dir(tmp_path):
    """A directory that looks like a set of pretrained weights."""
    d = tmp_path / "fake_model"
    d.mkdir()
    (d / "model.bin").write_bytes(b"weights")
    (d / "config.json").write_text(json.dumps({"arch": "whisper"}))
    return d
```

A pretend model: two tiny files. The registry does not care what is inside — it
moves files. So a 7-byte `model.bin` exercises exactly the same code path a 4 GB
one would, in microseconds.

> **A principle worth generalising:** test with the smallest input that
> exercises the logic. Size belongs in a separate performance test, not in
> every unit test.

## What the tests cover

### The contract

| Test | Proves |
|---|---|
| `test_register_returns_version` | returns `"1"`, and it is a `str` |
| `test_alias_decoupling` | repointing changes what a consumer gets |
| `test_rollback` | promoting backwards works |
| `test_pin_exact_version` | resolving by number gives `alias == ""` |
| `test_resolve_downloads_files` | the files actually arrive |
| `test_list_versions_and_models` | listing works |
| `test_list_versions_reports_aliases` | **the MLflow quirk from Chapter 8** |
| `test_find_version_by_tag` | importer idempotency works |

The seventh is a **regression test** — written after a bug, to stop it
returning:

```python
def test_list_versions_reports_aliases(reg, model_dir):
    """Regression: search_model_versions returns empty aliases, so list_versions
    has to source them from the registered model instead."""
```

The docstring records *why the test exists*. Without it, a future reader might
see redundant coverage and delete it, reopening the bug.

### The failure paths

The most valuable tests are the ones asserting what happens when things go
wrong.

```python
def test_failed_upload_leaves_no_model_shell(reg, model_dir, monkeypatch):
    """If artifact upload dies, nothing must appear in the registry. Observed
    for real: two killed imports left empty registered models behind."""
    def boom(*args, **kwargs):
        raise OSError("simulated upload failure")

    # reg._mlflow is the global mlflow module — patch via monkeypatch so the
    # stub is undone after this test instead of leaking into every later one.
    monkeypatch.setattr(reg._mlflow, "log_artifacts", boom)
    with pytest.raises(OSError):
        reg.register("never-registered", model_dir)

    assert "never-registered" not in reg.list_models()
```

This is the Chapter 4 ordering bug, locked down.

`boom` replaces `log_artifacts` with a function that always fails —
simulating a killed upload without needing to kill anything.
`*args, **kwargs` means "accept any arguments", so it can substitute for a
function of any signature.

`with pytest.raises(OSError):` asserts that the block **does** raise. If
`register()` somehow succeeded, the test fails.

The real assertion is the last line: after the failure, the registry is clean.

### The bug the tests themselves had

That comment about `monkeypatch` is there because of a mistake I made writing
these tests. The original was:

```python
reg._mlflow.log_artifacts = boom        # ✗ leaks
```

Direct assignment. It works within the test — and then never undoes itself.
`reg._mlflow` is not a copy; it *is* the global `mlflow` module, shared by every
test in the process. So `log_artifacts` stayed broken for the rest of the run,
and **six later tests failed** with `OSError: simulated upload failure` — a
message that has nothing to do with what those tests were checking.

Debugging that means noticing the failures are downstream of an earlier test,
which is not obvious when you are reading the failure you can see.

`monkeypatch.setattr` fixes it by restoring the original afterwards.

> **The lesson:** if a test modifies anything outside itself — environment,
> module, working directory, global — use a tool that undoes the change. Manual
> cleanup will eventually be forgotten, and the resulting failure will appear
> somewhere else entirely.

### The cache tests

Four tests for Chapter 7's logic. The sharpest asserts an **absence**:

```python
def test_second_resolve_is_served_from_cache(reg, model_dir, tmp_path, monkeypatch):
    v1 = reg.register("m", model_dir)
    reg.promote("m", v1, "production")
    first = reg.resolve("m", "production", cache_dir=tmp_path / "cache")
    assert first.from_cache is False

    def no_download(*a, **k):
        raise AssertionError("download_artifacts must not be called on a cache hit")

    monkeypatch.setattr(reg._mlflow.artifacts, "download_artifacts", no_download)
    second = reg.resolve("m", "production", cache_dir=tmp_path / "cache")
    assert second.from_cache is True
```

How do you prove no download happened? The obvious approach — measure the time —
is **flaky**: a slow machine fails a correct implementation.

Instead the download function is replaced with one that fails if called. If the
cache does not work, the test fails loudly and specifically. If it works, the
function is never touched.

> **Assert on behaviour, not on timing.** Timing assertions are the most common
> source of tests that pass locally and fail in CI.

And the dangerous case, fabricated deliberately:

```python
def test_incomplete_slot_is_discarded_and_refetched(reg, model_dir, tmp_path):
    """A slot without the completion marker is a half-finished download."""
    ...
    slot.mkdir(parents=True)
    (slot / "model.bin").write_bytes(b"trunc")      # partial file, no marker
    (slot / "garbage.tmp").write_bytes(b"x")

    resolved = reg.resolve("m", "production", cache_dir=tmp_path / "cache")
    assert resolved.from_cache is False
    assert (slot / "model.bin").read_bytes() == b"weights"
    assert not (slot / "garbage.tmp").exists()
    assert (slot / ".registry-complete").is_file()
```

A truncated file and stray junk, with no marker. The assertions prove the cache
threw it away, refetched, and left nothing behind.

## The importer tests

`tests/test_register_from_hf.py` tests the four pure helpers from Chapter 10 —
no network needed, because those functions have none.

```python
def test_selected_exclude_wins_over_include():
    # The gipformer repo ships the ONNX files AND a 279 MB training
    # checkpoint; and mms-lid ships the same weights as .bin and .safetensors.
    assert not rfh.selected("pytorch_model.bin", None, ["*.bin"])
    assert rfh.selected("model.safetensors", None, ["*.bin"])
    assert not rfh.selected("encoder.onnx", ["*.onnx"], ["encoder.*"])
```

The comment names the **real repositories** that motivated the rule. When
someone later wonders whether exclude-beats-include is correct, the answer is
right there.

```python
def test_apply_patches_refuses_when_target_missing(tmp_path):
    """Upstream changed the file: fail loudly rather than register an
    unpatched snapshot that only breaks at load time."""
    (tmp_path / "cfg.yaml").write_text("something else entirely\n")
    with pytest.raises(RuntimeError, match="patch target not found"):
        rfh.apply_patches(tmp_path, [{"file": "cfg.yaml", "replace": {"old": "absent", "new": "x"}}])
```

`match="..."` checks the message too — so a future refactor that raises a
*different* `RuntimeError` does not accidentally satisfy the test.

```python
def test_load_catalog_rejects_duplicate_keys(tmp_path):
    """A duplicated key must fail loudly — PyYAML's default silently keeps the
    last value, which is how models.yaml once shipped two `exclude:` lines."""
```

Another regression test, with the incident recorded in the docstring.

## What is *not* tested

Worth being explicit, because a test count can imply more coverage than exists.

**Nothing that needs a network** — the real HuggingFace download, GCS transfer,
Postgres. Those were verified by hand: real imports were run, and the resulting
versions resolved back and inspected (the patched file was checked for the new
class path, and the gipformer version for the exact filenames the loader wants).

**Nothing that needs Docker** — the compose file is validated by `docker compose
config` in CI, which checks syntax, not behaviour.

**No performance assertions** — no test says a cache hit is fast, only that it
downloads nothing.

That split is deliberate: unit tests cover logic, and integration is verified
deliberately and occasionally. Chasing a network test that runs everywhere would
cost more than it returns.

## Running them

```bash
make test                                         # all 27
.venv/bin/python -m pytest tests/ -v              # names of each
.venv/bin/python -m pytest tests/test_registry.py::test_alias_decoupling -v
.venv/bin/python -m pytest tests/ -k cache        # only cache tests
.venv/bin/python -m pytest tests/ -x              # stop at first failure
```

`-k` matches substrings of test names, which is the quickest way to run the
handful you care about while working.

---

## What you now know

- SQLite plus `tmp_path` removes every external dependency from the tests.
- Fixtures inject setup; `monkeypatch` guarantees it is undone.
- Assigning to a module attribute directly leaks across tests — a bug that hit
  this suite.
- Assert on behaviour, not timing.
- Regression tests should record the incident in their docstring.
- Network and Docker paths are verified by hand, not by tests, on purpose.

---

[← Previous](11-the-consumer.md) · [Contents](README.md) · [Next: Infrastructure →](13-infrastructure.md)
