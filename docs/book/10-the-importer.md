# Chapter 10 — The importer and the catalogue

[← Previous](09-scripts-and-cli.md) · [Contents](README.md) · [Next: A consumer →](11-the-consumer.md)

---

**Files:** `scripts/register_from_hf.py` (281 lines), `models.yaml` (120 lines).

The largest script in the repository, and the one that solves Problem 4 from
Chapter 1 — *the ground moves under you*.

## The problem, concretely

HuggingFace is a website hosting AI models. Code names a model by its repository
id, and a library downloads whatever is there:

```python
model = AutoModel.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
```

"Whatever is there" is the trouble. A repository is like a git repository: it has
commits, and its owner can push new ones whenever they like.

On **2026-08-21**, the owner of `g-group-ai-lab/gipformer-65M-rnnt` pushed a
commit titled *"Remove legacy epoch-35-avg-6 filenames"*. It renamed every
weight file. Code asking for the old names now gets a 404.

The speech worker requesting those names still runs — only because that machine
cached the files before August. Start it anywhere fresh and it fails. Nothing in
our code changed.

This script makes that impossible: download **one exact commit**, store the files
ourselves, record which commit it was.

## Usage

```bash
# everything in the catalogue (safe to re-run)
python scripts/register_from_hf.py --catalog models.yaml

# look but do not download
python scripts/register_from_hf.py --catalog models.yaml --dry-run

# just one entry
python scripts/register_from_hf.py --catalog models.yaml --only gipformer-asr-vi

# a one-off, no catalogue
python scripts/register_from_hf.py --repo openbmb/VoxCPM2 --name voxcpm2-tts
```

## `models.yaml` — the catalogue

Data, not code. It declares every model the system uses.

```yaml
defaults:
  exclude: ["*.h5", "*.msgpack", "*.ot", "*.tflite", "*.png"]
  tags:
    registered_by: catalog
    project: robo-be

models:
  - name: cohere-asr
    repo: CohereLabs/cohere-transcribe-03-2026
    revision: b1eacc2686a3d08ceaae5f24a88b1d519620bc09
    alias: production
    exclude: [".eval_results/*", "assets/*", "demo/*"]
    tags: {framework: transformers, role: asr, lang: vi+en}
```

| Field | Meaning |
|---|---|
| `name` | what it is called in our registry |
| `repo` | where it comes from upstream |
| `revision` | **the exact commit** — the whole point |
| `alias` | which alias to set, on first registration only |
| `include` | whitelist of files (omitted = everything) |
| `exclude` | files to skip, added to the defaults |
| `patches` | text edits to apply before registering |
| `tags` | labels recorded on the version |

`defaults` applies to every entry, so common exclusions are written once.

### This file is also documentation

Before it existed, *"what models does this system use?"* meant reading five
source files and a Dockerfile. Now it is one file — and it holds the reasoning
too. The most valuable comment in the repository lives here:

```yaml
  - name: gipformer-asr-vi
    repo: g-group-ai-lab/gipformer-65M-rnnt
    # PINNED BEHIND HEAD ON PURPOSE. Upstream renamed every weight file on
    # 2026-08-21 (commit 29621ec8 "Remove legacy epoch-35-avg-6 filenames").
    # robo_infer/models/asr_gipformer.py still requests the old names, so
    # HEAD 404s on any machine without a pre-August cache. This revision is
    # what production actually has. Bump it together with the loader code.
    revision: ba00dad1f33ce28ceec5677fb5890b9ade0d9d50
```

Without that comment, the next person sees a version pinned behind the latest
and "helpfully" updates it — breaking production. The comment converts a
mysterious choice into an informed one.

### Keeping only what is needed

```yaml
  - name: mms-lid-256
    # Repo ships identical weights as pytorch_model.bin AND model.safetensors
    # (3.9 GB each). Keep one.
    include: ["model.safetensors", "*.json", "langs.txt"]
```

Many repositories ship the same weights in several formats for different
frameworks. Downloading both doubles storage for nothing. `include` keeps one.

Similarly `gipformer-asr-vi` uses `include` to skip a 279 MB training
checkpoint that inference never touches.

### Patches — a fix that travels with the model

```yaml
  - name: voxlingua107-lid
    patches:
      - file: inference_wav2vec.yaml
        replace:
          old: speechbrain.lobes.models.huggingface_wav2vec.HuggingFaceWav2Vec2
          new: speechbrain.lobes.models.huggingface_transformers.wav2vec2.Wav2Vec2
```

This model ships a config file naming a Python class that **moved** in
SpeechBrain 1.0. Upstream has not updated it, so loading fails unless the file is
edited.

That edit used to live in a Dockerfile — meaning the fix applied only if you
built the image that way, and was invisible to anyone reading the model. Now it
is part of the registered version: the stored files are already correct, and the
reason is recorded next to them.

## The script — four pure helpers

"Pure" means: no network, no database, same input → same output. Easy to test,
easy to reason about. All four are tested in `tests/test_register_from_hf.py`.

### `selected()` (line 48) — which files to download

```python
def selected(path: str, include: list[str] | None, exclude: list[str]) -> bool:
    """Same semantics as snapshot_download's allow/ignore patterns, so the
    dry-run listing matches what a real download would fetch."""
    if include and not any(fnmatch(path, pat) for pat in include):
        return False
    return not any(fnmatch(path, pat) for pat in exclude)
```

`fnmatch` does shell-style glob matching: `*.onnx` matches `encoder.onnx`.

The logic is *whitelist first, then blacklist*:

1. If `include` is given and the file matches none of it → **no**.
2. If the file matches anything in `exclude` → **no**.
3. Otherwise → **yes**.

So exclude beats include. `any(...)` returns `True` if any item matches, and
short-circuits on the first hit.

The docstring explains why this mirrors the HuggingFace library's own rules:
`--dry-run` must list exactly what a real run would fetch. A dry run that lies is
worse than no dry run.

### `apply_patches()` (line 56) — edit files, or refuse

```python
def apply_patches(model_dir: Path, patches: list[dict]) -> None:
    """Apply declared text substitutions to files inside the snapshot.

    A patch whose target string is missing is an error, not a no-op: it
    means upstream changed, and registering a silently-unpatched model would
    ship a version that fails at load time with a confusing traceback.
    """
    for patch in patches or []:
        target = model_dir / patch["file"]
        old, new = patch["replace"]["old"], patch["replace"]["new"]
        text = target.read_text()
        if old not in text:
            raise RuntimeError(
                f"patch target not found in {patch['file']}: {old!r} — "
                "upstream changed; refusing to register an unpatched snapshot"
            )
        target.write_text(text.replace(old, new))
```

Mechanically it is find-and-replace. The judgement is in the `raise`.

The tempting alternative is to skip quietly when the text is absent — after all,
maybe upstream already fixed it. The cost of being wrong is severe: a version
gets registered that looks fine, sits in the registry, and fails at load time
with an error pointing at SpeechBrain rather than at the import.

Failing here is loud, immediate, and points at the real cause. **A patch that
silently does nothing is a bug generator.**

### `strip_download_metadata()` (line 75)

```python
def strip_download_metadata(model_dir: Path) -> None:
    """Remove huggingface_hub's bookkeeping from a snapshot before registering.

    snapshot_download(local_dir=...) writes a `.cache/huggingface/` directory
    of download metadata next to the files. It is not part of the model and
    would otherwise be uploaded as artifact junk on every version.
    """
    shutil.rmtree(model_dir / ".cache", ignore_errors=True)
```

Three lines, found by inspecting a real registered version and noticing a
`.cache/` folder that had been faithfully uploaded along with the weights.
Harmless, but noise in every version forever.

### `_strict_yaml_load()` (line 89) — reject duplicate keys

```python
class DuplicateKeyError(ValueError):
    pass


def _strict_yaml_load(text: str):
    """yaml.safe_load, but a repeated mapping key is an error instead of
    silently keeping the last value. The catalog is hand-edited; a duplicated
    `exclude:` would otherwise change what gets stored with no warning."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in seen:
                raise DuplicateKeyError(
                    f"duplicate key {key!r} at line {key_node.start_mark.line + 1}"
                )
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep)

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    return yaml.load(text, Loader=Loader)
```

The most advanced Python in the repository, written because of a real bug.

**What happened:** `models.yaml` ended up with `exclude:` written twice in the
same entry. Standard YAML keeps the last one and says nothing. The catalogue was
quietly excluding a different set of files than it appeared to.

**What this does:** subclasses PyYAML's loader and replaces how it builds
mappings. Before constructing a dict, it walks the keys, tracks them in a `set`,
and raises on a repeat — with the line number, from
`key_node.start_mark.line + 1` (`+ 1` because YAML counts from zero and humans
count from one).

`DuplicateKeyError` subclasses `ValueError` so callers can catch either the
specific error or the general category.

A custom `class Loader(yaml.SafeLoader): pass` is created rather than modifying
`SafeLoader` directly — otherwise every other YAML parse in the process would
inherit the strictness, which is not ours to impose.

> **The pattern:** when a library's default is "silently do something
> surprising", and correctness depends on it, override the default rather than
> relying on everyone remembering.

### `load_catalog()` (line 113) — merge defaults

```python
def load_catalog(path: Path) -> list[dict]:
    """Parse models.yaml, folding `defaults` into each entry."""
    doc = _strict_yaml_load(path.read_text()) or {}
    defaults = doc.get("defaults") or {}
    specs = []
    for entry in doc.get("models") or []:
        spec = dict(entry)
        spec["exclude"] = list(defaults.get("exclude") or []) + list(entry.get("exclude") or [])
        spec["tags"] = {**(defaults.get("tags") or {}), **(entry.get("tags") or {})}
        specs.append(spec)
    return specs
```

Two different merge strategies, each chosen for its field:

**`exclude` — concatenate.** Defaults and entry excludes are *added together*:

```python
["*.h5", "*.png"] + ["demo/*"] == ["*.h5", "*.png", "demo/*"]
```

An entry adds exclusions; it cannot remove the defaults.

**`tags` — override.** `{**a, **b}` unpacks both dicts into a new one, and later
keys win:

```python
{**{"team": "stt"}, **{"team": "tts"}} == {"team": "tts"}
```

So an entry can override a default tag.

The difference is intentional: excluding junk is cumulative, while labelling
something is specific.

## The main flow: `import_model()` (line 138)

```
1. resolve the exact commit SHA
2. list the files, filtered by include/exclude
3. already registered this SHA? → skip
4. --dry-run? → print and stop
5. download to a staging folder
6. strip metadata, apply patches
7. register with provenance tags
8. promote, if an alias is declared and this is a fresh registration
9. delete the staging folder
```

### Step 1 — pin the commit

```python
sha = api.model_info(repo, revision=spec.get("revision")).sha
```

Even when the catalogue gives a short SHA or a branch name, this resolves to the
**full 40-character commit hash**. The tag written on the version is therefore an
exact, permanent pointer — not `main`, which moves.

### Step 3 — the idempotency check

```python
existing = reg.find_version_by_tag(name, REVISION_TAG, sha)
if existing:
    # Deliberately do NOT touch aliases here. If someone rolled
    # @production back to an older version, a catalog re-run must not
    # silently undo that. Aliases are only set on a fresh registration.
    print(f"   already registered as version {existing} — skipping")
    return existing
```

This is what makes re-running safe (`find_version_by_tag` from Chapter 8).

The comment records a subtle decision. Suppose version 3 was bad and someone
rolled `production` back to version 2. Later, someone re-runs the catalogue. If
a skip also re-applied the alias, it would silently push the broken version back
into production.

So: **aliases are set only when a version is newly created.** A re-run is
genuinely a no-op.

### Step 5 — download to staging

```python
dest = staging_root / name
if dest.exists():
    shutil.rmtree(dest)
dest.mkdir(parents=True)

snapshot_download(
    repo_id=repo, revision=sha,
    allow_patterns=include, ignore_patterns=exclude or None,
    local_dir=str(dest), token=token,
)
```

A temporary folder, cleared first so a previous failed attempt cannot contribute
stray files. Deleted at the end unless `--keep` was passed.

### Step 7 — record where it came from

```python
tags = {
    "source": "huggingface",
    "hf_repo": repo,
    REVISION_TAG: sha,
    "base_model": repo,
    **spec.get("tags", {}),
}
version = reg.register(name=name, source_dir=dest, tags=tags, description=...)
```

**These tags are the whole value of the importer.** Any version can now be
traced to the exact upstream commit it came from. `{**spec.get("tags", {})}`
merges the catalogue's tags in, allowing them to override.

### Errors do not stop the run (line 246)

```python
for spec in specs:
    try:
        import_model(reg, spec, ...)
    except Exception as exc:  # keep going; report at the end
        print(f"   FAILED: {type(exc).__name__}: {exc}")
        failed.append(spec["name"])

if failed:
    raise SystemExit(f"\n{len(failed)} failed: {', '.join(failed)}")
```

Importing eight models takes a long time. If the fifth fails — say, a gated
repository needing a licence acceptance — stopping would waste the three that
would have succeeded.

So failures are collected, the run continues, and the script **still exits
non-zero** at the end. Automation sees a failure; a human gets the successes.
`type(exc).__name__` prints the exception class (`GatedRepoError`), which is
usually the most informative part.

## `--dry-run`

```bash
$ python scripts/register_from_hf.py --catalog models.yaml --dry-run --only gipformer-asr-vi

== gipformer-asr-vi  <-  g-group-ai-lab/gipformer-65M-rnnt @ ba00dad1
   + bpe.model
   + config.json
   + decoder-epoch-35-avg-6.onnx
   ...
   (dry run — nothing downloaded)
```

Metadata only — no gigabytes. Because `selected()` implements the same rules the
downloader uses, this listing is exactly what a real run would fetch. Use it
before every real import.

---

## What you now know

- Pinning a commit SHA is what stops upstream changes breaking you.
- The catalogue is both configuration and documentation; its comments record
  decisions that would otherwise look like mistakes.
- Patches travel with the model rather than living in a Dockerfile.
- A patch that cannot find its target **fails** rather than passing silently.
- Duplicate YAML keys are rejected, because they were once accepted.
- Re-running is a no-op, and never re-applies an alias.
- Batch failures are collected, not fatal — but still exit non-zero.

---

[← Previous](09-scripts-and-cli.md) · [Contents](README.md) · [Next: A consumer →](11-the-consumer.md)
