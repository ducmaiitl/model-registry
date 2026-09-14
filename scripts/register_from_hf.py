#!/usr/bin/env python3
"""Import HuggingFace models into the registry, pinned to an exact commit.

Why this exists: every robo-be worker hardcodes an HF repo id and loads
whatever is at HEAD. Upstream repos change under you — the gipformer repo
renamed every weight file on 2026-08-21 and the worker code still asks for
the old names; it only runs where a stale cache survives. This script
freezes a snapshot at one commit, applies any patch the consumer needs
(VoxLingua's SpeechBrain class-path fix), and registers it with the repo
and commit SHA as tags. After that the registry owns the files and HF Hub
stops being a runtime dependency.

    # One model
    python scripts/register_from_hf.py \\
        --repo g-group-ai-lab/gipformer-65M-rnnt --name gipformer-asr-vi \\
        --revision ba00dad1 --alias production \\
        --include "*.onnx" --include tokens.txt

    # The whole catalog. Idempotent: a version whose hf_revision is already
    # registered is skipped, so re-running is safe.
    python scripts/register_from_hf.py --catalog models.yaml
    python scripts/register_from_hf.py --catalog models.yaml --only gipformer-asr-vi --dry-run

Gated repos (Cohere) need HF_TOKEN in the environment or a cached
`huggingface-cli login`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from registry import ModelRegistry  # noqa: E402

# Tag that makes re-runs idempotent: one registry version per upstream commit.
REVISION_TAG = "hf_revision"


# -- pure helpers (unit-tested, no network) ----------------------------------

def selected(path: str, include: list[str] | None, exclude: list[str]) -> bool:
    """Same semantics as snapshot_download's allow/ignore patterns, so the
    dry-run listing matches what a real download would fetch."""
    if include and not any(fnmatch(path, pat) for pat in include):
        return False
    return not any(fnmatch(path, pat) for pat in exclude)


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


def strip_download_metadata(model_dir: Path) -> None:
    """Remove huggingface_hub's bookkeeping from a snapshot before registering.

    snapshot_download(local_dir=...) writes a `.cache/huggingface/` directory
    of download metadata next to the files. It is not part of the model and
    would otherwise be uploaded as artifact junk on every version.
    """
    shutil.rmtree(model_dir / ".cache", ignore_errors=True)


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


def parse_kv(pairs: list[str]) -> dict[str, str]:
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"expected key=value, got: {item!r}")
        key, value = item.split("=", 1)
        out[key] = value
    return out


# -- import ------------------------------------------------------------------

def import_model(
    reg: ModelRegistry,
    spec: dict,
    *,
    staging_root: Path,
    token: Optional[str],
    dry_run: bool,
    keep: bool,
) -> Optional[str]:
    """Register one HF snapshot. Returns the version, or None on dry-run."""
    from huggingface_hub import HfApi, snapshot_download

    repo, name = spec["repo"], spec["name"]
    api = HfApi(token=token)

    # Pin to a full commit SHA even when the catalog gives a short one or a
    # branch, so the tag on the version is an exact, reproducible pointer.
    sha = api.model_info(repo, revision=spec.get("revision")).sha
    include = spec.get("include") or None
    exclude = spec.get("exclude") or []
    files = [f for f in api.list_repo_files(repo, revision=sha) if selected(f, include, exclude)]

    print(f"\n== {name}  <-  {repo} @ {sha[:8]}")
    if not files:
        raise RuntimeError(f"{name}: include/exclude patterns selected zero files")

    existing = reg.find_version_by_tag(name, REVISION_TAG, sha)
    if existing:
        # Deliberately do NOT touch aliases here. If someone rolled
        # @production back to an older version, a catalog re-run must not
        # silently undo that. Aliases are only set on a fresh registration.
        print(f"   already registered as version {existing} — skipping")
        return existing

    for f in files:
        print(f"   + {f}")
    if dry_run:
        print("   (dry run — nothing downloaded)")
        return None

    dest = staging_root / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    print(f"   downloading {len(files)} files -> {dest}")
    snapshot_download(
        repo_id=repo,
        revision=sha,
        allow_patterns=include,
        ignore_patterns=exclude or None,
        local_dir=str(dest),
        token=token,
    )
    strip_download_metadata(dest)
    apply_patches(dest, spec.get("patches") or [])

    tags = {
        "source": "huggingface",
        "hf_repo": repo,
        REVISION_TAG: sha,
        "base_model": repo,
        **spec.get("tags", {}),
    }
    version = reg.register(
        name=name,
        source_dir=dest,
        tags=tags,
        description=spec.get("description") or f"{repo} @ {sha[:8]}",
    )
    print(f"   registered {name} version {version}")

    alias = spec.get("alias")
    if alias:
        reg.promote(name, version, alias)
        print(f"   promoted v{version} -> @{alias.lower()}")

    if not keep:
        shutil.rmtree(dest, ignore_errors=True)
    return version


def main() -> None:
    ap = argparse.ArgumentParser(description="Import HF models into the registry.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="HF repo id, e.g. openbmb/VoxCPM2")
    src.add_argument("--catalog", type=Path, help="models.yaml with a list of models")

    ap.add_argument("--name", help="registry model name (required with --repo)")
    ap.add_argument("--revision", help="commit SHA / branch / tag (default: HEAD)")
    ap.add_argument("--alias", help="promote to this alias after registering")
    ap.add_argument("--include", action="append", default=[], metavar="GLOB")
    ap.add_argument("--exclude", action="append", default=[], metavar="GLOB")
    ap.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--desc", default="")
    ap.add_argument("--only", action="append", default=[], metavar="NAME",
                    help="with --catalog: only these entries (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="resolve + list files, no download")
    ap.add_argument("--keep", action="store_true", help="keep the staging download after registering")
    ap.add_argument("--staging-dir", type=Path, default=Path("cache/hf-staging"))
    ap.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF token (default: $HF_TOKEN)")
    args = ap.parse_args()

    if args.catalog:
        specs = load_catalog(args.catalog)
        if args.only:
            wanted = set(args.only)
            unknown = wanted - {s["name"] for s in specs}
            if unknown:
                raise SystemExit(f"--only names not in catalog: {sorted(unknown)}")
            specs = [s for s in specs if s["name"] in wanted]
    else:
        if not args.name:
            ap.error("--name is required with --repo")
        specs = [{
            "name": args.name,
            "repo": args.repo,
            "revision": args.revision,
            "alias": args.alias,
            "include": args.include,
            "exclude": args.exclude,
            "tags": parse_kv(args.tag),
            "description": args.desc,
        }]

    reg = ModelRegistry()
    failed = []
    for spec in specs:
        try:
            import_model(
                reg, spec,
                staging_root=args.staging_dir, token=args.token,
                dry_run=args.dry_run, keep=args.keep,
            )
        except Exception as exc:  # keep going; report at the end
            print(f"   FAILED: {type(exc).__name__}: {exc}")
            failed.append(spec["name"])

    if failed:
        raise SystemExit(f"\n{len(failed)} failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
