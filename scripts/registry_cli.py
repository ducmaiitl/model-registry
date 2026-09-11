#!/usr/bin/env python3
"""Management CLI for the model registry.

    python scripts/registry_cli.py list
    python scripts/registry_cli.py versions --name whisper-stt
    python scripts/registry_cli.py promote --name whisper-stt --version 2
    python scripts/registry_cli.py resolve --name whisper-stt --ref production
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from registry import ModelRegistry  # noqa: E402


def main():
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

    args = ap.parse_args()
    reg = ModelRegistry()

    if args.cmd == "list":
        for name in reg.list_models():
            print(name)

    elif args.cmd == "versions":
        print(json.dumps(reg.list_versions(args.name), indent=2))

    elif args.cmd == "promote":
        reg.promote(args.name, args.version, args.alias)
        print(f"{args.name} v{args.version} -> @{args.alias.lower()}")

    elif args.cmd == "resolve":
        r = reg.resolve(args.name, args.ref, cache_dir=args.cache_dir)
        print(json.dumps({
            "name": r.name,
            "version": r.version,
            "alias": r.alias,
            "local_path": str(r.local_path),
            "metrics": r.metrics,
            "resolve_seconds": round(r.resolve_seconds, 4),
        }, indent=2))


if __name__ == "__main__":
    main()
