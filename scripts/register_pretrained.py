#!/usr/bin/env python3
"""Register a pretrained model directory as a new version.

No finetuning involved: this just takes a folder of downloaded weights and puts
it in the registry. When finetuning lands later the path is identical — the only
difference is that --metric gets real numbers instead of being omitted.

    python scripts/register_pretrained.py \
        --name whisper-stt --source-dir ./weights/whisper-small \
        --alias production --tag base_model=openai/whisper-small
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from registry import ModelRegistry  # noqa: E402


def _kv(pairs, cast=str):
    """Parse repeated key=value flags into a dict."""
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"expected key=value, got: {item!r}")
        key, value = item.split("=", 1)
        out[key] = cast(value)
    return out


def main():
    ap = argparse.ArgumentParser(description="Register a pretrained model.")
    ap.add_argument("--name", required=True, help="registered model name")
    ap.add_argument("--source-dir", required=True, help="directory of weight files")
    ap.add_argument("--alias", default=None, help="promote to this alias after registering")
    ap.add_argument("--desc", default="", help="version description")
    ap.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE",
                    help="version tag (repeatable)")
    ap.add_argument("--metric", action="append", default=[], metavar="KEY=VALUE",
                    help="metric (repeatable; pretrained models often have none)")
    args = ap.parse_args()

    tags = _kv(args.tag)
    try:
        metrics = _kv(args.metric, cast=float)
    except ValueError:
        raise SystemExit("--metric values must be numbers")

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


if __name__ == "__main__":
    main()
