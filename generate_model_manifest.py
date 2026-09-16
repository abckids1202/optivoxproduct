"""Generate a local checksum manifest for OptiVox model artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from deployment_security import MODEL_SUFFIXES, build_model_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("models/model_checksums.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    candidates = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES]
    manifest = build_model_manifest(candidates, root, args.output)
    print(f"Wrote {len(manifest['files'])} model checksum(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

