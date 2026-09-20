"""Validate and summarize an OptiVox dataset manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.data_registry import DatasetRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an OptiVox dataset manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--require-files", action="store_true")
    args = parser.parse_args()
    try:
        registry = DatasetRegistry.load(args.manifest, root=args.root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        return 2
    issues = registry.validate(require_files=args.require_files)
    result = registry.summary()
    result["validation"] = {"status": "PASS" if not issues else "FAIL", "issues": issues}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
