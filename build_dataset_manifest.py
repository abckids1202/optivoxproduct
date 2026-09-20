"""Build a provenance-aware OptiVox dataset manifest from local media.

The command inventories files only; it never copies media, creates augmented
samples, or assumes that a source is consented. Use separate source folders
for train, validation, holdout, and adversarial evidence and run the validator
afterwards with ``--require-files``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import List

from core.data_registry import (
    DatasetRecord,
    DatasetRegistry,
    SUPPORTED_SPLITS,
    ALLOWED_PROVENANCE,
    sha256_file,
)


MEDIA_SUFFIXES = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".mp4", ".avi", ".mov", ".mkv",
})


def _record_id(relative_path: str, task: str) -> str:
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
    prefix = re.sub(r"[^a-z0-9]+", "-", task.casefold()).strip("-") or "sample"
    return f"{prefix}-{digest}"


def build_records(root: Path, task: str, split: str, provenance: str) -> List[DatasetRecord]:
    """Create deterministic records without reading or modifying media."""
    records: List[DatasetRecord] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.casefold() not in MEDIA_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        parent_label = path.parent.name.strip() or None
        records.append(DatasetRecord(
            record_id=_record_id(relative, task),
            path=relative,
            task=task,
            split=split,
            label=parent_label,
            source_id=f"source:{relative}",
            provenance=provenance,
            content_sha256=sha256_file(path),
            metadata={"filename": path.name, "suffix": path.suffix.casefold()},
        ))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an OptiVox dataset manifest.")
    parser.add_argument("root", type=Path, help="Directory containing local media.")
    parser.add_argument("--output", type=Path, required=True, help="Manifest JSON path.")
    parser.add_argument("--task", default="face_verification")
    parser.add_argument("--split", choices=sorted(SUPPORTED_SPLITS), default="train")
    parser.add_argument("--provenance", choices=sorted(ALLOWED_PROVENANCE), default="internal")
    parser.add_argument("--require-files", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"dataset root is not a directory: {root}")
    registry = DatasetRegistry(
        root,
        build_records(root, args.task.strip(), args.split, args.provenance),
    )
    output = registry.save(args.output)
    issues = registry.validate(require_files=args.require_files)
    result = registry.summary()
    result.update({
        "manifest": str(output),
        "validation": {"status": "PASS" if not issues else "FAIL", "issues": issues},
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
