"""Small, provenance-aware dataset manifest for OptiVox evaluation.

The registry deliberately stores metadata rather than copying media. It is
used to prevent split leakage, reject paths outside the dataset root, and make
consent/licence provenance visible before a model is trained or evaluated.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SUPPORTED_SPLITS = frozenset({"train", "validation", "holdout", "adversarial"})
ALLOWED_PROVENANCE = frozenset({"consented", "licensed", "internal", "public_domain"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def sha256_file(path: Path) -> str:
    """Hash a media file incrementally so large videos do not fill memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetRecord:
    record_id: str
    path: str
    task: str
    split: str
    label: Optional[str] = None
    source_id: Optional[str] = None
    provenance: str = "internal"
    derived_from: Optional[str] = None
    transform: Optional[str] = None
    content_sha256: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "DatasetRecord":
        return cls(
            record_id=str(value.get("record_id") or value.get("id") or "").strip(),
            path=str(value.get("path") or "").strip(),
            task=str(value.get("task") or "").strip(),
            split=str(value.get("split") or "").strip().lower(),
            label=(str(value["label"]).strip() if value.get("label") is not None else None),
            source_id=(str(value["source_id"]).strip() if value.get("source_id") is not None else None),
            provenance=str(value.get("provenance") or "internal").strip().lower(),
            content_sha256=(
                str(value["content_sha256"]).strip().lower()
                if value.get("content_sha256") is not None else None
            ),
            derived_from=(str(value["derived_from"]).strip() if value.get("derived_from") is not None else None),
            transform=(str(value["transform"]).strip() if value.get("transform") is not None else None),
            metadata=dict(value.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "record_id": self.record_id,
            "path": self.path,
            "task": self.task,
            "split": self.split,
            "provenance": self.provenance,
        }
        for key in ("label", "source_id", "content_sha256", "derived_from", "transform"):
            item = getattr(self, key)
            if item is not None:
                value[key] = item
        if self.metadata:
            value["metadata"] = dict(self.metadata)
        return value


class DatasetRegistry:
    """Load, validate, and summarize a JSON dataset manifest."""

    schema_version = 1

    def __init__(self, root: Path | str, records: Iterable[DatasetRecord] = ()):
        self.root = Path(root).expanduser().resolve()
        self.records = list(records)

    @classmethod
    def load(cls, manifest_path: Path | str, root: Path | str | None = None) -> "DatasetRegistry":
        path = Path(manifest_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Dataset manifest must be a JSON object.")
        manifest_root = root or payload.get("root") or path.parent
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("Dataset manifest records must be a list.")
        return cls(manifest_root, [DatasetRecord.from_dict(item) for item in raw_records if isinstance(item, dict)])

    def save(self, manifest_path: Path | str) -> Path:
        path = Path(manifest_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "root": str(self.root),
            "records": [record.to_dict() for record in self.records],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def validate(self, require_files: bool = False, reject_exact_duplicates: bool = True) -> List[str]:
        issues: List[str] = []
        seen_ids = set()
        source_splits: Dict[str, set] = {}
        file_hashes: Dict[str, str] = {}
        record_by_id = {record.record_id: record for record in self.records if record.record_id}
        for record in self.records:
            if not record.record_id:
                issues.append("record is missing record_id")
            elif record.record_id in seen_ids:
                issues.append(f"duplicate record_id: {record.record_id}")
            seen_ids.add(record.record_id)
            if not record.path:
                issues.append(f"{record.record_id or '<unnamed>'}: missing path")
                continue
            if not record.task:
                issues.append(f"{record.record_id}: missing task")
            if record.split not in SUPPORTED_SPLITS:
                issues.append(f"{record.record_id}: unsupported split {record.split!r}")
            if record.provenance not in ALLOWED_PROVENANCE:
                issues.append(f"{record.record_id}: provenance must identify consent or licence")
            if record.content_sha256 is not None and not SHA256_PATTERN.fullmatch(
                str(record.content_sha256).casefold()
            ):
                issues.append(f"{record.record_id}: content_sha256 must be a 64-character SHA-256 hex digest")
            if record.derived_from and not record.transform:
                issues.append(f"{record.record_id}: derived record is missing transform")
            if record.derived_from:
                source = record_by_id.get(record.derived_from)
                if source is None:
                    issues.append(f"{record.record_id}: derived_from does not reference a known record")
                elif source.split != record.split:
                    issues.append(
                        f"{record.record_id}: derived record split {record.split!r} differs from source split {source.split!r}"
                    )
            resolved = (self.root / record.path).resolve()
            if not _inside(resolved, self.root):
                issues.append(f"{record.record_id}: path escapes dataset root")
            elif require_files and not resolved.is_file():
                issues.append(f"{record.record_id}: file is missing")
            groups = {value for value in (record.source_id, record.derived_from) if value}
            groups.add(record.record_id)
            for group in groups:
                source_splits.setdefault(group, set()).add(record.split)
            if require_files and resolved.is_file():
                digest = sha256_file(resolved)
                if record.content_sha256 and digest != record.content_sha256.casefold():
                    issues.append(
                        f"content hash mismatch: {record.record_id} expected {record.content_sha256} got {digest}"
                    )
                previous = file_hashes.get(digest)
                if previous and reject_exact_duplicates:
                    issues.append(f"exact duplicate media: {record.record_id} duplicates {previous}")
                file_hashes[digest] = record.record_id
        for group, splits in source_splits.items():
            if len(splits) > 1:
                issues.append(f"split leakage: source {group!r} appears in {sorted(splits)}")
        return sorted(set(issues))

    def summary(self) -> Dict[str, Any]:
        by_split: Dict[str, int] = {}
        by_task: Dict[str, int] = {}
        by_provenance: Dict[str, int] = {}
        for record in self.records:
            by_split[record.split] = by_split.get(record.split, 0) + 1
            by_task[record.task] = by_task.get(record.task, 0) + 1
            by_provenance[record.provenance] = by_provenance.get(record.provenance, 0) + 1
        return {
            "schema_version": self.schema_version,
            "root": str(self.root),
            "records": len(self.records),
            "by_split": dict(sorted(by_split.items())),
            "by_task": dict(sorted(by_task.items())),
            "by_provenance": dict(sorted(by_provenance.items())),
            "validation": {"status": "PASS" if not self.validate() else "FAIL"},
        }
