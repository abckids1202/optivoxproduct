"""Local model registry and promotion gate for the OptiVox edge agent.

The registry is metadata only. It never downloads, executes, or replaces a
model. A model is usable only when its local file, checksum, task metadata,
and evaluation/promotion state are explicit. This keeps future model changes
reversible and prevents an unverified artifact from looking production-ready.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


MODEL_SUFFIXES = frozenset({".onnx", ".pt", ".pth", ".engine", ".bin"})
CHECKSUM_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
PROMOTION_STATES = frozenset({"shadow", "canary", "promoted", "rollback"})


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    task: str
    version: str
    path: str
    sha256: str
    source: str = ""
    license: str = ""
    training_data_version: str = ""
    evaluation_report: str = ""
    promotion_status: str = "shadow"
    rollback_target: str = ""
    limitations: str = ""

    @classmethod
    def from_dict(cls, value: Any) -> Optional["ModelArtifact"]:
        if not isinstance(value, dict):
            return None
        required = ("name", "task", "version", "path", "sha256")
        if any(not str(value.get(key) or "").strip() for key in required):
            return None
        return cls(
            name=str(value["name"]).strip()[:120],
            task=str(value["task"]).strip()[:80],
            version=str(value["version"]).strip()[:80],
            path=str(value["path"]).strip(),
            sha256=str(value["sha256"]).strip().lower(),
            source=str(value.get("source") or "").strip()[:300],
            license=str(value.get("license") or "").strip()[:160],
            training_data_version=str(value.get("training_data_version") or "").strip()[:120],
            evaluation_report=str(value.get("evaluation_report") or "").strip()[:300],
            promotion_status=str(value.get("promotion_status") or "shadow").strip().lower(),
            rollback_target=str(value.get("rollback_target") or "").strip()[:80],
            limitations=str(value.get("limitations") or "").strip()[:500],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "version": self.version,
            "path": self.path,
            "sha256": self.sha256,
            "source": self.source,
            "license": self.license,
            "training_data_version": self.training_data_version,
            "evaluation_report": self.evaluation_report,
            "promotion_status": self.promotion_status,
            "rollback_target": self.rollback_target,
            "limitations": self.limitations,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelRegistry:
    """Validate a local, versioned model registry without model execution."""

    def __init__(self, root: str | Path, manifest_path: str | Path):
        self.root = Path(root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.artifacts: list[ModelArtifact] = []
        self.load_issues: list[str] = []

    def load(self) -> "ModelRegistry":
        self.artifacts = []
        self.load_issues = []
        try:
            self.manifest_path.relative_to(self.root)
        except ValueError:
            self.load_issues.append("registry manifest escapes registry root")
            return self
        if not self.manifest_path.is_file():
            self.load_issues.append("registry manifest is not configured")
            return self
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            self.load_issues.append("registry manifest is not valid JSON")
            return self
        if not isinstance(payload, dict) or int(payload.get("schema_version", 0) or 0) != 1:
            self.load_issues.append("registry schema_version must be 1")
            return self
        raw_models = payload.get("models")
        if isinstance(raw_models, dict):
            raw_models = list(raw_models.values())
        if not isinstance(raw_models, list):
            self.load_issues.append("registry models must be a list")
            return self
        for index, raw in enumerate(raw_models, start=1):
            artifact = ModelArtifact.from_dict(raw)
            if artifact is None:
                self.load_issues.append(f"model {index}: required metadata is missing")
                continue
            self.artifacts.append(artifact)
        return self

    def _resolved_path(self, artifact: ModelArtifact) -> Optional[Path]:
        candidate = Path(artifact.path)
        if candidate.is_absolute():
            return None
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            return None
        return resolved

    def validate(self) -> dict[str, Any]:
        issues = list(self.load_issues)
        models = []
        seen_names: set[str] = set()
        for artifact in self.artifacts:
            prefix = f"{artifact.name}@{artifact.version}"
            model_issues: list[str] = []
            if artifact.name.casefold() in seen_names:
                model_issues.append("duplicate model name")
            seen_names.add(artifact.name.casefold())
            if not CHECKSUM_PATTERN.fullmatch(artifact.sha256):
                model_issues.append("sha256 must be a 64-character hexadecimal digest")
            if artifact.promotion_status not in PROMOTION_STATES:
                model_issues.append("promotion_status is invalid")
            path = self._resolved_path(artifact)
            if path is None:
                model_issues.append("model path escapes registry root or is absolute")
            elif path.suffix.casefold() not in MODEL_SUFFIXES:
                model_issues.append("model path has an unsupported suffix")
            elif not path.is_file():
                model_issues.append("model file is missing")
            else:
                try:
                    actual = sha256_file(path)
                    if actual.casefold() != artifact.sha256:
                        model_issues.append("model checksum mismatch")
                except OSError:
                    model_issues.append("model file cannot be read")
            if not artifact.source:
                model_issues.append("model source is not documented")
            if not artifact.license:
                model_issues.append("model license is not documented")
            if not artifact.evaluation_report:
                model_issues.append("evaluation report is not documented")
            if model_issues:
                issues.extend(f"{prefix}: {item}" for item in model_issues)
            models.append({
                "name": artifact.name,
                "task": artifact.task,
                "version": artifact.version,
                "path": artifact.path,
                "promotion_status": artifact.promotion_status,
                "evaluation_report": artifact.evaluation_report,
                "status": "INVALID" if model_issues else "VALID",
                "issues": model_issues,
                "rollback_target": artifact.rollback_target or None,
            })
        status = "NOT_CONFIGURED" if not self.artifacts and not self.load_issues else (
            "VALID" if not issues else "INVALID")
        return {
            "schema_version": 1,
            "status": status,
            "manifest": str(self.manifest_path),
            "models": models,
            "issues": sorted(set(issues)),
        }

    def promotion_allowed(self, name: str, version: str, *, require_promoted: bool = True) -> bool:
        """Return whether a validated artifact may be selected by a caller.

        The registry does not load models itself. This helper gives a runtime
        or deployment tool a conservative gate: invalid artifacts never pass,
        and production-like selection requires an explicit ``promoted`` state.
        """
        report = self.validate()
        if report.get("status") == "NOT_CONFIGURED" or report.get("issues"):
            return False
        for artifact, result in zip(self.artifacts, report.get("models") or []):
            if artifact.name == str(name) and artifact.version == str(version):
                if result.get("status") != "VALID":
                    return False
                return (not require_promoted
                        or artifact.promotion_status == "promoted")
        return False

    def snapshot(self) -> dict[str, Any]:
        """Return safe registry metadata suitable for system-status APIs."""
        return self.validate()


def registry_snapshot(root: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    return ModelRegistry(root, manifest_path).load().snapshot()
