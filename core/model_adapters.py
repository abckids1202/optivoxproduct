"""Common capability and health boundary for OptiVox perception components.

The first version deliberately wraps existing inference rather than loading
another model or changing attendance semantics. A component can therefore be
measured and exposed as a versioned capability before it is trusted for an
operational decision.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


CAPABILITY_STATES = frozenset({
    "AVAILABLE",
    "DISABLED",
    "EXPERIMENTAL",
    "NOT_CONFIGURED",
    "FAILED_INTEGRITY",
    "FAILED_RUNTIME",
})
EXECUTION_MODES = frozenset({"active", "shadow"})
PROMOTION_STATES = frozenset({"not_applicable", "shadow", "canary", "promoted", "rollback"})
_FORBIDDEN_DECISION_KEYS = frozenset({
    "attendance", "official_attendance", "incident_resolution", "permissions",
    "delete_biometrics", "send_external_message", "access_grant",
})
_AGGREGATE_FORBIDDEN_KEYS = frozenset({
    "identity", "person_id", "person_name", "student_id", "attendance",
    "official_attendance", "access_grant", "disciplinary_action",
})


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: str
    version: str = "local"
    capability: str = "NOT_CONFIGURED"
    enabled: bool = True
    execution_mode: str = "active"
    cadence: str = "scheduled"
    roi_policy: str = "full_frame"
    resource_budget: str = "cpu"
    privacy_class: str = "operational"
    artifact_path: Optional[str] = None
    checksum_verified: Optional[bool] = None
    promotion_status: str = "not_applicable"
    evaluation_report: Optional[str] = None
    rollback_target: Optional[str] = None
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.capability not in CAPABILITY_STATES:
            raise ValueError(f"Unsupported capability state: {self.capability}")
        if self.execution_mode not in EXECUTION_MODES:
            raise ValueError(f"Unsupported execution mode: {self.execution_mode}")
        if self.promotion_status not in PROMOTION_STATES:
            raise ValueError(f"Unsupported promotion status: {self.promotion_status}")
        if not self.name.strip() or not self.task.strip():
            raise ValueError("Model name and task are required")


class ModelAdapterRegistry:
    """Register perception components and collect safe runtime telemetry."""

    def __init__(self, history_size: int = 120):
        self.history_size = max(10, int(history_size))
        self._specs: dict[str, ModelSpec] = {}
        self._history: dict[str, deque[float]] = {}
        self._calls: dict[str, int] = {}
        self._failures: dict[str, int] = {}
        self._last_error: dict[str, str | None] = {}
        self._last_call_at: dict[str, float | None] = {}

    def register(self, spec: ModelSpec) -> None:
        key = spec.name.strip()
        if key in self._specs:
            raise ValueError(f"Model adapter already registered: {key}")
        self._specs[key] = spec
        self._history[key] = deque(maxlen=self.history_size)
        self._calls[key] = 0
        self._failures[key] = 0
        self._last_error[key] = None
        self._last_call_at[key] = None

    def replace(self, spec: ModelSpec) -> None:
        """Replace metadata while preserving accumulated health counters."""
        key = spec.name.strip()
        if key not in self._specs:
            self.register(spec)
            return
        self._specs[key] = spec

    def apply_registry_snapshot(self, snapshot: Any) -> dict[str, Any]:
        """Synchronize matching adapter metadata from the model registry.

        Built-in components without a manifest entry remain ``not_applicable``.
        A manifest can only tighten runtime state; it cannot make a missing or
        invalid artifact look available.
        """
        models = snapshot.get("models") if isinstance(snapshot, dict) else None
        if not isinstance(models, list):
            return {"matched": [], "issues": ["model registry snapshot has no models list"]}
        by_name = {
            str(item.get("name")): item for item in models
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        }
        matched = []
        issues = list(snapshot.get("issues") or []) if isinstance(snapshot, dict) else []
        for name, spec in list(self._specs.items()):
            entry = by_name.get(name)
            if entry is None:
                continue
            status = str(entry.get("status") or "INVALID").upper()
            promotion = str(entry.get("promotion_status") or "shadow").lower()
            if promotion not in PROMOTION_STATES:
                promotion = "rollback"
            replacement = ModelSpec(
                **{
                    **spec.__dict__,
                    "checksum_verified": status == "VALID",
                    "promotion_status": promotion,
                    "evaluation_report": entry.get("evaluation_report") or spec.evaluation_report,
                    "rollback_target": entry.get("rollback_target") or spec.rollback_target,
                }
            )
            self.replace(replacement)
            matched.append(name)
            if status != "VALID":
                issues.append(f"{name}: registry artifact is {status}")
        return {"matched": matched, "issues": sorted(set(str(item) for item in issues))}

    def get(self, name: str) -> Optional[ModelSpec]:
        return self._specs.get(str(name))

    def record_inference(
        self,
        name: str,
        elapsed_ms: float,
        *,
        ok: bool = True,
        error: Optional[str] = None,
    ) -> None:
        key = str(name)
        if key not in self._specs:
            return
        try:
            elapsed = max(0.0, float(elapsed_ms))
        except (TypeError, ValueError):
            elapsed = 0.0
        self._history[key].append(elapsed)
        self._calls[key] += 1
        self._last_call_at[key] = time.time()
        if not ok:
            self._failures[key] += 1
            self._last_error[key] = str(error or "inference failed")[:240]

    def _effective_state(self, spec: ModelSpec) -> str:
        if not spec.enabled:
            return "DISABLED"
        if spec.checksum_verified is False:
            return "FAILED_INTEGRITY"
        if self._failures.get(spec.name, 0) and self._failures[spec.name] >= 3:
            return "FAILED_RUNTIME"
        return spec.capability

    @staticmethod
    def _summary(values: deque[float]) -> dict[str, Any]:
        if not values:
            return {"samples": 0, "p50_ms": None, "p95_ms": None, "average_ms": None}
        ordered = sorted(values)
        percentile = lambda fraction: ordered[min(
            len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]
        return {
            "samples": len(ordered),
            "p50_ms": round(percentile(0.50), 3),
            "p95_ms": round(percentile(0.95), 3),
            "average_ms": round(sum(ordered) / len(ordered), 3),
        }

    def snapshot(self) -> dict[str, Any]:
        models = []
        for name, spec in self._specs.items():
            models.append({
                "name": spec.name,
                "task": spec.task,
                "version": spec.version,
                "state": self._effective_state(spec),
                "execution_mode": spec.execution_mode,
                "cadence": spec.cadence,
                "roi_policy": spec.roi_policy,
                "resource_budget": spec.resource_budget,
                "privacy_class": spec.privacy_class,
                "artifact_path": spec.artifact_path,
                "checksum_verified": spec.checksum_verified,
                "promotion_status": spec.promotion_status,
                "evaluation_report": spec.evaluation_report,
                "rollback_target": spec.rollback_target,
                "limitations": list(spec.limitations),
                "calls": self._calls[name],
                "failures": self._failures[name],
                "last_error": self._last_error[name],
                "last_call_at": self._last_call_at[name],
                "latency": self._summary(self._history[name]),
            })
        return {"count": len(models), "models": models}

    def capability_snapshot(self) -> dict[str, str]:
        return {name: self._effective_state(spec) for name, spec in self._specs.items()}

    def promotion_allowed(self, name: str, *, require_promoted: bool = True) -> bool:
        """Apply the runtime gate to an explicitly evaluated model artifact."""
        spec = self._specs.get(str(name))
        if spec is None or not spec.enabled:
            return False
        if self._effective_state(spec) != "AVAILABLE":
            return False
        if spec.promotion_status == "rollback":
            return False
        if require_promoted and spec.promotion_status != "promoted":
            return False
        if spec.promotion_status != "not_applicable":
            return bool(
                spec.checksum_verified is True
                and str(spec.evaluation_report or "").strip()
            )
        return True

    def run(self, adapter: "CallableModelAdapter", frame: Any,
            context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Execute a registered adapter and record its health consistently."""
        name = str(adapter.spec.name)
        if name not in self._specs:
            raise ValueError(f"Model adapter is not registered: {name}")
        started = time.perf_counter()
        result = adapter.infer(frame, context)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        status = str(result.get("status") or "UNKNOWN") if isinstance(result, dict) else "INVALID_OUTPUT"
        self.record_inference(
            name,
            elapsed_ms,
            ok=status in {"OK", "NOT_CONFIGURED", "DISABLED", "ROLLBACK"},
            error=(result.get("error") if isinstance(result, dict) else "adapter returned non-object"),
        )
        if isinstance(result, dict):
            enriched = dict(result)
            enriched["registry_elapsed_ms"] = round(elapsed_ms, 3)
            return enriched
        return {
            "status": "INVALID_OUTPUT",
            "model": name,
            "task": adapter.spec.task,
            "observations": [],
            "registry_elapsed_ms": round(elapsed_ms, 3),
        }


def normalize_model_observations(output: Any, spec: ModelSpec,
                                 context: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    """Normalize model output into observation-shaped dictionaries.

    Model output is evidence only. Decision-shaped fields are rejected before
    the result can be handed to correlation or an operational worker.
    """
    if output is None:
        return []
    candidates = output if isinstance(output, list) else [output]
    if not all(isinstance(item, dict) for item in candidates):
        raise ValueError("model output must be a mapping or list of mappings")
    context = context or {}
    normalized = []
    for item in candidates:
        forbidden = sorted(set(item).intersection(_FORBIDDEN_DECISION_KEYS))
        if forbidden:
            raise ValueError("model output contains forbidden decision fields: " + ", ".join(forbidden))
        if spec.privacy_class == "sensitive_aggregate":
            aggregate_forbidden = sorted(set(item).intersection(_AGGREGATE_FORBIDDEN_KEYS))
            if aggregate_forbidden:
                raise ValueError(
                    "aggregate model output contains identity or decision fields: "
                    + ", ".join(aggregate_forbidden)
                )
        record = dict(item)
        event_type = str(
            record.get("event_type") or record.get("observation_type")
            or spec.task.upper()
        ).strip().upper()
        if not event_type:
            raise ValueError("model output must identify an observation type")
        record["event_type"] = event_type
        record["model_name"] = spec.name
        record["model_version"] = spec.version
        record["execution_mode"] = spec.execution_mode
        if "source_frame_id" not in record and context.get("source_frame_id") is not None:
            record["source_frame_id"] = context["source_frame_id"]
        if "camera_id" not in record and context.get("camera_id") is not None:
            record["camera_id"] = context["camera_id"]
        normalized.append(record)
    return normalized


class CallableModelAdapter:
    """Optional execution wrapper for future model implementations.

    It returns a structured result and never turns model output into an
    attendance, incident, or permission decision.
    """

    def __init__(self, spec: ModelSpec, infer_fn: Optional[Callable[..., Any]] = None):
        self.spec = spec
        self.infer_fn = infer_fn

    def capability_state(self) -> str:
        if not self.spec.enabled:
            return "DISABLED"
        if self.spec.checksum_verified is False:
            return "FAILED_INTEGRITY"
        if self.infer_fn is None:
            return self.spec.capability
        return self.spec.capability

    def operationally_allowed(self, *, require_promoted: bool = True) -> bool:
        """Whether this adapter may be selected for an operational path."""
        if self.capability_state() != "AVAILABLE":
            return False
        if self.spec.promotion_status == "rollback":
            return False
        if require_promoted and self.spec.promotion_status != "promoted":
            return False
        if self.spec.promotion_status != "not_applicable":
            return bool(
                self.spec.checksum_verified is True
                and str(self.spec.evaluation_report or "").strip()
            )
        return True

    def infer(self, frame: Any, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        state = self.capability_state()
        if self.spec.promotion_status == "rollback":
            state = "ROLLBACK"
        missing_promotion_evidence = (
            self.spec.promotion_status == "promoted"
            and (
                self.spec.checksum_verified is not True
                or not str(self.spec.evaluation_report or "").strip()
            )
        )
        if (
            self.spec.execution_mode == "active"
            and (
                self.spec.promotion_status not in {"not_applicable", "promoted"}
                or missing_promotion_evidence
            )
            and self.spec.promotion_status != "rollback"
        ):
            return {
                "status": "NOT_APPROVED",
                "model": self.spec.name,
                "task": self.spec.task,
                "observations": [],
                "reason": "active execution requires complete promoted model metadata",
            }
        if state in {"DISABLED", "NOT_CONFIGURED", "FAILED_INTEGRITY", "FAILED_RUNTIME", "ROLLBACK"}:
            return {
                "status": state,
                "model": self.spec.name,
                "task": self.spec.task,
                "observations": [],
            }
        if self.infer_fn is None:
            return {
                "status": "NOT_CONFIGURED",
                "model": self.spec.name,
                "task": self.spec.task,
                "observations": [],
            }
        started = time.perf_counter()
        try:
            context = context or {}
            output = self.infer_fn(frame, context)
            observations = normalize_model_observations(output, self.spec, context)
            return {
                "status": "OK",
                "model": self.spec.name,
                "task": self.spec.task,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "execution_mode": self.spec.execution_mode,
                "observations": observations,
            }
        except ValueError as exc:
            return {
                "status": "INVALID_OUTPUT",
                "model": self.spec.name,
                "task": self.spec.task,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "error": str(exc)[:240],
                "observations": [],
            }
        except Exception as exc:
            return {
                "status": "FAILED_RUNTIME",
                "model": self.spec.name,
                "task": self.spec.task,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "error": str(exc)[:240],
                "observations": [],
            }
