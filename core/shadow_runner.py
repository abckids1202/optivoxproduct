"""Bounded baseline-versus-candidate evaluation for model adapters.

The runner accepts caller-supplied frames or crops and never opens a camera,
writes operational records, or forwards candidate output to policy. It is a
comparison tool for shadow/canary evaluation, not a decision engine.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Iterable

from .model_adapters import CallableModelAdapter


class ShadowEvaluationError(ValueError):
    """Raised when a shadow evaluation input violates its contract."""


def _signature(observation: dict[str, Any]) -> tuple:
    """Build a comparison key without including model-specific provenance."""
    bbox = observation.get("bbox")
    if isinstance(bbox, (list, tuple)):
        bbox = tuple(round(float(value), 2) for value in bbox[:4])
    return (
        str(observation.get("event_type") or "").upper(),
        str(observation.get("class_name") or observation.get("value") or ""),
        bbox,
        observation.get("track_id"),
    )


def _latency_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"samples": 0, "p50_ms": None, "p95_ms": None, "average_ms": None}
    ordered = sorted(values)
    pick = lambda fraction: ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]
    return {
        "samples": len(ordered),
        "p50_ms": round(pick(0.5), 3),
        "p95_ms": round(pick(0.95), 3),
        "average_ms": round(sum(ordered) / len(ordered), 3),
    }


def run_shadow_evaluation(
    frames: Iterable[dict[str, Any]],
    baseline: CallableModelAdapter,
    candidate: CallableModelAdapter,
    *,
    max_frames: int = 10_000,
) -> dict[str, Any]:
    """Compare adapters and return a side-effect-free evaluation report."""
    if candidate.spec.execution_mode != "shadow":
        raise ShadowEvaluationError("candidate adapter must use execution_mode='shadow'")
    if max_frames < 1:
        raise ShadowEvaluationError("max_frames must be positive")

    baseline_latency: list[float] = []
    candidate_latency: list[float] = []
    disagreements: list[dict[str, Any]] = []
    statuses = Counter()
    processed = 0
    for index, item in enumerate(frames):
        if index >= max_frames:
            raise ShadowEvaluationError("shadow input exceeds max_frames")
        if not isinstance(item, dict) or "frame" not in item:
            raise ShadowEvaluationError("each shadow frame requires a frame field")
        frame = item["frame"]
        context = item.get("context") or {}
        if not isinstance(context, dict):
            raise ShadowEvaluationError("frame context must be an object")
        started = time.perf_counter()
        baseline_result = baseline.infer(frame, context)
        baseline_elapsed = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        candidate_result = candidate.infer(frame, context)
        candidate_elapsed = (time.perf_counter() - started) * 1000.0
        baseline_latency.append(baseline_elapsed)
        candidate_latency.append(candidate_elapsed)
        statuses[f"baseline:{baseline_result.get('status', 'UNKNOWN')}"] += 1
        statuses[f"candidate:{candidate_result.get('status', 'UNKNOWN')}"] += 1
        baseline_observations = baseline_result.get("observations") or []
        candidate_observations = candidate_result.get("observations") or []
        baseline_counts = Counter(_signature(item) for item in baseline_observations)
        candidate_counts = Counter(_signature(item) for item in candidate_observations)
        if baseline_counts != candidate_counts:
            disagreements.append({
                "frame_index": index,
                "source_frame_id": context.get("source_frame_id"),
                "baseline_only": [list(key) for key, count in (baseline_counts - candidate_counts).items() for _ in range(count)],
                "candidate_only": [list(key) for key, count in (candidate_counts - baseline_counts).items() for _ in range(count)],
            })
        processed += 1

    return {
        "schema_version": 1,
        "status": "MEASURED" if processed else "NOT_MEASURED",
        "frames": processed,
        "baseline": {
            "model": baseline.spec.name,
            "version": baseline.spec.version,
            "latency": _latency_summary(baseline_latency),
        },
        "candidate": {
            "model": candidate.spec.name,
            "version": candidate.spec.version,
            "execution_mode": candidate.spec.execution_mode,
            "latency": _latency_summary(candidate_latency),
        },
        "disagreements": disagreements[:500],
        "disagreement_count": len(disagreements),
        "statuses": dict(statuses),
        "decision_effect": "NONE",
        "notes": [
            "Shadow output is evidence for evaluation only.",
            "No attendance, incident, alert, permission, or biometric action is performed.",
        ],
    }
