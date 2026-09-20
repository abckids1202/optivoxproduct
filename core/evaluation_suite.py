"""Run and aggregate deterministic OptiVox replay evaluations.

This is a model-free evaluation boundary. It measures the decisions produced
by the correlation and policy layers when supplied with labelled perception
outputs; it does not claim that the underlying camera models are accurate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List

from .evaluation import evaluate_acceptance, evaluate_replay
from .replay import ReplayValidationError, run_replay


def load_scenarios(source: Path | str) -> List[dict]:
    """Load one scenario, a ``{"scenarios": [...]}` file, or a directory."""
    path = Path(source).expanduser().resolve()
    paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    scenarios: List[dict] = []
    for item in paths:
        payload = json.loads(item.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("scenarios"), list):
            values = payload["scenarios"]
        elif isinstance(payload, list):
            values = payload
        else:
            values = [payload]
        for scenario in values:
            if not isinstance(scenario, dict):
                raise ReplayValidationError(f"Scenario in {item.name} must be an object.")
            scenarios.append(scenario)
    if not scenarios:
        raise ReplayValidationError("No replay scenarios were found.")
    return scenarios


def _sum_metric(cases: Iterable[dict], section: str, key: str) -> int:
    return sum(int((case.get("evaluation", {}).get(section, {}).get(key) or 0)) for case in cases)


def _sum_numeric_metric(cases: Iterable[dict], section: str, key: str) -> float:
    return sum(float((case.get("evaluation", {}).get(section, {}).get(key) or 0.0)) for case in cases)


def run_suite(scenarios: Iterable[dict], acceptance_thresholds: dict | None = None) -> dict:
    """Run all cases and return a deterministic aggregate report."""
    cases = []
    seen_ids = set()
    for index, scenario in enumerate(scenarios, start=1):
        scenario_id = str(scenario.get("scenario_id") or f"scenario-{index}").strip()
        raw_tags = scenario.get("tags") or []
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        if not isinstance(raw_tags, list):
            raise ReplayValidationError(f"Scenario {scenario_id} tags must be a list.")
        tags = sorted({str(tag).strip().casefold() for tag in raw_tags if str(tag).strip()})
        if scenario_id in seen_ids:
            raise ReplayValidationError(f"Duplicate scenario_id: {scenario_id}")
        seen_ids.add(scenario_id)
        try:
            replay = run_replay(scenario)
            evaluation = evaluate_replay(scenario, replay)
            cases.append({
                "scenario_id": scenario_id,
                "tags": tags,
                "verdict": replay.get("verdict", "FAIL"),
                "evaluation": evaluation,
                "frames": replay.get("frames", 0),
            })
        except (ReplayValidationError, TypeError, ValueError, KeyError) as exc:
            cases.append({
                "scenario_id": scenario_id,
                "tags": tags,
                "verdict": "FAIL",
                "evaluation": {"status": "NOT_MEASURED"},
                "frames": 0,
                "error": str(exc),
            })

    recognition_samples = _sum_metric(cases, "recognition", "samples")
    recognition_correct = _sum_metric(cases, "recognition", "correct")
    recognition_unknown_correct = _sum_metric(cases, "recognition", "true_unknown_rejects")
    recognition_unknown_trials = _sum_metric(cases, "recognition", "unknown_trials")
    recognition_known_trials = _sum_metric(cases, "recognition", "known_trials")
    liveness_samples = _sum_metric(cases, "liveness", "samples")
    security_expected = _sum_metric(cases, "security", "true_positive")
    security_false_positive = _sum_metric(cases, "security", "false_positive")
    security_false_negative = _sum_metric(cases, "security", "false_negative")
    security_duration_hours = _sum_numeric_metric(cases, "security", "duration_hours")
    liveness_true_positive = _sum_metric(cases, "liveness", "true_positive")
    liveness_true_negative = _sum_metric(cases, "liveness", "true_negative")
    liveness_false_positive = _sum_metric(cases, "liveness", "false_positive")
    liveness_false_negative = _sum_metric(cases, "liveness", "false_negative")
    plate_samples = _sum_metric(cases, "plate", "samples")
    plate_readable = _sum_metric(cases, "plate", "readable_ground_truth")
    plate_exact = _sum_metric(cases, "plate", "exact_matches")
    demographic_samples = _sum_metric(cases, "demographics", "samples")
    demographic_correct = _sum_metric(cases, "demographics", "correct")
    density_samples = _sum_metric(cases, "density", "samples")
    density_error_sum = _sum_numeric_metric(cases, "density", "absolute_error_sum")
    measured_cases = sum(
        1 for case in cases
        if case.get("evaluation", {}).get("status") == "MEASURED"
    )
    replay_failures = sum(1 for case in cases if case.get("verdict") != "PASS")
    report = {
        "schema_version": 1,
        "status": "FAIL" if replay_failures else ("MEASURED" if measured_cases else "NOT_MEASURED"),
        "cases": len(cases),
        "replay_passed": len(cases) - replay_failures,
        "replay_failed": replay_failures,
        "measured_cases": measured_cases,
        "coverage": {
            "scenario_ids": sorted(case["scenario_id"] for case in cases),
            "scenario_tags": sorted({tag for case in cases for tag in case.get("tags", [])}),
            "case_count": len(cases),
        },
        "aggregate": {
            "recognition": {
                "status": "MEASURED" if recognition_samples else "NOT_MEASURED",
                "samples": recognition_samples,
                "observed_samples": _sum_metric(cases, "recognition", "observed_samples"),
                "deferred": _sum_metric(cases, "recognition", "deferred"),
                "liveness_blocked": _sum_metric(cases, "recognition", "liveness_blocked"),
                "correct": recognition_correct,
                "accuracy": recognition_correct / recognition_samples if recognition_samples else None,
                "false_accepts": _sum_metric(cases, "recognition", "false_accepts"),
                "false_rejects": _sum_metric(cases, "recognition", "false_rejects"),
                "true_unknown_rejects": recognition_unknown_correct,
                "false_accept_rate": (
                    _sum_metric(cases, "recognition", "false_accepts") /
                    recognition_unknown_trials if recognition_unknown_trials else None
                ),
                "false_reject_rate": (
                    _sum_metric(cases, "recognition", "false_rejects") /
                    recognition_known_trials if recognition_known_trials else None
                ),
                "unknown_rejection_accuracy": (
                    recognition_unknown_correct /
                    recognition_unknown_trials if recognition_unknown_trials else None
                ),
            },
            "liveness": {
                "status": "MEASURED" if liveness_samples else "NOT_MEASURED",
                "samples": liveness_samples,
                "false_accepts": _sum_metric(cases, "liveness", "false_positive"),
                "false_rejects": _sum_metric(cases, "liveness", "false_negative"),
                "false_accept_rate": (
                    liveness_false_positive /
                    (liveness_true_negative + liveness_false_positive)
                    if liveness_true_negative + liveness_false_positive else None
                ),
                "false_reject_rate": (
                    liveness_false_negative /
                    (liveness_true_positive + liveness_false_negative)
                    if liveness_true_positive + liveness_false_negative else None
                ),
            },
            "security": {
                "status": "MEASURED" if any(
                    case.get("evaluation", {}).get("security", {}).get("status") == "MEASURED"
                    for case in cases
                ) else "NOT_MEASURED",
                "true_positive": security_expected,
                "false_positive": security_false_positive,
                "false_negative": security_false_negative,
                "false_alerts_per_camera_hour": (
                    security_false_positive / security_duration_hours
                    if security_duration_hours > 0 else None
                ),
            },
            "tracking": {
                "status": "MEASURED" if _sum_metric(cases, "tracking", "matched_observations") else "NOT_MEASURED",
                "matched_observations": _sum_metric(cases, "tracking", "matched_observations"),
                "id_switches": _sum_metric(cases, "tracking", "id_switches"),
                "id_switch_rate": (
                    _sum_metric(cases, "tracking", "id_switches") /
                    _sum_metric(cases, "tracking", "matched_observations")
                    if _sum_metric(cases, "tracking", "matched_observations") else None
                ),
            },
            "vehicle": {
                "status": "MEASURED" if _sum_metric(cases, "vehicle", "samples") else "NOT_MEASURED",
                "samples": _sum_metric(cases, "vehicle", "samples"),
            },
            "plate": {
                "status": "MEASURED" if plate_samples else "NOT_MEASURED",
                "samples": plate_samples,
                "readable_ground_truth": plate_readable,
                "exact_matches": plate_exact,
                "exact_match_rate": plate_exact / plate_readable if plate_readable else None,
            },
            "demographics": {
                "status": "MEASURED" if demographic_samples else "NOT_MEASURED",
                "samples": demographic_samples,
                "correct": demographic_correct,
                "accuracy": demographic_correct / demographic_samples if demographic_samples else None,
                "decision_use": "aggregate_only",
            },
            "density": {
                "status": "MEASURED" if density_samples else "NOT_MEASURED",
                "samples": density_samples,
                "mean_absolute_error": density_error_sum / density_samples if density_samples else None,
            },
        },
        "cases_detail": cases,
        "notes": [
            "A PASS verdict means policy invariants held for the replay; it is not a model-accuracy claim.",
            "Aggregate accuracy is meaningful only when cases use independent holdout or adversarial labels.",
            "Malformed cases fail the suite and are retained with their error for triage.",
        ],
    }
    if acceptance_thresholds is not None:
        report["acceptance"] = evaluate_acceptance(report, acceptance_thresholds)
        if report["acceptance"]["status"] == "FAIL":
            report["status"] = "FAIL"
    return report
