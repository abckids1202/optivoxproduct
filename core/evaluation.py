"""Dependency-light evaluation metrics for OptiVox validation reports."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class BinaryMetrics:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    def to_dict(self) -> Dict[str, Any]:
        positives = self.true_positive + self.false_negative
        negatives = self.true_negative + self.false_positive
        predicted_positive = self.true_positive + self.false_positive
        return {
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": self.true_positive / predicted_positive if predicted_positive else None,
            "recall": self.true_positive / positives if positives else None,
            "false_accept_rate": self.false_positive / negatives if negatives else None,
            "false_reject_rate": self.false_negative / positives if positives else None,
            "samples": positives + negatives,
        }


def binary_metrics(labels: Sequence[bool], scores: Sequence[float], threshold: float) -> BinaryMetrics:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    counts = [0, 0, 0, 0]
    for expected, score in zip(labels, scores):
        actual = float(score) >= float(threshold)
        if bool(expected) and actual:
            counts[0] += 1
        elif not bool(expected) and not actual:
            counts[1] += 1
        elif not bool(expected) and actual:
            counts[2] += 1
        else:
            counts[3] += 1
    return BinaryMetrics(*counts)


def event_metrics(expected: Iterable[str], predicted: Iterable[str], duration_hours: Optional[float] = None) -> Dict[str, Any]:
    expected_set = {str(value) for value in expected}
    predicted_set = {str(value) for value in predicted}
    true_positive = len(expected_set & predicted_set)
    false_positive = len(predicted_set - expected_set)
    false_negative = len(expected_set - predicted_set)
    result = BinaryMetrics(true_positive, 0, false_positive, false_negative).to_dict()
    result["false_alerts_per_camera_hour"] = (
        false_positive / float(duration_hours)
        if duration_hours is not None and float(duration_hours) > 0 else None
    )
    return result


def mean_absolute_error(expected: Iterable[float], predicted: Iterable[float]) -> Optional[float]:
    pairs = [(float(left), float(right)) for left, right in zip(expected, predicted)]
    if not pairs:
        return None
    return sum(abs(left - right) for left, right in pairs) / len(pairs)


def categorical_metrics(expected: Sequence[Any], predicted: Sequence[Any]) -> Dict[str, Any]:
    """Measure exact categorical agreement without inventing missing labels."""
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted labels must have the same length")
    total = len(expected)
    correct = sum(1 for left, right in zip(expected, predicted)
                  if str(left).strip().casefold() == str(right).strip().casefold())
    return {
        "status": _evaluation_status(total),
        "samples": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
    }


def _normalize_plate(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = "".join(character for character in str(value).upper() if character.isalnum())
    return normalized or None


def plate_metrics(expected: Sequence[Any], predicted: Sequence[Any]) -> Dict[str, Any]:
    """Evaluate plate OCR conservatively, treating unreadable as uncertainty."""
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted plate labels must have the same length")
    expected_values = [_normalize_plate(value) for value in expected]
    predicted_values = [_normalize_plate(value) for value in predicted]
    readable = [index for index, value in enumerate(expected_values) if value is not None]
    unreadable = [index for index, value in enumerate(expected_values) if value is None]
    exact = sum(1 for index in readable if predicted_values[index] == expected_values[index])
    false_reads = sum(1 for index in unreadable if predicted_values[index] is not None)
    return {
        "status": _evaluation_status(len(expected_values)),
        "samples": len(expected_values),
        "readable_ground_truth": len(readable),
        "unreadable_ground_truth": len(unreadable),
        "exact_matches": exact,
        "exact_match_rate": exact / len(readable) if readable else None,
        "false_reads_on_unreadable": false_reads,
        "uncertainty_preservation_rate": (
            (len(unreadable) - false_reads) / len(unreadable) if unreadable else None
        ),
    }


def density_metrics(expected: Sequence[Any], predicted: Sequence[Any]) -> Dict[str, Any]:
    """Measure count/density error; this is not a crowd-safety decision."""
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted density labels must have the same length")
    values = []
    for left, right in zip(expected, predicted):
        try:
            values.append((float(left), float(right)))
        except (TypeError, ValueError):
            continue
    errors = [abs(left - right) for left, right in values]
    return {
        "status": _evaluation_status(len(values)),
        "samples": len(values),
        "absolute_error_sum": sum(errors) if errors else 0.0,
        "mean_absolute_error": sum(errors) / len(errors) if errors else None,
        "max_absolute_error": max(errors) if errors else None,
    }


def tracking_id_switches(matches: Iterable[Tuple[int, str, Optional[int]]]) -> Dict[str, Any]:
    """Count predicted-ID changes for each ground-truth identity.

    ``matches`` should contain ``(frame_number, ground_truth_id, predicted_id)``
    rows after a spatial matcher has paired detections. Missing predictions are
    ignored while the identity is occluded.
    """
    previous: Dict[str, int] = {}
    switches = 0
    usable = 0
    for _frame, truth_id, predicted_id in sorted(matches, key=lambda item: item[0]):
        if predicted_id is None:
            continue
        usable += 1
        key = str(truth_id)
        if key in previous and previous[key] != int(predicted_id):
            switches += 1
        previous[key] = int(predicted_id)
    return {"id_switches": switches, "matched_observations": usable}


def measured(value: Any) -> Dict[str, Any]:
    """Represent missing telemetry honestly instead of converting it to zero."""
    if value is None:
        return {"status": "NOT_MEASURED", "value": None}
    return {"status": "MEASURED", "value": value}


def _evaluation_status(samples: int) -> str:
    return "MEASURED" if int(samples) > 0 else "NOT_MEASURED"


def _safe_rate(numerator: int, denominator: int) -> Optional[float]:
    return float(numerator) / float(denominator) if int(denominator or 0) else None


def _expected_name(face: dict) -> Optional[str]:
    value = face.get("expected_name", face.get("expected_identity"))
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _predicted_face_by_frame(result: dict) -> Dict[Tuple[Any, int], dict]:
    """Index replay decisions without assuming frame IDs are contiguous."""
    indexed: Dict[Tuple[Any, int], dict] = {}
    for decision in result.get("attendance_decisions") or []:
        try:
            key = (decision.get("frame_id"), int(decision.get("track_id", decision.get("oid"))))
        except (TypeError, ValueError):
            continue
        indexed[key] = decision
    return indexed


def _identity_evaluation(scenario: dict, result: dict) -> dict:
    predictions = _predicted_face_by_frame(result)
    rows = []
    for frame in scenario.get("frames") or []:
        frame_id = frame.get("frame_id")
        for face in frame.get("faces") or []:
            if not isinstance(face, dict) or _expected_name(face) is None:
                continue
            try:
                track_id = int(face.get("oid"))
            except (TypeError, ValueError):
                continue
            expected = _expected_name(face)
            decision = predictions.get((frame_id, track_id), {})
            predicted = str(decision.get("person_name") or "UNKNOWN").strip()
            identity_state = str(
                decision.get("identity_state") or "UNRESOLVED"
            ).upper()
            liveness_state = str(
                decision.get("liveness_state") or "NOT_EVALUATED"
            ).upper()
            expected_unknown = expected.casefold() in {"unknown", "unresolved", "none"}
            predicted_unknown = predicted.casefold() in {"unknown", "unresolved", "none"}
            if expected_unknown and predicted_unknown:
                outcome = "true_unknown_reject"
            elif expected_unknown and not predicted_unknown:
                outcome = "false_accept"
            elif not expected_unknown and predicted.casefold() == expected.casefold():
                outcome = "true_identity"
            elif not expected_unknown and liveness_state in {"SUSPECT", "SPOOF_SUSPECT"}:
                # Recognition may be visually plausible while the separate
                # liveness gate correctly blocks the identity from becoming
                # an attendance decision. Keep that failure in liveness
                # reporting instead of misclassifying it as recognition FRR.
                outcome = "liveness_blocked"
            elif not expected_unknown and identity_state in {
                "UNRESOLVED", "CANDIDATE", "OCCLUDED"
            }:
                # Temporal confirmation and quality gating intentionally defer
                # a decision. A deferred frame is not a false rejection.
                outcome = "deferred"
            else:
                outcome = "false_reject"
            rows.append({
                "frame_id": frame_id,
                "track_id": track_id,
                "expected": expected,
                "predicted": predicted,
                "identity_state": identity_state,
                "liveness_state": liveness_state,
                "outcome": outcome,
            })
    counts = Counter(row["outcome"] for row in rows)
    deferred_outcomes = {"deferred", "liveness_blocked"}
    resolved_rows = [row for row in rows if row["outcome"] not in deferred_outcomes]
    resolved_samples = len(resolved_rows)
    correct = counts["true_identity"] + counts["true_unknown_reject"]
    unknown_trials = counts["true_unknown_reject"] + counts["false_accept"]
    known_trials = counts["true_identity"] + counts["false_reject"]
    return {
        "status": _evaluation_status(resolved_samples),
        # ``samples`` is the number of resolved recognition decisions used in
        # error rates. ``observed_samples`` retains the full labelled frame
        # count so temporal warm-up is visible without being misreported as a
        # false rejection.
        "samples": resolved_samples,
        "observed_samples": len(rows),
        "deferred": counts["deferred"],
        "liveness_blocked": counts["liveness_blocked"],
        "correct": correct,
        "accuracy": correct / resolved_samples if resolved_samples else None,
        "false_accepts": counts["false_accept"],
        "false_rejects": counts["false_reject"],
        "true_unknown_rejects": counts["true_unknown_reject"],
        "false_accept_rate": _safe_rate(counts["false_accept"], unknown_trials),
        "false_reject_rate": _safe_rate(counts["false_reject"], known_trials),
        "unknown_trials": unknown_trials,
        "known_trials": known_trials,
        "unknown_rejection_accuracy": (
            counts["true_unknown_reject"] /
            (counts["true_unknown_reject"] + counts["false_accept"])
            if counts["true_unknown_reject"] + counts["false_accept"] else None
        ),
        "outcomes": dict(counts),
        "rows": rows[:200],
    }


def _liveness_evaluation(scenario: dict, result: dict) -> dict:
    predictions = _predicted_face_by_frame(result)
    rows = []
    for frame in scenario.get("frames") or []:
        for face in frame.get("faces") or []:
            if not isinstance(face, dict) or "expected_live" not in face:
                continue
            try:
                track_id = int(face.get("oid"))
            except (TypeError, ValueError):
                continue
            decision = predictions.get((frame.get("frame_id"), track_id), {})
            predicted_live = str(decision.get("liveness_state") or "NOT_EVALUATED").upper() == "REAL"
            expected_live = bool(face.get("expected_live"))
            rows.append({
                "frame_id": frame.get("frame_id"),
                "track_id": track_id,
                "expected_live": expected_live,
                "predicted_live": predicted_live,
            })
    labels = [row["expected_live"] for row in rows]
    scores = [1.0 if row["predicted_live"] else 0.0 for row in rows]
    metrics = binary_metrics(labels, scores, 0.5).to_dict() if rows else {
        "true_positive": 0, "true_negative": 0, "false_positive": 0,
        "false_negative": 0, "precision": None, "recall": None,
        "false_accept_rate": None, "false_reject_rate": None, "samples": 0,
    }
    metrics.update({"status": _evaluation_status(len(rows)), "rows": rows[:200]})
    return metrics


def _tracking_evaluation(scenario: dict) -> dict:
    matches = []
    for index, frame in enumerate(scenario.get("frames") or []):
        for track in frame.get("tracks") or []:
            if not isinstance(track, dict) or track.get("ground_truth_id") is None:
                continue
            try:
                predicted = int(track.get("id"))
            except (TypeError, ValueError):
                predicted = None
            matches.append((index, str(track["ground_truth_id"]), predicted))
    result = tracking_id_switches(matches)
    result["status"] = _evaluation_status(len(matches))
    result["id_switch_rate"] = _safe_rate(
        result["id_switches"], result["matched_observations"])
    return result


def _security_evaluation(scenario: dict, result: dict) -> dict:
    expected = scenario.get("expected_security_events")
    if not isinstance(expected, list):
        return {"status": "NOT_MEASURED", "expected": [], "predicted": []}
    predicted = [str(item.get("event_type")) for item in result.get("security_events") or []]
    expected_counts = Counter(map(str, expected))
    predicted_counts = Counter(predicted)
    true_positive = sum(min(expected_counts[key], predicted_counts[key]) for key in expected_counts)
    false_positive = sum(max(0, predicted_counts[key] - expected_counts[key]) for key in predicted_counts)
    false_negative = sum(max(0, expected_counts[key] - predicted_counts[key]) for key in expected_counts)
    metrics = BinaryMetrics(true_positive, 0, false_positive, false_negative).to_dict()
    duration_hours = scenario.get("duration_hours")
    metrics["false_alerts_per_camera_hour"] = (
        false_positive / float(duration_hours)
        if duration_hours is not None and float(duration_hours) > 0 else None
    )
    metrics.update({"status": _evaluation_status(len(expected)), "duration_hours": duration_hours,
                    "expected": sorted(set(map(str, expected))),
                    "predicted": sorted(set(predicted))})
    return metrics


def _vehicle_evaluation(scenario: dict) -> dict:
    rows = scenario.get("vehicle_speed_labels")
    if not isinstance(rows, list):
        return {"status": "NOT_MEASURED", "samples": 0, "mean_absolute_error_kmh": None}
    expected = []
    predicted = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            expected.append(float(row["expected_kmh"]))
            predicted.append(float(row["predicted_kmh"]))
        except (KeyError, TypeError, ValueError):
            continue
    return {
        "status": _evaluation_status(len(expected)),
        "samples": len(expected),
        "mean_absolute_error_kmh": mean_absolute_error(expected, predicted),
        "calibration_required": True,
    }


def _plate_evaluation(scenario: dict) -> dict:
    rows = scenario.get("plate_labels")
    if not isinstance(rows, list):
        return {"status": "NOT_MEASURED", "samples": 0}
    expected = [row.get("expected") for row in rows if isinstance(row, dict)]
    predicted = [row.get("predicted") for row in rows if isinstance(row, dict)]
    return plate_metrics(expected, predicted)


def _demographic_evaluation(scenario: dict) -> dict:
    rows = scenario.get("age_band_labels")
    if not isinstance(rows, list):
        return {"status": "NOT_MEASURED", "samples": 0}
    expected = [row.get("expected_band") for row in rows if isinstance(row, dict)]
    predicted = [row.get("predicted_band") for row in rows if isinstance(row, dict)]
    result = categorical_metrics(expected, predicted)
    result["decision_use"] = "aggregate_only"
    return result


def _density_evaluation(scenario: dict) -> dict:
    rows = scenario.get("density_labels")
    if not isinstance(rows, list):
        return {"status": "NOT_MEASURED", "samples": 0}
    expected = [row.get("expected_count") for row in rows if isinstance(row, dict)]
    predicted = [row.get("predicted_count") for row in rows if isinstance(row, dict)]
    return density_metrics(expected, predicted)


def evaluate_replay(scenario: dict, result: dict) -> dict:
    """Build a capability report from a replay and optional ground-truth labels.

    A replay without ``expected_*`` labels still proves policy invariants, but
    it is intentionally reported as ``NOT_MEASURED`` for model accuracy.
    """
    recognition = _identity_evaluation(scenario, result)
    liveness = _liveness_evaluation(scenario, result)
    tracking = _tracking_evaluation(scenario)
    security = _security_evaluation(scenario, result)
    vehicle = _vehicle_evaluation(scenario)
    plates = _plate_evaluation(scenario)
    demographics = _demographic_evaluation(scenario)
    density = _density_evaluation(scenario)
    measured_sections = (
        recognition["samples"] > 0,
        liveness["samples"] > 0,
        tracking["matched_observations"] > 0,
        security["status"] == "MEASURED",
        vehicle["status"] == "MEASURED",
        plates["status"] == "MEASURED",
        demographics["status"] == "MEASURED",
        density["status"] == "MEASURED",
    )
    return {
        "schema_version": 1,
        "status": "MEASURED" if any(measured_sections) else "NOT_MEASURED",
        "recognition": recognition,
        "liveness": liveness,
        "tracking": tracking,
        "security": security,
        "vehicle": vehicle,
        "plate": plates,
        "demographics": demographics,
        "density": density,
        "coverage": {
            "frames": len(scenario.get("frames") or []),
            "labelled_identity_samples": recognition["observed_samples"],
            "labelled_liveness_samples": liveness["samples"],
            "labelled_tracking_samples": tracking["matched_observations"],
        },
        "notes": [
            "Replay evaluates correlation and policy behavior; it does not replace live model validation.",
            "Accuracy metrics require independent expected_* labels and a holdout or adversarial split.",
            "Vehicle speed accuracy is meaningful only when predicted values came from a valid calibration profile.",
        ],
    }


def evaluate_acceptance(report: dict, thresholds: Optional[dict]) -> dict:
    """Apply explicit promotion thresholds to an aggregate evaluation.

    Thresholds use a deliberately small, auditable schema:

    ``{"recognition": {"accuracy_min": 0.95,
    "false_accept_rate_max": 0.01},
    "coverage": {"required_tags": ["known", "unknown", "occlusion"]}}``.

    Missing or unmeasured metrics never pass. No default threshold is
    invented because acceptable error depends on the deployment and risk
    policy. This gate is for promotion decisions, not for tuning a camera.
    """
    if thresholds is None:
        return {"status": "NOT_CONFIGURED", "checks": [], "thresholds": {}}
    if not isinstance(thresholds, dict):
        raise ValueError("acceptance thresholds must be an object")
    aggregate = report.get("aggregate") or {}
    checks = []
    for section, rules in thresholds.items():
        if not isinstance(rules, dict):
            raise ValueError(f"thresholds for {section!r} must be an object")
        if str(section) == "coverage":
            coverage = report.get("coverage") or {}
            for rule_name, expected in rules.items():
                if rule_name not in {"required_scenarios", "required_tags"}:
                    raise ValueError(
                        f"coverage threshold {rule_name!r} must be required_scenarios or required_tags")
                if not isinstance(expected, list) or not expected:
                    raise ValueError(f"coverage.{rule_name} must be a non-empty list")
                available_key = "scenario_ids" if rule_name == "required_scenarios" else "scenario_tags"
                available = {
                    str(item).strip().casefold()
                    for item in (coverage.get(available_key) or [])
                }
                for required in expected:
                    requested = str(required).strip()
                    if not requested:
                        raise ValueError(f"coverage.{rule_name} cannot contain an empty value")
                    present = requested.casefold() in available
                    checks.append({
                        "section": "coverage",
                        "metric": rule_name,
                        "required": requested,
                        "actual": requested if present else None,
                        "status": "PASS" if present else "FAIL",
                        "reason": "scenario coverage present" if present else "required scenario coverage is missing",
                    })
            continue
        metrics = aggregate.get(str(section))
        if not isinstance(metrics, dict):
            raise ValueError(f"unknown evaluation section: {section}")
        for rule, expected in rules.items():
            rule_name = str(rule)
            if rule_name.endswith("_min"):
                metric_name, operator = rule_name[:-4], ">="
            elif rule_name.endswith("_max"):
                metric_name, operator = rule_name[:-4], "<="
            else:
                raise ValueError(
                    f"threshold {section}.{rule_name} must end with _min or _max")
            try:
                limit = float(expected)
            except (TypeError, ValueError):
                raise ValueError(f"threshold {section}.{rule_name} must be numeric")
            value = metrics.get(metric_name)
            status = "PASS"
            reason = "within threshold"
            if metrics.get("status") != "MEASURED" or value is None:
                status = "NOT_MEASURED"
                reason = "metric is not measured"
            else:
                try:
                    actual = float(value)
                except (TypeError, ValueError):
                    status = "NOT_MEASURED"
                    reason = "metric is not numeric"
                else:
                    passed = actual >= limit if operator == ">=" else actual <= limit
                    if not passed:
                        status = "FAIL"
                        reason = f"{actual:g} {operator} {limit:g} is false"
            checks.append({
                "section": str(section),
                "metric": metric_name,
                "operator": operator,
                "threshold": limit,
                "actual": value,
                "status": status,
                "reason": reason,
            })
    statuses = {check["status"] for check in checks}
    overall = (
        "FAIL" if "FAIL" in statuses
        else "NOT_MEASURED" if "NOT_MEASURED" in statuses or not checks
        else "PASS"
    )
    return {"status": overall, "checks": checks, "thresholds": thresholds}
