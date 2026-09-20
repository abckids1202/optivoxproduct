"""Model-agnostic evaluation for OptiVox face-embedding galleries.

This module evaluates embedding decisions without loading cameras, models, or
the production biometric database. It is intentionally local and report-only:
the returned result contains metrics and sample counts, never vectors.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


UNKNOWN_LABELS = frozenset({"", "unknown", "unresolved", "none", "stranger"})
GALLERY_KINDS = frozenset({"original", "augmented"})


class RecognitionEvaluationError(ValueError):
    """Raised when an embedding evaluation input is unsafe or malformed."""


def _label(value: Any) -> str:
    return str(value or "").strip()


def _is_unknown(value: Any) -> bool:
    return _label(value).casefold() in UNKNOWN_LABELS


def _vector(value: Any, field: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise RecognitionEvaluationError(f"{field} must be a numeric vector") from exc
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise RecognitionEvaluationError(f"{field} must be finite and non-empty")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RecognitionEvaluationError(f"{field} must have a non-zero norm")
    return vector / norm


def _sample(value: dict, field: str) -> dict:
    if not isinstance(value, dict):
        raise RecognitionEvaluationError(f"{field} sample must be an object")
    identity = _label(value.get("identity"))
    if not identity or _is_unknown(identity):
        raise RecognitionEvaluationError(f"{field} gallery samples require a known identity")
    kind = _label(value.get("kind") or "original").casefold()
    if kind not in GALLERY_KINDS:
        raise RecognitionEvaluationError(f"{field} sample has unsupported kind {kind!r}")
    sample_id = _label(value.get("sample_id") or value.get("id"))
    if not sample_id:
        raise RecognitionEvaluationError(f"{field} sample requires sample_id")
    source_id = _label(value.get("source_id")) or None
    if kind == "augmented" and not (_label(value.get("derived_from")) or source_id):
        raise RecognitionEvaluationError(
            f"{field} augmented sample requires derived_from or source_id"
        )
    return {
        "sample_id": sample_id,
        "identity": identity,
        "kind": kind,
        "source_id": source_id,
        "split": _label(value.get("split") or "train").casefold(),
        "embedding": _vector(value.get("embedding"), f"{field}.{sample_id}.embedding"),
    }


def _gallery_by_identity(gallery: Iterable[dict], include_augmented: bool) -> dict[str, list[np.ndarray]]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    seen_ids: set[str] = set()
    seen_sources: dict[str, set[str]] = defaultdict(set)
    dimensions: int | None = None
    for index, raw in enumerate(gallery):
        sample = _sample(raw, f"gallery[{index}]")
        if sample["sample_id"] in seen_ids:
            raise RecognitionEvaluationError(f"duplicate gallery sample_id: {sample['sample_id']}")
        seen_ids.add(sample["sample_id"])
        if sample["kind"] == "augmented" and not include_augmented:
            continue
        vector = sample["embedding"]
        dimensions = dimensions or int(vector.size)
        if int(vector.size) != dimensions:
            raise RecognitionEvaluationError("gallery embeddings have inconsistent dimensions")
        if sample["source_id"]:
            seen_sources[sample["identity"]].add(sample["source_id"])
        grouped[sample["identity"]].append(vector)
    if not grouped:
        raise RecognitionEvaluationError("gallery has no usable samples")
    return dict(grouped)


def _score_probe(probe: dict, gallery: dict[str, list[np.ndarray]], threshold: float,
                 margin: float) -> dict:
    vector = _vector(probe.get("embedding"), "probe.embedding")
    try:
        scores = sorted([
            (
                float(max(float(np.dot(vector, sample)) for sample in samples)),
                identity,
            )
            for identity, samples in gallery.items()
        ], reverse=True)
    except ValueError as exc:
        raise RecognitionEvaluationError(
            "probe and gallery embeddings have inconsistent dimensions"
        ) from exc
    best_score, best_identity = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else -1.0
    score_margin = best_score - second_score if len(scores) > 1 else best_score
    accepted = best_score >= threshold and score_margin >= margin
    return {
        "predicted_identity": best_identity if accepted else "UNKNOWN",
        "best_score": round(best_score, 6),
        "second_score": round(second_score, 6) if len(scores) > 1 else None,
        "margin": round(score_margin, 6),
        "accepted": accepted,
    }


def evaluate_embedding_gallery(
    gallery: Iterable[dict],
    probes: Iterable[dict],
    threshold: float = 0.62,
    min_margin: float = 0.08,
    include_augmented: bool = False,
    require_independent_probes: bool = False,
) -> dict[str, Any]:
    """Evaluate a gallery against labelled probes without exposing vectors.

    ``require_independent_probes`` is the promotion-safe mode. It requires
    gallery samples to come from train/validation and probes to declare an
    explicit holdout/adversarial split plus a source identifier. The default
    remains permissive for small local smoke tests, but those tests must not
    be presented as independent accuracy evidence.
    """
    try:
        threshold = float(threshold)
        min_margin = float(min_margin)
    except (TypeError, ValueError) as exc:
        raise RecognitionEvaluationError("thresholds must be numeric") from exc
    if not 0.0 <= threshold <= 1.0:
        raise RecognitionEvaluationError("threshold must be between 0 and 1")
    if not 0.0 <= min_margin <= 1.0:
        raise RecognitionEvaluationError("min_margin must be between 0 and 1")

    # Materialize once because callers may provide generators. The same
    # validated records are used for scoring and provenance checks.
    gallery_list = list(gallery)
    probe_list = list(probes)
    grouped = _gallery_by_identity(gallery_list, include_augmented=include_augmented)
    # Re-parse the gallery only for provenance checks. The scoring structure
    # intentionally contains vectors in memory, but no vector is ever put in
    # the returned report.
    gallery_records = [
        _sample(raw, f"gallery[{index}]")
        for index, raw in enumerate(gallery_list)
    ]
    if require_independent_probes:
        invalid_gallery_splits = sorted({
            record["split"] for record in gallery_records
            if record["split"] not in {"train", "validation"}
        })
        if invalid_gallery_splits:
            raise RecognitionEvaluationError(
                "independent evaluation gallery must use train or validation splits: "
                + ", ".join(invalid_gallery_splits)
            )
    gallery_sources = {
        record["source_id"]
        for record in gallery_records
        if record["source_id"]
    }
    gallery_vectors = [record["embedding"] for record in gallery_records]
    rows = []
    for index, raw_probe in enumerate(probe_list):
        if not isinstance(raw_probe, dict):
            raise RecognitionEvaluationError(f"probes[{index}] must be an object")
        expected = _label(raw_probe.get("identity") or raw_probe.get("expected_identity"))
        if not expected:
            raise RecognitionEvaluationError(f"probes[{index}] requires identity")
        probe_split = _label(raw_probe.get("split")).casefold()
        if require_independent_probes and probe_split not in {"holdout", "adversarial"}:
            raise RecognitionEvaluationError(
                f"probes[{index}] must use holdout or adversarial split"
            )
        probe_source = _label(raw_probe.get("source_id")) or None
        if require_independent_probes and not probe_source:
            raise RecognitionEvaluationError(
                f"probes[{index}] requires source_id in independent mode"
            )
        if probe_source and probe_source in gallery_sources:
            raise RecognitionEvaluationError(
                f"probe source overlaps gallery source: {probe_source}"
            )
        probe_vector = _vector(raw_probe.get("embedding"), f"probe[{index}].embedding")
        if any(
            probe_vector.size == gallery_vector.size
            and float(np.dot(probe_vector, gallery_vector)) >= 0.9999999
            for gallery_vector in gallery_vectors
        ):
            raise RecognitionEvaluationError(
                f"probe[{index}] is an exact embedding duplicate of gallery evidence"
            )
        decision = _score_probe(raw_probe, grouped, threshold, min_margin)
        predicted = decision["predicted_identity"]
        expected_unknown = _is_unknown(expected)
        predicted_unknown = _is_unknown(predicted)
        if expected_unknown and predicted_unknown:
            outcome = "true_unknown_reject"
        elif expected_unknown:
            outcome = "false_accept"
        elif predicted.casefold() == expected.casefold():
            outcome = "true_identity"
        else:
            outcome = "false_reject"
        rows.append({
            "sample_id": _label(raw_probe.get("sample_id") or raw_probe.get("id")) or f"probe-{index + 1}",
            "expected": expected,
            "predicted": predicted,
            "outcome": outcome,
            "best_score": decision["best_score"],
            "second_score": decision["second_score"],
            "margin": decision["margin"],
        })

    known = [row for row in rows if not _is_unknown(row["expected"])]
    unknown = [row for row in rows if _is_unknown(row["expected"])]
    correct = sum(row["outcome"] in {"true_identity", "true_unknown_reject"} for row in rows)
    false_accepts = sum(row["outcome"] == "false_accept" for row in rows)
    false_rejects = sum(row["outcome"] == "false_reject" for row in rows)
    true_unknown_rejects = sum(row["outcome"] == "true_unknown_reject" for row in rows)
    return {
        "status": "MEASURED" if rows else "NOT_MEASURED",
        "gallery_mode": "original_plus_augmented" if include_augmented else "original_only",
        "identities": len(grouped),
        "gallery_samples": sum(len(samples) for samples in grouped.values()),
        "probes": len(rows),
        "known_trials": len(known),
        "unknown_trials": len(unknown),
        "correct": int(correct),
        "accuracy": correct / len(rows) if rows else None,
        "false_accepts": int(false_accepts),
        "false_rejects": int(false_rejects),
        "true_unknown_rejects": int(true_unknown_rejects),
        "false_accept_rate": false_accepts / len(unknown) if unknown else None,
        "false_reject_rate": false_rejects / len(known) if known else None,
        "unknown_rejection_accuracy": true_unknown_rejects / len(unknown) if unknown else None,
        "rows": rows[:200],
        "notes": [
            "Embedding evaluation measures gallery decisions, not detector or liveness accuracy.",
            "Use independent validation, holdout, and adversarial probes; do not reuse enrollment images.",
        ],
    }


def compare_embedding_galleries(
    original_gallery: Iterable[dict],
    augmented_gallery: Iterable[dict],
    probes: Iterable[dict],
    threshold: float = 0.62,
    min_margin: float = 0.08,
    require_independent_probes: bool = False,
) -> dict[str, Any]:
    """Compare augmentation and reject it when false accepts increase."""
    probe_list = list(probes)
    original = evaluate_embedding_gallery(
        original_gallery, probe_list, threshold, min_margin,
        include_augmented=False,
        require_independent_probes=require_independent_probes,
    )
    augmented = evaluate_embedding_gallery(
        augmented_gallery, probe_list, threshold, min_margin,
        include_augmented=True,
        require_independent_probes=require_independent_probes,
    )
    original_far = original["false_accept_rate"]
    augmented_far = augmented["false_accept_rate"]
    original_frr = original["false_reject_rate"]
    augmented_frr = augmented["false_reject_rate"]
    if original_far is None or augmented_far is None:
        recommendation = "INSUFFICIENT_UNKNOWN_PROBES"
    elif augmented_far > original_far:
        recommendation = "REJECT_AUGMENTATION_FALSE_ACCEPT_INCREASE"
    elif original_frr is not None and augmented_frr is not None and augmented_frr < original_frr:
        recommendation = "KEEP_AUGMENTATION"
    else:
        recommendation = "NO_CLEAR_GAIN"
    return {
        "status": "MEASURED" if original["status"] == "MEASURED" and augmented["status"] == "MEASURED" else "NOT_MEASURED",
        "original_only": original,
        "original_plus_augmented": augmented,
        "delta": {
            "accuracy": _delta(original["accuracy"], augmented["accuracy"]),
            "false_accept_rate": _delta(original_far, augmented_far),
            "false_reject_rate": _delta(original_frr, augmented_frr),
            "unknown_rejection_accuracy": _delta(
                original["unknown_rejection_accuracy"],
                augmented["unknown_rejection_accuracy"],
            ),
        },
        "recommendation": recommendation,
        "notes": [
            "Augmentation is not accepted solely because accuracy increases.",
            "A false-accept increase is a hard rejection for attendance identity use.",
        ],
    }


def _delta(before: Any, after: Any) -> float | None:
    if before is None or after is None:
        return None
    return round(float(after) - float(before), 6)
