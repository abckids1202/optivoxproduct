"""Evaluate original and augmented embedding galleries without exposing vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.recognition_evaluation import (
    RecognitionEvaluationError,
    compare_embedding_galleries,
    evaluate_embedding_gallery,
)


def _load_records(path: Path, key: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get(key), list):
        records = payload[key]
    else:
        raise ValueError(f"{path} must be a list or an object containing {key!r}")
    if any(not isinstance(item, dict) for item in records):
        raise ValueError(f"{path} contains a non-object record")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate local face-embedding galleries against labelled probes."
    )
    parser.add_argument("--gallery", type=Path, required=True,
                        help="JSON gallery containing samples with embeddings.")
    parser.add_argument("--probes", type=Path, required=True,
                        help="JSON probes containing embeddings and identities.")
    parser.add_argument("--augmented-gallery", type=Path, default=None,
                        help="Optional JSON gallery for original-plus-augmented comparison.")
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--min-margin", type=float, default=0.08)
    parser.add_argument(
        "--require-independent-probes",
        action="store_true",
        help="Require train/validation galleries and holdout/adversarial probes with source IDs.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        gallery = _load_records(args.gallery, "samples")
        probes = _load_records(args.probes, "probes")
        if args.augmented_gallery:
            report = compare_embedding_galleries(
                gallery,
                _load_records(args.augmented_gallery, "samples"),
                probes,
            threshold=args.threshold,
            min_margin=args.min_margin,
            require_independent_probes=args.require_independent_probes,
        )
        else:
            report = evaluate_embedding_gallery(
                gallery,
                probes,
                threshold=args.threshold,
                min_margin=args.min_margin,
                include_augmented=False,
            )
        encoded = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
        return 0
    except (OSError, ValueError, json.JSONDecodeError, RecognitionEvaluationError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
