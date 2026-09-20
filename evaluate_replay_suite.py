"""Evaluate one or more labelled OptiVox replay scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.evaluation_suite import load_scenarios, run_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic OptiVox replay evaluation suite.")
    parser.add_argument("source", type=Path, help="Scenario JSON file or directory of JSON scenarios")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--thresholds", type=Path, default=None,
        help="Optional JSON object containing promotion thresholds")
    args = parser.parse_args()
    try:
        thresholds = None
        if args.thresholds is not None:
            thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
        report = run_suite(load_scenarios(args.source), thresholds)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        return 2
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if report["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
