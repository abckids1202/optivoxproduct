"""Run a deterministic OptiVox policy replay without opening a camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.replay import run_replay
from core.evaluation import evaluate_replay


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay OptiVox trust and security decisions.")
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
    result = run_replay(scenario)
    result["evaluation"] = evaluate_replay(scenario, result)
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
