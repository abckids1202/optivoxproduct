"""Collect a non-invasive benchmark from a running OptiVox edge runtime.

This observer never opens the camera and never writes to the operational
database. It samples the atomic runtime heartbeat/live-state files, aggregates
the available measurements, and preserves missing hardware values as null.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = ROOT / "runtime"
REPORT_DIR = ROOT / "reports"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _numbers(values: list[Any]) -> list[float]:
    output = []
    for value in values:
        try:
            if value is not None:
                output.append(float(value))
        except (TypeError, ValueError):
            continue
    return output


def _summary(values: list[Any]) -> dict[str, Any]:
    numbers = _numbers(values)
    if not numbers:
        return {"status": "NOT_MEASURED", "samples": 0, "average": None,
                "p95": None, "minimum": None, "maximum": None}
    ordered = sorted(numbers)
    p95 = ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))]
    return {
        "status": "MEASURED",
        "samples": len(numbers),
        "average": round(statistics.fmean(numbers), 3),
        "p95": round(p95, 3),
        "minimum": round(min(numbers), 3),
        "maximum": round(max(numbers), 3),
    }


def _metric_average(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name) or {}
    average = value.get("average") if isinstance(value, dict) else None
    return float(average) if average is not None else None


def _counter_rate(samples: list[dict[str, Any]], field: str) -> float | None:
    """Convert a cumulative runtime counter into a measured per-second rate."""
    usable = [sample for sample in samples if sample.get(field) is not None]
    if len(usable) < 2:
        return None
    first, last = usable[0], usable[-1]
    try:
        elapsed = float(last["sampled_monotonic"]) - float(first["sampled_monotonic"])
        delta = float(last[field]) - float(first[field])
    except (KeyError, TypeError, ValueError):
        return None
    if elapsed <= 0 or delta < 0:
        return None
    return round(delta / elapsed, 3)


def _canonical_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Project the observer report into the runtime summary contract."""
    metrics = result.get("metrics") or {}
    stage_metrics = result.get("stage_metrics") or {}
    model_metrics = {}
    for source_name, target_name in (
        ("face_detection_latency_ms", "face_detection"),
        ("recognition_latency_ms", "identity_matching"),
        ("yolo_latency_ms", "yolo"),
        ("pose_latency_ms", "pose"),
    ):
        summary = stage_metrics.get(source_name) or {
            "status": "NOT_MEASURED", "average": None,
            "p95": None, "samples": 0,
        }
        model_metrics[target_name] = {
            "calls_per_second": None,
            "average_latency_ms": summary.get("average"),
            "p95_latency_ms": summary.get("p95"),
            "window_calls": summary.get("samples", 0),
        }
    recognition_rate = _metric_average(metrics, "recognition_attempts_per_second")
    if recognition_rate is not None:
        model_metrics["identity_matching"]["calls_per_second"] = recognition_rate
    return {
        "schema_version": 2,
        "measurement_status": result.get("measurement_status", "NOT_MEASURED"),
        "capture_fps": _metric_average(metrics, "capture_fps"),
        "inference_fps": _metric_average(metrics, "inference_fps"),
        "display_fps": _metric_average(metrics, "display_fps"),
        "latency_ms": {
            "frame_age_p95": _metric_average(metrics, "frame_age_ms_p95"),
            "inference_avg": _metric_average(metrics, "inference_latency_ms"),
            "inference_p95": _metric_average(metrics, "inference_p95_ms"),
            "end_to_end_p95": None,
        },
        "vision": {
            "models": model_metrics,
            "identity_timing": {
                "mean_time_to_confirm_sec": (
                    _metric_average(metrics, "identity_confirmation_ms") or 0.0
                ) / 1000.0 if _metric_average(metrics, "identity_confirmation_ms") is not None else None,
            },
        },
        "resource": {
            "cpu_percent": _metric_average(metrics, "cpu_percent"),
            "gpu_percent": _metric_average(metrics, "gpu_percent"),
            "vram_used_mb": _metric_average(metrics, "vram_used_mb"),
        },
        "counters": {
            "frames_replaced": result.get("totals", {}).get("frames_replaced"),
            "stale_frames_dropped": result.get("totals", {}).get("stale_frames_dropped"),
            "side_effect_queue_drops": result.get("totals", {}).get("side_effect_queue_drops"),
            "critical_queue_drops": result.get("totals", {}).get("critical_queue_drops"),
        },
        "benchmark": {
            "started_at": result.get("started_at"),
            "ended_at": result.get("ended_at"),
            "duration_sec": result.get("duration_seconds"),
            "source": "benchmark_runtime.py observer",
        },
    }


def collect(duration_seconds: float = 600.0, interval_seconds: float = 1.0,
            runtime_dir: Path = RUNTIME_DIR) -> dict[str, Any]:
    started = time.time()
    samples: list[dict[str, Any]] = []
    last_sample_key = None
    attendance_decisions = 0
    security_observations = 0
    while time.time() - started < max(1.0, duration_seconds):
        heartbeat = _read_json(runtime_dir / "heartbeat.json") or {}
        live = _read_json(runtime_dir / "live_state.json") or {}
        performance = heartbeat.get("performance") or live.get("performance") or {}
        counters = performance.get("counters") or {}
        frame_ids = performance.get("frame_ids") or {}
        current_frame_id = frame_ids.get("source")
        sample_key = current_frame_id if current_frame_id is not None else heartbeat.get("timestamp")
        if sample_key is not None and sample_key != last_sample_key:
            last_sample_key = sample_key
            samples.append({
                "sampled_monotonic": time.monotonic(),
                "timestamp": heartbeat.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                "fps": heartbeat.get("fps"),
                "capture_fps": performance.get("capture_fps"),
                "inference_fps": performance.get("inference_fps"),
                "display_fps": performance.get("display_fps"),
                "frame_age_ms": (performance.get("latency_ms") or {}).get("latest_frame_age"),
                "inference_latency_ms": (performance.get("latency_ms") or {}).get("inference_avg"),
                "inference_p95_ms": (performance.get("latency_ms") or {}).get("inference_p95"),
                "cpu_percent": (performance.get("resource") or {}).get("cpu_percent"),
                "gpu_percent": (performance.get("resource") or {}).get("gpu_percent"),
                "vram_used_mb": (performance.get("resource") or {}).get("vram_used_mb"),
                "frames_replaced": counters.get("frames_replaced"),
                "stale_frames_dropped": counters.get("stale_frames_dropped"),
                "side_effect_queue_drops": counters.get("side_effect_queue_drops"),
                "critical_queue_drops": counters.get("critical_queue_drops"),
                "recognition_attempts": (performance.get("vision") or {}).get("recognition_attempts"),
                "cache_hits": (performance.get("vision") or {}).get("recognition_cache_hits"),
                "identity_confirmations": (performance.get("vision") or {}).get("recognition_confirmations"),
                "face_detection_latency_ms": ((performance.get("vision") or {}).get("models") or {}).get("face_detection", {}).get("average_latency_ms"),
                "recognition_latency_ms": ((performance.get("vision") or {}).get("models") or {}).get("identity_matching", {}).get("average_latency_ms"),
                "yolo_latency_ms": ((performance.get("vision") or {}).get("models") or {}).get("yolo", {}).get("average_latency_ms"),
                "pose_latency_ms": ((performance.get("vision") or {}).get("models") or {}).get("pose", {}).get("average_latency_ms"),
                "identity_confirmation_ms": (((performance.get("vision") or {}).get("identity_timing") or {}).get("mean_time_to_confirm_sec") * 1000
                                             if ((performance.get("vision") or {}).get("identity_timing") or {}).get("mean_time_to_confirm_sec") is not None else None),
            })
        attendance_decisions = max(attendance_decisions, int((live.get("summary") or {}).get("presentToday", 0) or 0))
        security_observations = max(security_observations, int((live.get("security") or {}).get("active_event_count", 0) or 0))
        time.sleep(max(0.1, interval_seconds))

    fields = [
        "capture_fps", "inference_fps", "display_fps", "frame_age_ms",
        "inference_latency_ms", "inference_p95_ms", "cpu_percent",
        "gpu_percent", "vram_used_mb", "recognition_attempts", "cache_hits",
        "identity_confirmations", "frames_replaced", "stale_frames_dropped",
        "side_effect_queue_drops", "critical_queue_drops",
    ]
    metrics = {field: _summary([sample.get(field) for sample in samples]) for field in fields}
    metrics.update({
        "frame_age_ms_p95": _summary([sample.get("frame_age_ms") for sample in samples]),
        "face_detection_latency_ms": _summary([sample.get("face_detection_latency_ms") for sample in samples]),
        "recognition_latency_ms": _summary([sample.get("recognition_latency_ms") for sample in samples]),
        "yolo_latency_ms": _summary([sample.get("yolo_latency_ms") for sample in samples]),
        "pose_latency_ms": _summary([sample.get("pose_latency_ms") for sample in samples]),
        "identity_confirmation_ms": _summary([sample.get("identity_confirmation_ms") for sample in samples]),
        "recognition_attempts_per_second": _summary(
            [_counter_rate(samples[:index + 1], "recognition_attempts") for index in range(len(samples))]
        ),
    })
    totals = {
        field: next((sample.get(field) for sample in reversed(samples) if sample.get(field) is not None), None)
        for field in ("frames_replaced", "stale_frames_dropped", "side_effect_queue_drops", "critical_queue_drops")
    }
    return {
        "schema_version": 1,
        "measurement_status": "MEASURED" if samples else "NOT_MEASURED",
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - started, 3),
        "sample_count": len(samples),
        "runtime_dir": str(runtime_dir),
        "metrics": metrics,
        "stage_metrics": {
            name: metrics[name]
            for name in ("face_detection_latency_ms", "recognition_latency_ms", "yolo_latency_ms", "pose_latency_ms")
        },
        "totals": totals,
        "observed_outcomes": {
            "max_present_today": attendance_decisions,
            "max_active_security_events": security_observations,
        },
        "samples": samples,
        "notes": [
            "This observer reports runtime telemetry; it does not prove recognition accuracy.",
            "GPU and VRAM remain NOT_MEASURED unless a collector is configured.",
            "Run the scenario checklist separately for known, unknown, occluded, spoof, and intrusion cases.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Observe a running OptiVox runtime.")
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--publish-runtime-summary", action="store_true",
        help="Also publish a dashboard-compatible summary to runtime/performance_summary.json.",
    )
    args = parser.parse_args()
    result = collect(args.seconds, args.interval)
    REPORT_DIR.mkdir(exist_ok=True)
    output = args.output or REPORT_DIR / f"benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.publish_runtime_summary:
        runtime_summary = RUNTIME_DIR / "performance_summary.json"
        runtime_summary.write_text(
            json.dumps(_canonical_summary(result), indent=2), encoding="utf-8")
        print(f"[BENCHMARK] runtime summary={runtime_summary}")
    print(f"[BENCHMARK] {result['measurement_status']} samples={result['sample_count']} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
