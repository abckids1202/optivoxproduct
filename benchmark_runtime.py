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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = ROOT / "runtime"
REPORT_DIR = ROOT / "reports"
MIN_FRESH_SAMPLES = 2
MAX_CLOCK_SKEW_SECONDS = 5.0
ACTIVE_ENGINE_STATES = {"ONLINE", "DEGRADED"}
ACTIVE_CAMERA_STATES = {"HEALTHY", "DEGRADED", "CONNECTED", "ONLINE"}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _timestamp_age_seconds(value: Any) -> float | None:
    """Return wall-clock age for a runtime timestamp, or None if invalid."""
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()
        # A large future timestamp is not fresh telemetry. Treating it as age
        # zero would allow clock errors or forged files to pass the observer's
        # liveness gate. Small skew is tolerated for ordinary host drift.
        if age < -MAX_CLOCK_SKEW_SECONDS:
            return None
        return max(0.0, age)
    except (TypeError, ValueError, OverflowError):
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


def _metric_stat(metrics: dict[str, Any], name: str, stat: str) -> float | None:
    value = metrics.get(name) or {}
    selected = value.get(stat) if isinstance(value, dict) else None
    try:
        return float(selected) if selected is not None else None
    except (TypeError, ValueError):
        return None


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


def _is_active_heartbeat(heartbeat: dict[str, Any]) -> bool:
    """Require an online engine and usable camera before counting telemetry."""
    engine = str(heartbeat.get("engine_status") or "").upper()
    camera = str(heartbeat.get("camera_status") or "").upper()
    return engine in ACTIVE_ENGINE_STATES and camera in ACTIVE_CAMERA_STATES


def _ratio_summary(samples: list[dict[str, Any]], numerator: str,
                   denominator: str) -> dict[str, Any]:
    ratios = []
    for sample in samples:
        try:
            top = float(sample.get(numerator) or 0)
            bottom = float(sample.get(denominator) or 0)
            if bottom > 0:
                ratios.append(top / bottom)
        except (TypeError, ValueError):
            continue
    return _summary([value * 100.0 for value in ratios])


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
            "frame_age_p95": (
                _metric_stat(metrics, "frame_age_consumed_ms_p95", "p95")
                or _metric_stat(metrics, "frame_age_ms_p95", "p95")
            ),
            "stale_frame_age_p95": _metric_stat(metrics, "stale_frame_age_ms", "p95"),
            "inference_avg": _metric_average(metrics, "inference_latency_ms"),
            "inference_p95": _metric_stat(metrics, "inference_p95_ms", "p95"),
            "end_to_end_p95": _metric_stat(metrics, "end_to_end_latency_ms", "p95"),
        },
        "vision": {
            "models": model_metrics,
            "identity_timing": {
                "mean_time_to_confirm_sec": (
                    _metric_average(metrics, "identity_confirmation_ms") or 0.0
                ) / 1000.0 if _metric_average(metrics, "identity_confirmation_ms") is not None else None,
                "p95_time_to_confirm_sec": (
                    _metric_stat(metrics, "identity_confirmation_p95_ms", "p95") or 0.0
                ) / 1000.0 if _metric_stat(metrics, "identity_confirmation_p95_ms", "p95") is not None else None,
            },
        },
        "resource": {
            "cpu_percent": _metric_average(metrics, "cpu_percent"),
            "gpu_percent": _metric_average(metrics, "gpu_percent"),
            "vram_used_mb": _metric_average(metrics, "vram_used_mb"),
        },
        "counters": {
            "frames_captured": result.get("totals", {}).get("frames_captured"),
            "frames_consumed": result.get("totals", {}).get("frames_consumed"),
            "frames_inferred": result.get("totals", {}).get("frames_inferred"),
            "frames_displayed": result.get("totals", {}).get("frames_displayed"),
            "frame_ids_skipped": result.get("totals", {}).get("frame_ids_skipped"),
            "frames_replaced": result.get("totals", {}).get("frames_replaced"),
            "stale_frames_dropped": result.get("totals", {}).get("stale_frames_dropped"),
            "side_effect_queue_drops": result.get("totals", {}).get("side_effect_queue_drops"),
            "critical_queue_drops": result.get("totals", {}).get("critical_queue_drops"),
            "tasks_processed": result.get("totals", {}).get("tasks_processed"),
            "attendance_decisions": result.get("totals", {}).get("attendance_decisions"),
            "attendance_eligible": result.get("totals", {}).get("attendance_eligible"),
            "attendance_rejected": result.get("totals", {}).get("attendance_rejected"),
            "events_persisted": result.get("totals", {}).get("events_persisted"),
            "alert_events": result.get("totals", {}).get("alert_events"),
            "incidents_synced": result.get("totals", {}).get("incidents_synced"),
        },
        "recognition": {
            "cache_hit_rate_percent": _metric_average(metrics, "recognition_cache_hit_rate_percent"),
            "attempts_per_second": recognition_rate,
            "decisions": result.get("totals", {}).get("recognition_decisions"),
        },
        "benchmark": {
            "started_at": result.get("started_at"),
            "ended_at": result.get("ended_at"),
            "duration_sec": result.get("duration_seconds"),
            "required_duration_sec": result.get("required_duration_seconds"),
            "duration_satisfied": result.get("duration_satisfied", False),
            "frame_progress_verified": result.get("frame_progress_verified", False),
            "source": "benchmark_runtime.py observer",
        },
    }


def collect(duration_seconds: float = 600.0, interval_seconds: float = 1.0,
            runtime_dir: Path = RUNTIME_DIR,
            min_fresh_samples: int = MIN_FRESH_SAMPLES,
            required_duration_seconds: float = 600.0) -> dict[str, Any]:
    started = time.time()
    samples: list[dict[str, Any]] = []
    last_sample_key = None
    last_numeric_frame_id = None
    verified_frame_samples = 0
    attendance_decisions = 0
    security_observations = 0
    health_counts = Counter()
    # Keep the caller's explicit minimum intact.  The default remains two
    # samples, but tests and approved protocols may intentionally require a
    # single fresh sample while still relying on the duration gate.
    min_fresh_samples = max(1, int(min_fresh_samples))
    try:
        required_duration_seconds = max(0.0, float(required_duration_seconds))
    except (TypeError, ValueError):
        required_duration_seconds = 600.0
    while time.time() - started < max(1.0, duration_seconds):
        heartbeat = _read_json(runtime_dir / "heartbeat.json") or {}
        live = _read_json(runtime_dir / "live_state.json") or {}
        heartbeat_age = _timestamp_age_seconds(heartbeat.get("timestamp"))
        # A stale file from a previous run is not a live measurement. Without
        # this guard an offline runtime could produce a falsely "MEASURED"
        # report from one old frame ID.
        if heartbeat_age is None:
            health_counts["invalid_timestamp"] += 1
            time.sleep(max(0.1, interval_seconds))
            continue
        if heartbeat_age > 10.0:
            health_counts["stale"] += 1
            time.sleep(max(0.1, interval_seconds))
            continue
        if not _is_active_heartbeat(heartbeat):
            health_counts["inactive"] += 1
            time.sleep(max(0.1, interval_seconds))
            continue
        health_counts["active"] += 1
        performance = heartbeat.get("performance") or live.get("performance") or {}
        counters = performance.get("counters") or {}
        frame_ids = performance.get("frame_ids") or heartbeat.get("frame_ids") or {}
        current_frame_id = frame_ids.get("source")
        sample_key = current_frame_id if current_frame_id is not None else heartbeat.get("timestamp")
        if sample_key is not None and sample_key != last_sample_key:
            # A heartbeat timestamp proves that telemetry changed, not that a
            # newer camera frame was processed. Require strictly increasing
            # source frame IDs before counting a sample as fresh-frame
            # evidence; otherwise a runtime can look healthy while repeating
            # the same frame or publishing incomplete telemetry.
            numeric_frame_id = None
            try:
                if current_frame_id is not None:
                    numeric_frame_id = int(current_frame_id)
            except (TypeError, ValueError):
                health_counts["invalid_frame_id"] += 1
            if (numeric_frame_id is not None and last_numeric_frame_id is not None
                    and numeric_frame_id <= last_numeric_frame_id):
                health_counts["non_monotonic_frame_id"] += 1
                time.sleep(max(0.1, interval_seconds))
                continue
            last_sample_key = sample_key
            frame_progress_verified = numeric_frame_id is not None
            if frame_progress_verified:
                last_numeric_frame_id = numeric_frame_id
                verified_frame_samples += 1
            samples.append({
                "sampled_monotonic": time.monotonic(),
                "timestamp": heartbeat.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                "heartbeat_age_seconds": round(heartbeat_age, 3),
                "source_frame_id": numeric_frame_id,
                "frame_progress_verified": frame_progress_verified,
                "fps": heartbeat.get("fps"),
                "capture_fps": performance.get("capture_fps"),
                "inference_fps": performance.get("inference_fps"),
                "display_fps": performance.get("display_fps"),
                "frame_age_ms": (performance.get("latency_ms") or {}).get("latest_frame_age"),
                "frame_age_consumed_ms": (performance.get("latency_ms") or {}).get("frame_age_consumed_p95"),
                "stale_frame_age_ms": (performance.get("latency_ms") or {}).get("stale_frame_age_p95"),
                "end_to_end_latency_ms": (performance.get("latency_ms") or {}).get("end_to_end_p95"),
                "inference_latency_ms": (performance.get("latency_ms") or {}).get("inference_avg"),
                "inference_p95_ms": (performance.get("latency_ms") or {}).get("inference_p95"),
                "cpu_percent": (performance.get("resource") or {}).get("cpu_percent"),
                "gpu_percent": (performance.get("resource") or {}).get("gpu_percent"),
                "vram_used_mb": (performance.get("resource") or {}).get("vram_used_mb"),
                "frames_replaced": counters.get("frames_replaced"),
                "frames_captured": counters.get("frames_captured"),
                "frames_consumed": counters.get("frames_consumed"),
                "frames_inferred": counters.get("frames_inferred"),
                "frames_displayed": counters.get("frames_displayed"),
                "frame_ids_skipped": counters.get("frame_ids_skipped"),
                "stale_frames_dropped": counters.get("stale_frames_dropped"),
                "side_effect_queue_drops": counters.get("side_effect_queue_drops"),
                "critical_queue_drops": counters.get("critical_queue_drops"),
                "recognition_attempts": (performance.get("vision") or {}).get("recognition_attempts"),
                "cache_hits": (performance.get("vision") or {}).get("recognition_cache_hits"),
                "cache_misses": (performance.get("vision") or {}).get("recognition_cache_misses"),
                "recognition_decisions": (performance.get("vision") or {}).get("recognition_decisions"),
                "cache_hit_rate_percent": (performance.get("vision") or {}).get("recognition_cache_hit_rate_percent"),
                "identity_confirmations": (performance.get("vision") or {}).get("recognition_confirmations"),
                "face_detection_latency_ms": ((performance.get("vision") or {}).get("models") or {}).get("face_detection", {}).get("average_latency_ms"),
                "recognition_latency_ms": ((performance.get("vision") or {}).get("models") or {}).get("identity_matching", {}).get("average_latency_ms"),
                "yolo_latency_ms": ((performance.get("vision") or {}).get("models") or {}).get("yolo", {}).get("average_latency_ms"),
                "pose_latency_ms": ((performance.get("vision") or {}).get("models") or {}).get("pose", {}).get("average_latency_ms"),
                "identity_confirmation_ms": (((performance.get("vision") or {}).get("identity_timing") or {}).get("mean_time_to_confirm_sec") * 1000
                                             if ((performance.get("vision") or {}).get("identity_timing") or {}).get("mean_time_to_confirm_sec") is not None else None),
                "identity_confirmation_p95_ms": (((performance.get("vision") or {}).get("identity_timing") or {}).get("p95_sec") * 1000
                                                 if ((performance.get("vision") or {}).get("identity_timing") or {}).get("p95_sec") is not None else None),
                "tasks_processed": (performance.get("operations") or {}).get("tasks_processed"),
                "attendance_decisions": (performance.get("operations") or {}).get("attendance_decisions"),
                "attendance_eligible": (performance.get("operations") or {}).get("attendance_eligible"),
                "attendance_rejected": (performance.get("operations") or {}).get("attendance_rejected"),
                "events_persisted": (performance.get("operations") or {}).get("events_persisted"),
                "alert_events": (performance.get("operations") or {}).get("alert_events"),
                "incidents_synced": (performance.get("operations") or {}).get("incidents_synced"),
            })
        attendance_decisions = max(attendance_decisions, int((live.get("summary") or {}).get("presentToday", 0) or 0))
        security_observations = max(security_observations, int((live.get("security") or {}).get("active_event_count", 0) or 0))
        time.sleep(max(0.1, interval_seconds))

    fields = [
        "capture_fps", "inference_fps", "display_fps", "frame_age_ms",
        "frame_age_consumed_ms", "stale_frame_age_ms",
        "inference_latency_ms", "inference_p95_ms", "cpu_percent",
        "gpu_percent", "vram_used_mb", "recognition_attempts", "cache_hits",
        "cache_misses", "recognition_decisions", "cache_hit_rate_percent",
        "identity_confirmations", "frames_captured", "frames_consumed",
        "frames_inferred", "frames_displayed", "frame_ids_skipped",
        "frames_replaced", "stale_frames_dropped",
        "side_effect_queue_drops", "critical_queue_drops",
        "tasks_processed", "attendance_decisions", "attendance_eligible",
        "attendance_rejected", "events_persisted", "alert_events", "incidents_synced",
    ]
    metrics = {field: _summary([sample.get(field) for sample in samples]) for field in fields}
    metrics.update({
        "frame_age_ms_p95": _summary([sample.get("frame_age_ms") for sample in samples]),
        "frame_age_consumed_ms_p95": _summary([sample.get("frame_age_consumed_ms") for sample in samples]),
        "stale_frame_age_ms": _summary([sample.get("stale_frame_age_ms") for sample in samples]),
        "end_to_end_latency_ms": _summary([sample.get("end_to_end_latency_ms") for sample in samples]),
        "face_detection_latency_ms": _summary([sample.get("face_detection_latency_ms") for sample in samples]),
        "recognition_latency_ms": _summary([sample.get("recognition_latency_ms") for sample in samples]),
        "yolo_latency_ms": _summary([sample.get("yolo_latency_ms") for sample in samples]),
        "pose_latency_ms": _summary([sample.get("pose_latency_ms") for sample in samples]),
        "identity_confirmation_ms": _summary([sample.get("identity_confirmation_ms") for sample in samples]),
        "identity_confirmation_p95_ms": _summary([sample.get("identity_confirmation_p95_ms") for sample in samples]),
        "recognition_attempts_per_second": _summary(
            [_counter_rate(samples[:index + 1], "recognition_attempts") for index in range(len(samples))]
        ),
        # Cache hits and misses are decision counts. Recognition attempts are
        # only embedding/matching calls and are therefore the wrong
        # denominator for cache effectiveness.
        "recognition_cache_hit_rate_percent": (
            _summary([sample.get("cache_hit_rate_percent") for sample in samples])
            if any(sample.get("cache_hit_rate_percent") is not None for sample in samples)
            else _ratio_summary(samples, "cache_hits", "recognition_decisions")
        ),
    })
    totals = {
        field: next((sample.get(field) for sample in reversed(samples) if sample.get(field) is not None), None)
        for field in ("frames_captured", "frames_consumed", "frames_inferred", "frames_displayed",
                      "frame_ids_skipped", "frames_replaced", "stale_frames_dropped",
                      "side_effect_queue_drops", "critical_queue_drops")
    }
    totals.update({
        field: next((sample.get(field) for sample in reversed(samples) if sample.get(field) is not None), None)
        for field in ("tasks_processed", "attendance_decisions", "attendance_eligible",
                      "attendance_rejected", "events_persisted", "alert_events", "incidents_synced")
    })
    totals.update({
        field: next((sample.get(field) for sample in reversed(samples) if sample.get(field) is not None), None)
        for field in ("recognition_attempts", "cache_hits", "cache_misses",
                      "recognition_decisions", "identity_confirmations")
    })
    elapsed_seconds = round(time.time() - started, 3)
    samples_satisfied = verified_frame_samples >= min_fresh_samples
    duration_satisfied = elapsed_seconds >= required_duration_seconds
    measurement_status = (
        "MEASURED" if samples_satisfied and duration_satisfied else
        "PARTIAL" if verified_frame_samples else "NOT_MEASURED"
    )
    return {
        "schema_version": 2,
        "measurement_status": measurement_status,
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": elapsed_seconds,
        "required_duration_seconds": required_duration_seconds,
        "duration_satisfied": duration_satisfied,
        "fresh_samples_satisfied": samples_satisfied,
        "sample_count": len(samples),
        "fresh_sample_count": verified_frame_samples,
        "frame_progress_verified": verified_frame_samples >= min_fresh_samples,
        "collection_quality": (
            "COMPLETE" if measurement_status == "MEASURED" else
            "PARTIAL" if measurement_status == "PARTIAL" else
            "NOT_MEASURED"
        ),
        "minimum_fresh_samples": min_fresh_samples,
        "runtime_health": {
            "status": measurement_status,
            "active_heartbeat_checks": health_counts.get("active", 0),
            "active_unique_samples": len(samples),
            "verified_frame_samples": verified_frame_samples,
            "ignored_stale_checks": health_counts.get("stale", 0),
            "ignored_inactive_checks": health_counts.get("inactive", 0),
            "ignored_invalid_timestamp_checks": health_counts.get("invalid_timestamp", 0),
            "ignored_invalid_frame_id_checks": health_counts.get("invalid_frame_id", 0),
            "ignored_non_monotonic_frame_id_checks": health_counts.get("non_monotonic_frame_id", 0),
            "required_engine_states": sorted(ACTIVE_ENGINE_STATES),
            "required_camera_states": sorted(ACTIVE_CAMERA_STATES),
            "required_duration_seconds": required_duration_seconds,
            "duration_satisfied": duration_satisfied,
        },
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
            "Fresh-frame status requires numeric, strictly increasing source frame IDs.",
            "GPU and VRAM remain NOT_MEASURED unless a collector is configured.",
            "Run the scenario checklist separately for known, unknown, occluded, spoof, and intrusion cases.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Observe a running OptiVox runtime.")
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--required-seconds", type=float, default=600.0,
        help="Minimum duration required for a COMPLETE ten-minute benchmark.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--publish-runtime-summary", action="store_true",
        help="Also publish a dashboard-compatible summary to runtime/performance_summary.json.",
    )
    args = parser.parse_args()
    result = collect(
        args.seconds,
        args.interval,
        required_duration_seconds=args.required_seconds,
    )
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
