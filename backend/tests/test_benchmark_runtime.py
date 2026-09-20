import json
from datetime import datetime, timezone

from benchmark_runtime import _canonical_summary, _counter_rate, _summary, collect


def _fresh_timestamp():
    return datetime.now(timezone.utc).isoformat()


def test_metric_summary_preserves_unmeasured_values():
    assert _summary([None, ""]) == {
        "status": "NOT_MEASURED",
        "samples": 0,
        "average": None,
        "p95": None,
        "minimum": None,
        "maximum": None,
    }
    assert _summary([10, 20])["status"] == "MEASURED"


def test_runtime_observer_collects_atomic_telemetry(tmp_path):
    heartbeat = {
        "timestamp": _fresh_timestamp(),
        "engine_status": "online",
        "camera_status": "healthy",
        "fps": 8.0,
        "performance": {
            "capture_fps": 24.0,
            "inference_fps": 8.0,
            "display_fps": 30.0,
            "latency_ms": {"latest_frame_age": 42.0, "inference_avg": 110.0, "inference_p95": 140.0},
            "resource": {"cpu_percent": 65.0, "gpu_percent": None, "vram_used_mb": None},
            "counters": {"frames_replaced": 3, "stale_frames_dropped": 0},
            "frame_ids": {"source": 10, "inference": 9},
            "vision": {"recognition_attempts": 2, "recognition_cache_hits": 5, "recognition_confirmations": 1},
        },
    }
    (tmp_path / "heartbeat.json").write_text(json.dumps(heartbeat), encoding="utf-8")
    (tmp_path / "live_state.json").write_text(json.dumps({"summary": {"presentToday": 1}, "security": {"active_event_count": 2}}), encoding="utf-8")

    result = collect(duration_seconds=1.0, interval_seconds=0.1, runtime_dir=tmp_path)

    assert result["measurement_status"] == "PARTIAL"
    assert result["sample_count"] == 1
    assert result["metrics"]["capture_fps"]["average"] == 24.0
    assert result["metrics"]["gpu_percent"]["status"] == "NOT_MEASURED"
    assert result["runtime_health"]["status"] == "PARTIAL"
    assert result["collection_quality"] == "PARTIAL"
    assert result["duration_satisfied"] is False


def test_short_fresh_collection_cannot_claim_complete_benchmark(tmp_path):
    heartbeat = {
        "timestamp": _fresh_timestamp(),
        "engine_status": "online",
        "camera_status": "healthy",
        "performance": {"capture_fps": 24.0},
        "frame_ids": {"source": 1},
    }
    (tmp_path / "heartbeat.json").write_text(json.dumps(heartbeat), encoding="utf-8")

    result = collect(
        duration_seconds=1.0,
        interval_seconds=0.1,
        runtime_dir=tmp_path,
        min_fresh_samples=1,
        required_duration_seconds=600.0,
    )

    assert result["measurement_status"] == "PARTIAL"
    assert result["fresh_samples_satisfied"] is True
    assert result["duration_satisfied"] is False
    assert result["collection_quality"] == "PARTIAL"


def test_heartbeat_without_source_frame_progress_is_not_measured(tmp_path):
    heartbeat = {
        "timestamp": _fresh_timestamp(),
        "engine_status": "online",
        "camera_status": "healthy",
        "performance": {"capture_fps": 24.0},
    }
    (tmp_path / "heartbeat.json").write_text(json.dumps(heartbeat), encoding="utf-8")

    result = collect(
        duration_seconds=1.0,
        interval_seconds=0.1,
        runtime_dir=tmp_path,
        min_fresh_samples=1,
        required_duration_seconds=0.0,
    )

    assert result["sample_count"] == 1
    assert result["fresh_sample_count"] == 0
    assert result["frame_progress_verified"] is False
    assert result["measurement_status"] == "NOT_MEASURED"
    assert result["collection_quality"] == "NOT_MEASURED"


def test_runtime_observer_ignores_stale_heartbeat_files(tmp_path):
    heartbeat = {
        "timestamp": "2020-01-01T00:00:00+00:00",
        "engine_status": "offline",
        "camera_status": "disconnected",
        "performance": {"capture_fps": 24.0},
        "frame_ids": {"source": 999},
    }
    (tmp_path / "heartbeat.json").write_text(json.dumps(heartbeat), encoding="utf-8")

    result = collect(duration_seconds=1.0, interval_seconds=0.1, runtime_dir=tmp_path)

    assert result["measurement_status"] == "NOT_MEASURED"
    assert result["sample_count"] == 0
    assert result["metrics"]["capture_fps"]["status"] == "NOT_MEASURED"


def test_runtime_observer_rejects_large_future_clock_skew(tmp_path):
    heartbeat = {
        "timestamp": "2099-01-01T00:00:00+00:00",
        "engine_status": "online",
        "camera_status": "healthy",
        "performance": {"capture_fps": 24.0, "frame_ids": {"source": 1}},
    }
    (tmp_path / "heartbeat.json").write_text(json.dumps(heartbeat), encoding="utf-8")

    result = collect(duration_seconds=1.0, interval_seconds=0.1, runtime_dir=tmp_path)

    assert result["measurement_status"] == "NOT_MEASURED"
    assert result["runtime_health"]["ignored_invalid_timestamp_checks"] > 0


def test_runtime_observer_ignores_fresh_but_inactive_heartbeat(tmp_path):
    heartbeat = {
        "timestamp": _fresh_timestamp(),
        "engine_status": "offline",
        "camera_status": "disconnected",
        "performance": {"capture_fps": 24.0},
        "frame_ids": {"source": 999},
    }
    (tmp_path / "heartbeat.json").write_text(json.dumps(heartbeat), encoding="utf-8")

    result = collect(duration_seconds=1.0, interval_seconds=0.1, runtime_dir=tmp_path)

    assert result["measurement_status"] == "NOT_MEASURED"
    assert result["runtime_health"]["ignored_inactive_checks"] > 0


def test_runtime_observer_requires_two_fresh_samples_for_measured_status(tmp_path):
    heartbeat = {
        "timestamp": _fresh_timestamp(),
        "engine_status": "online",
        "camera_status": "healthy",
        "performance": {"capture_fps": 24.0},
        "frame_ids": {"source": 1},
    }
    (tmp_path / "heartbeat.json").write_text(json.dumps(heartbeat), encoding="utf-8")

    result = collect(duration_seconds=1.0, interval_seconds=0.1, runtime_dir=tmp_path)

    assert result["sample_count"] == 1
    assert result["measurement_status"] == "PARTIAL"


def test_counter_rate_is_not_reported_as_a_cumulative_average():
    samples = [
        {"sampled_monotonic": 10.0, "recognition_attempts": 4},
        {"sampled_monotonic": 12.0, "recognition_attempts": 10},
    ]
    assert _counter_rate(samples, "recognition_attempts") == 3.0


def test_canonical_summary_keeps_missing_stage_values_explicit():
    result = {
        "measurement_status": "MEASURED",
        "started_at": "2026-09-15T10:00:00+00:00",
        "ended_at": "2026-09-15T10:01:00+00:00",
        "duration_seconds": 60.0,
        "metrics": {
            "capture_fps": {"average": 24.0},
            "inference_fps": {"average": 8.0},
            "display_fps": {"average": 30.0},
            "frame_age_ms_p95": {"average": 42.0, "p95": 65.0},
            "inference_latency_ms": {"average": 110.0},
            "inference_p95_ms": {"average": 110.0, "p95": 140.0},
            "end_to_end_latency_ms": {"average": 160.0, "p95": 210.0},
            "recognition_attempts_per_second": {"average": 1.5},
            "recognition_cache_hit_rate_percent": {"average": 72.0},
            "identity_confirmation_ms": {"average": 900.0},
            "cpu_percent": {"average": 65.0},
            "gpu_percent": {"average": None},
            "vram_used_mb": {"average": None},
        },
        "stage_metrics": {},
        "totals": {"frames_replaced": 3},
    }
    summary = _canonical_summary(result)
    assert summary["inference_fps"] == 8.0
    assert summary["latency_ms"]["frame_age_p95"] == 65.0
    assert summary["latency_ms"]["inference_p95"] == 140.0
    assert summary["latency_ms"]["end_to_end_p95"] == 210.0
    assert summary["vision"]["models"]["face_detection"]["average_latency_ms"] is None
    assert summary["vision"]["models"]["identity_matching"]["calls_per_second"] == 1.5
    assert summary["recognition"]["cache_hit_rate_percent"] == 72.0
    assert summary["counters"]["frames_replaced"] == 3


def test_canonical_summary_prefers_consumed_age_over_latest_snapshot_age():
    result = {
        "measurement_status": "MEASURED",
        "metrics": {
            "frame_age_ms_p95": {"p95": 80.0},
            "frame_age_consumed_ms_p95": {"p95": 125.0},
            "stale_frame_age_ms": {"p95": 410.0},
        },
        "stage_metrics": {},
        "totals": {},
    }
    summary = _canonical_summary(result)
    assert summary["latency_ms"]["frame_age_p95"] == 125.0
    assert summary["latency_ms"]["stale_frame_age_p95"] == 410.0


def test_cache_effectiveness_uses_decision_count_not_embedding_attempts():
    result = {
        "measurement_status": "MEASURED",
        "metrics": {
            "recognition_cache_hit_rate_percent": {"average": 75.0, "p95": 80.0},
            "identity_confirmation_p95_ms": {"average": 1200.0, "p95": 1500.0},
        },
        "stage_metrics": {},
        "totals": {
            "cache_hits": 75,
            "cache_misses": 25,
            "recognition_decisions": 100,
        },
    }
    summary = _canonical_summary(result)

    assert summary["recognition"]["cache_hit_rate_percent"] == 75.0
    assert summary["recognition"]["decisions"] == 100
    assert summary["vision"]["identity_timing"]["p95_time_to_confirm_sec"] == 1.5
