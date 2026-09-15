import json

from benchmark_runtime import _canonical_summary, _counter_rate, _summary, collect


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
        "timestamp": "2026-09-15T10:00:00+00:00",
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

    assert result["measurement_status"] == "MEASURED"
    assert result["sample_count"] == 1
    assert result["metrics"]["capture_fps"]["average"] == 24.0
    assert result["metrics"]["gpu_percent"]["status"] == "NOT_MEASURED"


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
            "frame_age_ms_p95": {"average": 42.0},
            "inference_latency_ms": {"average": 110.0},
            "inference_p95_ms": {"average": 140.0},
            "recognition_attempts_per_second": {"average": 1.5},
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
    assert summary["vision"]["models"]["face_detection"]["average_latency_ms"] is None
    assert summary["vision"]["models"]["identity_matching"]["calls_per_second"] == 1.5
    assert summary["counters"]["frames_replaced"] == 3
