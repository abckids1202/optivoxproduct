import time

from runtime_performance import (
    AdaptiveInferenceScheduler,
    LatestFrameBuffer,
    LatestInferenceState,
    PerformanceProfiler,
)


def test_latest_frame_buffer_replaces_stale_frames_and_keeps_frame_ids():
    buffer = LatestFrameBuffer(max_age_ms=250)
    first_id = buffer.publish("first", captured_at=time.time())
    second_id = buffer.publish("second", captured_at=time.time())

    packet = buffer.wait_for_latest(first_id, timeout=0.01)

    assert first_id == 1
    assert second_id == 2
    assert packet["frame"] == "second"
    assert packet["frame_id"] == 2
    assert buffer.metrics()["frames_replaced"] == 1


def test_latest_frame_buffer_unblocks_when_closed():
    buffer = LatestFrameBuffer()
    buffer.close()

    assert buffer.wait_for_latest(0, timeout=0.01) is None
    assert buffer.metrics()["closed"] is True


def test_profiler_reports_latency_percentiles_and_counters():
    profiler = PerformanceProfiler(window_size=10)
    profiler.record_capture(replaced=True)
    profiler.record_inference(7, frame_age_ms=14.0, total_ms=20.0,
                              stages={"face": 8.0, "render": 2.0})
    profiler.record_stale_drop()
    profiler.record_queue_drop(critical=True)

    snapshot = profiler.snapshot(
        queue_depths={"side_effects": 2}, latest_frame_age_ms=18.0)

    assert snapshot["window_frames"] == 1
    assert snapshot["latency_ms"]["inference_p95"] == 20.0
    assert snapshot["stages_ms"]["face"]["avg"] == 8.0
    assert snapshot["latency_ms"]["latest_frame_age"] == 18.0
    assert snapshot["counters"]["stale_frames_dropped"] == 1
    assert snapshot["counters"]["critical_queue_drops"] == 1
    assert snapshot["queue_depths"]["side_effects"] == 2


def test_latest_inference_state_keeps_only_newest_result():
    state = LatestInferenceState()
    state.publish({"frame_id": 1})
    state.publish({"frame_id": 2})

    assert state.snapshot() == {"frame_id": 2}


def test_adaptive_scheduler_uses_idle_and_active_intervals():
    scheduler = AdaptiveInferenceScheduler({
        "enabled": True,
        "target_total_ms": 120,
        "face_detect_active_every_n_frames": 2,
        "face_detect_idle_every_n_frames": 6,
    })

    assert scheduler.should_run("face_detect", 1, 4, has_signal=False, force=True)
    assert scheduler.complete("face_detect", 1, 4, has_signal=False) == 6
    assert not scheduler.should_run("face_detect", 4, 4, has_signal=False)
    assert scheduler.should_run("face_detect", 7, 4, has_signal=True)
    assert scheduler.complete("face_detect", 7, 4, has_signal=True) == 2


def test_adaptive_scheduler_expands_when_total_runtime_is_over_budget():
    scheduler = AdaptiveInferenceScheduler({
        "enabled": True,
        "target_total_ms": 100,
        "face_detect_active_every_n_frames": 2,
    })
    scheduler.record_total_ms(250)
    scheduler.record_total_ms(250)

    interval = scheduler.complete("face_detect", 10, 2, has_signal=True)

    assert interval > 2
