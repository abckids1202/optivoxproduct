import time

from runtime_performance import LatestFrameBuffer, LatestInferenceState, PerformanceProfiler


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
