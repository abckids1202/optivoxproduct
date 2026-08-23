"""Small, dependency-free primitives for the OptiVox real-time runtime.

These classes intentionally do not import OpenCV or any model package.  That
keeps the concurrency contract easy to test and makes it safe to reuse from
the edge agent without loading the vision stack.
"""

from collections import deque
import threading
import time


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = int(round((len(ordered) - 1) * percentile))
    return ordered[max(0, min(index, len(ordered) - 1))]


class PerformanceProfiler:
    """Thread-safe rolling performance metrics for the live pipeline."""

    def __init__(self, window_size=120, enabled=True):
        self.window_size = max(10, int(window_size))
        self.enabled = bool(enabled)
        self._lock = threading.RLock()
        self._started_at = time.monotonic()
        self._records = deque(maxlen=self.window_size)
        self._display_times = deque(maxlen=self.window_size)
        self._capture_times = deque(maxlen=self.window_size)
        self._inference_times = deque(maxlen=self.window_size)
        self._counters = {
            "frames_captured": 0,
            "frames_replaced": 0,
            "frames_inferred": 0,
            "frames_displayed": 0,
            "stale_frames_dropped": 0,
            "inference_errors": 0,
            "side_effect_queue_drops": 0,
            "critical_queue_drops": 0,
        }

    def _increment(self, key, amount=1):
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + int(amount)

    def record_capture(self, replaced=False, timestamp=None):
        now = float(timestamp or time.monotonic())
        with self._lock:
            self._counters["frames_captured"] += 1
            if replaced:
                self._counters["frames_replaced"] += 1
            self._capture_times.append(now)

    def record_inference(self, frame_id, frame_age_ms, total_ms, stages=None):
        with self._lock:
            self._counters["frames_inferred"] += 1
            self._inference_times.append(time.monotonic())
            self._records.append({
                "frame_id": int(frame_id),
                "frame_age_ms": float(frame_age_ms),
                "total_ms": float(total_ms),
                "stages_ms": dict(stages or {}),
            })

    def record_display(self):
        with self._lock:
            self._counters["frames_displayed"] += 1
            self._display_times.append(time.monotonic())

    def record_stale_drop(self):
        self._increment("stale_frames_dropped")

    def record_inference_error(self):
        self._increment("inference_errors")

    def record_queue_drop(self, critical=False):
        self._increment("critical_queue_drops" if critical else "side_effect_queue_drops")

    @staticmethod
    def _rate(timestamps, now):
        if len(timestamps) < 2:
            return 0.0
        span = max(0.001, float(timestamps[-1] - timestamps[0]))
        return (len(timestamps) - 1) / span

    def snapshot(self, queue_depths=None, latest_frame_age_ms=None):
        with self._lock:
            now = time.monotonic()
            records = list(self._records)
            stage_values = {}
            for record in records:
                for name, value in record.get("stages_ms", {}).items():
                    stage_values.setdefault(name, []).append(value)
            totals = [record["total_ms"] for record in records]
            ages = [record["frame_age_ms"] for record in records]
            data = {
                "enabled": self.enabled,
                "window_frames": len(records),
                "uptime_sec": round(now - self._started_at, 3),
                "capture_fps": round(self._rate(self._capture_times, now), 2),
                "inference_fps": round(self._rate(self._inference_times, now), 2),
                "display_fps": round(self._rate(self._display_times, now), 2),
                "latency_ms": {
                    "inference_avg": round(sum(totals) / len(totals), 2) if totals else 0.0,
                    "inference_p50": round(_percentile(totals, 0.50), 2),
                    "inference_p95": round(_percentile(totals, 0.95), 2),
                    "frame_age_avg": round(sum(ages) / len(ages), 2) if ages else 0.0,
                    "frame_age_p95": round(_percentile(ages, 0.95), 2),
                    "latest_frame_age": round(float(latest_frame_age_ms or 0.0), 2),
                },
                "stages_ms": {
                    name: {
                        "avg": round(sum(values) / len(values), 2),
                        "p95": round(_percentile(values, 0.95), 2),
                    }
                    for name, values in sorted(stage_values.items())
                },
                "counters": dict(self._counters),
            }
            if queue_depths:
                data["queue_depths"] = dict(queue_depths)
            return data


class LatestFrameBuffer:
    """A one-slot latest-frame buffer with monotonic frame identity."""

    def __init__(self, max_age_ms=250):
        self.max_age_ms = max(1, int(max_age_ms))
        self._condition = threading.Condition(threading.RLock())
        self._frame = None
        self._frame_id = 0
        self._captured_at = None
        self._closed = False
        self._frames_captured = 0
        self._frames_replaced = 0

    def publish(self, frame, captured_at=None):
        if frame is None:
            return None
        captured_at = float(captured_at or time.time())
        with self._condition:
            if self._closed:
                return None
            replaced = self._frame is not None
            self._frame_id += 1
            self._frame = frame
            self._captured_at = captured_at
            self._frames_captured += 1
            if replaced:
                self._frames_replaced += 1
            self._condition.notify_all()
            return self._frame_id

    def wait_for_latest(self, after_frame_id=0, timeout=0.2):
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while (not self._closed and
                   (self._frame is None or self._frame_id <= int(after_frame_id))):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._frame is None or self._frame_id <= int(after_frame_id):
                return None
            return self._packet_locked()

    def snapshot(self):
        with self._condition:
            if self._frame is None:
                return None
            return self._packet_locked()

    def _packet_locked(self):
        captured_at = self._captured_at
        age_ms = max(0.0, (time.time() - captured_at) * 1000.0) if captured_at else 0.0
        return {
            "frame": self._frame,
            "frame_id": self._frame_id,
            "captured_at": captured_at,
            "age_ms": age_ms,
        }

    def metrics(self):
        with self._condition:
            packet = self._packet_locked() if self._frame is not None else None
            return {
                "frame_id": self._frame_id,
                "frames_captured": self._frames_captured,
                "frames_replaced": self._frames_replaced,
                "latest_frame_age_ms": packet["age_ms"] if packet else 0.0,
                "closed": self._closed,
            }

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class LatestInferenceState:
    """One-slot state for the newest completed inference result."""

    def __init__(self):
        self._lock = threading.RLock()
        self._result = None

    def publish(self, result):
        with self._lock:
            self._result = result

    def snapshot(self):
        with self._lock:
            return self._result
