"""Small, dependency-free primitives for the OptiVox real-time runtime.

These classes intentionally do not import OpenCV or any model package.  That
keeps the concurrency contract easy to test and makes it safe to reuse from
the edge agent without loading the vision stack.
"""

from collections import defaultdict, deque
import threading
import time

try:
    import psutil
except ImportError:
    psutil = None


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
        self._resource_profiler = ResourceProfiler()
        self._records = deque(maxlen=self.window_size)
        self._display_times = deque(maxlen=self.window_size)
        self._capture_times = deque(maxlen=self.window_size)
        self._inference_times = deque(maxlen=self.window_size)
        self._counters = {
            "frames_captured": 0,
            "frames_consumed": 0,
            "frames_replaced": 0,
            "frame_ids_skipped": 0,
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

    def record_capture(self, replaced=False, frame_id=None, timestamp=None):
        now = float(timestamp or time.monotonic())
        with self._lock:
            self._counters["frames_captured"] += 1
            if replaced:
                self._counters["frames_replaced"] += 1
            self._capture_times.append(now)

    def record_frame_consumed(self, frame_id, frame_age_ms, skipped=0):
        with self._lock:
            self._counters["frames_consumed"] += 1
            self._counters["frame_ids_skipped"] += max(0, int(skipped or 0))

    def record_inference(self, frame_id, frame_age_ms, total_ms, stages=None,
                         frame_age_start_ms=None, frame_age_end_ms=None):
        with self._lock:
            self._counters["frames_inferred"] += 1
            self._inference_times.append(time.monotonic())
            self._records.append({
                "frame_id": int(frame_id),
                "frame_age_ms": float(frame_age_ms),
                "frame_age_start_ms": float(
                    frame_age_start_ms if frame_age_start_ms is not None else frame_age_ms),
                "frame_age_end_ms": float(
                    frame_age_end_ms if frame_age_end_ms is not None else frame_age_ms),
                "total_ms": float(total_ms),
                "stages_ms": dict(stages or {}),
            })

    def record_display(self, frame_id=None, frame_age_ms=None):
        with self._lock:
            self._counters["frames_displayed"] += 1
            self._display_times.append(time.monotonic())
            if frame_age_ms is not None:
                if not hasattr(self, "_display_ages"):
                    self._display_ages = deque(maxlen=self.window_size)
                self._display_ages.append(float(frame_age_ms))

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

    def snapshot(self, queue_depths=None, latest_frame_age_ms=None,
                 current_source_frame_id=None, current_inference_frame_id=None):
        with self._lock:
            now = time.monotonic()
            records = list(self._records)
            stage_values = {}
            for record in records:
                for name, value in record.get("stages_ms", {}).items():
                    stage_values.setdefault(name, []).append(value)
            records_with_inference = [record for record in records if record["total_ms"] > 0]
            totals = [record["total_ms"] for record in records_with_inference]
            ages = [record.get("frame_age_end_ms", record.get("frame_age_ms", 0.0))
                    for record in records_with_inference]
            start_ages = [record.get("frame_age_start_ms", record.get("frame_age_ms", 0.0))
                          for record in records_with_inference]
            display_ages = list(getattr(self, "_display_ages", ()))
            data = {
                "enabled": self.enabled,
                "window_frames": len(records),
                "uptime_sec": round(now - self._started_at, 3),
                "capture_fps": round(self._rate(self._capture_times, now), 2) if len(self._capture_times) >= 2 else None,
                "inference_fps": round(self._rate(self._inference_times, now), 2) if len(self._inference_times) >= 2 else None,
                "display_fps": round(self._rate(self._display_times, now), 2) if len(self._display_times) >= 2 else None,
                "latency_ms": {
                    "inference_avg": round(sum(totals) / len(totals), 2) if totals else None,
                    "inference_min": round(min(totals), 2) if totals else None,
                    "inference_p50": round(_percentile(totals, 0.50), 2) if totals else None,
                    "inference_p95": round(_percentile(totals, 0.95), 2) if totals else None,
                    "inference_max": round(max(totals), 2) if totals else None,
                    "frame_age_start_avg": round(sum(start_ages) / len(start_ages), 2)
                    if start_ages else None,
                    "frame_age_start_p95": round(_percentile(start_ages, 0.95), 2) if start_ages else None,
                    "frame_age_avg": round(sum(ages) / len(ages), 2) if ages else None,
                    "frame_age_p95": round(_percentile(ages, 0.95), 2) if ages else None,
                    "frame_age_end_avg": round(sum(ages) / len(ages), 2) if ages else None,
                    "frame_age_end_p95": round(_percentile(ages, 0.95), 2) if ages else None,
                    "display_frame_age_avg": round(sum(display_ages) / len(display_ages), 2)
                    if display_ages else None,
                    "display_frame_age_p95": round(_percentile(display_ages, 0.95), 2) if display_ages else None,
                    # Display frame age is the measured capture-to-display
                    # latency for the latest-frame pipeline.
                    "end_to_end_avg": round(sum(display_ages) / len(display_ages), 2)
                    if display_ages else None,
                    "end_to_end_p95": round(_percentile(display_ages, 0.95), 2)
                    if display_ages else None,
                    "latest_frame_age": round(float(latest_frame_age_ms), 2) if latest_frame_age_ms is not None else None,
                },
                "stages_ms": {
                    name: {
                        "avg": round(sum(values) / len(values), 2),
                        "p95": round(_percentile(values, 0.95), 2),
                    }
                    for name, values in sorted(stage_values.items())
                },
                "counters": dict(self._counters),
                "resource": self._resource_profiler.snapshot(),
            }
            if current_source_frame_id is not None or current_inference_frame_id is not None:
                data["frame_ids"] = {
                    "source": current_source_frame_id,
                    "inference": current_inference_frame_id,
                }
            if queue_depths:
                data["queue_depths"] = dict(queue_depths)
            return data


class ModelCallProfiler:
    """Rolling call and latency telemetry for expensive vision subsystems."""

    def __init__(self, window_size=120):
        self.window_size = max(10, int(window_size))
        self._lock = threading.RLock()
        self._calls = defaultdict(lambda: deque(maxlen=self.window_size))
        self._totals = defaultdict(int)

    def record(self, name, latency_ms):
        now = time.monotonic()
        with self._lock:
            self._calls[str(name)].append({
                "timestamp": now,
                "latency_ms": max(0.0, float(latency_ms)),
            })
            self._totals[str(name)] += 1

    def snapshot(self):
        with self._lock:
            now = time.monotonic()
            output = {}
            for name, records in sorted(self._calls.items()):
                values = [float(record["latency_ms"]) for record in records]
                timestamps = [float(record["timestamp"]) for record in records]
                output[name] = {
                    "calls_total": int(self._totals[name]),
                    "calls_per_second": round(PerformanceProfiler._rate(timestamps, now), 2),
                    "average_latency_ms": round(sum(values) / len(values), 2) if values else 0.0,
                    "p95_latency_ms": round(_percentile(values, 0.95), 2),
                    "last_latency_ms": round(values[-1], 2) if values else 0.0,
                    "window_calls": len(records),
                }
            return output


class ResourceProfiler:
    """Best-effort process resource telemetry without inventing GPU values."""

    def __init__(self):
        self.available = psutil is not None
        self._process = None
        if self.available:
            try:
                self._process = psutil.Process()
                self._process.cpu_percent(None)
            except Exception:
                self.available = False
                self._process = None

    def snapshot(self):
        if not self.available or self._process is None:
            return {
                "available": False,
                "cpu_percent": None,
                "process_rss_mb": None,
                "gpu_percent": None,
                "vram_used_mb": None,
                "gpu_temperature_c": None,
                "note": "psutil unavailable; GPU telemetry not configured",
            }
        try:
            memory = self._process.memory_info()
            return {
                "available": True,
                "cpu_percent": round(float(self._process.cpu_percent(None)), 2),
                "process_rss_mb": round(float(memory.rss) / (1024 * 1024), 2),
                "gpu_percent": None,
                "vram_used_mb": None,
                "gpu_temperature_c": None,
                "note": "GPU telemetry not configured",
            }
        except Exception as exc:
            return {
                "available": False,
                "cpu_percent": None,
                "process_rss_mb": None,
                "gpu_percent": None,
                "vram_used_mb": None,
                "gpu_temperature_c": None,
                "note": f"resource read failed: {exc}",
            }


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
        self._consumed_frame_id = 0
        self._frames_consumed = 0
        self._frame_ids_skipped = 0
        self._last_publish_replaced = False

    def publish(self, frame, captured_at=None):
        if frame is None:
            return None
        captured_at = float(captured_at or time.time())
        with self._condition:
            if self._closed:
                return None
            replaced = self._frame is not None and self._frame_id > self._consumed_frame_id
            self._frame_id += 1
            self._frame = frame
            self._captured_at = captured_at
            self._frames_captured += 1
            if replaced:
                self._frames_replaced += 1
            self._last_publish_replaced = replaced
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
            frame_id = self._frame_id
            skipped = max(0, frame_id - int(after_frame_id) - 1)
            self._consumed_frame_id = frame_id
            self._frames_consumed += 1
            self._frame_ids_skipped += skipped
            return self._packet_locked(frames_skipped=skipped)

    def snapshot(self):
        with self._condition:
            if self._frame is None:
                return None
            return self._packet_locked()

    def _packet_locked(self, frames_skipped=0):
        captured_at = self._captured_at
        age_ms = max(0.0, (time.time() - captured_at) * 1000.0) if captured_at else 0.0
        return {
            "frame": self._frame,
            "frame_id": self._frame_id,
            "captured_at": captured_at,
            "age_ms": age_ms,
            "frames_skipped": int(frames_skipped),
        }

    def metrics(self):
        with self._condition:
            packet = self._packet_locked() if self._frame is not None else None
            return {
                "frame_id": self._frame_id,
                "frames_captured": self._frames_captured,
                "frames_replaced": self._frames_replaced,
                "frames_consumed": self._frames_consumed,
                "frame_ids_skipped": self._frame_ids_skipped,
                "last_publish_replaced": self._last_publish_replaced,
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


class AdaptiveInferenceScheduler:
    """Frame-based scheduler with active/idle rates and load feedback."""

    def __init__(self, config=None):
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        self.target_total_ms = float(self.config.get("target_total_ms", 120.0))
        self._next_due = {}
        self._ema_total_ms = None
        self._last_intervals = {}
        self._lock = threading.RLock()

    def should_run(self, name, frame_id, base_interval, has_signal=False, force=False):
        base_interval = max(1, int(base_interval))
        if force:
            return True
        if not self.enabled:
            return int(frame_id) % base_interval == 0
        with self._lock:
            return int(frame_id) >= self._next_due.get(name, int(frame_id))

    def _configured_interval(self, name, base_interval, has_signal):
        base_interval = max(1, int(base_interval))
        mode = "active" if has_signal else "idle"
        value = self.config.get(f"{name}_{mode}_every_n_frames")
        return max(1, int(value if value is not None else base_interval))

    def complete(self, name, frame_id, base_interval, has_signal=False):
        interval = self._configured_interval(name, base_interval, has_signal)
        with self._lock:
            pressure = 1.0
            if self._ema_total_ms is not None and self.target_total_ms > 0:
                pressure = self._ema_total_ms / self.target_total_ms
            if self.enabled and pressure > 1.10:
                interval += min(3, max(1, int(round(pressure - 1.0))))
            elif self.enabled and pressure < 0.60 and interval > 1:
                interval -= 1
            self._last_intervals[name] = interval
            self._next_due[name] = int(frame_id) + interval
        return interval

    def record_total_ms(self, duration_ms):
        duration_ms = max(0.0, float(duration_ms))
        with self._lock:
            alpha = 0.15
            self._ema_total_ms = duration_ms if self._ema_total_ms is None else (
                alpha * duration_ms + (1.0 - alpha) * self._ema_total_ms
            )

    def snapshot(self):
        with self._lock:
            return {
                "enabled": self.enabled,
                "target_total_ms": self.target_total_ms,
                "ema_total_ms": round(self._ema_total_ms or 0.0, 2),
                "next_due": dict(self._next_due),
                "intervals": dict(self._last_intervals),
            }
