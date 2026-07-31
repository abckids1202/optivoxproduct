"""Thread-safe shared MJPEG frame buffer."""
import threading
import time


class FrameBuffer:
    def __init__(self):
        self._frame = None
        self._seq = 0
        self._meta = {}
        self._lock = threading.Condition()
        self.started_at = time.time()

    def publish(self, jpeg_bytes, meta=None):
        with self._lock:
            self._frame = jpeg_bytes
            self._seq += 1
            if meta is not None:
                self._meta = dict(meta)
            self._lock.notify_all()

    def wait_for_next(self, last_seq, timeout=2.0):
        with self._lock:
            if self._frame is not None and self._seq != last_seq:
                return self._frame, self._seq
            self._lock.wait(timeout=timeout)
            return self._frame, self._seq

    def meta(self):
        with self._lock:
            return dict(self._meta)


buffer = FrameBuffer()


def mjpeg_generator(fps=20):
    boundary = b"--frame"
    last_seq = -1
    min_interval = 1.0 / max(fps, 1)
    while True:
        frame, seq = buffer.wait_for_next(last_seq, timeout=2.0)
        if frame is None:
            time.sleep(0.1)
            continue
        last_seq = seq
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(min_interval)