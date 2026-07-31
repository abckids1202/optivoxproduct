"""Bridge between the local OptiVox cv2 system and the web dashboard."""
import importlib
import threading
import time
import traceback

import cv2
import numpy as np

from .stream import buffer


class VisionBridge:
    def __init__(self, config):
        self.cfg = config
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self.mode = "idle"
        self.last_error = None
        self._vision = None
        self._db = None
        self._attendance = None
        self._alert_manager = None
        self._assistant = None
        self._severe = set()
        self._severity = {}
        self._last_state = {
            "frame": {"faces": [], "objects": [], "held_objects": [], "tracked_people": 0},
            "active_strangers": [],
            "toggles": {},
        }

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="VisionBridge")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=4.0)

    def status(self):
        return {"mode": self.mode, "error": self.last_error}

    def live_state(self):
        with self._lock:
            state = dict(self._last_state)
            state["bridge"] = self.status()
            state["stream_meta"] = buffer.meta()
            return state

    def set_toggle(self, name, value):
        with self._lock:
            if not self._vision:
                return False, "vision system is not ready"
            cfg = getattr(self._vision, "cfg", None)
            if cfg is None:
                return False, "vision config is unavailable"
            mapping = {
                "show_heatmap": "SHOW_HEATMAP",
                "show_object_boxes": "SHOW_OBJECT_BOXES",
                "show_pose_landmarks": "SHOW_POSE_LANDMARKS",
                "show_age_gender": "SHOW_AGE_GENDER",
                "show_zones_grid": "SHOW_ZONES_GRID",
            }
            if name == "danger_detection":
                danger = cfg.setdefault("DANGER_DETECTION", {})
                danger["ENABLED"] = bool(value)
                detector = getattr(self._vision, "danger_detector", None)
                if detector is not None:
                    detector.enabled = bool(value)
                elif bool(value):
                    try:
                        mod = importlib.import_module(self.cfg.VISION_MODULE)
                        self._vision.danger_detector = mod.DangerDetector(cfg)
                    except Exception as exc:
                        danger["ENABLED"] = False
                        return False, f"danger detector could not start: {exc}"
                return True, "updated"
            key = mapping.get(name)
            if not key:
                return False, f"unknown toggle: {name}"
            cfg[key] = bool(value)
            return True, "updated"

    def enroll_visible_face(self, person_name):
        person_name = str(person_name or "").strip()
        if not person_name:
            return False, "name is required"
        with self._lock:
            if not self._vision:
                return False, "vision system is not ready"
            fn = getattr(self._vision, "enroll_best_visible_face", None)
            if not callable(fn):
                return False, "your vision module does not expose enroll_best_visible_face"
            ok = bool(fn(person_name))
            if not ok:
                return False, "no visible UNKNOWN or STRANGER face was available"
            if self._db:
                try:
                    self._db.upsert_person(person_name)
                    self._db.log_audit("WEB_ENROLL_VISIBLE_FACE", person_name, {"mode": "dashboard"})
                except Exception:
                    pass
            return True, f"{person_name} enrolled"

    def manual_clock_in(self, person_name):
        with self._lock:
            if not self._db:
                return False, "database is not ready"
            pid = self._db.get_person_id(person_name) or self._db.upsert_person(person_name)
            result = self._db.attendance_clock_in(pid, self.cfg.CAMERA_ID, self.cfg.CAMERA_LOCATION)
            self._db.log_event(
                "ATTENDANCE_CLOCKIN",
                person_id=pid,
                confidence=1.0,
                details={"manual": True, "source": "web"},
                camera_id=self.cfg.CAMERA_ID,
                location=self.cfg.CAMERA_LOCATION,
                severity=0,
            )
            return True, result

    def manual_clock_out(self, person_name):
        with self._lock:
            if self._attendance and hasattr(self._attendance, "manual_clock_out"):
                return True, self._attendance.manual_clock_out(person_name)
            if not self._db:
                return False, "database is not ready"
            pid = self._db.get_person_id(person_name)
            if pid is None:
                return False, f"unknown person: {person_name}"
            return True, self._db.attendance_clock_out(pid)

    def ask_assistant(self, question):
        question = str(question or "").strip()
        if not question:
            return False, "question is required"
        with self._lock:
            if self._assistant and hasattr(self._assistant, "ask"):
                return True, self._assistant.ask(question)
            return False, "assistant is not configured in the running vision module"

    def _load_vision_system(self):
        mod = importlib.import_module(self.cfg.VISION_MODULE)
        try:
            mod.CONFIG["DATABASE_FILE"] = self.cfg.DATABASE_FILE
        except Exception:
            pass

        self._db = mod.EventDatabase(self.cfg.DATABASE_FILE)
        self._db.setup_database()
        self._vision = mod.VisionSystem()
        try:
            self._vision.import_known_faces_folder(self._db)
        except Exception:
            pass
        self._attendance = mod.AttendanceManager(self._db)
        self._severe = set(getattr(mod, "_SEVERE_EVENT_TYPES", set()))
        self._severity = dict(getattr(mod, "_SEVERITY_MAP", {}))

        alert_cls = getattr(mod, "AlertManager", None)
        if alert_cls:
            try:
                self._alert_manager = alert_cls(db=self._db)
            except Exception:
                self._alert_manager = None
        assistant_cls = getattr(mod, "AIAssistant", None)
        if assistant_cls:
            try:
                self._assistant = assistant_cls(
                    db=self._db,
                    alert_manager=self._alert_manager,
                    vision_system=self._vision,
                    attendance_manager=self._attendance,
                    config={"openai_api_key": None, "stt_engine": "text"},
                )
            except Exception:
                self._assistant = None

    def _open_camera(self):
        source = self.cfg.CAMERA_SOURCE
        try:
            source = int(source)
        except (TypeError, ValueError):
            pass
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def _run(self):
        try:
            self._load_vision_system()
        except Exception as exc:
            self.last_error = f"vision module '{self.cfg.VISION_MODULE}' not importable: {exc}"
            if self.cfg.ALLOW_SYNTHETIC:
                self._run_synthetic()
            else:
                self.mode = "error"
            return

        cap = self._open_camera()
        if cap is None:
            self.last_error = f"camera source '{self.cfg.CAMERA_SOURCE}' could not be opened"
            if self.cfg.ALLOW_SYNTHETIC:
                self._run_synthetic()
            else:
                self.mode = "error"
            return

        self.mode = "live"
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.cfg.JPEG_QUALITY]
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            try:
                frame = cv2.resize(frame, (1280, 720))
                display, faces, hands, objects, events, tracked_count, elapsed = self._vision.process(frame)
                self._handle_attendance(faces)
                self._handle_events(events)
                self._update_state(faces, objects, tracked_count, events, elapsed)
                ok2, enc = cv2.imencode(".jpg", display, encode_params)
                if ok2:
                    buffer.publish(enc.tobytes(), meta=self._last_state.get("stream_meta", {}))
            except Exception as exc:
                print(f"[BRIDGE] frame error: {exc}")
                traceback.print_exc()
                time.sleep(0.05)

        cap.release()
        self.mode = "idle"

    def _handle_attendance(self, faces):
        if not self._attendance:
            return
        for face in faces:
            name = face.get("name", "UNKNOWN")
            if face.get("is_real", True) and name not in ("UNKNOWN", "SPOOF") and not str(name).startswith("STRANGER_"):
                try:
                    self._attendance.handle_recognition(name, self.cfg.CAMERA_ID, self.cfg.CAMERA_LOCATION)
                except Exception:
                    pass

    def _handle_events(self, events):
        if not self._db:
            return
        for event in events:
            try:
                event_type = event[0]
                target = event[1] if len(event) > 1 else "SYSTEM"
                confidence = float(event[2]) if len(event) > 2 else 0.0
                details = event[3] if len(event) > 3 else ""
            except (IndexError, TypeError, ValueError):
                continue
            pid = None
            if target not in ("SYSTEM", "UNKNOWN", "PERSON") and not str(target).startswith(("STRANGER_", "ID_", "AREA_", "ZONE_")):
                try:
                    pid = self._db.get_person_id(target)
                except Exception:
                    pid = None
            try:
                self._db.log_event(
                    event_type=event_type,
                    person_id=pid,
                    confidence=confidence,
                    details=details,
                    camera_id=self.cfg.CAMERA_ID,
                    location=self.cfg.CAMERA_LOCATION,
                    severity=self._severity.get(event_type, 0),
                )
            except Exception:
                pass

    def _update_state(self, faces, objects, tracked_count, events, elapsed):
        vision_summary = getattr(self._vision, "_last_frame_summary", {}) or {}
        active_strangers = []
        for oid, sb in getattr(self._vision, "stranger_buffer", {}).items():
            active_strangers.append({
                "track_id": oid,
                "label": sb.get("label"),
                "frames": sb.get("frames"),
                "last_seen_sec_ago": round(time.time() - sb.get("last_seen", time.time()), 1),
            })
        cfg = getattr(self._vision, "cfg", {})
        toggles = {
            "show_heatmap": bool(cfg.get("SHOW_HEATMAP", False)),
            "show_object_boxes": bool(cfg.get("SHOW_OBJECT_BOXES", True)),
            "show_pose_landmarks": bool(cfg.get("SHOW_POSE_LANDMARKS", False)),
            "show_age_gender": bool(cfg.get("SHOW_AGE_GENDER", False)),
            "show_zones_grid": bool(cfg.get("SHOW_ZONES_GRID", False)),
            "danger_detection": bool(cfg.get("DANGER_DETECTION", {}).get("ENABLED", False)),
        }
        with self._lock:
            self._last_state = {
                "frame": {
                    "timestamp_utc": vision_summary.get("timestamp_utc"),
                    "tracked_people": tracked_count,
                    "faces": vision_summary.get("faces", faces)[:20],
                    "objects": vision_summary.get("objects", objects)[:30],
                    "held_objects": vision_summary.get("held_objects", []),
                    "events": [e[0] for e in events][-10:],
                    "process_ms": round(elapsed * 1000, 1),
                },
                "active_strangers": active_strangers,
                "toggles": toggles,
                "stream_meta": {
                    "faces": len(faces),
                    "objects": len(objects),
                    "tracked": tracked_count,
                    "process_ms": round(elapsed * 1000, 1),
                    "mode": self.mode,
                },
            }

    def _run_synthetic(self):
        self.mode = "synthetic"
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.cfg.JPEG_QUALITY]
        start = time.time()
        while not self._stop.is_set():
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            frame[:] = (10, 8, 12)
            for x in range(0, 1280, 64):
                cv2.line(frame, (x, 0), (x, 720), (32, 28, 40), 1)
            for y in range(0, 720, 64):
                cv2.line(frame, (0, y), (1280, y), (32, 28, 40), 1)
            t = time.time() - start
            cx = int(640 + 300 * np.sin(t * 0.7))
            cy = int(360 + 120 * np.cos(t * 0.5))
            cv2.rectangle(frame, (cx - 70, cy - 95), (cx + 70, cy + 95), (46, 197, 255), 2)
            cv2.putText(frame, "STRANGER_1 0.71 ID:1", (cx - 70, cy - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (46, 197, 255), 1)
            cv2.putText(frame, "OPTIVOX SYNTHETIC STREAM - CONNECT REAL MODULE FOR AI", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 230, 150), 2)
            cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"), (30, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            state = {
                "frame": {
                    "tracked_people": 1,
                    "faces": [{"track_id": 1, "name": "STRANGER_1", "confidence": 0.71, "is_real": True}],
                    "objects": [{"class_name": "person", "confidence": 0.89, "category": "person"}],
                    "held_objects": [],
                    "events": ["SYNTHETIC_STREAM"],
                    "process_ms": 0.8,
                },
                "active_strangers": [{"track_id": 1, "label": "STRANGER_1", "frames": int(t * 20), "last_seen_sec_ago": 0}],
                "toggles": {
                    "show_heatmap": False,
                    "show_object_boxes": True,
                    "show_pose_landmarks": False,
                    "show_age_gender": False,
                    "show_zones_grid": False,
                    "danger_detection": False,
                },
                "stream_meta": {"mode": self.mode, "faces": 1, "objects": 1, "tracked": 1, "process_ms": 0.8},
            }
            with self._lock:
                self._last_state = state
            ok, enc = cv2.imencode(".jpg", frame, encode_params)
            if ok:
                buffer.publish(enc.tobytes(), meta=state["stream_meta"])
            time.sleep(1.0 / max(self.cfg.STREAM_FPS, 1))
        self.mode = "idle"