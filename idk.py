# Section 1: Imports
import os
import sys
import cv2
import time
import json
import math
import pickle
import sqlite3
import smtplib
import threading
import traceback
import queue
import hashlib
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from collections import OrderedDict, deque, defaultdict
from typing import Optional, Any, Dict, List, Tuple
from datetime import datetime as dt_datetime, timedelta, date as dt_date

#Optional try and error
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[WARN] mediapipe not installed - pose/hand/face-mesh disabled.")

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[CRITICAL] insightface required. pip install insightface onnxruntime")
    sys.exit(1)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARN] faiss not installed - using numpy fallback (slower).")

try:
    from scipy.spatial.distance import cosine as _cos_dist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] torch not installed - ML anti-spoof disabled.")

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed - object detection disabled.")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[WARN] pyttsx3 not installed - voice disabled.")

try:
    import speech_recognition as sr
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[WARN] openai not installed - AI assistant in demo mode.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[WARN] requests not installed - Telegram/webhook disabled.")

# Section 2: Configuration
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG: Dict[str, Any] = {
    #Paths
    "BASE_DIR": _BASE_DIR,
    "DATABASE_FILE": os.path.join(_BASE_DIR, "security.db"),
    "FACE_DB_FILE": os.path.join(_BASE_DIR, "data", "face_db.pkl"),
    "SNAPSHOT_DIR": os.path.join(_BASE_DIR, "snapshots"),
    "EXPORT_DIR": os.path.join(_BASE_DIR, "exports"),
    "MODEL_PATH": "yolov8n.pt",         
    "ALERT_CONFIG_FILE": os.path.join(_BASE_DIR, "alert_config.json"),
    "KNOWN_FACES_DIR": os.path.join(_BASE_DIR, "known_faces"),

    #Report
    "REPORTING": {
    "ENABLED": True,
    "EMAIL_ON_SHUTDOWN": True,
    "REPORT_DIR": os.path.join(_BASE_DIR, "reports"),
    "RECENT_EVENTS_LIMIT": 200,
    },
    #Camera
    "CAMERA_INDEX": 0,
    "FRAME_WIDTH": 854,
    "FRAME_HEIGHT": 480,
    "TARGET_FPS": 25,
    "CAMERAS": [                         
        {"id": "cam_0", "source": 0, "location": "Class", "enabled": True},
    ],

    #Detection thresholds
    "YOLO_CONF": 0.40,
    "YOLO_IOU": 0.45,
    "HAND_DETECTION_CONF": 0.5,
    "POSE_DETECTION_CONF": 0.5,
    "FACE_RECOG_THRESHOLD": 0.6,        
    "RECOGNITION_HISTORY_LENGTH": 10,

    #Danger detection
    "DANGER_DETECTION": {
    "ENABLED": False,
    "MODEL_PATHS": [
        os.path.join(_BASE_DIR, "models", "best_weapon.onnx"),
    ],
    "CONF": 0.60,
    "IOU": 0.45,
    "EVERY_N_FRAMES": 10,
    "CONFIRM_FRAMES": 2,
    "HISTORY_FRAMES": 4,
    "ALLOWED_CLASSES": {"gun", "pistol", "rifle", "knife", "scissors"},
    "MIN_CONF_BY_CLASS": {
        "gun": 0.65,
        "pistol": 0.65,
        "rifle": 0.65,
        "knife": 0.60,
        "scissors": 0.60,
    },
    },

    "ENABLE_FIRE_SMOKE_HEURISTICS": False,

    #Object detection
    "ACCESSORY_OBJECTS": {
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "cup", "cell phone", "remote", "book",
    "mask", "helmet", "glasses", "sunglasses",
    },

    #Image quality
    "BLUR_THRESHOLD": 100.0,
    "BRIGHTNESS_MIN": 50,
    "BRIGHTNESS_MAX": 210,
    "CONTRAST_THRESHOLD": 40,
    "ENROLLMENT_QUALITY_THRESHOLD": 65,
    "ENROLLMENT_QUALITY_BLUR_MIN": 80.0,
    "MIN_ENROLLMENT_EMBEDDINGS": 5,
    "MAX_ENROLLMENT_EMBEDDINGS": 10,
    "MULTI_EMBEDDING_POOLING": "max",   
    "AUTO_THRESHOLD_MARGIN": 0.12,

    #Tracking
    "TRACKER_MAX_DISAPPEARED": 30,
    "TRACKER_MAX_DISTANCE": 150,
    "STRANGER_TRACKING_ENABLED": True,
    "STRANGER_REID_THRESHOLD": 0.68,
    "STRANGER_BUFFER_MAX_AGE_SEC": 600,

    #Behavior scoring
    "SUSPICION_DECAY_RATE": 0.85,
    "SUSPICION_DECAY_INTERVAL": 1.0,
    "SUSPICION_POINTS": {
        "HESITATION": 3, "PACING": 5, "SCANNING": 2,
        "SPATIAL_ANOMALY": 2, "LOITERING": 4, "RUNNING": 3,
        "OBJECT_INTERACTION": 2, "CROWD_FORMING": 2,
        "FALL_DETECTED": 8, "HANDS_RAISED": 6,
    },
    "STRESS_THRESHOLDS": {"LOW": 20, "MEDIUM": 50, "HIGH": 80},
    "SUSPICION_MIN_TRACK_FRAMES": 30,
    "SUSPICION_BEHAVIOR_COOLDOWN": 10.0,
    "HESITATION_SPEED_THRESHOLD": 2.0,
    "HESITATION_STOP_TIME_SEC": 5.0,
    "PACING_WINDOW_SEC": 10.0,
    "PACING_DIRECTION_CHANGES": 6,
    "SCANNING_VAR_THRESHOLD": 250.0,
    "SCANNING_DISP_THRESHOLD": 50.0,
    "OBJECT_INTERACTION_COOLDOWN_SEC": 8.0,

    #Heatmap and spatial
    "SPATIAL_GRID_SIZE": (20, 20),
    "SPATIAL_ANOMALY_THRESHOLD": 0.02,
    "HEATMAP_UPDATE_INTERVAL": 5.0,
    "LOITERING_ZONE": (400, 100, 800, 400),
    "COUNT_LINE_Y": 360,

    #Crowd intelligence
    "CROWD_INTELLIGENCE": {
        "ENABLED": True,
        "HEATMAP_GRID": (40, 30),
        "HEATMAP_DECAY": 0.998,
        "HEATMAP_GAUSSIAN_RADIUS": 2.5,
        "HEATMAP_GAUSSIAN_STRENGTH": 1.0,
        "SHOW_DENSITY_HEATMAP": False,
        "HEATMAP_OPACITY": 0.35,
        "CONGESTION_GRID": (3, 3),
        "CONGESTION_THRESHOLD": 5,
        "CROWD_MIN_SIZE": 4,
        "CROWD_RADIUS": 100,
        "EVAC_AVG_SPEED_THRESHOLD": 25.0,
        "EVAC_MIN_PEOPLE": 4,
    },

    #Anti-spoofing
    "ANTI_SPOOFING": {
    "ENABLED": True,
    "STRICTNESS": "normal",
    "ENABLE_ML_CLASSIFIER": False,
    "ML_CLASSIFIER_THRESHOLD": 0.6,
    "EAR_HISTORY_FRAMES": 30,
    "BLINK_EAR_THRESHOLD": 0.19,
    "MIN_TIME_BETWEEN_BLINKS_SEC": 0.4,
    "MIN_BLINKS_FOR_LIVENESS": 1,
    "LIVENESS_TIME_WINDOW_SEC": 20.0,
    "POSE_HISTORY_FRAMES": 30,
    "DEPTH_VARIANCE_THRESHOLD": 0.004,
    "ANTI_SPOOF_MODEL_PATH": "antispoof_model.bin",
    "REQUIRED_CHECKS_PASSED": 1,
    "WARMUP_FRAMES": 12,
    "SUSPECT_CONFIRM_FRAMES": 4,
    "SUSPECT_WINDOW_FRAMES": 8,
    "UNCERTAIN_BLOCKS_ATTENDANCE": False,
},

    "CUSTOM_OBJECTS": {
    "ENABLED": True,
    "DB_FILE": os.path.join(_BASE_DIR, "data", "custom_objects.pkl"),
    "MATCH_EVERY_N_FRAMES": 5,
    "MATCH_THRESHOLD": 0.58,
    "MIN_ORB_MATCHES": 12,
    "HAND_CROP_PADDING": 80,
},

    "SHOW_AGE_GENDER": False,
    "SHOW_ZONES_GRID": False,
    "ZONES_GRID": (3, 3),

    "DANGEROUS_OBJECTS": {
    "gun", "pistol", "rifle", "knife", "scissors",
    },

    "PPE_REQUIRED_OBJECTS": {"helmet", "mask", "vest"},   

    "PERFORMANCE": {
    "YOLO_EVERY_N_FRAMES": 8,
    "FACE_DETECT_EVERY_N_FRAMES": 4,
    "FACE_RECOG_EVERY_N_FRAMES": 8,
    "RECOGNITION_CACHE_TTL_SEC": 4.0,
    },

    #Attendance
    "ATTENDANCE": {
        "ENABLED": True,
        "MIN_RECOGNITION_FRAMES": 8,       
        "RECOGNITION_WINDOW_SEC": 5.0,
        "DAILY_CLOCKOUT_TIMEOUT_MIN": 720,  
        "WORK_START_HOUR": 9,
        "LATE_GRACE_MIN": 15,
        "ANNOUNCE_ARRIVAL": True,
    },

    #Voice and alerts
    "VOICE": {
        "ENABLED": True,
        "RATE": 175,
        "VOLUME": 0.95,
        "QUEUE_MAX": 8,
        "DEDUP_WINDOW_SEC": 12.0,
        "COOLDOWN_PER_EVENT_SEC": 30.0,
    },

    #Toggles
    "DISPLAY_FPS": True,
    "SHOW_COUNT_LINE": False,
    "SHOW_HEATMAP": False,
    "SHOW_FACE_MESH": False,
    "SHOW_HAND_LANDMARKS": False,
    "SHOW_POSE_LANDMARKS": False,
    "SHOW_OBJECT_BOXES": True,

    #Hand detection
    "HAND_MAX_NUM": 2,
    "HAND_MIN_DETECTION": 0.5,
    "HAND_MIN_TRACKING": 0.5,

    #Object display
    "OBJECT_DISPLAY_CATEGORIES": {
        "person": (0, 255, 0), "vehicle": (255, 165, 0),
        "electronics": (255, 0, 255), "furniture": (0, 165, 255),
        "animal": (0, 255, 255), "food": (0, 128, 0),
        "utensil": (128, 0, 128), "sports": (255, 200, 0),
        "accessory": (180, 130, 255), "clothing": (100, 200, 200),
        "household": (200, 150, 100), "infrastructure": (150, 150, 150),
        "misc": (180, 180, 180), "toy": (255, 150, 200),
        "dangerous": (0, 0, 255), "ppe": (0, 255, 128),
        "default": (200, 200, 200),
    },
}

#Full object detection using YOLO
_YOLO_CATEGORY_MAP = {
    0: "person",
    **{i: "vehicle" for i in range(1, 9)},
    **{i: "infrastructure" for i in range(9, 13)},
    13: "furniture",
    **{i: "animal" for i in range(14, 24)},
    **{i: "accessory" for i in [24, 25, 26, 28]},
    27: "clothing",
    **{i: "sports" for i in range(29, 39)},
    39: "household",
    **{i: "utensil" for i in range(40, 46)},
    **{i: "food" for i in range(46, 56)},
    **{i: "furniture" for i in range(56, 62)},
    **{i: "electronics" for i in [62, 63, 64, 65, 66, 67, 68, 69, 70]},
    71: "household", 72: "electronics",
    **{i: "misc" for i in [73, 74, 75, 79]},
    76: "utensil", 77: "toy", 78: "electronics",
}

_YOLO_NAME_TO_CATEGORY_OVERRIDE = {
    "knife": "dangerous",
    "scissors": "dangerous",
    "gun": "dangerous",
    "pistol": "dangerous",
    "rifle": "dangerous",

    "helmet": "ppe",
    "mask": "accessory",
    "vest": "ppe",
    "glasses": "accessory",
    "sunglasses": "accessory",

    "backpack": "accessory",
    "umbrella": "accessory",
    "handbag": "accessory",
    "tie": "accessory",
    "suitcase": "accessory",
    "cell phone": "accessory",
}

# Section 3: Utilities
def _utc_now() -> str:
    return dt_datetime.utcnow().isoformat()

def _today_iso() -> str:
    return dt_date.today().isoformat()

def _format_duration(seconds: float) -> str:
    if seconds < 60: return f"{int(seconds)}s"
    if seconds < 3600: return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    h = int(seconds // 3600); m = int((seconds % 3600) // 60)
    if h < 24: return f"{h}h {m}m"
    d = h // 24; h = h % 24
    return f"{d}d {h}h {m}m"

def _safe_json_parse(value) -> Any:
    if value is None: return {}
    try: return json.loads(value)
    except (json.JSONDecodeError, TypeError): return {"raw": str(value)}

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

def _compute_intra_class_variance(embeddings):
    if len(embeddings) < 2: return 0.0, 0.0
    distances = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            d = 1.0 - _cosine(embeddings[i], embeddings[j])
            distances.append(d)
    return float(np.mean(distances)), float(np.std(distances))

def _auto_threshold(mean_var, std_var, margin=None):
    if margin is None: margin = CONFIG["AUTO_THRESHOLD_MARGIN"]
    base = 1.0 - (mean_var + 2 * std_var)
    return float(np.clip(base - margin, 0.30, 0.65))

def get_dominant_color(image, k=3):
    if image is None or image.size == 0: return None
    try:
        small = cv2.resize(image, (50, 50))
        data = small.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(labels.flatten())
        dominant = centers[np.argmax(counts)]
        return (int(dominant[2]), int(dominant[1]), int(dominant[0]))
    except Exception:
        return None

def ensure_dirs():
    for k in ("SNAPSHOT_DIR", "EXPORT_DIR"):
        os.makedirs(CONFIG[k], exist_ok=True)
    os.makedirs(CONFIG.get("REPORTING", {}).get("REPORT_DIR", os.path.join(_BASE_DIR, "reports")), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["FACE_DB_FILE"]), exist_ok=True)

class DatabaseMigrationManager:
    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock):
        self.conn = conn
        self.lock = lock

    def table_columns(self, table_name: str) -> set:
        rows = self.conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {row[1] for row in rows}

    def add_column_if_missing(self, table_name: str, column_name: str, column_sql: str):
        if column_name in self.table_columns(table_name):
            return
        print(f"[DB] Migration: adding {table_name}.{column_name}")
        self.conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")

    def run(self):
        with self.lock:
            self.add_column_if_missing("people", "role", "role TEXT")
            self.add_column_if_missing("people", "thumbnail_path", "thumbnail_path TEXT")
            self.add_column_if_missing("people", "metadata_json", "metadata_json TEXT")
            self.add_column_if_missing("people", "created_at", "created_at TEXT")
            self.add_column_if_missing("people", "updated_at", "updated_at TEXT")

            people_cols = self.table_columns("people")
            if "face_embedding" in people_cols:
                print("[DB] Legacy people.face_embedding column detected; keeping it for compatibility.")

            self.add_column_if_missing("events", "camera_id", "camera_id TEXT DEFAULT 'cam_0'")
            self.add_column_if_missing("events", "location", "location TEXT")
            self.add_column_if_missing("events", "severity", "severity INTEGER DEFAULT 0")

            self.add_column_if_missing("attendance", "work_minutes", "work_minutes INTEGER DEFAULT 0")
            self.add_column_if_missing("attendance", "late_minutes", "late_minutes INTEGER DEFAULT 0")
            self.add_column_if_missing("attendance", "camera_id", "camera_id TEXT")
            self.add_column_if_missing("attendance", "location", "location TEXT")
            self.add_column_if_missing("attendance", "notes", "notes TEXT")

            now = _utc_now()
            if "created_at" in self.table_columns("people"):
                self.conn.execute("UPDATE people SET created_at=? WHERE created_at IS NULL", (now,))
            if "updated_at" in self.table_columns("people"):
                self.conn.execute("UPDATE people SET updated_at=? WHERE updated_at IS NULL", (now,))
            self.conn.commit()


class EventCooldown:
    def __init__(self):
        self._last: Dict[str, float] = {}

    def allowed(self, key: str, cooldown_sec: float) -> bool:
        now = time.time()
        if now - self._last.get(key, 0.0) < cooldown_sec:
            return False
        self._last[key] = now
        return True
    
# Section 4: Voice manager
class VoiceManager:
    PRIORITY = {"CRITICAL": 0, "WARN": 1, "INFO": 2}

    def __init__(self, cfg=None):
        self.cfg = (cfg or CONFIG).get("VOICE", {})
        self.enabled = self.cfg.get("ENABLED", True) and TTS_AVAILABLE
        self._engine = None
        self._lock = threading.Lock()
        self._queue: "queue.PriorityQueue" = queue.PriorityQueue(
            maxsize=self.cfg.get("QUEUE_MAX", 8))
        self._last_said: Dict[str, float] = {}
        self._stop_evt = threading.Event()
        self._counter = 0
        if self.enabled:
            self._init_engine()
            self._thread = threading.Thread(target=self._worker, daemon=True, name="VoiceMgr")
            self._thread.start()

    def _init_engine(self):
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.cfg.get("RATE", 175))
            self._engine.setProperty("volume", self.cfg.get("VOLUME", 0.95))
            voices = self._engine.getProperty("voices") or []
            for v in voices:
                if "english" in (v.name or "").lower() or "en" in (v.id or "").lower():
                    self._engine.setProperty("voice", v.id); break
        except Exception as e:
            print(f"[VOICE] init failed: {e}")
            self.enabled = False

    def say(self, text: str, priority: str = "INFO", dedup_key: Optional[str] = None, force: bool = False):
        if not text:
            return
        if not self.enabled:
            print(f"[VOICE-mute] {text}")
            return

        prio = self.PRIORITY.get(priority, 2)
        key = dedup_key or text.lower().strip()
        now = time.time()
        dedup_win = self.cfg.get("DEDUP_WINDOW_SEC", 12.0)

        if not force and now - self._last_said.get(key, 0) < dedup_win and priority != "CRITICAL":
            return

        self._last_said[key] = now

        try:
            self._counter += 1

            if force:
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break

            try:
                self._queue.put_nowait((prio, self._counter, text))
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait((prio, self._counter, text))

        except Exception as e:
            print(f"[VOICE] queue failed: {e}")

        print(f"[VOICE-{priority}] {text}")

    def _worker(self):
        while not self._stop_evt.is_set():
            try:
                prio, _, text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._lock:
                try:
                    self._engine.stop()
                    time.sleep(0.05)
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as e:
                    print(f"[VOICE] runtime error: {e}")
                    try: self._init_engine()
                    except Exception: pass

    def stop(self):
        self._stop_evt.set()
        with self._lock:
            try: 
                if self._engine: self._engine.stop()
            except Exception: pass

#Global voice
VOICE: Optional[VoiceManager] = None

def voice(text: str, priority: str = "INFO", dedup_key: Optional[str] = None, force: bool = False):
    if VOICE is not None:
        VOICE.say(text, priority, dedup_key, force=force)
    else:
        print(f"[VOICE-nil] {text}")

# Section 5: Database
class EventDatabase:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or CONFIG["DATABASE_FILE"]
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self._configure()

    def _configure(self):
        with self.lock:
            for p in ["PRAGMA journal_mode=WAL",
                      "PRAGMA synchronous=NORMAL",
                      "PRAGMA foreign_keys=ON",
                      "PRAGMA cache_size=-32000"]:
                self.conn.execute(p)
            self.conn.commit()

    def _fetchall(self, sql, params=()):
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def _fetchone(self, sql, params=()):
        with self.lock:
            return self.conn.execute(sql, params).fetchone()
    
    def _table_columns(self, table_name: str) -> set:
        with self.lock:
            rows = self.conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            return {row[1] for row in rows}

    def setup_database(self):
        with self.lock:
            cur = self.conn.cursor()
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS people (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    name            TEXT    UNIQUE NOT NULL,
                    role            TEXT,
                    thumbnail_path  TEXT,
                    metadata_json   TEXT,
                    created_at      TEXT    NOT NULL,
                    updated_at      TEXT    NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id       INTEGER,
                    event_type      TEXT    NOT NULL,
                    confidence      REAL,
                    details_json    TEXT,
                    snapshot_path   TEXT,
                    camera_id       TEXT    DEFAULT 'cam_0',
                    location        TEXT,
                    severity        INTEGER DEFAULT 0,
                    timestamp       TEXT    NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE SET NULL
                );
                CREATE TABLE IF NOT EXISTS attendance (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id       INTEGER NOT NULL,
                    date            TEXT    NOT NULL,
                    clock_in        TEXT,
                    clock_out       TEXT,
                    work_minutes    INTEGER DEFAULT 0,
                    late_minutes    INTEGER DEFAULT 0,
                    camera_id       TEXT,
                    location        TEXT,
                    notes           TEXT,
                    UNIQUE(person_id, date),
                    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS event_stats_daily (
                    date            TEXT NOT NULL,
                    event_type      TEXT NOT NULL,
                    count           INTEGER DEFAULT 0,
                    PRIMARY KEY (date, event_type)
                );
                CREATE TABLE IF NOT EXISTS event_stats_hourly (
                    hour            TEXT NOT NULL,
                    event_type      TEXT NOT NULL,
                    count           INTEGER DEFAULT 0,
                    PRIMARY KEY (hour, event_type)
                );
                CREATE TABLE IF NOT EXISTS behavior_profiles (
                    person_id       INTEGER PRIMARY KEY,
                    profile_json    TEXT NOT NULL,
                    updated_at      TEXT NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS zone_activity (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    zone_id         TEXT NOT NULL,
                    camera_id       TEXT,
                    timestamp       TEXT NOT NULL,
                    person_count    INTEGER DEFAULT 0,
                    avg_dwell_sec   REAL    DEFAULT 0.0,
                    max_dwell_sec   REAL    DEFAULT 0.0
                );
                CREATE TABLE IF NOT EXISTS alert_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel         TEXT NOT NULL,
                    event_type      TEXT NOT NULL,
                    target          TEXT,
                    status          TEXT,
                    error           TEXT,
                    timestamp       TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    action          TEXT NOT NULL,
                    target          TEXT,
                    details_json    TEXT,
                    timestamp       TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_people_name      ON people(name);
                CREATE INDEX IF NOT EXISTS idx_events_time      ON events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_type      ON events(event_type);
                CREATE INDEX IF NOT EXISTS idx_events_person    ON events(person_id);
                CREATE INDEX IF NOT EXISTS idx_events_severity  ON events(severity);
                CREATE INDEX IF NOT EXISTS idx_attendance_date  ON attendance(date);
                CREATE INDEX IF NOT EXISTS idx_attendance_pers  ON attendance(person_id);
                CREATE INDEX IF NOT EXISTS idx_audit_time       ON audit_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_alert_time       ON alert_log(timestamp);
            """)
            self.conn.commit()
            DatabaseMigrationManager(self.conn, self.lock).run()
            print("[DB] Schema checked and migrations applied.")

    #People
    def upsert_person(self, name: str, role: str = None, thumbnail_path: str = None,
                  metadata: dict = None) -> int:
        now = _utc_now()
        meta = json.dumps(metadata or {})

        with self.lock:
            cols_info = self.conn.execute("PRAGMA table_info(people)").fetchall()
            cols = {row[1]: row for row in cols_info}

            insert_cols = ["name"]
            insert_vals = [name]

            optional_values = {
                "role": role,
                "thumbnail_path": thumbnail_path,
                "metadata_json": meta,
                "created_at": now,
                "updated_at": now,
                "face_embedding": b"",
            }

            for col, val in optional_values.items():
                if col in cols:
                    insert_cols.append(col)
                    insert_vals.append(val)

            placeholders = ", ".join(["?"] * len(insert_cols))
            col_sql = ", ".join(insert_cols)

            update_parts = []
            for col in ("role", "thumbnail_path", "metadata_json", "updated_at"):
                if col in cols:
                    if col in ("role", "thumbnail_path"):
                        update_parts.append(f"{col}=COALESCE(excluded.{col}, {col})")
                    else:
                        update_parts.append(f"{col}=excluded.{col}")

            update_sql = ", ".join(update_parts) or "name=excluded.name"

            self.conn.execute(f"""
                INSERT INTO people ({col_sql})
                VALUES ({placeholders})
                ON CONFLICT(name) DO UPDATE SET {update_sql}
            """, tuple(insert_vals))
            self.conn.commit()

            row = self.conn.execute("SELECT id FROM people WHERE name=?", (name,)).fetchone()
            return row["id"] if row else None

    def get_person_id(self, name: str):
        row = self._fetchone("SELECT id FROM people WHERE name=?", (name,))
        return row["id"] if row else None

    def get_known_face_names(self, limit=1000, offset=0):
        return self._fetchall(
            "SELECT id, name, role, thumbnail_path, created_at FROM people "
            "ORDER BY name LIMIT ? OFFSET ?", (limit, offset))

    def get_face_count(self) -> int:
        row = self._fetchone("SELECT COUNT(*) AS c FROM people")
        return row["c"] if row else 0

    #Events
    def log_event(self, event_type, person_id=None, confidence=None, details=None,
                  snapshot_path=None, camera_id="cam_0", location=None, severity=0):
        details_json = json.dumps(details) if isinstance(details, dict) else json.dumps(
            {"message": str(details)} if details else {})
        with self.lock:
            self.conn.execute("""
                INSERT INTO events (person_id, event_type, confidence, details_json,
                    snapshot_path, camera_id, location, severity, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (person_id, event_type, confidence, details_json,
                  snapshot_path, camera_id, location, severity, _utc_now()))
            self.conn.execute("""
                INSERT INTO event_stats_daily (date, event_type, count) VALUES (?, ?, 1)
                ON CONFLICT(date, event_type) DO UPDATE SET count = count + 1
            """, (_today_iso(), event_type))
            self.conn.execute("""
                INSERT INTO event_stats_hourly (hour, event_type, count) VALUES (?, ?, 1)
                ON CONFLICT(hour, event_type) DO UPDATE SET count = count + 1
            """, (dt_datetime.utcnow().strftime("%Y-%m-%d %H:00"), event_type))
            self.conn.commit()

    def get_recent_events(self, limit=100):
        return self._fetchall("""
            SELECT e.*, p.name AS person_name FROM events e
            LEFT JOIN people p ON e.person_id = p.id
            ORDER BY e.timestamp DESC LIMIT ?""", (limit,))

    def get_event_summary(self, days=7):
        since = (dt_datetime.utcnow() - timedelta(days=days)).date().isoformat()
        return self._fetchall("""
            SELECT event_type, SUM(count) AS total FROM event_stats_daily
            WHERE date >= ? GROUP BY event_type ORDER BY total DESC""", (since,))

    def get_events_by_type(self, event_type, limit=50):
        return self._fetchall("""
            SELECT e.*, p.name AS person_name FROM events e
            LEFT JOIN people p ON e.person_id = p.id
            WHERE e.event_type=? ORDER BY e.timestamp DESC LIMIT ?""",
            (event_type, limit))

    def get_person_timeline(self, person_id, limit=200):
        return self._fetchall("""
            SELECT * FROM events WHERE person_id=?
            ORDER BY timestamp DESC LIMIT ?""", (person_id, limit))

    def search_events(self, query, days=7, limit=30):
        since = (dt_datetime.utcnow() - timedelta(days=days)).isoformat()
        like = f"%{query}%"
        return self._fetchall("""
            SELECT e.*, p.name AS person_name FROM events e
            LEFT JOIN people p ON e.person_id = p.id
            WHERE e.timestamp >= ?
              AND (e.event_type LIKE ? OR e.details_json LIKE ?
                   OR p.name LIKE ? OR e.location LIKE ?)
            ORDER BY e.timestamp DESC LIMIT ?""",
            (since, like, like, like, like, limit))

    #Attendance
    def attendance_clock_in(self, person_id: int, camera_id: str = None,
                            location: str = None) -> dict:
        today = _today_iso()
        now = _utc_now()
        existing = self._fetchone(
            "SELECT * FROM attendance WHERE person_id=? AND date=?", (person_id, today))
        if existing and existing["clock_in"]:
            return {"already_clocked_in": True, "clock_in": existing["clock_in"]}
        work_start = CONFIG["ATTENDANCE"]["WORK_START_HOUR"]
        grace = CONFIG["ATTENDANCE"]["LATE_GRACE_MIN"]
        now_local = dt_datetime.now()
        scheduled = now_local.replace(hour=work_start, minute=0, second=0, microsecond=0)
        late = max(0, int((now_local - scheduled).total_seconds() / 60) - grace)
        with self.lock:
            self.conn.execute("""
                INSERT INTO attendance (person_id, date, clock_in, late_minutes, camera_id, location)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(person_id, date) DO UPDATE SET
                    clock_in      = COALESCE(attendance.clock_in, excluded.clock_in),
                    late_minutes  = excluded.late_minutes,
                    camera_id     = excluded.camera_id,
                    location      = excluded.location
            """, (person_id, today, now, late, camera_id, location))
            self.conn.commit()
        return {"clocked_in_at": now, "late_minutes": late}

    def attendance_clock_out(self, person_id: int) -> dict:
        today = _today_iso()
        existing = self._fetchone(
            "SELECT * FROM attendance WHERE person_id=? AND date=?", (person_id, today))
        if not existing or not existing["clock_in"]:
            return {"error": "Not clocked in today"}
        now = _utc_now()
        try:
            ci = dt_datetime.fromisoformat(existing["clock_in"])
            co = dt_datetime.fromisoformat(now)
            work_min = max(0, int((co - ci).total_seconds() / 60))
        except Exception:
            work_min = 0
        with self.lock:
            self.conn.execute("""UPDATE attendance SET clock_out=?, work_minutes=?
                                 WHERE person_id=? AND date=?""",
                              (now, work_min, person_id, today))
            self.conn.commit()
        return {"clocked_out_at": now, "work_minutes": work_min}

    def attendance_report(self, days=7, person_id=None):
        since = (dt_date.today() - timedelta(days=days)).isoformat()
        if person_id is not None:
            return self._fetchall("""
                SELECT a.*, p.name FROM attendance a
                JOIN people p ON a.person_id = p.id
                WHERE a.date >= ? AND a.person_id=?
                ORDER BY a.date DESC""", (since, person_id))
        return self._fetchall("""
            SELECT a.*, p.name FROM attendance a
            JOIN people p ON a.person_id = p.id
            WHERE a.date >= ?
            ORDER BY a.date DESC, p.name""", (since,))

    #Audit and alert
    def log_audit(self, action, target=None, details=None):
        with self.lock:
            self.conn.execute(
                "INSERT INTO audit_log (action, target, details_json, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (action, target, json.dumps(details or {}), _utc_now()))
            self.conn.commit()

    def get_audit_log(self, action=None, limit=100):
        if action:
            return self._fetchall(
                "SELECT * FROM audit_log WHERE action=? ORDER BY timestamp DESC LIMIT ?",
                (action, limit))
        return self._fetchall(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,))

    def log_alert(self, channel, event_type, target=None, status="sent", error=None):
        with self.lock:
            self.conn.execute("""INSERT INTO alert_log (channel, event_type, target,
                                 status, error, timestamp) VALUES (?, ?, ?, ?, ?, ?)""",
                              (channel, event_type, target, status, error, _utc_now()))
            self.conn.commit()

    def get_alert_log(self, limit=50):
        return self._fetchall("SELECT * FROM alert_log ORDER BY timestamp DESC LIMIT ?",
                              (limit,))

    #Behavior profiles
    def update_behavior_profile(self, person_id, profile):
        with self.lock:
            self.conn.execute("""
                INSERT INTO behavior_profiles (person_id, profile_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at   = excluded.updated_at
            """, (person_id, json.dumps(profile), _utc_now()))
            self.conn.commit()

    def get_behavior_profile(self, person_id):
        row = self._fetchone(
            "SELECT profile_json FROM behavior_profiles WHERE person_id=?", (person_id,))
        return json.loads(row["profile_json"]) if row else {}

# Section 6: Alert
class AlertManager:
    DEFAULT_RATE_LIMIT = {
        "DANGEROUS_OBJECT": 60,  
        "SPOOF_DETECTED": 60,
        "FIRE_DETECTED": 30,
        "FALL_DETECTED": 60,
        "EVACUATION_ALERT": 60,
        "HANDS_RAISED": 90,
        "_default": 120,
    }

    def __init__(self, config_path=None, db: EventDatabase = None):
        self.db = db
        self.config_path = config_path or CONFIG["ALERT_CONFIG_FILE"]
        self.config = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f: self.config = json.load(f)
            except Exception as e:
                print(f"[ALERT] failed to load {self.config_path}: {e}")

        self.enabled = self.config.get("enabled", True)

        email = self.config.get("email", {})
        self.email_enabled = email.get("enabled", False)
        self.smtp_server   = email.get("smtp_server", "smtp.gmail.com")
        self.smtp_port     = email.get("smtp_port", 587)
        self.smtp_user     = email.get("smtp_user", "")
        self.smtp_pass     = email.get("smtp_pass", "")
        self.email_from    = email.get("from", self.smtp_user)
        self.email_to      = email.get("to", [])
        if isinstance(self.email_to, str): self.email_to = [self.email_to]

        tg = self.config.get("telegram", {})
        self.telegram_enabled = tg.get("enabled", False) and REQUESTS_AVAILABLE
        self.tg_bot_token = tg.get("bot_token", "")
        self.tg_chat_id   = tg.get("chat_id", "")

        wh = self.config.get("webhook", {})
        self.webhook_enabled = wh.get("enabled", False) and REQUESTS_AVAILABLE
        self.webhook_url = wh.get("url", "")
        self.discord_enabled = False
        self.sms_enabled = False

        self._last_alert: Dict[str, float] = {}
        self._rate_limit = {**self.DEFAULT_RATE_LIMIT,
                            **self.config.get("rate_limit_sec", {})}
        self._lock = threading.Lock()
        print(f"[INFO] AlertManager ready (email={self.email_enabled}, "
              f"telegram={self.telegram_enabled}, webhook={self.webhook_enabled})")

    def _allowed(self, event_type: str, target: str) -> bool:
        key = f"{event_type}|{target}"
        gap = self._rate_limit.get(event_type, self._rate_limit["_default"])
        now = time.time()
        with self._lock:
            if now - self._last_alert.get(key, 0) < gap:
                return False
            self._last_alert[key] = now
        return True

    def _send_email(self, subject, body, snapshot_path=None) -> Tuple[bool, str]:
        if not self.email_enabled or not self.smtp_user: return False, "disabled"
        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_from
            msg["To"]   = ", ".join(self.email_to)
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))
            if snapshot_path and os.path.exists(snapshot_path):
                with open(snapshot_path, "rb") as f:
                    img = MIMEImage(f.read(), name=os.path.basename(snapshot_path))
                    msg.attach(img)
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=20) as srv:
                srv.starttls(); srv.login(self.smtp_user, self.smtp_pass)
                srv.send_message(msg)
            return True, "ok"
        except Exception as e:
            return False, str(e)
        
    def send_report_email(self, report_path: str) -> Tuple[bool, str]:
        if not self.email_enabled or not self.smtp_user:
            return False, "email disabled"

        try:
            subject = "[REPORT] Security System Shutdown Summary"
            body = (
                "Security system shutdown report attached.\n\n"
                f"Report: {report_path}\n"
                f"Time: {_utc_now()}\n"
            )

            msg = MIMEMultipart()
            msg["From"] = self.email_from
            msg["To"] = ", ".join(self.email_to)
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            if report_path and os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    part = MIMEApplication(f.read(), Name=os.path.basename(report_path))
                part["Content-Disposition"] = f'attachment; filename="{os.path.basename(report_path)}"'
                msg.attach(part)

            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=20) as srv:
                srv.starttls()
                srv.login(self.smtp_user, self.smtp_pass)
                srv.send_message(msg)

            if self.db:
                self.db.log_alert("email", "SHUTDOWN_REPORT", report_path, "sent", None)

            return True, "ok"

        except Exception as e:
            if self.db:
                self.db.log_alert("email", "SHUTDOWN_REPORT", report_path, "failed", str(e))
            return False, str(e)

    def _send_telegram(self, text, snapshot_path=None) -> Tuple[bool, str]:
        if not self.telegram_enabled: return False, "disabled"
        try:
            if snapshot_path and os.path.exists(snapshot_path):
                url = f"https://api.telegram.org/bot{self.tg_bot_token}/sendPhoto"
                with open(snapshot_path, "rb") as f:
                    r = requests.post(url, data={"chat_id": self.tg_chat_id,
                                                 "caption": text[:1024]},
                                      files={"photo": f}, timeout=15)
            else:
                url = f"https://api.telegram.org/bot{self.tg_bot_token}/sendMessage"
                r = requests.post(url, data={"chat_id": self.tg_chat_id,
                                             "text": text[:4000]}, timeout=15)
            if r.status_code == 200: return True, "ok"
            return False, f"http {r.status_code}: {r.text[:200]}"
        except Exception as e:
            return False, str(e)

    def _send_webhook(self, payload: dict) -> Tuple[bool, str]:
        if not self.webhook_enabled: return False, "disabled"
        try:
            r = requests.post(self.webhook_url, json=payload, timeout=10)
            return (r.status_code in (200, 201, 204)), f"http {r.status_code}"
        except Exception as e:
            return False, str(e)

    def test_alert(self):
        subj = "[TEST] Security Alert"
        body = f"Test from Security System at {_utc_now()}"
        if self.email_enabled:
            ok, msg = self._send_email(subj, body)
            print(f"[ALERT-test/email] {'OK' if ok else 'FAIL'}: {msg}")
        if self.telegram_enabled:
            ok, msg = self._send_telegram(body)
            print(f"[ALERT-test/telegram] {'OK' if ok else 'FAIL'}: {msg}")
        if not (self.email_enabled or self.telegram_enabled):
            print("[ALERT-test] No channel enabled.")

    def check_and_alert(self, event_type, name, confidence, details, snapshot_path):
        if not self.enabled: return
        if not self._allowed(event_type, str(name)): return

        subject = f"[ALERT] {event_type}"
        if name: subject += f": {name}"
        body = (f"Event:      {event_type}\n"
                f"Target:     {name}\n"
                f"Confidence: {confidence}\n"
                f"Details:    {details}\n"
                f"Time:       {_utc_now()}\n")
        if snapshot_path: body += f"Snapshot:   {snapshot_path}\n"

        payload = {"event_type": event_type, "target": str(name),
                   "confidence": confidence, "details": str(details),
                   "snapshot": snapshot_path, "ts": _utc_now()}

        for channel, sender in (("email", lambda: self._send_email(subject, body, snapshot_path)),
                                ("telegram", lambda: self._send_telegram(body, snapshot_path)),
                                ("webhook", lambda: self._send_webhook(payload))):
            enabled = getattr(self, f"{channel}_enabled")
            if not enabled: continue
            ok, msg = sender()
            status = "sent" if ok else "failed"
            if self.db:
                self.db.log_alert(channel, event_type, str(name), status,
                                  None if ok else msg)
            print(f"[ALERT-{channel}] {status}: {msg}")

    def get_alert_history(self, limit=50):
        if not self.db: return []
        rows = self.db.get_alert_log(limit=limit)
        return [{k: r[k] for k in r.keys()} for r in rows]

    @staticmethod
    def create_sample_config(path=None):
        path = path or CONFIG["ALERT_CONFIG_FILE"]
        if os.path.exists(path): return
        sample = {
            "enabled": True,
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "smtp_user": "your_email@gmail.com",
                "smtp_pass": "your_app_password",
                "from": "your_email@gmail.com",
                "to": ["recipient@gmail.com"]
            },
            "telegram": {
                "enabled": False,
                "bot_token": "0000:AAA",
                "chat_id":   "0000"
            },
            "webhook": {
                "enabled": False,
                "url":     "https://hooks.example.com/your-endpoint"
            },
            "rate_limit_sec": {
                "DANGEROUS_OBJECT": 60,
                "SPOOF_DETECTED":   60,
                "FALL_DETECTED":    60,
                "_default":         120
            }
        }
        with open(path, "w") as f: json.dump(sample, f, indent=2)
        print(f"[INFO] Sample alert config created: {path}")

# Section 7: Vision pipeline 
#7.1 Faiss indexer
class FAISSIndexer:
    def __init__(self, dim=512):
        self.dim = dim
        self.id_to_name: List[str] = []
        self.id_to_threshold: List[float] = []
        self._np_index: Optional[np.ndarray] = None
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1: x = x[np.newaxis, :]
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, 1e-10)

    def add_embeddings(self, names, embeddings, thresholds=None):
        if embeddings is None or len(embeddings) == 0: return
        normed = self._normalize(np.array(embeddings, dtype=np.float32))
        if self.index is not None:
            self.index.add(normed)
        else:
            self._np_index = normed if self._np_index is None else np.vstack([self._np_index, normed])
        for i, n in enumerate(names):
            self.id_to_name.append(n)
            th = thresholds[i] if thresholds and i < len(thresholds) else CONFIG["FACE_RECOG_THRESHOLD"]
            self.id_to_threshold.append(th)

    def search(self, query_embedding, k=5):
        if not self.id_to_name: return [], [], []
        q = self._normalize(np.array(query_embedding, dtype=np.float32))
        if self.index is not None:
            if self.index.ntotal == 0: return [], [], []
            sims, idxs = self.index.search(q, min(k, self.index.ntotal))
            sims, idxs = sims[0], idxs[0]
        else:
            if self._np_index is None or self._np_index.shape[0] == 0: return [], [], []
            sims = (self._np_index @ q.T).flatten()
            order = np.argsort(-sims)[:k]
            idxs = order; sims = sims[order]
        names, dists, ths = [], [], []
        for d, ix in zip(sims, idxs):
            if 0 <= ix < len(self.id_to_name):
                names.append(self.id_to_name[ix])
                dists.append(float(d))
                ths.append(self.id_to_threshold[ix])
        return names, dists, ths

    def reset(self):
        if self.index is not None: self.index.reset()
        self._np_index = None
        self.id_to_name = []
        self.id_to_threshold = []


class CustomObjectManager:
    def __init__(self, cfg=None):
        self.cfg = (cfg or CONFIG).get("CUSTOM_OBJECTS", {})
        self.enabled = self.cfg.get("ENABLED", True)
        self.db_file = self.cfg.get("DB_FILE")
        self.objects: Dict[str, List[dict]] = defaultdict(list)
        self.orb = cv2.ORB_create(nfeatures=700)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._load()

    def _load(self):
        if self.db_file and os.path.exists(self.db_file):
            try:
                with open(self.db_file, "rb") as f:
                    self.objects = pickle.load(f)
                print(f"[CUSTOM-OBJ] Loaded {len(self.objects)} custom object(s)")
            except Exception as e:
                print(f"[CUSTOM-OBJ] Load failed: {e}")

    def save(self):
        if not self.db_file:
            return
        try:
            os.makedirs(os.path.dirname(self.db_file), exist_ok=True)
            with open(self.db_file, "wb") as f:
                pickle.dump(self.objects, f)
        except Exception as e:
            print(f"[CUSTOM-OBJ] Save failed: {e}")

    def _hand_bbox(self, frame, hand):
        h, w = frame.shape[:2]
        lm = hand.get("landmarks")
        if lm is None:
            return None

        pts = []
        for p in lm.landmark:
            pts.append((int(p.x * w), int(p.y * h)))

        if not pts:
            return None

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = int(self.cfg.get("HAND_CROP_PADDING", 80))

        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = min(w, max(xs) + pad)
        y2 = min(h, max(ys) + pad)

        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _features(self, crop):
        if crop is None or crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hist = hist.flatten().astype(np.float32)

        return {
            "des": des,
            "hist": hist,
            "shape": crop.shape[:2],
        }

    def enroll_from_hand(self, frame, hands, object_name: str) -> bool:
        if not self.enabled:
            print("[CUSTOM-OBJ] Disabled.")
            return False
        if not hands:
            print("[CUSTOM-OBJ] No hand detected. Hold the object in your hand.")
            return False

        bbox = self._hand_bbox(frame, hands[0])
        if bbox is None:
            print("[CUSTOM-OBJ] Could not crop hand/object region.")
            return False

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        feat = self._features(crop)

        if feat is None:
            print("[CUSTOM-OBJ] Could not extract object features.")
            return False

        feat["bbox_hint"] = bbox
        feat["created_at"] = _utc_now()
        self.objects[object_name].append(feat)
        self.save()

        print(f"[CUSTOM-OBJ] Enrolled '{object_name}' with {len(self.objects[object_name])} template(s)")
        return True

    def _score(self, feat_a, feat_b):
        hist_score = float(cv2.compareHist(
            feat_a["hist"].astype(np.float32),
            feat_b["hist"].astype(np.float32),
            cv2.HISTCMP_CORREL
        ))
        hist_score = max(0.0, min(1.0, hist_score))

        des_a = feat_a.get("des")
        des_b = feat_b.get("des")

        orb_score = 0.0
        if des_a is not None and des_b is not None and len(des_a) >= 2 and len(des_b) >= 2:
            try:
                matches = self.matcher.knnMatch(des_a, des_b, k=2)
                good = []
                for pair in matches:
                    if len(pair) == 2:
                        m, n = pair
                        if m.distance < 0.75 * n.distance:
                            good.append(m)
                min_matches = self.cfg.get("MIN_ORB_MATCHES", 12)
                orb_score = min(1.0, len(good) / max(min_matches, 1))
            except Exception:
                orb_score = 0.0

        return 0.55 * orb_score + 0.45 * hist_score

    def detect_from_hands(self, frame, hands):
        if not self.enabled or not self.objects or not hands:
            return []

        detections = []
        threshold = float(self.cfg.get("MATCH_THRESHOLD", 0.58))

        for hand in hands:
            bbox = self._hand_bbox(frame, hand)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            feat = self._features(crop)
            if feat is None:
                continue

            best_name, best_score = None, 0.0

            for name, templates in self.objects.items():
                for tmpl in templates:
                    score = self._score(feat, tmpl)
                    if score > best_score:
                        best_name, best_score = name, score

            if best_name and best_score >= threshold:
                detections.append({
                    "class_id": -100,
                    "class_name": best_name,
                    "confidence": best_score,
                    "bbox": bbox,
                    "category": "custom_object",
                    "color": (255, 120, 0),
                    "source_model": "custom_object",
                    "event_type": "CUSTOM_OBJECT_SEEN",
                })

        return detections
    
#7.2: Image quality scorer
class ImageQualityScorer:
    def __init__(self, cfg=None):
        self.cfg = cfg or CONFIG

    def score(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        bright = float(np.mean(gray))
        contrast = float(gray.std())
        blur_score = min(blur / 200.0, 1.0) * 40
        if self.cfg["BRIGHTNESS_MIN"] <= bright <= self.cfg["BRIGHTNESS_MAX"]:
            bright_score = 20.0
        else:
            dist = min(abs(bright - self.cfg["BRIGHTNESS_MIN"]),
                       abs(bright - self.cfg["BRIGHTNESS_MAX"]))
            bright_score = max(0, 20.0 - dist * 0.2)
        contrast_score = min(contrast / 80.0, 1.0) * 20
        sharp_score = min(blur / 300.0, 1.0) * 20
        return (blur_score + bright_score + contrast_score + sharp_score,
                {"blur": blur, "brightness": bright, "contrast": contrast})

    def is_acceptable(self, image):
        score, m = self.score(image)
        return (score >= self.cfg["ENROLLMENT_QUALITY_THRESHOLD"]
                and m["blur"] >= self.cfg["ENROLLMENT_QUALITY_BLUR_MIN"]), score

#7.3: Centroid tracker
class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=150):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.next_id += 1
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        return self.next_id

    def deregister(self, oid):
        self.objects.pop(oid, None)
        self.disappeared.pop(oid, None)

    def update(self, rects):
        valid = []
        for r in rects:
            try:
                if len(r) == 4: valid.append(tuple(int(x) for x in r))
            except TypeError:
                continue
        rects = valid

        if not rects:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects.copy()

        input_centroids = np.zeros((len(rects), 2), dtype=np.int32)
        for i, (x1, y1, x2, y2) in enumerate(rects):
            input_centroids[i] = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        if not self.objects:
            for c in input_centroids: self.register(tuple(c))
            return self.objects.copy()

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values())).reshape(-1, 2)
        dists = np.linalg.norm(object_centroids[:, None] - input_centroids[None, :], axis=2)

        if dists.size == 0: return self.objects.copy()

        rows = dists.min(axis=1).argsort()
        used_rows, used_cols = set(), set()
        for r in rows:
            c = int(np.argmin(dists[r]))
            if r in used_rows or c in used_cols: continue
            if dists[r, c] > self.max_distance: continue
            oid = object_ids[r]
            self.objects[oid] = tuple(input_centroids[c])
            self.disappeared[oid] = 0
            used_rows.add(r); used_cols.add(c)

        for r in range(dists.shape[0]):
            if r not in used_rows:
                oid = object_ids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        for c in range(len(input_centroids)):
            if c not in used_cols:
                self.register(tuple(input_centroids[c]))

        return self.objects.copy()


#7.4: Face analyzer
class FaceAnalyzer:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def detect(self, frame): return self.app.get(frame)
    def get_embedding(self, face): return face.embedding
    def get_bbox(self, face):
        b = face.bbox.astype(int)
        return int(b[0]), int(b[1]), int(b[2]), int(b[3])
    def get_landmarks(self, face):
        for attr in ("landmark_2d_106", "landmark_3d_68", "kps"):
            kps = getattr(face, attr, None)
            if kps is not None:
                kps = np.array(kps)
                if kps.ndim == 1: kps = kps.reshape(-1, 2)
                return kps[:, :2]   # drop z if 3d
        return None
    def get_age(self, face): return int(face.age) if getattr(face, "age", None) else None
    def get_gender(self, face): 
        g = getattr(face, "gender", None)
        return "M" if g == 1 else ("F" if g == 0 else "?")


#7.5: Hand detector
class HandDetector:
    def __init__(self, cfg=None):
        self.cfg = cfg or CONFIG
        self.enabled = MEDIAPIPE_AVAILABLE
        if self.enabled:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.cfg["HAND_MAX_NUM"],
                min_detection_confidence=self.cfg["HAND_MIN_DETECTION"],
                min_tracking_confidence=self.cfg["HAND_MIN_TRACKING"])
            self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        if not self.enabled: return []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        out = []
        if res.multi_hand_landmarks:
            for idx, lm in enumerate(res.multi_hand_landmarks):
                handedness = "Right"
                if res.multi_handedness and idx < len(res.multi_handedness):
                    handedness = res.multi_handedness[idx].classification[0].label
                out.append({"landmarks": lm, "handedness": handedness})
        return out

    def draw(self, frame, hands):
        if not self.enabled or not self.cfg.get("SHOW_HAND_LANDMARKS", False):
            return frame
        for h in hands:
            self.mp_draw.draw_landmarks(
                frame, h["landmarks"], self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2))
        return frame


#7.6: Pose detection
class PoseDetector:
    def __init__(self, cfg=None):
        self.cfg = cfg or CONFIG
        self.enabled = MEDIAPIPE_AVAILABLE
        if self.enabled:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=self.cfg["POSE_DETECTION_CONF"],
                min_tracking_confidence=0.5)
            self.mp_draw = mp.solutions.drawing_utils

    def analyze(self, frame) -> dict:
        """Returns dict with keys: pose (landmarks), is_fallen, hands_raised."""
        if not self.enabled: return {"pose": None, "is_fallen": False, "hands_raised": False}
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks: return {"pose": None, "is_fallen": False, "hands_raised": False}
        lm = res.pose_landmarks.landmark
        h, w = frame.shape[:2]
        def pt(i): return (lm[i].x * w, lm[i].y * h, lm[i].visibility)
        nose = pt(0); l_sh = pt(11); r_sh = pt(12); l_hip = pt(23); r_hip = pt(24)
        l_an = pt(27); r_an = pt(28); l_wr = pt(15); r_wr = pt(16)
        is_fallen = False
        if min(l_sh[2], r_sh[2], l_hip[2], r_hip[2]) > 0.4:
            sh_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
            hip_mid = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
            dx = abs(hip_mid[0] - sh_mid[0])
            dy = abs(hip_mid[1] - sh_mid[1])
            if dx > dy * 1.2 and abs(nose[1] - hip_mid[1]) < h * 0.18:
                is_fallen = True

        hands_raised = False
        if l_wr[2] > 0.4 and r_wr[2] > 0.4 and nose[2] > 0.4:
            if l_wr[1] < nose[1] and r_wr[1] < nose[1]:
                hands_raised = True

        return {"pose": res.pose_landmarks, "is_fallen": is_fallen, "hands_raised": hands_raised,
                "landmarks_px": {"nose": nose, "l_sh": l_sh, "r_sh": r_sh,
                                 "l_hip": l_hip, "r_hip": r_hip}}

    def draw(self, frame, result):
        if not self.enabled or not result or not result.get("pose"): return frame
        if not self.cfg.get("SHOW_POSE_LANDMARKS", False): return frame
        self.mp_draw.draw_landmarks(
            frame, result["pose"], self.mp_pose.POSE_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
            self.mp_draw.DrawingSpec(color=(255, 100, 0), thickness=2))
        return frame


#7.7: Full object detection
class ObjectDetector:
    def __init__(self, cfg=None):
        self.cfg = cfg or CONFIG
        self.model = None
        self.names: Dict[int, str] = {}
        if not YOLO_AVAILABLE: return
        try:
            mp_ = self.cfg["MODEL_PATH"]
            if not os.path.exists(mp_):
                print(f"[INFO] YOLO model {mp_} not present, will auto-download.")
            self.model = YOLO(mp_)
            self.names = self.model.names if hasattr(self.model, "names") else {}
            print(f"[INFO] YOLO loaded: {mp_} ({len(self.names)} classes)")
        except Exception as e:
            print(f"[ERROR] YOLO init failed: {e}")
            self.model = None

    def detect(self, frame):
        if self.model is None: return []
        try:
            results = self.model(frame, conf=self.cfg["YOLO_CONF"],
                                 iou=self.cfg.get("YOLO_IOU", 0.45),
                                 verbose=False)
        except Exception as e:
            print(f"[ERROR] YOLO inference failed: {e}")
            return []
        dets = []
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                if box.xyxy is None or box.xyxy.numel() == 0: continue
                if box.conf is None or box.cls is None: continue
                coords = np.array(box.xyxy[0].cpu().numpy()).flatten()
                if len(coords) != 4: continue
                x1, y1, x2, y2 = (int(c) for c in coords)
                cls_id = int(box.cls[0]); conf = float(box.conf[0])
                cls_name = self.names.get(cls_id, f"id_{cls_id}")
                accessories = self.cfg.get("ACCESSORY_OBJECTS", set())
                cls_lower = str(cls_name).lower()

                if cls_lower in accessories:
                    cat = "accessory"
                else:
                    cat = _YOLO_NAME_TO_CATEGORY_OVERRIDE.get(cls_name, _YOLO_CATEGORY_MAP.get(cls_id, "misc"))
                    
                color = self.cfg["OBJECT_DISPLAY_CATEGORIES"].get(
                    cat, self.cfg["OBJECT_DISPLAY_CATEGORIES"]["default"])
                dets.append({
                    "class_id": cls_id, "class_name": cls_name,
                    "confidence": conf, "bbox": (x1, y1, x2, y2),
                    "category": cat, "color": color,
                })
        return dets

    def draw_detections(self, frame, detections, skip_person=False):
        for d in detections:
            if skip_person and d["class_name"] == "person": continue
            x1, y1, x2, y2 = d["bbox"]
            color = d.get("color", (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            prefix = "DANGER: " if d.get("category") == "dangerous" else ""
            src = d.get("source_model")
            label = f"{prefix}{d['class_name']} {d['confidence']:.2f}"
            if src:
                label += f" [{src}]"
            cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

class DangerDetector:
    def __init__(self, cfg=None):
        self.cfg = (cfg or CONFIG).get("DANGER_DETECTION", {})
        self.enabled = self.cfg.get("ENABLED", True) and YOLO_AVAILABLE
        self.models = []
        self.history = deque(maxlen=self.cfg.get("HISTORY_FRAMES", 4))

        if not self.enabled:
            return

        for model_path in self.cfg.get("MODEL_PATHS", []):
            if not os.path.exists(model_path):
                print(f"[WARN] Danger model missing: {model_path}")
                continue
            try:
                model = YOLO(model_path, task="detect")
                names = model.names if hasattr(model, "names") else {}
                self.models.append((model_path, model, names))
                print(f"[INFO] Danger model loaded: {model_path} ({len(names)} classes)")
            except Exception as e:
                print(f"[ERROR] Danger model load failed {model_path}: {e}")

    def _event_type_for_class(self, cls_name: str) -> str:
        n = cls_name.lower()
        if "fire" in n or "flame" in n:
            return "FIRE_DETECTED"
        if "smoke" in n:
            return "SMOKE_DETECTED"
        if any(w in n for w in ("gun", "pistol", "rifle", "weapon", "knife", "blade")):
            return "WEAPON_DETECTED"
        return "DANGEROUS_OBJECT"

    def detect(self, frame):
        if not self.enabled or not self.models:
            return []

        raw_dets = []
        conf_th = self.cfg.get("CONF", 0.45)
        iou_th = self.cfg.get("IOU", 0.45)

        for model_path, model, names in self.models:
            try:
                results = model(frame, conf=conf_th, iou=iou_th, verbose=False)
            except Exception as e:
                print(f"[ERROR] Danger model inference failed {model_path}: {e}")
                continue

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    if box.xyxy is None or box.conf is None or box.cls is None:
                        continue

                    coords = np.array(box.xyxy[0].cpu().numpy()).flatten()
                    if len(coords) != 4:
                        continue

                    x1, y1, x2, y2 = (int(c) for c in coords)
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = str(names.get(cls_id, f"id_{cls_id}"))

                    raw_dets.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                        "category": "dangerous",
                        "color": (0, 0, 255),
                        "source_model": os.path.basename(model_path),
                        "event_type": self._event_type_for_class(cls_name),
                    })

        frame_classes = {d["class_name"].lower() for d in raw_dets}
        self.history.append(frame_classes)

        confirmed = []
        need = self.cfg.get("CONFIRM_FRAMES", 2)
        for d in raw_dets:
            cls_key = d["class_name"].lower()
            hits = sum(1 for frame_set in self.history if cls_key in frame_set)
            if hits >= need:
                confirmed.append(d)
        
        if raw_dets:
            labels = ", ".join(
                f"{d['class_name']}:{d['confidence']:.2f}" for d in raw_dets[:5]
            )
            print(f"[DANGER] raw={len(raw_dets)} confirmed={len(confirmed)} {labels}")

        return confirmed
    
#7.8: Supplementary detection
class SupplementaryDetector:
    def __init__(self, cfg=None):
        self.cfg = cfg or CONFIG
        self.fire_lower = np.array([0, 120, 150], dtype=np.uint8)
        self.fire_upper = np.array([25, 255, 255], dtype=np.uint8)
        self.smoke_lower = np.array([0, 0, 100], dtype=np.uint8)
        self.smoke_upper = np.array([180, 50, 220], dtype=np.uint8)
        self.min_fire_area = 2000
        self.min_smoke_area = 8000
        self.prev_fire_mask = None

    def detect_fire(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.fire_lower, self.fire_upper)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            a = cv2.contourArea(c)
            if a < self.min_fire_area: continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 15 or bh < 15: continue
            flicker = True
            if self.prev_fire_mask is not None:
                overlap = cv2.bitwise_and(mask, self.prev_fire_mask)
                ratio = np.sum(overlap > 0) / max(np.sum(mask > 0), 1)
                flicker = 0.3 < ratio < 0.95
            if flicker:
                out.append({"class_id": -1, "class_name": "fire",
                            "confidence": min(a / 10000.0, 0.95),
                            "bbox": (x, y, x + bw, y + bh),
                            "category": "dangerous", "color": (0, 0, 255)})
        self.prev_fire_mask = mask.copy()
        return out

    def detect_all(self, frame):
        return self.detect_fire(frame)


#7.9: Anti spoofing
class BlinkDetector:
    LEFT_EYE_106  = [33, 35, 40, 39, 42, 41]
    RIGHT_EYE_106 = [87, 89, 95, 93, 96, 94]

    def __init__(self, cfg=None):
        self.cfg = (cfg or CONFIG).get("ANTI_SPOOFING", {})
        self.ear_history = deque(maxlen=self.cfg.get("EAR_HISTORY_FRAMES", 30))
        self.blink_ts: List[float] = []
        self.last_blink = 0.0

    def compute_ear_from_indices(self, landmarks, indices):
        if landmarks is None or len(landmarks) <= max(indices, default=0):
            return None
        pts = np.array([landmarks[i] for i in indices], dtype=np.float32)
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h_ = np.linalg.norm(pts[0] - pts[3])
        return None if h_ < 1e-6 else (v1 + v2) / (2.0 * h_)

    def update(self, landmarks, now):
        ear_l = self.compute_ear_from_indices(landmarks, self.LEFT_EYE_106)
        ear_r = self.compute_ear_from_indices(landmarks, self.RIGHT_EYE_106)
        if ear_l is None or ear_r is None: return False
        ear = (ear_l + ear_r) / 2.0
        self.ear_history.append(ear)
        if len(self.ear_history) < 3: return False
        thr = self.cfg.get("BLINK_EAR_THRESHOLD", 0.21)
        gap = self.cfg.get("MIN_TIME_BETWEEN_BLINKS_SEC", 0.4)
        if (self.ear_history[-2] < thr and self.ear_history[-1] >= self.ear_history[-2]
                and (now - self.last_blink) >= gap):
            self.last_blink = now
            self.blink_ts.append(now)
            win = self.cfg.get("LIVENESS_TIME_WINDOW_SEC", 20.0)
            cutoff = now - win
            self.blink_ts = [t for t in self.blink_ts if t >= cutoff]
            return True
        return False

    def get_blink_count(self, within=None):
        within = within or self.cfg.get("LIVENESS_TIME_WINDOW_SEC", 20.0)
        cutoff = time.time() - within
        return sum(1 for t in self.blink_ts if t >= cutoff)


class HeadPoseEstimator:
    def __init__(self, cfg=None):
        self.cfg = (cfg or CONFIG).get("ANTI_SPOOFING", {})
        self.history = deque(maxlen=self.cfg.get("POSE_HISTORY_FRAMES", 30))

    def update(self, landmarks):
        if landmarks is None or len(landmarks) < 5: return
        center = np.mean(landmarks[:5], axis=0)
        self.history.append(tuple(center))

    def variance(self):
        if len(self.history) < 5: return float("inf")
        arr = np.array(self.history)
        return float(np.mean(np.var(arr, axis=0)))


class DepthEstimator:
    def __init__(self, cfg=None):
        self.cfg = (cfg or CONFIG).get("ANTI_SPOOFING", {})
        self.history = deque(maxlen=30)
        self.ref_size = None
        self.ref_depth = 0.5

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        size = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if size < 1: return
        if self.ref_size is None:
            self.ref_size = size; return
        depth = self.ref_depth * (self.ref_size / size)
        depth = float(np.clip(depth, 0.1, 5.0))
        self.history.append(depth)

    def is_consistent(self):
        if len(self.history) < 10: return True
        var = float(np.var(np.array(self.history)))
        return var >= self.cfg.get("DEPTH_VARIANCE_THRESHOLD", 0.01)


class AntiSpoofDetector:
    REAL = "REAL"
    UNCERTAIN = "UNCERTAIN"
    SUSPECT = "SUSPECT"

    def __init__(self, cfg=None):
        self.cfg = (cfg or CONFIG).get("ANTI_SPOOFING", {})
        self.enabled = self.cfg.get("ENABLED", True)
        self.blink = BlinkDetector(cfg)
        self.pose = HeadPoseEstimator(cfg)
        self.depth = DepthEstimator(cfg)
        self._frames_seen = 0

    def analyze(self, face_crop, frame_shape, landmarks=None, bbox=None) -> Tuple[str, dict]:
        if not self.enabled:
            return self.REAL, {"method": "disabled", "status": self.REAL}

        self._frames_seen += 1
        details = {"frames_seen": self._frames_seen}

        if landmarks is not None:
            self.pose.update(landmarks)
            self.blink.update(landmarks, time.time())
        if bbox is not None:
            self.depth.update(bbox)

        if self._frames_seen < int(self.cfg.get("WARMUP_FRAMES", 12)):
            details.update({"reason": "anti-spoof warmup", "status": self.UNCERTAIN})
            return self.UNCERTAIN, details

        if landmarks is None or len(landmarks) < 30:
            details.update({"reason": "insufficient landmarks", "status": self.UNCERTAIN})
            return self.UNCERTAIN, details

        pose_var = self.pose.variance()
        blink_n = self.blink.get_blink_count()
        depth_ok = self.depth.is_consistent()

        strictness = self.cfg.get("STRICTNESS", "normal").lower()
        movement_thr = {"low": 0.20, "normal": 0.35, "high": 0.65}.get(strictness, 0.35)
        required = int(self.cfg.get("REQUIRED_CHECKS_PASSED", 1))

        has_movement = pose_var > movement_thr
        has_blinks = blink_n >= int(self.cfg.get("MIN_BLINKS_FOR_LIVENESS", 1))
        checks = sum([bool(has_movement), bool(depth_ok), bool(has_blinks)])

        details.update({
            "head_pose_variance": pose_var,
            "movement_threshold": movement_thr,
            "has_movement": has_movement,
            "depth_consistent": depth_ok,
            "blink_count": blink_n,
            "has_blinks": has_blinks,
            "checks_passed": checks,
            "required_checks": required,
            "strictness": strictness,
        })

        if checks >= required:
            details["status"] = self.REAL
            return self.REAL, details

        if checks == 0 and strictness == "high":
            details["status"] = self.SUSPECT
            return self.SUSPECT, details

        details["status"] = self.UNCERTAIN
        return self.UNCERTAIN, details

#7.10: Suspicion scorer and behavior analyzer
class SuspicionScorer:
    def __init__(self, cfg=None):
        self.cfg = cfg or CONFIG
        self.scores: Dict[int, float] = {}
        self.last_decay = time.time()

    def add(self, oid, pts):
        self.scores[oid] = self.scores.get(oid, 0.0) + pts

    def decay(self, active_ids):
        now = time.time()
        if now - self.last_decay < self.cfg["SUSPICION_DECAY_INTERVAL"]: return
        self.last_decay = now
        rate = self.cfg["SUSPICION_DECAY_RATE"]
        for oid in list(self.scores.keys()):
            self.scores[oid] *= rate
            if oid not in active_ids or self.scores[oid] < 0.5:
                if oid not in active_ids: self.scores.pop(oid, None)

    def get(self, oid): return self.scores.get(oid, 0.0)

    def stress(self, oid):
        s = self.get(oid)
        t = self.cfg["STRESS_THRESHOLDS"]
        if s >= t.get("HIGH", 80):   return "Critical"
        if s >= t["MEDIUM"]:         return "High"
        if s >= t["LOW"]:            return "Medium"
        return "Low"


class BehaviorAnalyzer:
    def __init__(self, cfg=None):
        self.cfg = cfg or CONFIG
        self.history: Dict[int, dict] = {}
        self.scorer = SuspicionScorer(cfg)
        self.heatmap = np.zeros(self.cfg.get("SPATIAL_GRID_SIZE", (20, 20)), dtype=np.float32)
        self.heatmap_last_update = 0.0
        self.last_behavior_flag: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.current_time = time.time()
        self.frame_size = (720, 1280)  # (h, w) — set by VisionSystem

    def set_frame_size(self, h, w): self.frame_size = (h, w)

    def update(self, tracked: Dict[int, Tuple[int, int]]) -> List[tuple]:
        self.current_time = time.time()
        events = []
        for oid, c in tracked.items():
            if oid not in self.history:
                self.history[oid] = {
                    "centroids": deque(maxlen=300), "speeds": deque(maxlen=300),
                    "directions": deque(maxlen=300), "timestamps": deque(maxlen=300),
                    "behaviors": set(), "first_seen": self.current_time,
                    "profile": {"avg_speed": 0.0, "speed_variance": 0.0,
                                "visit_count": 0, "total_time": 0.0},
                }
            h = self.history[oid]
            h["centroids"].append(c); h["timestamps"].append(self.current_time)
            if len(h["centroids"]) >= 2:
                prev = h["centroids"][-2]
                dt = self.current_time - h["timestamps"][-2]
                if dt > 0:
                    dist = math.hypot(c[0] - prev[0], c[1] - prev[1])
                    h["speeds"].append(dist / dt)
                    h["directions"].append(math.atan2(c[1] - prev[1], c[0] - prev[0]))
            h["profile"]["visit_count"] += 1
            self._update_heatmap(c)

        min_frames = self.cfg["SUSPICION_MIN_TRACK_FRAMES"]
        cooldown = self.cfg["SUSPICION_BEHAVIOR_COOLDOWN"]

        for oid in list(self.history.keys()):
            if oid not in tracked: continue
            h = self.history[oid]
            if len(h["centroids"]) < min_frames: continue
            h["behaviors"].clear()
            for name, fn in (("HESITATION", self._hesitation),
                             ("PACING", self._pacing),
                             ("SCANNING", self._scanning)):
                r = fn(h)
                if r and (self.current_time - self.last_behavior_flag[oid][name]) >= cooldown:
                    events.append((name, f"ID_{oid}", 1.0, r))
                    h["behaviors"].add(name)
                    self.scorer.add(oid, self.cfg["SUSPICION_POINTS"].get(name, 1))
                    self.last_behavior_flag[oid][name] = self.current_time
            self._update_profile(h)

        for oid, c in tracked.items():
            if oid not in self.history or len(self.history[oid]["centroids"]) < min_frames:
                continue
            if self._spatial_anomaly(c) and (self.current_time -
                    self.last_behavior_flag[oid]["SPATIAL_ANOMALY"]) >= cooldown:
                events.append(("SPATIAL_ANOMALY", f"ID_{oid}", 1.0,
                               f"ID_{oid} in unusual location."))
                self.scorer.add(oid, self.cfg["SUSPICION_POINTS"]["SPATIAL_ANOMALY"])
                self.last_behavior_flag[oid]["SPATIAL_ANOMALY"] = self.current_time

        self.scorer.decay(set(tracked.keys()))
        return events

    def get_state(self, oid):
        if oid not in self.history: return None
        h = self.history[oid]
        return {"suspicion": self.scorer.get(oid),
                "stress_level": self.scorer.stress(oid),
                "active_behaviors": list(h.get("behaviors", set())),
                "avg_speed": h["profile"]["avg_speed"],
                "speed_variance": h["profile"]["speed_variance"],
                "visit_count": h["profile"]["visit_count"]}

    def cleanup(self, active_ids):
        for oid in set(self.history) - set(active_ids):
            self.history.pop(oid, None)
            self.scorer.scores.pop(oid, None)
            self.last_behavior_flag.pop(oid, None)

    def _update_heatmap(self, c):
        fh, fw = self.frame_size
        gh, gw = self.heatmap.shape
        gx = int(np.clip((c[0] / fw) * gw, 0, gw - 1))
        gy = int(np.clip((c[1] / fh) * gh, 0, gh - 1))
        self.heatmap[gy, gx] += 1.0
        if self.current_time - self.heatmap_last_update > self.cfg["HEATMAP_UPDATE_INTERVAL"]:
            self.heatmap *= self.cfg["CROWD_INTELLIGENCE"]["HEATMAP_DECAY"]
            self.heatmap_last_update = self.current_time

    def _hesitation(self, h):
        if len(h["speeds"]) < 10: return None
        speeds = list(h["speeds"])[-30:]
        avg = float(np.mean(speeds))
        if avg < self.cfg["HESITATION_SPEED_THRESHOLD"]:
            ts = list(h["timestamps"]); sp = list(h["speeds"])
            slow_dur = 0.0
            for i in range(len(sp) - 1, 0, -1):
                if sp[i] < self.cfg["HESITATION_SPEED_THRESHOLD"]:
                    slow_dur += ts[i] - ts[i - 1]
                else: break
            if slow_dur >= self.cfg["HESITATION_STOP_TIME_SEC"]:
                return f"Stationary {slow_dur:.1f}s (avg {avg:.1f}px/s)"
        return None

    def _pacing(self, h):
        if len(h["directions"]) < 10: return None
        win = self.cfg["PACING_WINDOW_SEC"]
        min_ch = self.cfg["PACING_DIRECTION_CHANGES"]
        ts = list(h["timestamps"]); ds = list(h["directions"])
        cutoff = self.current_time - win
        recent = [d for t, d in zip(ts, ds) if t >= cutoff]
        if len(recent) < 5: return None
        ch = 0
        for i in range(1, len(recent)):
            diff = (recent[i] - recent[i - 1] + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) > math.pi / 4: ch += 1
        return f"{ch} direction changes / {win:.0f}s" if ch >= min_ch else None

    def _scanning(self, h):
        if len(h["centroids"]) < 15: return None
        pos = np.array(list(h["centroids"])[-60:])
        var = float(np.sum(np.var(pos, axis=0)))
        disp = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
        if var > self.cfg["SCANNING_VAR_THRESHOLD"] and disp > self.cfg["SCANNING_DISP_THRESHOLD"]:
            return f"variance={var:.0f} disp={disp:.0f}"
        return None

    def _spatial_anomaly(self, c):
        total = float(np.sum(self.heatmap))
        if total < 5.0: return False
        fh, fw = self.frame_size
        gh, gw = self.heatmap.shape
        gx = int(np.clip((c[0] / fw) * gw, 0, gw - 1))
        gy = int(np.clip((c[1] / fh) * gh, 0, gh - 1))
        normed = self.heatmap / np.max(self.heatmap)
        return normed[gy, gx] < self.cfg["SPATIAL_ANOMALY_THRESHOLD"]

    def _update_profile(self, h):
        if h["speeds"]:
            s = np.array(h["speeds"])
            h["profile"]["avg_speed"] = float(np.mean(s))
            h["profile"]["speed_variance"] = float(np.var(s))
        if h["timestamps"]:
            h["profile"]["total_time"] = h["timestamps"][-1] - h["timestamps"][0]


#7.11: Crowd Intelligence
class CrowdIntelligence:
    def __init__(self, cfg=None):
        self.cfg = (cfg or CONFIG)["CROWD_INTELLIGENCE"]
        self.enabled = self.cfg.get("ENABLED", True)
        gh, gw = self.cfg.get("HEATMAP_GRID", (40, 30))
        self.density_heatmap = np.zeros((gh, gw), dtype=np.float32)

    def update(self, tracked: Dict[int, Tuple[int, int]], frame_size=(720, 1280)):
        """frame_size = (h, w) ; returns events list."""
        if not self.enabled: return []
        events = []
        gh, gw = self.density_heatmap.shape
        fh, fw = frame_size
        self.density_heatmap *= self.cfg.get("HEATMAP_DECAY", 0.998)
        radius = self.cfg.get("HEATMAP_GAUSSIAN_RADIUS", 2.5)
        strength = self.cfg.get("HEATMAP_GAUSSIAN_STRENGTH", 1.0)
        krad = max(1, int(math.ceil(radius * 1.5)))   # FIX: derive radius from config
        for c in tracked.values():
            gx = int(np.clip((c[0] / fw) * gw, 0, gw - 1))
            gy = int(np.clip((c[1] / fh) * gh, 0, gh - 1))
            for dy in range(-krad, krad + 1):
                for dx in range(-krad, krad + 1):
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < gh and 0 <= nx < gw:
                        d2 = dx * dx + dy * dy
                        self.density_heatmap[ny, nx] += strength * math.exp(
                            -d2 / (2 * radius * radius))

        positions = list(tracked.values())
        min_size = self.cfg.get("CROWD_MIN_SIZE", 4)
        radius_px = self.cfg.get("CROWD_RADIUS", 100)
        if len(positions) >= min_size:
            for i, p1 in enumerate(positions):
                near = sum(1 for p2 in positions
                           if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < radius_px)
                if near >= min_size:
                    events.append(("CROWD_FORMING", f"AREA_{i}", 1.0,
                                   f"{near} people within {radius_px}px"))
                    break

        cong_grid = self.cfg.get("CONGESTION_GRID", (3, 3))
        thr = self.cfg.get("CONGESTION_THRESHOLD", 5)
        if len(positions) >= thr:
            cw, ch = fw / cong_grid[1], fh / cong_grid[0]
            for row in range(cong_grid[0]):
                for col in range(cong_grid[1]):
                    cx, cy = cw * (col + 0.5), ch * (row + 0.5)
                    cnt = sum(1 for p in positions
                              if abs(p[0] - cx) < cw / 2 and abs(p[1] - cy) < ch / 2)
                    if cnt >= thr:
                        events.append(("CONGESTION", f"ZONE_{row}_{col}", 1.0,
                                       f"{cnt} people in zone"))
        return events

    def get_density_overlay(self, frame):
        if not self.enabled: return None
        h, w = frame.shape[:2]
        resized = cv2.resize(self.density_heatmap, (w, h))
        norm = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        op = self.cfg.get("HEATMAP_OPACITY", 0.35)
        return cv2.addWeighted(frame, 1 - op, colored, op, 0)

# Section 8: Vision system
class VisionSystem:
    def __init__(self, cfg=None):
        self.cfg = cfg or CONFIG
        ensure_dirs()
        print("[INFO] Initializing VisionSystem...")
        self._init_components()
        self._init_state()
        print("[INFO] VisionSystem ready.")

    def _init_components(self):
        self.face_analyzer    = FaceAnalyzer()
        self.hand_detector    = HandDetector(self.cfg)
        self.object_detector  = ObjectDetector(self.cfg) if YOLO_AVAILABLE else None
        self.danger_detector = DangerDetector(self.cfg) if YOLO_AVAILABLE and self.cfg.get("DANGER_DETECTION", {}).get("ENABLED", False) else None
        self.pose_detector    = PoseDetector(self.cfg)
        self.anti_spoof       = AntiSpoofDetector(self.cfg)
        self.custom_objects = CustomObjectManager(self.cfg)
        self.quality_scorer   = ImageQualityScorer(self.cfg)
        self.behavior         = BehaviorAnalyzer(self.cfg)
        self.crowd_intel      = CrowdIntelligence(self.cfg)
        self.supplementary    = SupplementaryDetector(self.cfg)
        self.person_tracker   = CentroidTracker(self.cfg["TRACKER_MAX_DISAPPEARED"], self.cfg["TRACKER_MAX_DISTANCE"])
        self.face_indexer     = FAISSIndexer(dim=512)
        self.face_db: Dict[str, Dict[str, Any]] = {}
        self.stranger_buffer: Dict[int, Dict[str, Any]] = {}
        self.stranger_counter = 0
        self._load_face_db()

    def _init_state(self):
        self._current_face_labels: Dict[int, Tuple[str, float, np.ndarray]] = {}
        self._person_colors: Dict[int, Tuple[int, int, int]] = {}
        self._frame_count = 0; self._fps = 0.0
        self._start_time = time.time()
        self._last_enrollment_time = 0
        self._face_recog_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        self._object_interaction_last: Dict[str, float] = {}
        self._last_face_objects = []
        self._last_object_detections = []
        self._last_danger_detections = []
        self._recognition_cache: Dict[int, dict] = {}
        self._spoof_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.cfg["ANTI_SPOOFING"].get("SUSPECT_WINDOW_FRAMES", 8)))
        self._event_cooldown = EventCooldown()
        self._stranger_gallery: Dict[str, dict] = {}
        self._next_stranger_label = 1
        self._object_seen_counts: Dict[str, int] = defaultdict(int)
        self._danger_seen_counts: Dict[str, int] = defaultdict(int)
        self._last_seen_people: Dict[str, dict] = {}
        self._last_objects_seen: List[dict] = []
        self._last_faces_seen: List[dict] = []
        self._last_frame_summary: dict = {}
        self._last_custom_object_detections = []
        self._last_held_objects: List[dict] = []

    def import_known_faces_folder(self, db: EventDatabase = None):
        root = self.cfg.get("KNOWN_FACES_DIR")
        if not root or not os.path.isdir(root):
            print(f"[FACES] Known faces folder not found: {root}")
            return

        imported_people = 0
        imported_embeddings = 0
        quality = self.quality_scorer

        for person_name in os.listdir(root):
            person_dir = os.path.join(root, person_name)

            if not os.path.isdir(person_dir):
                continue

            added_for_person = 0

            for fname in os.listdir(person_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    continue

                path = os.path.join(person_dir, fname)
                img = cv2.imread(path)
                if img is None:
                    print(f"[FACES] Could not read {path}")
                    continue

                faces = self.face_analyzer.detect(img)
                if len(faces) != 1:
                    print(f"[FACES] Skip {path}: expected 1 face, got {len(faces)}")
                    continue

                x1, y1, x2, y2 = self.face_analyzer.get_bbox(faces[0])
                crop = img[max(0, y1):min(img.shape[0], y2),
                        max(0, x1):min(img.shape[1], x2)]

                if crop.size > 0:
                    ok, score = quality.is_acceptable(crop)
                    if not ok:
                        print(f"[FACES] Skip {path}: low quality {score:.0f}")
                        continue

                emb = self.face_analyzer.get_embedding(faces[0])
                if emb is None:
                    print(f"[FACES] Skip {path}: no embedding")
                    continue

                self.add_embedding_for_name(person_name, emb)
                added_for_person += 1
                imported_embeddings += 1

            if added_for_person:
                imported_people += 1
                if db is not None:
                    try:
                        db.upsert_person(person_name)
                    except Exception as e:
                        print(f"[FACES] DB upsert failed for {person_name}: {e}")
                print(f"[FACES] Imported {added_for_person} image(s) for {person_name}")

        if imported_embeddings:
            self._rebuild_index()
            self.save_face_db()

        print(f"[FACES] Import complete: {imported_people} people, {imported_embeddings} embeddings")

    def _load_face_db(self):
        path = self.cfg["FACE_DB_FILE"]
        if not os.path.exists(path): return
        try:
            with open(path, "rb") as f: self.face_db = pickle.load(f)
            print(f"[INFO] Face DB loaded: {len(self.face_db)} person(s)")
            self._rebuild_index()
        except Exception as e:
            print(f"[WARN] Face DB load failed: {e}")
            self.face_db = {}

    def save_face_db(self):
        path = self.cfg["FACE_DB_FILE"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "wb") as f: pickle.dump(self.face_db, f)
        except Exception as e:
            print(f"[WARN] Face DB save failed: {e}")

    def _rebuild_index(self):
        self.face_indexer.reset()
        pool_mode = self.cfg.get("MULTI_EMBEDDING_POOLING", "max")
        names, embs, ths = [], [], []
        for name, data in self.face_db.items():
            e = data.get("embeddings", [])
            if not e: continue
            arr = np.array(e, dtype=np.float32)
            pooled = np.mean(arr, axis=0) if pool_mode == "mean" else np.max(arr, axis=0)
            names.append(name); embs.append(pooled)
            ths.append(data.get("threshold", self.cfg["FACE_RECOG_THRESHOLD"]))
        if names:
            self.face_indexer.add_embeddings(names, embs, ths)
            print(f"[INFO] FAISS index rebuilt: {len(names)} person(s)")

    def recognize_face(self, embedding) -> Tuple[str, float, str]:
        if embedding is None: return "UNKNOWN", 0.0, "No embedding"
        names, sims, ths = self.face_indexer.search(embedding, k=3)
        if not names: return "UNKNOWN", 0.0, "No enrollments"
        name, sim, th = names[0], sims[0], ths[0]
        if sim >= th:
            return name, float(np.clip(sim, 0, 1)), f"sim={sim:.3f}>={th:.3f}"
        return "UNKNOWN", float(sim), f"below_threshold sim={sim:.3f}<{th:.3f}"

    def _smooth_recognize(self, oid, embedding):
        name, conf, reason = self.recognize_face(embedding)
        self._face_recog_history[oid].append(name)
        if conf > 0.5:
            counts = defaultdict(int)
            for n in self._face_recog_history[oid]: counts[n] += 1
            name = max(counts.items(), key=lambda kv: kv[1])[0]
        return name, conf, reason

    def get_current_unknowns(self) -> List[Tuple[int, str, float]]:
        out = []
        for oid, (name, conf, _) in self._current_face_labels.items():
            if name == "UNKNOWN" or name.startswith("STRANGER_"):
                out.append((oid, name, conf))
        return out
    
    def _summarize_held_objects(self, frame, hand_dets, object_detections):
        held = []

        if not hand_dets:
            return held

        hand_boxes = []
        if hasattr(self, "custom_objects") and self.custom_objects is not None:
            for h in hand_dets:
                hb = self.custom_objects._hand_bbox(frame, h)
                if hb is not None:
                    hand_boxes.append(hb)

        for d in object_detections:
            if d.get("class_name") == "person":
                continue

            x1, y1, x2, y2 = d.get("bbox", (0, 0, 0, 0))
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            for hx1, hy1, hx2, hy2 in hand_boxes:
                pad = 40
                if hx1 - pad <= cx <= hx2 + pad and hy1 - pad <= cy <= hy2 + pad:
                    held.append({
                        "class_name": d.get("class_name"),
                        "confidence": float(d.get("confidence", 0.0)),
                        "category": d.get("category"),
                        "source_model": d.get("source_model", "general"),
                        "bbox": d.get("bbox"),
                    })
                    break

        return held

    def enroll_unknown_face(self, oid: int, person_name: str) -> bool:
        if oid not in self._current_face_labels:
            print(f"[ENROLL] Track {oid} not present."); return False
        _, _, embedding = self._current_face_labels[oid]
        if embedding is None:
            print(f"[ENROLL] No embedding for track {oid}."); return False
        self.stranger_buffer.pop(oid, None)
        if person_name not in self.face_db:
            self.face_db[person_name] = {"embeddings": [],
                                          "threshold": self.cfg["FACE_RECOG_THRESHOLD"]}
        embs = self.face_db[person_name]["embeddings"]
        if len(embs) >= self.cfg["MAX_ENROLLMENT_EMBEDDINGS"]:
            embs.pop(0)
        embs.append(embedding)
        if len(embs) >= 2:
            mv, sv = _compute_intra_class_variance(embs)
            self.face_db[person_name]["threshold"] = _auto_threshold(mv, sv)
        self._rebuild_index()
        self.save_face_db()
        self._last_enrollment_time = time.time()
        print(f"[ENROLL] {person_name}: {len(embs)}/{self.cfg['MAX_ENROLLMENT_EMBEDDINGS']} embeddings")
        return True
    
    def enroll_best_visible_face(self, person_name: str) -> bool:
        candidates = []

        for oid, data in self._current_face_labels.items():
            try:
                name, conf, embedding = data
            except Exception:
                continue

            if embedding is None:
                continue

            if name == "UNKNOWN" or str(name).startswith("STRANGER_"):
                candidates.append((oid, name, conf, embedding))

        if not candidates:
            print("[ENROLL] No visible unknown/stranger face to enroll.")
            return False

        oid, old_name, conf, embedding = max(candidates, key=lambda x: x[2])

        ok = self.add_embedding_for_name(person_name, embedding)
        if not ok:
            print("[ENROLL] Failed to add embedding.")
            return False

        self._current_face_labels[oid] = (person_name, 1.0, embedding)

        if oid in self.stranger_buffer:
            self.stranger_buffer[oid]["label"] = person_name
            self.stranger_buffer[oid]["embedding"] = embedding
            self.stranger_buffer[oid]["last_seen"] = time.time()

        self._recognition_cache[oid] = {
            "name": person_name,
            "conf": 1.0,
            "reason": "manual_visible_enrollment",
            "embedding": embedding,
            "ts": time.time(),
        }

        self._face_recog_history[oid].clear()
        self._face_recog_history[oid].append(person_name)

        print(f"[ENROLL] {old_name} on track {oid} is now enrolled as {person_name}.")
        return True

    def add_embedding_for_name(self, person_name: str, embedding: np.ndarray) -> bool:
        if embedding is None: return False
        if person_name not in self.face_db:
            self.face_db[person_name] = {"embeddings": [],
                                         "threshold": self.cfg["FACE_RECOG_THRESHOLD"]}
        embs = self.face_db[person_name]["embeddings"]
        if len(embs) >= self.cfg["MAX_ENROLLMENT_EMBEDDINGS"]: embs.pop(0)
        embs.append(embedding)
        if len(embs) >= 2:
            mv, sv = _compute_intra_class_variance(embs)
            self.face_db[person_name]["threshold"] = _auto_threshold(mv, sv)
        self._rebuild_index()
        self.save_face_db()
        return True

    def _extract_shirt_colors(self, frame, obj_dets, tracked):
        person_dets = [d for d in obj_dets if d["class_name"] == "person"]
        if not person_dets or not tracked: return
        H, W = frame.shape[:2]
        for oid, c in tracked.items():
            best, best_d = None, float("inf")
            for pd in person_dets:
                bx1, by1, bx2, by2 = pd["bbox"]
                pc = ((bx1 + bx2) / 2, (by1 + by2) / 2)
                d = math.hypot(c[0] - pc[0], c[1] - pc[1])
                if d < best_d and d < 150: best, best_d = pd, d
            if not best: continue
            x1, y1, x2, y2 = best["bbox"]
            h, w = y2 - y1, x2 - x1
            ty1 = max(0, y1 + int(h * 0.30))
            ty2 = min(H, y1 + int(h * 0.65))
            tx1 = max(0, x1 + int(w * 0.15))
            tx2 = min(W, x2 - int(w * 0.15))
            if ty2 > ty1 and tx2 > tx1:
                torso = frame[ty1:ty2, tx1:tx2]
                color = get_dominant_color(torso, k=3)
                if color: self._person_colors[oid] = color

    @staticmethod
    def _draw_face_box(frame, x1, y1, x2, y2, name, conf, oid, is_real):
        if not is_real: color = (0, 0, 255)
        elif name == "UNKNOWN" or name.startswith("STRANGER_"): color = (0, 165, 255)
        else: color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} ({conf:.2f}) ID:{oid}"
        cv2.putText(frame, label, (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def _draw_person_info(self, frame, fobj, x1, y2, oid, age, gender, emotion):
        st = self.behavior.get_state(oid)
        if not st: return
        y = y2 + 20
        if oid in self._person_colors:
            r, g, b = self._person_colors[oid]
            cv2.rectangle(frame, (x1, y - 12), (x1 + 14, y + 2), (int(b), int(g), int(r)), -1)
            cv2.rectangle(frame, (x1, y - 12), (x1 + 14, y + 2), (255, 255, 255), 1)
            cv2.putText(frame, "Shirt", (x1 + 18, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (200, 200, 200), 1); y += 20
        cv2.putText(frame, f"Age:{age} G:{gender} E:{emotion}", (x1, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1); y += 20
        sus = st.get("suspicion", 0)
        if sus > 5:
            cv2.putText(frame, f"Suspicion: {sus:.1f}", (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1); y += 20
        stress = st.get("stress_level", "Low")
        sc = {"Low": (0, 255, 0), "Medium": (0, 255, 255),
              "High": (0, 100, 255), "Critical": (0, 0, 255)}.get(stress, (0, 255, 0))
        cv2.putText(frame, f"Stress: {stress}", (x1, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, sc, 1); y += 20
        beh = st.get("active_behaviors", [])
        if beh:
            cv2.putText(frame, ", ".join(beh), (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)
            
    def _draw_zones_grid(self, frame):
        if not self.cfg.get("SHOW_ZONES_GRID", False):
            return frame

        h, w = frame.shape[:2]
        rows, cols = self.cfg.get("ZONES_GRID", (3, 3))

        for r in range(1, rows):
            y = int(h * r / rows)
            cv2.line(frame, (0, y), (w, y), (180, 180, 180), 1)

        for c in range(1, cols):
            x = int(w * c / cols)
            cv2.line(frame, (x, 0), (x, h), (180, 180, 180), 1)

        zone_id = 1
        for r in range(rows):
            for c in range(cols):
                x = int(w * c / cols) + 8
                y = int(h * r / rows) + 22
                cv2.putText(frame, f"Z{zone_id}", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                zone_id += 1

        return frame

    def _recognize_and_draw(self, frame, face_objects, tracked):
        info = []
        for fobj in face_objects:
            x1, y1, x2, y2 = self.face_analyzer.get_bbox(fobj)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            embedding = self.face_analyzer.get_embedding(fobj)
            oid = -1; best_d = float("inf")
            for t_oid, t_c in tracked.items():
                d = math.hypot(cx - t_c[0], cy - t_c[1])
                if d < best_d and d < 150: best_d, oid = d, t_oid

            spoof_status = AntiSpoofDetector.REAL
            spoof_details = {}
            spoof_confirmed = False

            if self.anti_spoof.enabled:
                landmarks = self.face_analyzer.get_landmarks(fobj)
                crop = frame[max(0, y1):min(frame.shape[0], y2),
                             max(0, x1):min(frame.shape[1], x2)]
                spoof_status, spoof_details = self.anti_spoof.analyze(
                    crop, frame.shape, landmarks=landmarks, bbox=(x1, y1, x2, y2))

                if oid > 0:
                    self._spoof_history[oid].append(spoof_status)
                    needed = self.cfg["ANTI_SPOOFING"].get("SUSPECT_CONFIRM_FRAMES", 4)
                    recent_suspects = sum(
                        1 for s in self._spoof_history[oid]
                        if s == AntiSpoofDetector.SUSPECT
                    )
                    spoof_confirmed = recent_suspects >= needed

            is_real = not spoof_confirmed

            if spoof_confirmed:
                self._draw_face_box(frame, x1, y1, x2, y2, "SPOOF", 0.0, oid, False)
                info.append({"oid": oid, "name": "SPOOF", "confidence": 0.0,
                             "bbox": (x1, y1, x2, y2), "is_real": False,
                             "spoof_status": spoof_status,
                             "spoof_details": spoof_details})
                continue

            perf = self.cfg.get("PERFORMANCE", {})
            recog_every = max(1, int(perf.get("FACE_RECOG_EVERY_N_FRAMES", 3)))
            cache_ttl = float(perf.get("RECOGNITION_CACHE_TTL_SEC", 2.0))
            cached = self._recognition_cache.get(oid)

            if (cached and time.time() - cached.get("ts", 0) <= cache_ttl
                    and self._frame_count % recog_every != 0):
                name, conf, reason = cached["name"], cached["conf"], cached["reason"]
            else:
                name, conf, reason = self._smooth_recognize(oid, embedding)
                self._recognition_cache[oid] = {
                    "name": name,
                    "conf": conf,
                    "reason": reason,
                    "embedding": embedding,
                    "ts": time.time(),
                }
            if name == "UNKNOWN" and self.cfg.get("STRANGER_TRACKING_ENABLED", True):
                if oid not in self.stranger_buffer:
                    matched_label = self._reid_stranger(embedding)
                    if matched_label:
                        name = matched_label
                    else:
                        name = f"STRANGER_{self._next_stranger_label}"
                        self._next_stranger_label += 1
                        self._stranger_gallery[name] = {
                            "embedding": np.asarray(embedding, dtype=np.float32),
                            "first_seen": time.time(),
                            "last_seen": time.time(),
                            "seen_count": 1,
                        }
                    self.stranger_buffer[oid] = {
                        "embedding": embedding, "label": name,
                        "first_seen": time.time(), "last_seen": time.time(),
                        "frames": 1, "bbox": (x1, y1, x2, y2),
                    }
                else:
                    sb = self.stranger_buffer[oid]
                    sb["last_seen"] = time.time(); sb["frames"] += 1
                    sb["bbox"] = (x1, y1, x2, y2)
                    name = sb["label"]
            self._current_face_labels[oid] = (name, conf, embedding)
            self._draw_face_box(frame, x1, y1, x2, y2, name, conf, oid, is_real)
            age = None
            gender = None
            if self.cfg.get("SHOW_AGE_GENDER", False):
                age = self.face_analyzer.get_age(fobj) or "?"
                gender = self.face_analyzer.get_gender(fobj)
                emotion = "neutral"
                if oid > 0:
                    self._draw_person_info(frame, fobj, x1, y2, oid, age, gender, emotion)

            info.append({"oid": oid, "name": name, "confidence": conf,
                         "bbox": (x1, y1, x2, y2), "is_real": is_real,
                         "reason": reason, "age": age, "gender": gender})
        return info

    def _reid_stranger(self, embedding) -> Optional[str]:
        if embedding is None or not self._stranger_gallery:
            return None

        thr = self.cfg.get("STRANGER_REID_THRESHOLD", 0.62)
        best_label, best_sim = None, -1.0

        for label, data in self._stranger_gallery.items():
            sim = _cosine(embedding, data["embedding"])
            if sim > best_sim:
                best_sim, best_label = sim, label

        if best_sim >= thr:
            self._stranger_gallery[best_label]["embedding"] = (
                0.8 * self._stranger_gallery[best_label]["embedding"] +
                0.2 * embedding
            )
            self._stranger_gallery[best_label]["last_seen"] = time.time()
            self._stranger_gallery[best_label]["seen_count"] += 1
            return best_label

        return None

    def _cleanup_strangers(self):
        max_age = self.cfg["STRANGER_BUFFER_MAX_AGE_SEC"]
        now = time.time()
        for oid in list(self.stranger_buffer.keys()):
            if now - self.stranger_buffer[oid]["last_seen"] > max_age:
                self.stranger_buffer.pop(oid, None)

    def _check_object_interactions(self, tracked, obj_dets) -> List[tuple]:
        events = []
        cooldown = self.cfg["OBJECT_INTERACTION_COOLDOWN_SEC"]
        now = time.time()
        for d in obj_dets:
            if d["class_name"] == "person": continue
            ox = (d["bbox"][0] + d["bbox"][2]) / 2
            oy = (d["bbox"][1] + d["bbox"][3]) / 2
            for oid, c in tracked.items():
                if math.hypot(c[0] - ox, c[1] - oy) < 80:
                    key = f"{oid}|{d['class_name']}"
                    if now - self._object_interaction_last.get(key, 0) < cooldown:
                        continue
                    self._object_interaction_last[key] = now
                    events.append(("OBJECT_INTERACTION", f"ID_{oid}", 1.0,
                                   f"ID_{oid} near {d['class_name']}"))
        return events

    def _draw_ui(self, frame, events, tracked, obj_dets):
        h, w = frame.shape[:2]
        if self.cfg.get("SHOW_COUNT_LINE", False):
            ly = self.cfg.get("COUNT_LINE_Y", 360)
            cv2.line(frame, (0, ly), (w, ly), (255, 255, 0), 2)

        danger_overlay_enabled = self.cfg.get("DANGER_DETECTION", {}).get("ENABLED", False)
        if danger_overlay_enabled:
            dangerous = self.cfg.get("DANGEROUS_OBJECTS", set())
            for d in obj_dets:
                if d["class_name"].lower() in dangerous or d["category"] == "dangerous":
                    bx1, by1, bx2, by2 = d["bbox"]
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
                    cv2.putText(frame, f"DANGER: {d['class_name'].upper()}",
                                (bx1, max(15, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

        y = 30
        for evt in events[-6:]:
            t = evt[0]; det = evt[3] if len(evt) > 3 else ""
            color = {"DANGEROUS_OBJECT": (0, 0, 255), "SPOOF_DETECTED": (0, 0, 255),
                     "EVACUATION_ALERT": (0, 0, 255), "FIRE_DETECTED": (0, 0, 255),
                     "FALL_DETECTED": (0, 0, 255), "HANDS_RAISED": (0, 165, 255),
                     "HESITATION": (0, 165, 255), "PACING": (0, 165, 255),
                     "SCANNING": (0, 165, 255), "SPATIAL_ANOMALY": (0, 165, 255),
                     "LOITERING": (0, 165, 255), "CROWD_FORMING": (0, 165, 255),
                     "RECOGNITION": (0, 255, 0)}.get(t, (200, 200, 200))
            txt = f"[{t}] {evt[1] if len(evt) > 1 else ''} - {det[:60]}"
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y += 20

        if self.cfg.get("SHOW_HEATMAP", False):
            ov = self.crowd_intel.get_density_overlay(frame)
            if ov is not None: frame[:] = ov

        cv2.putText(frame, f"Tracked: {len(tracked)}", (w - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if self.cfg.get("DISPLAY_FPS", True):
            el = time.time() - self._start_time
            self._fps = self._frame_count / el if el > 0 else 0
            cv2.putText(frame, f"FPS: {self._fps:.1f}", (w - 120, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        self._draw_zones_grid(frame)
        return frame

    def process(self, frame):
        t0 = time.time()
        events: List[tuple] = []
        h, w = frame.shape[:2]
        self.behavior.set_frame_size(h, w)
        analysis = frame.copy(); display = frame.copy()
        self._frame_count += 1
        perf = self.cfg.get("PERFORMANCE", {})
        face_every = max(1, int(perf.get("FACE_DETECT_EVERY_N_FRAMES", 2)))
        if self._frame_count % face_every == 0 or not self._last_face_objects:
            try:
                face_objects = self.face_analyzer.detect(analysis)
                self._last_face_objects = face_objects
            except Exception as e:
                print(f"[ERROR] face detect: {e}")
                face_objects = self._last_face_objects or []
        else:
            face_objects = self._last_face_objects

        hand_dets = self.hand_detector.detect(analysis)

        if self.cfg.get("SHOW_HAND_LANDMARKS", False):
            self.hand_detector.draw(display, hand_dets)

        object_detections = []
        yolo_every = max(1, int(perf.get("YOLO_EVERY_N_FRAMES", 3)))

        if self.object_detector is not None:
            if self._frame_count % yolo_every == 0 or not self._last_object_detections:
                object_detections = self.object_detector.detect(analysis)
                self._last_object_detections = object_detections
            else:
                object_detections = list(self._last_object_detections)

        if self.cfg.get("ENABLE_FIRE_SMOKE_HEURISTICS", False):
            object_detections.extend(self.supplementary.detect_all(analysis))

        if self.danger_detector is not None:
            danger_cfg = self.cfg.get("DANGER_DETECTION", {})
            danger_every = max(1, int(danger_cfg.get("EVERY_N_FRAMES", 10)))
            allowed = {c.lower() for c in danger_cfg.get("ALLOWED_CLASSES", set())}
            min_conf_by_class = {
                str(k).lower(): float(v)
                for k, v in danger_cfg.get("MIN_CONF_BY_CLASS", {}).items()
            }

            if self._frame_count % danger_every == 0:
                raw_danger_dets = self.danger_detector.detect(analysis)
                danger_dets = []

                for d in raw_danger_dets:
                    cls_name = str(d.get("class_name", "")).lower()
                    conf = float(d.get("confidence", 0.0))

                    matched = any(token in cls_name for token in allowed)
                    if not matched:
                        continue

                    required_conf = max(
                        min_conf_by_class.get(cls_name, danger_cfg.get("CONF", 0.55)),
                        danger_cfg.get("CONF", 0.55),
                    )
                    if conf < required_conf:
                        continue

                    d["category"] = "dangerous"
                    danger_dets.append(d)

                self._last_danger_detections = danger_dets
            else:
                danger_dets = list(self._last_danger_detections)

            object_detections.extend(danger_dets)

        if self.cfg.get("SHOW_OBJECT_BOXES", True) and self.object_detector is not None:
            self.object_detector.draw_detections(display, object_detections, skip_person=True)
        
        if self.custom_objects is not None and self.custom_objects.enabled:
            custom_cfg = self.cfg.get("CUSTOM_OBJECTS", {})
            custom_every = max(1, int(custom_cfg.get("MATCH_EVERY_N_FRAMES", 5)))

            if self._frame_count % custom_every == 0:
                custom_dets = self.custom_objects.detect_from_hands(analysis, hand_dets)
                self._last_custom_object_detections = custom_dets
            else:
                custom_dets = list(self._last_custom_object_detections)

            object_detections.extend(custom_dets)

        person_rects = [d["bbox"] for d in object_detections if d["class_name"] == "person"]
        tracked = self.person_tracker.update(person_rects)

        self._extract_shirt_colors(analysis, object_detections, tracked)

        events.extend(self.behavior.update(tracked))
        events.extend(self.crowd_intel.update(tracked, frame_size=(h, w)))

        try:
            pose_result = self.pose_detector.analyze(analysis)
            if pose_result.get("is_fallen"):
                events.append(("FALL_DETECTED", "PERSON", 0.85, "Possible fall: torso horizontal"))
            if pose_result.get("hands_raised"):
                events.append(("HANDS_RAISED", "PERSON", 0.8, "Both hands raised above head"))
            if self.cfg.get("SHOW_POSE_LANDMARKS", False):
                self.pose_detector.draw(display, pose_result)
        except Exception as e:
            print(f"[WARN] pose analysis failed: {e}")

        faces_info = self._recognize_and_draw(display, face_objects, tracked)

        self._last_faces_seen = []
        for fi in faces_info:
            oid = fi.get("oid", -1)
            shirt_color = None
            if oid in self._person_colors:
                r, g, b = self._person_colors[oid]
                shirt_color = {"rgb": [int(r), int(g), int(b)]}

            self._last_faces_seen.append({
                "track_id": oid,
                "name": fi.get("name"),
                "confidence": float(fi.get("confidence", 0.0)),
                "is_real": bool(fi.get("is_real", True)),
                "bbox": fi.get("bbox"),
                "shirt_color": shirt_color,
            })

        self._last_objects_seen = [
            {
                "class_name": d.get("class_name"),
                "confidence": float(d.get("confidence", 0.0)),
                "bbox": d.get("bbox"),
                "category": d.get("category"),
            }
            for d in object_detections[:30]
        ]

        self._last_held_objects = self._summarize_held_objects(
            analysis,
            hand_dets,
            object_detections,
        )

        self._last_frame_summary = {
        "timestamp_utc": _utc_now(),
        "tracked_people": len(tracked),
        "faces": self._last_faces_seen,
        "objects": self._last_objects_seen,
        "held_objects": self._last_held_objects,
    }

        for fi in faces_info:
            name = fi.get("name", "UNKNOWN")
            oid = fi.get("oid", -1)
            if name in ("UNKNOWN", "SPOOF") or str(name).startswith("STRANGER_"):
                continue

            shirt_color = None
            if oid in self._person_colors:
                r, g, b = self._person_colors[oid]
                shirt_color = {"rgb": [int(r), int(g), int(b)]}

            self._last_seen_people[name] = {
                "last_seen_utc": _utc_now(),
                "track_id": oid,
                "confidence": float(fi.get("confidence", 0.0)),
                "shirt_color": shirt_color,
            }

        danger_enabled = self.cfg.get("DANGER_DETECTION", {}).get("ENABLED", False)
        dangerous = {x.lower() for x in self.cfg.get("DANGEROUS_OBJECTS", set())}
        for d in object_detections:
            cls_name = str(d.get("class_name", "unknown"))
            self._object_seen_counts[cls_name] += 1
            if d.get("category") == "dangerous" or d.get("event_type") in ("FIRE_DETECTED", "SMOKE_DETECTED", "WEAPON_DETECTED"):
                self._danger_seen_counts[cls_name] += 1
                
            is_danger = d.get("category") == "dangerous" or d["class_name"].lower() in dangerous
            if not is_danger:
                continue

            event_type = d.get("event_type", "DANGEROUS_OBJECT")
            key = f"{event_type}:{d['class_name']}:{d.get('source_model', 'general')}"

            if hasattr(self, "_event_cooldown"):
                if not self._event_cooldown.allowed(key, 20.0):
                    continue

            events.append((
                event_type,
                d["class_name"],
                d["confidence"],
                f"Detected {d['class_name']} at {d['bbox']} via {d.get('source_model', 'general')}"
            ))

        events.extend(self._check_object_interactions(tracked, object_detections))

        for fi in faces_info:
            if not fi.get("is_real", True):
                oid = fi.get("oid", -1)
                key = f"spoof_event:{oid}"
                if self._event_cooldown.allowed(key, 30.0):
                    events.append(("SPOOF_DETECTED", fi.get("name", "?"), 0.9,
                                   f"Confirmed repeated spoof suspicion at bbox {fi['bbox']}"))

        active_ids = set(tracked.keys())
        for fi in faces_info:
            if fi["oid"] > 0: active_ids.add(fi["oid"])
        self.behavior.cleanup(active_ids)
        self._cleanup_strangers()

        self._draw_ui(display, events, tracked, object_detections)
        dt = time.time() - t0
        return (display, faces_info, hand_dets, object_detections,
                events, len(tracked), dt)

# Section 9: Attendance Manager
class AttendanceManager:
    def __init__(self, db: EventDatabase, cfg=None):
        self.db = db
        self.cfg = (cfg or CONFIG)["ATTENDANCE"]
        self.enabled = self.cfg.get("ENABLED", True)
        self._recognitions: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.cfg.get("MIN_RECOGNITION_FRAMES", 8) * 4))
        self._today_clocked_in: set = set()
        self._announce = self.cfg.get("ANNOUNCE_ARRIVAL", True)
        self._date = _today_iso()

    def _maybe_rollover(self):
        today = _today_iso()
        if today != self._date:
            self._today_clocked_in.clear()
            self._date = today

    def handle_recognition(self, person_name: str, camera_id: str = "cam_0", location: str = None):
        if not self.enabled: return
        if person_name in ("UNKNOWN", "SPOOF") or person_name.startswith("STRANGER_"):
            return
        self._maybe_rollover()
        now = time.time()
        self._recognitions[person_name].append(now)
        # purge old
        win = self.cfg.get("RECOGNITION_WINDOW_SEC", 5.0)
        while (self._recognitions[person_name]
               and now - self._recognitions[person_name][0] > win):
            self._recognitions[person_name].popleft()
        min_frames = self.cfg.get("MIN_RECOGNITION_FRAMES", 8)
        if (len(self._recognitions[person_name]) >= min_frames
                and person_name not in self._today_clocked_in):
            pid = self.db.get_person_id(person_name)
            if pid is None:
                # auto-create lightweight people row
                pid = self.db.upsert_person(person_name)
            result = self.db.attendance_clock_in(pid, camera_id, location)
            self._today_clocked_in.add(person_name)
            if "clocked_in_at" in result:
                late = result.get("late_minutes", 0)
                msg = f"Welcome {person_name}."
                if late > 0: msg = f"Welcome {person_name}. You are {late} minutes late."
                if self._announce: voice(msg, "INFO", dedup_key=f"clockin:{person_name}")
                self.db.log_event("ATTENDANCE_CLOCKIN", person_id=pid, confidence=1.0,
                                  details={"late_min": late}, camera_id=camera_id,
                                  location=location, severity=0)

    def manual_clock_out(self, person_name: str) -> dict:
        pid = self.db.get_person_id(person_name)
        if pid is None: return {"error": f"Unknown person: {person_name}"}
        r = self.db.attendance_clock_out(pid)
        if "clocked_out_at" in r:
            self._today_clocked_in.discard(person_name)
            self.db.log_event("ATTENDANCE_CLOCKOUT", person_id=pid, confidence=1.0,
                              details={"work_min": r.get("work_minutes", 0)})
            if self._announce:
                voice(f"Goodbye {person_name}.", "INFO", dedup_key=f"clockout:{person_name}")
        return r

# Section 9: AI assistant
TOOL_DEFINITIONS = [
    {"type": "function", "function": {
        "name": "get_system_status",
        "description": "Get system overall status: uptime, enrolled faces, today's event count, channel availability.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_enrolled_faces",
        "description": "List all people enrolled in face recognition.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_recent_events",
        "description": "Get the most recent security events (any type).",
        "parameters": {"type": "object", "properties": {
            "limit": {"type": "integer", "description": "Max events (default 20, max 100)"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "get_event_summary",
        "description": "Statistical summary of events grouped by type over N days.",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "Days to look back (default 7)"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "get_events_by_type",
        "description": "Filter events by a specific event_type (e.g. DANGEROUS_OBJECT, FALL_DETECTED).",
        "parameters": {"type": "object", "properties": {
            "event_type": {"type": "string"},
            "limit": {"type": "integer"}},
            "required": ["event_type"]}}},
    {"type": "function", "function": {
        "name": "get_person_events",
        "description": "All events related to a specific enrolled person.",
        "parameters": {"type": "object", "properties": {
            "person_name": {"type": "string"},
            "limit": {"type": "integer"}},
            "required": ["person_name"]}}},
    {"type": "function", "function": {
        "name": "get_person_behavior_profile",
        "description": "Get the behavior profile (suspicion / patterns) for a person.",
        "parameters": {"type": "object", "properties": {
            "person_name": {"type": "string"}},
            "required": ["person_name"]}}},
    {"type": "function", "function": {
        "name": "get_alert_history",
        "description": "Recent alert delivery log (email/telegram/webhook).",
        "parameters": {"type": "object", "properties": {
            "limit": {"type": "integer"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "get_audit_log",
        "description": "Audit log for system actions (enrollments, config changes).",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string"},
            "limit": {"type": "integer"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "search_events",
        "description": "Free-text search across events (event_type, details, person, location).",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "days": {"type": "integer"},
            "limit": {"type": "integer"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_attendance_report",
        "description": "Attendance report (clock-ins / outs) across the last N days, optionally per person.",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer"},
            "person_name": {"type": "string"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "get_today_attendance",
        "description": "Who is currently clocked in today.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_active_strangers",
        "description": "Strangers currently being tracked (re-identified across appearances).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

DEFAULT_AI_CONFIG = {
    "openai_api_key": None,
    "openai_model": "gpt-4o-mini",
    "openai_whisper_model": "gpt-4o-mini-transcribe",
    "tts_engine": "system",
    "stt_engine": "whisper",
    "conversation_history_limit": 6,
    "max_tool_iterations": 3,
    "greeting_message": "Security assistant online. How can I help?",
    "farewell_message": "Security assistant offline.",
}

def _build_system_prompt() -> str:
    return (
        "You are an AI assistant embedded in an intelligent security and attendance system.\n"
        "You can call tools to query the database for: system status, enrolled faces, events, "
        "attendance, alerts, audit logs, behavior profiles, and active strangers.\n\n"
        "Rules:\n"
        " - Be concise. Use bullet points for lists.\n"
        " - Use relative times ('3 hours ago').\n"
        " - For security events, lead with severity.\n"
        " - If user asks about someone, fetch their events AND attendance.\n"
        " - Never invent data. If a tool returns nothing, say so.\n"
        f"Current UTC time: {dt_datetime.utcnow().strftime('%Y-%m-%d %H:%M')}\n"
    )


class AIAssistant:
    def __init__(self, db: EventDatabase, alert_manager: AlertManager = None, vision_system: VisionSystem = None, attendance_manager: AttendanceManager = None, config=None):
        self.db = db
        self.alert_manager = alert_manager
        self.vision_system = vision_system
        self.attendance = attendance_manager
        self.config = {**DEFAULT_AI_CONFIG, **(config or {})}
        self._start_time = time.time()
        self._is_listening = False
        self._stop_evt = threading.Event()
        self._thread = None
        self._history: List[dict] = []
        self._lock = threading.Lock()
        self._openai_client = None
        self._stt_recognizer = None
        self._init_openai()
        self._init_stt()

    def _init_openai(self):
        if not OPENAI_AVAILABLE: return
        key = self.config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        if not key or "PLACEHOLDER" in str(key) or key == "OPENAI_API_KEY_PLACEHOLDER":
            print("[AI] No valid OpenAI key. Running in demo mode."); return
        try:
            self._openai_client = openai.OpenAI(api_key=key)
            print(f"[AI] OpenAI client ready (model: {self.config['openai_model']})")
        except Exception as e:
            print(f"[AI] OpenAI init failed: {e}")
            self._openai_client = None

    def _init_stt(self):
        if not STT_AVAILABLE: return
        try:
            self._stt_recognizer = sr.Recognizer()
            self._stt_recognizer.energy_threshold = 300
        except Exception as e:
            print(f"[AI] STT init failed: {e}")

    def is_ready(self) -> bool: return self._openai_client is not None
    def is_listening(self) -> bool: return self._is_listening

    def start_listening(self):
        if self._is_listening:
            return

        if self.config.get("stt_engine", "text") == "text":
            print("[AI] Text-mode background listening is disabled to avoid input conflicts.")
            print("[AI] Use assistant.ask('status') from code, or set stt_engine to microphone/whisper later.")
            return

        if self._openai_client is None:
            print("[AI] Cannot start: no valid OpenAI key.")
            return

        self._is_listening = True
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="AIAsst")
        self._thread.start()

    def stop_listening(self):
        if not self._is_listening: return
        self._stop_evt.set()
        self._is_listening = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def ask(self, question: str) -> str:
        with self._lock:
            direct = self._live_direct_answer(question)
            if direct:
                return direct
            return self._process(question)

    def _live_context(self) -> str:
        ctx = {
            "current_time_utc": _utc_now(),
            "live_camera": {},
            "recent_events": [],
            "today_attendance": [],
            "enrolled_people": [],
        }

        try:
            if self.vision_system:
                ctx["live_camera"] = {
                    "frame_summary": getattr(self.vision_system, "_last_frame_summary", {}),
                    "active_strangers": [
                        {
                            "track_id": oid,
                            "label": sb.get("label"),
                            "frames": sb.get("frames"),
                            "last_seen_sec_ago": round(time.time() - sb.get("last_seen", time.time()), 1),
                        }
                        for oid, sb in self.vision_system.stranger_buffer.items()
                    ],
                }
        except Exception as e:
            ctx["live_camera_error"] = str(e)

        try:
            ctx["recent_events"] = [
                {
                    "timestamp": e["timestamp"],
                    "event_type": e["event_type"],
                    "person": e["person_name"] if "person_name" in e.keys() else None,
                    "details": _safe_json_parse(e["details_json"]),
                }
                for e in self.db.get_recent_events(limit=20)
            ]
        except Exception as e:
            ctx["recent_events_error"] = str(e)

        try:
            ctx["today_attendance"] = [
                {k: r[k] for k in r.keys()}
                for r in self.db.attendance_report(days=1)
            ]
        except Exception as e:
            ctx["attendance_error"] = str(e)

        try:
            ctx["enrolled_people"] = [
                {"id": r["id"], "name": r["name"], "role": r["role"]}
                for r in self.db.get_known_face_names(limit=100)
            ]
        except Exception as e:
            ctx["people_error"] = str(e)

        return json.dumps(ctx, default=str)
    
    def _live_direct_answer(self, question: str) -> Optional[str]:
        q = question.lower()

        if any(p in q for p in ("what am i holding", "what i'm holding", "what is in my hand", "what object am i holding")):
            if not self.vision_system:
                return "I do not have live camera access right now."

            held = getattr(self.vision_system, "_last_held_objects", [])
            if not held:
                objects = getattr(self.vision_system, "_last_objects_seen", [])
                if objects:
                    names = ", ".join(
                        f"{o.get('class_name')} ({float(o.get('confidence', 0.0)):.2f})"
                        for o in objects[:5]
                    )
                    return f"I do not see a confirmed object in your hand, but I currently see: {names}."
                return "I do not see a clear object in your hand right now."

            names = ", ".join(
                f"{o.get('class_name')} ({float(o.get('confidence', 0.0)):.2f})"
                for o in held[:5]
            )
            return f"You appear to be holding: {names}."

        if any(p in q for p in ("who is visible", "who can you see", "who is on camera")):
            faces = getattr(self.vision_system, "_last_faces_seen", []) if self.vision_system else []
            if not faces:
                return "I do not currently see any recognized faces."
            names = ", ".join(
                f"{f.get('name')} ({float(f.get('confidence', 0.0)):.2f})"
                for f in faces[:8]
            )
            return f"I currently see: {names}."

        if any(p in q for p in ("what objects", "what do you see", "objects do you see")):
            objects = getattr(self.vision_system, "_last_objects_seen", []) if self.vision_system else []
            if not objects:
                return "I do not currently see any clear objects."

            counts = defaultdict(int)
            best_conf = {}
            for o in objects:
                name = o.get("class_name", "unknown")
                counts[name] += 1
                best_conf[name] = max(best_conf.get(name, 0.0), float(o.get("confidence", 0.0)))

            summary = ", ".join(
                f"{name} x{count} ({best_conf[name]:.2f})"
                for name, count in list(counts.items())[:8]
            )
            return f"I currently see: {summary}."

        return None

    def ask_voice_once(self, timeout: float = 6.0, phrase_time_limit: float = 12.0) -> str:
        if self._openai_client is None:
            return "AI assistant is not ready. Check your OpenAI API key."

        if not STT_AVAILABLE:
            return "speech_recognition is not installed. Install it with pip install SpeechRecognition pyaudio."

        try:
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True

            print("[AI] Listening... speak now.")
            voice("Listening.", "INFO", dedup_key="ai_listening")

            with sr.Microphone() as src:
                recognizer.adjust_for_ambient_noise(src, duration=0.5)
                audio = recognizer.listen(
                    src,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )

            import tempfile

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio.get_wav_data())
                    tmp_path = f.name

                print("[AI] Transcribing...")
                with open(tmp_path, "rb") as af:
                    transcript = self._openai_client.audio.transcriptions.create(
                        model=self.config.get("openai_whisper_model", "gpt-4o-mini-transcribe"),
                        file=af,
                        response_format="text",
                    )

                question = transcript.strip() if isinstance(transcript, str) else transcript.text.strip()
                if not question:
                    return "I did not hear a clear question."

                print(f"[USER voice] {question}")
                answer = self.ask(question)
                return answer

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        except sr.WaitTimeoutError:
            return "I did not hear anything."
        except sr.UnknownValueError:
            return "I could not understand the audio."
        except Exception as e:
            return f"Voice assistant failed: {e}"

    def _loop(self):
        try:
            greeting = self.config.get("greeting_message")
            print(f"[AI] {greeting}"); voice(greeting, "INFO")
            while not self._stop_evt.is_set():
                user = self._listen()
                if not user: continue
                if any(c in user.lower() for c in ("exit", "quit", "stop assistant",
                                                   "goodbye assistant", "bye assistant")):
                    voice("Assistant deactivated.", "INFO"); break
                print(f"[USER] {user}\n[AI] Thinking...")
                try:
                    a = self.ask(user)
                    print(f"[AI] {a}"); voice(a[:300], "INFO", dedup_key=f"answer:{hash(a)%10000}")
                except Exception as e:
                    print(f"[AI ERR] {e}")
        except Exception as e:
            print(f"[AI] Loop failed: {e}")
        finally:
            self._is_listening = False
            print("[AI] Session ended.")

    def _listen(self) -> Optional[str]:
        engine = self.config.get("stt_engine", "text")
        if engine == "text" or not self._stt_recognizer:
            try: return input("[AI] You: ").strip() or None
            except EOFError: return None
        try:
            with sr.Microphone() as src:
                self._stt_recognizer.adjust_for_ambient_noise(src, duration=0.5)
                audio = self._stt_recognizer.listen(src, timeout=8, phrase_time_limit=20)
            if engine == "whisper" and self._openai_client:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio.get_wav_data()); path = f.name
                with open(path, "rb") as af:
                    r = self._openai_client.audio.transcriptions.create(
                        model=self.config.get("openai_whisper_model", "whisper-1"),
                        file=af, language="en")
                os.unlink(path)
                return r.text.strip()
            return self._stt_recognizer.recognize_google(audio).strip()
        except sr.WaitTimeoutError: return None
        except sr.UnknownValueError: return None
        except Exception as e:
            print(f"[AI] STT err: {e}"); return None

    def _process(self, question: str) -> str:
        if self._openai_client is None: return self._demo(question)
        system = _build_system_prompt() + (
        "\n\nYou have access to live camera/database context below. "
        "Use it directly when answering questions about who is visible, objects, clothing colors, attendance, and events. "
        "If the context does not contain the answer, say what is missing.\n"
        f"{self._live_context()}\n"
    )
        self._history.append({"role": "user", "content": question})
        limit = self.config.get("conversation_history_limit", 10)
        if len(self._history) > limit * 2:
            self._history = self._history[-(limit * 2):]
        messages = [{"role": "system", "content": system}] + self._history
        last_content = None
        for _ in range(self.config.get("max_tool_iterations", 5)):
            try:
                resp = self._openai_client.chat.completions.create(
                    model=self.config.get("openai_model", "gpt-4o-mini"),
                    messages=messages, tools=TOOL_DEFINITIONS,
                    tool_choice="auto", temperature=0.3, max_tokens=400, timeout=20.0)
            except Exception as e:
                return f"OpenAI error: {e}"
            msg = resp.choices[0].message
            messages.append(msg)
            if msg.content: last_content = msg.content
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    fn = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except Exception: args = {}
                    result = self._execute_tool(fn, args)
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": result})
                continue
            ans = msg.content or "Done."
            self._history.append({"role": "assistant", "content": ans})
            return ans
        if last_content:
            self._history.append({"role": "assistant", "content": last_content})
            return last_content
        return "I gathered information but need you to be more specific."

    def _demo(self, q: str) -> str:
        ql = q.lower()
        if "status" in ql:
            return f"System online. {self.db.get_face_count()} faces enrolled."
        if "enrolled" in ql or "who" in ql:
            f = self.db.get_known_face_names(limit=50)
            if not f: return "No faces enrolled."
            return f"Enrolled ({len(f)}): " + ", ".join(r["name"] for r in f[:20])
        if "event" in ql or "recent" in ql:
            ev = self.db.get_recent_events(limit=5)
            if not ev: return "No recent events."
            return "Recent:\n" + "\n".join(
                f"  [{e['timestamp']}] {e['event_type']} - {e['person_name'] or 'System'}"
                for e in ev)
        if "attendance" in ql:
            a = self.db.attendance_report(days=1)
            if not a: return "No attendance records today."
            return "Today:\n" + "\n".join(
                f"  {r['name']}: in={r['clock_in'] or '-'} out={r['clock_out'] or '-'}"
                for r in a[:20])
        return "Demo mode (no OpenAI key). Try: status, enrolled, events, attendance."

    def _execute_tool(self, name: str, args: dict) -> str:
        try:
            fn = {
                "get_system_status":           self._tool_status,
                "get_enrolled_faces":          self._tool_faces,
                "get_recent_events":           lambda: self._tool_recent(min(args.get("limit", 20), 100)),
                "get_event_summary":           lambda: self._tool_summary(min(args.get("days", 7), 365)),
                "get_events_by_type":          lambda: self._tool_by_type(args.get("event_type", ""), min(args.get("limit", 20), 100)),
                "get_person_events":           lambda: self._tool_person_events(args.get("person_name", ""), min(args.get("limit", 50), 200)),
                "get_person_behavior_profile": lambda: self._tool_profile(args.get("person_name", "")),
                "get_alert_history":           lambda: self._tool_alerts(min(args.get("limit", 20), 100)),
                "get_audit_log":               lambda: self._tool_audit(args.get("action"), min(args.get("limit", 20), 100)),
                "search_events":               lambda: self._tool_search(args.get("query", ""), min(args.get("days", 7), 365), min(args.get("limit", 30), 100)),
                "get_attendance_report":       lambda: self._tool_attendance(min(args.get("days", 7), 90), args.get("person_name")),
                "get_today_attendance":        self._tool_today_attendance,
                "get_active_strangers":        self._tool_strangers,
            }.get(name)
            if not fn: return json.dumps({"error": f"Unknown tool: {name}"})
            return fn() if callable(fn) and not args else (fn() if not args else fn())
        except Exception as e:
            return json.dumps({"error": f"Tool {name} failed: {e}"})

    def _tool_status(self) -> str:
        fc = self.db.get_face_count()
        rec = self.db.get_recent_events(limit=1)
        last = rec[0]["timestamp"] if rec else None
        ch = {}
        if self.alert_manager:
            ch = {"email": self.alert_manager.email_enabled,
                  "telegram": self.alert_manager.telegram_enabled,
                  "webhook": self.alert_manager.webhook_enabled}
        try:
            today_cnt = self.db._fetchone(
                "SELECT COUNT(*) AS c FROM events WHERE date(timestamp)=?", (_today_iso(),))["c"]
        except Exception: today_cnt = 0
        return json.dumps({
            "status": "online",
            "uptime": _format_duration(time.time() - self._start_time),
            "enrolled_faces": fc, "events_today": today_cnt,
            "last_event_time": last, "alert_channels": ch,
            "current_time_utc": _utc_now(),
        }, default=str)

    def _tool_faces(self) -> str:
        rows = self.db.get_known_face_names(limit=200)
        return json.dumps({"total": len(rows),
                           "faces": [{"id": r["id"], "name": r["name"],
                                      "role": r["role"], "registered": r["created_at"]}
                                     for r in rows]}, default=str)

    def _tool_recent(self, limit) -> str:
        rows = self.db.get_recent_events(limit=limit)
        return json.dumps({"count": len(rows), "events": [self._fmt_event(r) for r in rows]},
                          default=str)

    def _tool_summary(self, days) -> str:
        s = self.db.get_event_summary(days=days)
        return json.dumps({"period_days": days,
                           "summary": [{"event_type": r["event_type"],
                                        "total": int(r["total"])} for r in s]})

    def _tool_by_type(self, et, limit) -> str:
        rows = self.db.get_events_by_type(event_type=et, limit=limit)
        return json.dumps({"event_type": et, "count": len(rows),
                           "events": [self._fmt_event(r) for r in rows]}, default=str)

    def _tool_person_events(self, name, limit) -> str:
        pid = self.db.get_person_id(name)
        if pid is None: return json.dumps({"error": f"Person '{name}' not found."})
        rows = self.db.get_person_timeline(person_id=pid, limit=limit)
        prof = self.db.get_behavior_profile(pid)
        return json.dumps({"person": name, "person_id": pid,
                           "total_events": len(rows),
                           "events": [self._fmt_event(r) for r in rows],
                           "behavior_profile": prof}, default=str)

    def _tool_profile(self, name) -> str:
        pid = self.db.get_person_id(name)
        if pid is None: return json.dumps({"error": f"Person '{name}' not found."})
        prof = self.db.get_behavior_profile(pid)
        try:
            breakdown = self.db._fetchall(
                "SELECT event_type, COUNT(*) AS c FROM events WHERE person_id=? "
                "GROUP BY event_type ORDER BY c DESC", (pid,))
            br = [{"event_type": r["event_type"], "count": r["c"]} for r in breakdown]
        except Exception: br = []
        return json.dumps({"person": name, "behavior_profile": prof, "breakdown": br},
                          default=str)

    def _tool_alerts(self, limit) -> str:
        if self.alert_manager:
            return json.dumps({"count": 0, "alerts": self.alert_manager.get_alert_history(limit)},
                              default=str)
        return json.dumps({"error": "Alert manager not configured."})

    def _tool_audit(self, action, limit) -> str:
        rows = self.db.get_audit_log(action=action, limit=limit)
        return json.dumps({"count": len(rows),
                           "entries": [{"id": r["id"], "timestamp": r["timestamp"],
                                        "action": r["action"], "target": r["target"],
                                        "details": _safe_json_parse(r["details_json"])}
                                       for r in rows]}, default=str)

    def _tool_search(self, q, days, limit) -> str:
        rows = self.db.search_events(q, days=days, limit=limit)
        return json.dumps({"query": q, "period_days": days, "count": len(rows),
                           "events": [self._fmt_event(r) for r in rows]}, default=str)

    def _tool_attendance(self, days, person_name) -> str:
        pid = None
        if person_name:
            pid = self.db.get_person_id(person_name)
            if pid is None: return json.dumps({"error": f"Person '{person_name}' not found."})
        rows = self.db.attendance_report(days=days, person_id=pid)
        return json.dumps({"days": days,
                           "records": [{k: r[k] for k in r.keys()} for r in rows]},
                          default=str)

    def _tool_today_attendance(self) -> str:
        rows = self.db.attendance_report(days=1)
        return json.dumps({"date": _today_iso(),
                           "records": [{k: r[k] for k in r.keys()} for r in rows]},
                          default=str)

    def _tool_strangers(self) -> str:
        if not self.vision_system: return json.dumps({"strangers": []})
        out = []
        for oid, sb in self.vision_system.stranger_buffer.items():
            out.append({
                "track_id": oid, "label": sb["label"],
                "first_seen_sec_ago": round(time.time() - sb["first_seen"], 1),
                "last_seen_sec_ago":  round(time.time() - sb["last_seen"], 1),
                "frames_observed": sb["frames"]})
        return json.dumps({"count": len(out), "strangers": out})

    @staticmethod
    def _fmt_event(r) -> dict:
        return {"id": r["id"], "timestamp": r["timestamp"], "event_type": r["event_type"],
                "person": r.get("person_name") if "person_name" in r.keys() else None,
                "confidence": round(r["confidence"], 3) if r["confidence"] else None,
                "severity": r["severity"], "location": r.get("location"),
                "camera": r["camera_id"],
                "details": _safe_json_parse(r.get("details_json"))}

    def cleanup(self):
        self.stop_listening()

# Section 11: Main entry + Additional functions
def _select_primary_camera(cfg: dict):
    for c in cfg.get("CAMERAS", []):
        if c.get("enabled", True):
            return c
    return {"id": "cam_0", "source": 0, "location": "Default Camera", "enabled": True}


def _open_capture(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened() and isinstance(source, int):
        alt = 1 if source == 0 else 0
        print(f"[WARN] Camera index {source} failed, trying {alt}...")
        cap.release()
        cap = cv2.VideoCapture(alt)
    return cap


def _multi_angle_enrollment(vision: VisionSystem, db: EventDatabase, cap: cv2.VideoCapture, person_name: str, min_embeddings: int = 5, max_embeddings: int = 10) -> bool:
    print(f"\n[ENROLL] Passive enrollment for '{person_name}'")
    print(f"[ENROLL] Collecting {min_embeddings}-{max_embeddings} good samples automatically.")
    print("[ENROLL] Keep one face visible. Press ESC to cancel, ENTER to finish after minimum.")
    voice(f"Enrollment started for {person_name}. Please face the camera naturally.", "INFO")

    captured_embeddings = []
    captured = 0
    last_capture_ts = 0.0
    min_gap_sec = 0.35
    quality_scorer = vision.quality_scorer

    while captured < max_embeddings:
        ret, frame = cap.read()
        if not ret:
            print("[ENROLL] Camera read failed.")
            return False

        display = cv2.resize(frame, (1280, 720))
        faces = vision.face_analyzer.detect(display)

        cv2.putText(display, f"ENROLLMENT: {person_name}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(display, f"Good samples: {captured}/{min_embeddings} required",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        status = "Show exactly one clear, well-lit face"
        status_color = (0, 165, 255)

        if len(faces) == 1:
            x1, y1, x2, y2 = vision.face_analyzer.get_bbox(faces[0])
            face_crop = display[max(0, y1):min(display.shape[0], y2),
                                max(0, x1):min(display.shape[1], x2)]

            if face_crop.size > 0:
                ok, score = quality_scorer.is_acceptable(face_crop)
                color = (0, 255, 0) if ok else (0, 165, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
                cv2.putText(display, f"Quality: {score:.0f}",
                            (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                now = time.time()
                if ok and now - last_capture_ts >= min_gap_sec:
                    embedding = vision.face_analyzer.get_embedding(faces[0])
                    if embedding is not None:
                        captured_embeddings.append(np.asarray(embedding, dtype=np.float32))
                        captured += 1
                        last_capture_ts = now
                        status = f"Accepted sample {captured}"
                        status_color = (0, 255, 0)
                        print(f"[ENROLL] Accepted {captured}/{min_embeddings} quality={score:.0f}")
        elif len(faces) > 1:
            status = "Multiple faces detected"
            status_color = (0, 0, 255)
        else:
            status = "No face detected"
            status_color = (0, 0, 255)

        cv2.putText(display, status, (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(display, "[ENTER] finish after minimum   [ESC] cancel",
                    (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

        cv2.imshow("Enrollment Wizard", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            cv2.destroyWindow("Enrollment Wizard")
            voice("Enrollment cancelled.", "INFO")
            print("[ENROLL] Cancelled.")
            return False

        if key == 13 and captured >= min_embeddings:
            break

    cv2.destroyWindow("Enrollment Wizard")

    if len(captured_embeddings) < min_embeddings:
        print("[ENROLL] Not enough good samples.")
        return False

    for emb in captured_embeddings[:max_embeddings]:
        vision.add_embedding_for_name(person_name, emb)

    db.upsert_person(person_name)
    db.log_audit("ENROLL_PERSON", person_name, {"embeddings": len(captured_embeddings)})

    voice(f"Enrollment complete. Welcome, {person_name}.", "INFO")
    print(f"[ENROLL] SUCCESS: {person_name} enrolled with {len(captured_embeddings)} embeddings.")
    return True


def _save_snapshot(snapshot_dir: str, event_type: str, name: str, frame: np.ndarray) -> Optional[str]:
    try:
        ts = dt_datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_name = "".join(c for c in str(name) if c.isalnum() or c in "_-")[:32] or "x"
        fname = os.path.join(snapshot_dir, f"{event_type}_{safe_name}_{ts}.jpg")
        cv2.imwrite(fname, frame)
        return fname
    except Exception as e:
        print(f"[ERROR] Snapshot save failed: {e}")
        return None

def _export_shutdown_report(db: EventDatabase, vision: VisionSystem, cam_id: str = "cam_0", location: str = None, started_at: float = None) -> Optional[str]:
    try:
        report_cfg = CONFIG.get("REPORTING", {})
        if not report_cfg.get("ENABLED", True):
            return None

        report_dir = report_cfg.get("REPORT_DIR", os.path.join(_BASE_DIR, "reports"))
        os.makedirs(report_dir, exist_ok=True)

        recent_limit = int(report_cfg.get("RECENT_EVENTS_LIMIT", 200))
        recent_events = db.get_recent_events(limit=recent_limit)
        attendance_today = db.attendance_report(days=1)
        people = db.get_known_face_names(limit=1000)

        event_counts = defaultdict(int)
        danger_events = []

        for e in recent_events:
            event_counts[e["event_type"]] += 1
            if e["event_type"] in _SEVERE_EVENT_TYPES:
                danger_events.append({
                    "timestamp": e["timestamp"],
                    "event_type": e["event_type"],
                    "person": e["person_name"] if "person_name" in e.keys() else None,
                    "confidence": e["confidence"],
                    "location": e["location"] if "location" in e.keys() else None,
                    "details": _safe_json_parse(e["details_json"]),
                })

        people_log = []
        for r in attendance_today:
            name = r["name"]
            runtime = getattr(vision, "_last_seen_people", {}).get(name, {})
            people_log.append({
                "name": name,
                "date": r["date"],
                "clock_in": r["clock_in"],
                "clock_out": r["clock_out"],
                "work_minutes": r["work_minutes"],
                "late_minutes": r["late_minutes"],
                "camera_id": r["camera_id"],
                "location": r["location"],
                "last_seen": runtime.get("last_seen_utc"),
                "recognition_confidence": runtime.get("confidence"),
                "shirt_color": runtime.get("shirt_color"),
            })

        active_strangers = []
        for oid, sb in getattr(vision, "stranger_buffer", {}).items():
            active_strangers.append({
                "track_id": oid,
                "label": sb.get("label"),
                "frames": sb.get("frames"),
                "first_seen_sec_ago": round(time.time() - sb.get("first_seen", time.time()), 1),
                "last_seen_sec_ago": round(time.time() - sb.get("last_seen", time.time()), 1),
            })

        report = {
            "report_type": "shutdown_summary",
            "generated_at_utc": _utc_now(),
            "camera": {
                "id": cam_id,
                "location": location,
            },
            "runtime": {
                "started_at_epoch": started_at,
                "duration_sec": round(time.time() - started_at, 1) if started_at else None,
                "duration_human": _format_duration(time.time() - started_at) if started_at else None,
            },
            "summary": {
                "enrolled_people_count": len(people),
                "attendance_records_today": len(attendance_today),
                "recent_event_count": len(recent_events),
                "danger_event_count": len(danger_events),
                "active_stranger_count": len(active_strangers),
            },
            "people": [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "role": p["role"] if "role" in p.keys() else None,
                    "created_at": p["created_at"] if "created_at" in p.keys() else None,
                    "thumbnail_path": p["thumbnail_path"] if "thumbnail_path" in p.keys() else None,
                }
                for p in people
            ],
            "people_log_today": people_log,
            "objects_seen_runtime": dict(sorted(
                getattr(vision, "_object_seen_counts", {}).items(),
                key=lambda kv: kv[1],
                reverse=True,
            )),
            "danger_objects_seen_runtime": dict(sorted(
                getattr(vision, "_danger_seen_counts", {}).items(),
                key=lambda kv: kv[1],
                reverse=True,
            )),
            "event_counts_recent": dict(sorted(event_counts.items())),
            "danger_events_recent": danger_events[:50],
            "active_strangers": active_strangers,
        }

        ts = dt_datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(report_dir, f"shutdown_report_{ts}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"[REPORT] Saved shutdown report -> {path}")
        return path

    except Exception as e:
        print(f"[REPORT] Failed to export shutdown report: {e}")
        traceback.print_exc()
        return None

_SEVERE_EVENT_TYPES = {
    "DANGEROUS_OBJECT", "WEAPON_DETECTED", "SPOOF_DETECTED",
    "FIRE_DETECTED", "SMOKE_DETECTED", "EVACUATION_ALERT",
    "FALL_DETECTED", "HANDS_RAISED", "CONGESTION",
}

_SEVERITY_MAP = {
    "DANGEROUS_OBJECT": 3,
    "WEAPON_DETECTED": 3,
    "FIRE_DETECTED": 3,
    "EVACUATION_ALERT": 3,
    "SMOKE_DETECTED": 2,
    "SPOOF_DETECTED": 2,
    "FALL_DETECTED": 2,
    "HANDS_RAISED": 1,
    "CONGESTION": 1,
    "CROWD_FORMING": 1,
    "HESITATION": 0,
    "PACING": 0,
    "SCANNING": 0,
    "SPATIAL_ANOMALY": 0,
    "LOITERING": 0,
    "OBJECT_INTERACTION": 0,
}

#Main function
def main():
    print("=" * 70)
    print("  INTELLIGENT SECURITY & ATTENDANCE SYSTEM  v2.0")
    print("=" * 70)
    ensure_dirs()

    global VOICE
    VOICE = VoiceManager(CONFIG)
    if VOICE.enabled:
        print("[INFO] Voice manager ready.")
    else:
        print("[WARN] Voice manager disabled or pyttsx3 unavailable.")

    db = EventDatabase()
    db.setup_database()
    print("[INFO] Database ready.")

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "alert_config.json")
    if not os.path.exists(cfg_path):
        AlertManager.create_sample_config(cfg_path)
        print(f"[INFO] Sample alert config written to {cfg_path}. "
              "Edit it to enable notifications.")
    alert_mgr = AlertManager(config_path=cfg_path, db=db)
    if alert_mgr.enabled:
        print(f"[INFO] AlertManager active. Channels: "
              f"email={alert_mgr.email_enabled} "
              f"telegram={alert_mgr.telegram_enabled} "
              f"webhook={alert_mgr.webhook_enabled}")
    else:
        print("[WARN] AlertManager disabled in config.")

    vision = VisionSystem()
    vision.import_known_faces_folder(db)

    attendance = AttendanceManager(db)
    if attendance.enabled:
        print("[INFO] Attendance tracking enabled.")

    api_key = os.environ.get("OPENAI_API_KEY")
    assistant = AIAssistant(
        db=db,
        alert_manager=alert_mgr,
        vision_system=vision,
        attendance_manager=attendance,
        config={
            "openai_api_key": api_key,
            "openai_model": "gpt-4o-mini",
            "openai_whisper_model": "gpt-4o-mini-transcribe",
            "stt_engine": "whisper",
        },
    )
    if assistant.is_ready():
        print("[INFO] AI Assistant ready (OpenAI configured).")
    else:
        print("[INFO] AI Assistant running in DEMO mode "
              "(set OPENAI_API_KEY for full features).")

    cam_spec = _select_primary_camera(CONFIG)
    cam_id = cam_spec["id"]
    cam_location = cam_spec.get("location", "Unknown")
    print(f"[INFO] Opening camera '{cam_id}' source={cam_spec['source']} ({cam_location})...")
    cap = _open_capture(cam_spec["source"])
    if not cap.isOpened():
        print("[FATAL] No camera available. Exiting.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera active at {actual_w}x{actual_h}.")

    snapshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    print()
    print("=" * 70)
    print("  CONTROLS")
    print("    q  -- quit")
    print("    a  -- toggle AI assistant (interactive Q&A)")
    print("    e  -- multi-angle enrollment wizard")
    print("    c  -- manual clock-out wizard")
    print("    i  -- manual clock-in wizard")
    print("    t  -- test alert (send via configured channels)")
    print("    s  -- save manual snapshot")
    print("    h  -- toggle heatmap   o  -- toggle object boxes")
    print("    p  -- toggle pose overlay")
    print("    r  -- print system report")
    print("    u  -- enroll handheld custom object")
    print("    d  -- toggle danger detection")
    print("    g  -- toggle age/gender")
    print("    z  -- toggle zones/grid")
    print("=" * 70)
    voice("Security system online.", "INFO")

    frame_count = 0
    start_time = time.time()
    last_raw_frame = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Camera read failed, retrying...")
                time.sleep(0.3)
                continue
            frame = cv2.resize(frame, (1280, 720))
            last_raw_frame = frame.copy()

            try:
                (display, faces_info, hand_dets, obj_dets,
                 events, tracked_count, dt) = vision.process(frame)
            except Exception as e:
                print(f"[ERROR] Vision pipeline failed (skipping frame): {e}")
                traceback.print_exc()
                continue

            for fi in faces_info:
                name = fi.get("name", "UNKNOWN")
                is_real = fi.get("is_real", True)
                if (is_real and name not in ("UNKNOWN", "SPOOF")
                        and not name.startswith("STRANGER_")):
                    try:
                        attendance.handle_recognition(name, cam_id, cam_location)
                    except Exception as e:
                        print(f"[ERROR] Attendance update failed: {e}")

            for evt in events:
                try:
                    event_type = evt[0]
                    target = evt[1] if len(evt) > 1 else "SYSTEM"
                    confidence = float(evt[2]) if len(evt) > 2 else 0.0
                    details = evt[3] if len(evt) > 3 else ""
                except (IndexError, TypeError, ValueError):
                    continue

                snapshot_path = None
                if event_type in _SEVERE_EVENT_TYPES and last_raw_frame is not None:
                    snapshot_path = _save_snapshot(
                        snapshot_dir, event_type, target, last_raw_frame)

                pid = None
                if (target not in ("SYSTEM", "UNKNOWN", "PERSON")
                        and not str(target).startswith(("STRANGER_", "ID_", "AREA_", "ZONE_"))):
                    pid = db.get_person_id(target)

                try:
                    db.log_event(
                        event_type=event_type,
                        person_id=pid,
                        confidence=confidence,
                        details=details,
                        snapshot_path=snapshot_path,
                        camera_id=cam_id,
                        location=cam_location,
                        severity=_SEVERITY_MAP.get(event_type, 0),
                    )
                except Exception as e:
                    print(f"[ERROR] log_event failed: {e}")

                if event_type in _SEVERE_EVENT_TYPES:
                    try:
                        alert_mgr.check_and_alert(
                            event_type=event_type,
                            name=str(target),
                            confidence=confidence,
                            details=details,
                            snapshot_path=snapshot_path,
                        )
                    except Exception as e:
                        print(f"[ERROR] Alert dispatch failed: {e}")

                if event_type == "DANGEROUS_OBJECT":
                    voice(f"Warning. Dangerous object detected: {target}.",
                        "CRITICAL", dedup_key=f"danger:{target}")
                elif event_type == "WEAPON_DETECTED":
                    voice(f"Warning. Weapon detected: {target}.",
                        "CRITICAL", dedup_key=f"weapon:{target}")
                elif event_type == "FIRE_DETECTED":
                    voice("Warning. Fire or flame detected.",
                        "CRITICAL", dedup_key="fire")
                elif event_type == "SMOKE_DETECTED":
                    voice("Warning. Smoke detected.",
                        "CRITICAL", dedup_key="smoke")
                elif event_type == "SPOOF_DETECTED":
                    voice("Warning. Possible spoofed face detected.",
                        "WARN", dedup_key=f"spoof:{target}")
                elif event_type == "FALL_DETECTED":
                    voice("Alert. Possible fall detected. Please check.",
                          "WARN", dedup_key="fall")
                elif event_type == "EVACUATION_ALERT":
                    voice("Emergency. Possible evacuation in progress.",
                          "CRITICAL", dedup_key="evac")
                elif event_type == "CONGESTION":
                    voice(f"Notice. Crowd congestion at {target}.",
                          "WARN", dedup_key=f"cong:{target}")

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            h_disp = display.shape[0]
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            ai_color = (0, 255, 255) if assistant.is_listening() else (140, 140, 140)
            ai_text = "AI: LIVE" if assistant.is_listening() else "AI: idle"
            cv2.putText(display, ai_text, (10, h_disp - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, ai_color, 2)
            cv2.putText(display, f"Cam: {cam_id} ({cam_location})",
                        (220, h_disp - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Security & Attendance Feed", display)

            #Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            elif key == ord('a'):
                print("[AI] Voice question mode.")
                try:
                    answer = assistant.ask_voice_once(timeout=6.0, phrase_time_limit=12.0)
                    print(f"[AI] {answer}")
                    voice(answer[:300], "INFO", dedup_key=f"ai_answer:{hash(answer) % 10000}", force=True)
                except Exception as e:
                    print(f"[AI] Voice question failed: {e}")
                    voice("AI voice question failed.", "WARN", dedup_key="ai_voice_failed")

            elif key == ord('t'):
                print("[TEST] Dispatching test alert...")
                voice("Sending test alert.", "INFO")
                try:
                    results = alert_mgr.test_alert()
                    print(f"[TEST] Result: {results}")
                except Exception as e:
                    print(f"[TEST] Failed: {e}")

            elif key == ord('s'):
                if last_raw_frame is not None:
                    path = _save_snapshot(snapshot_dir, "MANUAL", "operator",
                                          last_raw_frame)
                    if path:
                        print(f"[SNAPSHOT] Saved -> {path}")
                        voice("Snapshot saved.", "INFO")
                        db.log_audit("MANUAL_SNAPSHOT", path, {})

            elif key == ord('h'):
                CONFIG["SHOW_HEATMAP"] = not CONFIG.get("SHOW_HEATMAP", False)
                CONFIG.setdefault("CROWD_INTELLIGENCE", {})["SHOW_DENSITY_HEATMAP"] = CONFIG["SHOW_HEATMAP"]
                print(f"[UI] Heatmap = {CONFIG['SHOW_HEATMAP']}")

            elif key == ord('o'):
                CONFIG["SHOW_OBJECT_BOXES"] = not CONFIG.get("SHOW_OBJECT_BOXES", True)
                print(f"[UI] Object boxes = {CONFIG['SHOW_OBJECT_BOXES']}")

            elif key == ord('p'):
                CONFIG["SHOW_POSE_LANDMARKS"] = not CONFIG.get("SHOW_POSE_LANDMARKS", False)
                print(f"[UI] Pose overlay = {CONFIG['SHOW_POSE_LANDMARKS']}")

            elif key == ord('e'):
                cv2.destroyWindow("Security & Attendance Feed")
                name = input("\n[ENROLL] Enter name for visible stranger/unknown (blank = cancel): ").strip()

                if not name:
                    print("[ENROLL] Cancelled.")
                else:
                    try:
                        ok = vision.enroll_best_visible_face(name)
                        if ok:
                            db.upsert_person(name)
                            db.log_audit("ENROLL_VISIBLE_FACE", name, {"mode": "instant"})
                            voice(f"{name} enrolled.", "INFO", dedup_key=f"enroll:{name}")
                        else:
                            print("[ENROLL] No visible stranger found. Starting passive enrollment instead.")
                            _multi_angle_enrollment(
                                vision, db, cap, name,
                                min_embeddings=CONFIG.get("MIN_ENROLLMENT_EMBEDDINGS", 5),
                                max_embeddings=CONFIG.get("MAX_ENROLLMENT_EMBEDDINGS", 10),
                            )
                    except Exception as e:
                        print(f"[ENROLL] Failed: {e}")

            elif key == ord('c'):
                cv2.destroyWindow("Security & Attendance Feed")
                rows = db.get_known_face_names(limit=200)
                if not rows:
                    print("[CLOCKOUT] No enrolled people.")
                else:
                    print("\n[CLOCKOUT] Enrolled people:")
                    names = []
                    for i, r in enumerate(rows):
                        names.append(r["name"])
                        print(f"  [{i:>2}] {r['name']}")
                    choice = input("Select index (or name) to clock out, blank = cancel: ").strip()
                    target = None
                    if not choice:
                        pass
                    elif choice.isdigit() and 0 <= int(choice) < len(names):
                        target = names[int(choice)]
                    elif choice in names:
                        target = choice
                    if target:
                        result = attendance.manual_clock_out(target)
                        print(f"[CLOCKOUT] {target}: {result}")
            elif key == ord('i'):
                cv2.destroyWindow("Security & Attendance Feed")
                rows = db.get_known_face_names(limit=200)

                if not rows:
                    print("[CLOCKIN] No enrolled people.")
                else:
                    print("\n[CLOCKIN] Enrolled people:")
                    names = []
                    for i, r in enumerate(rows):
                        names.append(r["name"])
                        print(f"  [{i:>2}] {r['name']}")

                    choice = input("Select index (or name) to clock in, blank = cancel: ").strip()
                    target = None

                    if not choice:
                        pass
                    elif choice.isdigit() and 0 <= int(choice) < len(names):
                        target = names[int(choice)]
                    elif choice in names:
                        target = choice

                    if target:
                        pid = db.get_person_id(target)
                        if pid is None:
                            pid = db.upsert_person(target)

                        result = db.attendance_clock_in(pid, cam_id, cam_location)
                        print(f"[CLOCKIN] {target}: {result}")

                        if "clocked_in_at" in result:
                            db.log_event(
                                "ATTENDANCE_CLOCKIN",
                                person_id=pid,
                                confidence=1.0,
                                details={"manual": True, "late_min": result.get("late_minutes", 0)},
                                camera_id=cam_id,
                                location=cam_location,
                                severity=0,
                            )
                            voice(f"{target} clocked in.", "INFO", dedup_key=f"manual_clockin:{target}")
            elif key == ord('r'):
                try:
                    print("\n===== SYSTEM REPORT =====")
                    print(f"Faces enrolled: {db.get_face_count()}")

                    print("\nRecent events:")
                    for e in db.get_recent_events(limit=10):
                        person = e["person_name"] if "person_name" in e.keys() else None
                        print(f"  {e['timestamp']} | {e['event_type']} | {person or 'SYSTEM'} | {e['details_json']}")

                    print("\nToday attendance:")
                    for r in db.attendance_report(days=1):
                        print(f"  {r['name']} | in={r['clock_in'] or '-'} | out={r['clock_out'] or '-'} | late={r['late_minutes']} min")

                    print("=========================\n")
                except Exception as e:
                    print(f"[REPORT] Failed: {e}")
            elif key == ord('u'):
                cv2.destroyWindow("Security & Attendance Feed")
                obj_name = input("\n[CUSTOM-OBJ] Enter object name (blank = cancel): ").strip()

                if not obj_name:
                    print("[CUSTOM-OBJ] Cancelled.")
                elif last_raw_frame is None:
                    print("[CUSTOM-OBJ] No frame available.")
                else:
                    try:
                        hands = vision.hand_detector.detect(last_raw_frame)
                        ok = vision.custom_objects.enroll_from_hand(last_raw_frame, hands, obj_name)

                        if ok:
                            voice(f"{obj_name} enrolled.", "INFO", dedup_key=f"custom_obj:{obj_name}")
                            db.log_audit("ENROLL_CUSTOM_OBJECT", obj_name, {"mode": "handheld"})
                        else:
                            voice("Object enrollment failed. Hold the object clearly in your hand.",
                                "WARN", dedup_key="custom_obj_failed")
                    except Exception as e:
                        print(f"[CUSTOM-OBJ] Failed: {e}")
            elif key == ord('d'):
                danger_cfg = CONFIG.setdefault("DANGER_DETECTION", {})
                danger_cfg["ENABLED"] = not danger_cfg.get("ENABLED", False)

                if danger_cfg["ENABLED"] and getattr(vision, "danger_detector", None) is None:
                    try:
                        vision.danger_detector = DangerDetector(CONFIG)
                    except Exception as e:
                        print(f"[DANGER] Failed to initialize: {e}")
                        danger_cfg["ENABLED"] = False

                if getattr(vision, "danger_detector", None) is not None:
                    vision.danger_detector.enabled = danger_cfg["ENABLED"]

                print(f"[DANGER] Detection = {danger_cfg['ENABLED']}")
                voice(f"Danger detection {'enabled' if danger_cfg['ENABLED'] else 'disabled'}.",
                    "INFO", dedup_key="danger_toggle")
            elif key == ord('g'):
                CONFIG["SHOW_AGE_GENDER"] = not CONFIG.get("SHOW_AGE_GENDER", False)
                print(f"[UI] Age/Gender = {CONFIG['SHOW_AGE_GENDER']}")
            elif key == ord('z'):
                CONFIG["SHOW_ZONES_GRID"] = not CONFIG.get("SHOW_ZONES_GRID", False)
                print(f"[UI] Zones/Grid = {CONFIG['SHOW_ZONES_GRID']}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"[FATAL] Unhandled error: {e}")
        traceback.print_exc()
    finally:
        print("[INFO] Shutting down...")

        report_path = None
        try:
            report_path = _export_shutdown_report(
                db=db,
                vision=vision,
                cam_id=cam_id,
                location=cam_location,
                started_at=start_time,
            )
        except Exception as e:
            print(f"[REPORT] Shutdown report failed: {e}")

        try:
            if report_path and CONFIG.get("REPORTING", {}).get("EMAIL_ON_SHUTDOWN", True):
                ok, msg = alert_mgr.send_report_email(report_path)
                print(f"[REPORT-email] {'sent' if ok else 'failed'}: {msg}")
        except Exception as e:
            print(f"[REPORT-email] Failed: {e}")

        try: assistant.cleanup()
        except Exception as e: print(f"[WARN] Assistant cleanup: {e}")
        try:
            if cap and cap.isOpened(): cap.release()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass
        try: vision.save_face_db()
        except Exception as e: print(f"[WARN] Face DB save: {e}")
        print("[INFO] Done.")

#Main loop
if __name__ == "__main__":
    main()