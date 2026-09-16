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
import textwrap
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from collections import OrderedDict, deque, defaultdict
from typing import Optional, Any, Dict, List, Tuple
from datetime import datetime as dt_datetime, timedelta, date as dt_date, timezone as dt_timezone
from zoneinfo import ZoneInfo
from runtime_performance import (
    AdaptiveInferenceScheduler,
    LatestFrameBuffer,
    LatestInferenceState,
    ModelCallProfiler,
    PerformanceProfiler,
)
from core.correlation import CorrelationCore
from core.entities import normalize_identity_state
from core.face_augmentation import generate_variants
from core.security import SecuritySignalEngine
from core.liveness import LivenessChallenge
from biometric_storage import BiometricStorageError, load_face_database, save_face_database
from deployment_security import (
    CameraHealthMonitor,
    DeploymentSecurityError,
    append_runtime_health_event,
    is_strict_mode,
    process_identity,
    validate_alert_config,
    validate_camera_source,
    validate_edge_configuration,
    validate_webhook_url,
    verify_face_model_cache,
    watchdog_status,
)

#Optional try and error
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[WARN] mediapipe not installed - pose/hand/face-mesh disabled.")

try:
    from insightface.app import FaceAnalysis
    from insightface.app.common import Face as InsightFace
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


def _env_int_list(name, default):
    values = []
    for item in os.environ.get(name, "").split(","):
        try:
            number = int(item.strip())
        except (TypeError, ValueError):
            continue
        if 0 <= number <= 6:
            values.append(number)
    return values or list(default)


def _env_string_list(name):
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]

CONFIG: Dict[str, Any] = {
    #Paths
    "BASE_DIR": _BASE_DIR,
    "DATABASE_FILE": os.path.join(_BASE_DIR, "security.db"),
    "FACE_DB_FILE": os.path.join(_BASE_DIR, "data", "face_db.pkl"),
    "SNAPSHOT_DIR": os.path.join(_BASE_DIR, "snapshots"),
    "EXPORT_DIR": os.path.join(_BASE_DIR, "exports"),
    "MODEL_PATH": "yolov8n.pt",         
    "MODEL_MANIFEST_PATH": os.environ.get(
        "OPTIVOX_MODEL_MANIFEST", os.path.join(_BASE_DIR, "models", "model_checksums.json")),
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
    "CAPTURE_WIDTH": 1280,
    "CAPTURE_HEIGHT": 720,
    "DISPLAY_WIDTH": 1280,
    "DISPLAY_HEIGHT": 720,
    "PROCESSING_WIDTH": 1280,
    "PROCESSING_HEIGHT": 720,
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
    "FACE_RECOG_ACCEPT_THRESHOLD": 0.62,
    "FACE_RECOG_MIN_MARGIN": 0.08,
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
    "MULTI_EMBEDDING_POOLING": "mean",  
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
    "LOITERING_DURATION_SEC": 15.0,
    "RUNNING_SPEED_THRESHOLD": 280.0,
    "RUNNING_MIN_SEC": 0.6,
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
        "EVAC_CONFIRM_SECONDS": 1.0,
        "EVAC_COOLDOWN_SECONDS": 30.0,
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
    "REQUIRED_CHECKS_PASSED": 2,
    "WARMUP_FRAMES": 12,
    "SUSPECT_CONFIRM_FRAMES": 4,
    "SUSPECT_WINDOW_FRAMES": 8,
    "UNCERTAIN_BLOCKS_ATTENDANCE": True,
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

    # Deterministic security policy. Model-dependent capabilities remain
    # disabled until a compatible detector is explicitly configured.
    "SECURITY": {
        "ENABLED": True,
        "ZONES": [],
        "RUNNING": {
            "SPEED_THRESHOLD_PX_SEC": 280.0,
            "CONFIRM_SECONDS": 0.6,
        },
        "EVACUATION": {
            "MIN_PEOPLE": 4,
            "AVERAGE_SPEED_THRESHOLD_PX_SEC": 25.0,
            "CONFIRM_SECONDS": 1.0,
            "COOLDOWN_SECONDS": 30.0,
        },
        "PPE": {
            "ENABLED": False,
            "AVAILABLE": False,
            "REQUIRED_OBJECTS": {"helmet", "mask", "vest"},
            "MIN_CONFIDENCE": 0.55,
            "MIN_OVERLAP": 0.15,
        },
    },

    "PERFORMANCE": {
    "YOLO_EVERY_N_FRAMES": 8,
    "FACE_DETECT_EVERY_N_FRAMES": 4,
    "FACE_RECOG_EVERY_N_FRAMES": 8,
    "HAND_EVERY_N_FRAMES": 4,
    "POSE_EVERY_N_FRAMES": 4,
    "RECOGNITION_CACHE_TTL_SEC": 4.0,
    "ENABLE_PROFILING": True,
    "PROFILER_WINDOW_FRAMES": 120,
    "MAX_INFERENCE_FRAME_AGE_MS": 250,
    "CAPTURE_BUFFER_SIZE": 1,
    "SIDE_EFFECT_QUEUE_SIZE": 256,
    "CRITICAL_QUEUE_SIZE": 64,
    "DISPLAY_LOOP_FPS": 30,
    "ADAPTIVE_SCHEDULING": True,
    "ADAPTIVE_TARGET_TOTAL_MS": 120.0,
    "FACE_DETECT_ACTIVE_EVERY_N_FRAMES": 3,
    "FACE_DETECT_IDLE_EVERY_N_FRAMES": 8,
    "YOLO_ACTIVE_EVERY_N_FRAMES": 8,
    "YOLO_IDLE_EVERY_N_FRAMES": 12,
    "HAND_ACTIVE_EVERY_N_FRAMES": 4,
    "HAND_IDLE_EVERY_N_FRAMES": 8,
    "POSE_ACTIVE_EVERY_N_FRAMES": 4,
    "POSE_IDLE_EVERY_N_FRAMES": 8,
    "DANGER_ACTIVE_EVERY_N_FRAMES": 8,
    "DANGER_IDLE_EVERY_N_FRAMES": 12,
    "CUSTOM_ACTIVE_EVERY_N_FRAMES": 5,
    "CUSTOM_IDLE_EVERY_N_FRAMES": 8,
    "FACE_QUALITY_GATE_ENABLED": True,
    "FACE_QUALITY_MIN_SCORE": 50.0,
    "FACE_QUALITY_MIN_BLUR": 45.0,
        "FACE_RECOG_STABLE_FRAMES": 3,
        "IDENTITY_CONFIRMATION_OBSERVATIONS": 3,
    "FACE_RECOG_REFRESH_SEC": 2.5,
    "FACE_RECOG_REFRESH_MOVEMENT_PX": 36.0,
    "ROI_INFERENCE_ENABLED": True,
    "ROI_PADDING_RATIO": 0.12,
    "ROI_MAX_FRAME_RATIO": 0.88,
    "RECOGNITION_UNRESOLVED_EVERY_SEC": 0.20,
    "RECOGNITION_CANDIDATE_EVERY_SEC": 0.30,
        "MAX_ATTENDANCE_IDENTITY_AGE_SEC": 2.5,
        "RECOGNITION_CONTRADICTION_CONFIRMATIONS": 2,
        "RECOGNITION_CACHE_MIN_SIMILARITY": 0.45,
        "TRACK_SWITCH_RESET_PX": 110.0,
        "AUTO_AUGMENT_SPARSE_DATASET": True,
        "AUGMENT_WHEN_SOURCE_COUNT_BELOW": 3,
        "AUGMENTATIONS_PER_SOURCE_IMAGE": 3,
        "MAX_AUGMENTED_EMBEDDINGS_PER_PERSON": 8,
        "AUGMENT_MIN_SOURCE_SIMILARITY": 0.86,
        "AUGMENTATION_VERSION": 1,
    },

    # Correlation Core keeps model outputs fresh, attributable, and bounded
    # without changing the existing recognition or attendance policies.
    "CORRELATION_CORE": {
        "ENABLED": True,
        "MAX_ENTITIES": 128,
        "MAX_OBSERVATIONS_PER_TYPE": 12,
        "ENTITY_CLOSE_AFTER_SEC": 10.0,
        "TTL_MS": {
            "PERSON_DETECTED": 350,
            "FACE_DETECTED": 350,
            "FACE_QUALITY": 750,
            "FACE_HEAD_POSE": 500,
            "FACE_IDENTITY_RESULT": 3000,
            "LIVENESS_RESULT": 1500,
            "POSE_STATE": 350,
            "OBJECT_DETECTED": 750,
            "SECURITY_SIGNAL": 1500,
            "TRACK_MOTION": 500,
        },
        "FACE_VISIBLE_AFTER_SEC": 1.0,
        "QUALITY_VALID_AFTER_SEC": 1.5,
        "LIVENESS_VALID_AFTER_SEC": 2.0,
    },

    #Attendance
    "ATTENDANCE": {
        "ENABLED": True,
        "MIN_RECOGNITION_FRAMES": 8,       
        "RECOGNITION_WINDOW_SEC": 5.0,
        "CENTER_MODE_DWELL_SEC": 3.0,
        "CENTER_MODE_STABLE_FRAMES": 6,
        "CENTER_MODE_REQUIRE_LIVENESS": True,
        "CENTER_MODE_YAW_THRESHOLD": 0.12,
        "CENTER_MODE_NEUTRAL_THRESHOLD": 0.07,
        "CENTER_MODE_TURN_HOLD_SEC": 0.35,
        "CENTER_MODE_CHALLENGE_TIMEOUT_SEC": 15.0,
        "CENTER_MODE_X_MIN": 0.34,
        "CENTER_MODE_X_MAX": 0.66,
        "CENTER_MODE_Y_MIN": 0.12,
        "CENTER_MODE_Y_MAX": 0.82,
        "DAILY_CLOCKOUT_TIMEOUT_MIN": 720,  
        "AUTO_CLOCKOUT_TIMEOUT_MIN": 15,
        "AUTO_CLOCKOUT_ON_SHUTDOWN": True,
        "PRESENCE_SESSION_CLOSE_AFTER_SEC": 5.0,
        "SCHOOL_DAYS": _env_int_list("OPTIVOX_SCHOOL_DAYS", [0, 1, 2, 3, 4]),
        "HOLIDAYS": _env_string_list("OPTIVOX_SCHOOL_HOLIDAYS"),
        "RECOGNITION_EVIDENCE_MIN_INTERVAL_SEC": 0.75,
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
    # Compute switches are separate from the display-only overlay switches.
    "ENABLE_OBJECT_DETECTION": True,
    "ENABLE_DANGER_INFERENCE": True,
    "ENABLE_POSE_INFERENCE": True,
    "ENABLE_HAND_INFERENCE": True,
    "ENABLE_CUSTOM_OBJECT_INFERENCE": True,
    "ENABLE_LIVENESS_INFERENCE": True,
    "ENABLE_AGE_GENDER_INFERENCE": False,

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
_APP_TIMEZONE = ZoneInfo(os.environ.get("OPTIVOX_TIMEZONE", "Asia/Jakarta"))


def _utc_datetime() -> dt_datetime:
    return dt_datetime.now(dt_timezone.utc)


def _local_datetime() -> dt_datetime:
    return dt_datetime.now(_APP_TIMEZONE)


def _utc_now() -> str:
    return _utc_datetime().isoformat(timespec="milliseconds")

def _today_iso() -> str:
    return _local_datetime().date().isoformat()


def _is_school_day(local_now=None) -> bool:
    """Return whether automatic attendance is allowed on this local date."""
    local_now = local_now or _local_datetime()
    attendance_cfg = CONFIG.get("ATTENDANCE", {})
    raw_days = attendance_cfg.get("SCHOOL_DAYS", range(5))
    school_days = set()
    for day in raw_days:
        try:
            day = int(day)
        except (TypeError, ValueError):
            continue
        if 0 <= day <= 6:
            school_days.add(day)
    holidays = {
        str(day).strip() for day in attendance_cfg.get("HOLIDAYS", [])
        if str(day).strip()
    }
    return local_now.weekday() in school_days and local_now.date().isoformat() not in holidays


def _parse_timestamp(value: str) -> dt_datetime:
    parsed = dt_datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt_timezone.utc)
    return parsed

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
            self.add_column_if_missing("events", "entity_id", "entity_id TEXT")
            self.add_column_if_missing("events", "presence_session_id", "presence_session_id INTEGER")
            self.add_column_if_missing("events", "source_frame_id", "source_frame_id INTEGER")
            self.add_column_if_missing("events", "observation_type", "observation_type TEXT")
            self.add_column_if_missing("events", "evidence_path", "evidence_path TEXT")
            self.add_column_if_missing("events", "review_status", "review_status TEXT DEFAULT 'open'")
            self.add_column_if_missing("events", "review_note", "review_note TEXT")
            self.add_column_if_missing("events", "reviewed_at", "reviewed_at TEXT")
            self.add_column_if_missing("events", "reviewed_by", "reviewed_by TEXT")
            self.add_column_if_missing("alert_log", "source_event_id", "source_event_id INTEGER")

            self.add_column_if_missing("attendance", "work_minutes", "work_minutes INTEGER DEFAULT 0")
            self.add_column_if_missing("attendance", "late_minutes", "late_minutes INTEGER DEFAULT 0")
            self.add_column_if_missing("attendance", "camera_id", "camera_id TEXT")
            self.add_column_if_missing("attendance", "location", "location TEXT")
            self.add_column_if_missing("attendance", "notes", "notes TEXT")
            self.add_column_if_missing("attendance", "presence_session_id", "presence_session_id INTEGER")
            self.add_column_if_missing("attendance", "recognition_evidence_id", "recognition_evidence_id INTEGER")
            self.add_column_if_missing("attendance", "decision_source", "decision_source TEXT DEFAULT 'automatic'")
            self.add_column_if_missing("attendance", "identity_state", "identity_state TEXT")
            self.add_column_if_missing("attendance", "liveness_status", "liveness_status TEXT")
            self.add_column_if_missing("attendance", "source_frame_id", "source_frame_id INTEGER")
            self.add_column_if_missing("attendance", "recognition_confidence", "recognition_confidence REAL")
            self.add_column_if_missing("attendance", "evidence_path", "evidence_path TEXT")
            self.add_column_if_missing("attendance", "attendance_status", "attendance_status TEXT DEFAULT 'recorded'")
            self.add_column_if_missing("attendance", "absence_type", "absence_type TEXT")
            self.add_column_if_missing("attendance", "is_official", "is_official INTEGER DEFAULT 1")
            self.add_column_if_missing("attendance", "subject", "subject TEXT")
            self.add_column_if_missing("attendance", "schedule_key", "schedule_key TEXT")
            self.add_column_if_missing("attendance", "expected_start", "expected_start TEXT")
            self.add_column_if_missing("attendance", "expected_end", "expected_end TEXT")
            self.add_column_if_missing("attendance", "early_departure_minutes", "early_departure_minutes INTEGER DEFAULT 0")
            self.add_column_if_missing("attendance", "last_seen_at", "last_seen_at TEXT")
            self.add_column_if_missing("attendance", "clock_out_source", "clock_out_source TEXT")
            self.add_column_if_missing("presence_sessions", "track_generation", "track_generation INTEGER DEFAULT 1")
            self.add_column_if_missing("presence_sessions", "closed_reason", "closed_reason TEXT")

            now = _utc_now()
            if "created_at" in self.table_columns("people"):
                self.conn.execute("UPDATE people SET created_at=? WHERE created_at IS NULL", (now,))
            if "updated_at" in self.table_columns("people"):
                self.conn.execute("UPDATE people SET updated_at=? WHERE updated_at IS NULL", (now,))
            self.conn.execute("UPDATE events SET review_status='open' WHERE review_status IS NULL")
            self.conn.execute("UPDATE attendance SET decision_source='automatic' WHERE decision_source IS NULL")
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS enrollment_operations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    person_name TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sample_count INTEGER DEFAULT 0,
                    quality_json TEXT,
                    provenance_json TEXT,
                    actor_id TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE SET NULL
                );
                CREATE TABLE IF NOT EXISTS attendance_decisions (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_key            TEXT NOT NULL UNIQUE,
                    person_id               INTEGER,
                    entity_id               TEXT,
                    presence_session_id    INTEGER,
                    recognition_evidence_id INTEGER,
                    decision                TEXT NOT NULL,
                    reason                  TEXT,
                    identity_state          TEXT,
                    liveness_status         TEXT,
                    quality_score           REAL,
                    recognition_confidence REAL,
                    source_frame_id        INTEGER,
                    observed_at            TEXT NOT NULL,
                    details_json           TEXT,
                    created_at             TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE SET NULL,
                    FOREIGN KEY(presence_session_id) REFERENCES presence_sessions(id) ON DELETE SET NULL,
                    FOREIGN KEY(recognition_evidence_id) REFERENCES recognition_evidence(id) ON DELETE SET NULL
                );
                CREATE TABLE IF NOT EXISTS absence_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    absence_date TEXT NOT NULL,
                    subject TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'inferred',
                    reason TEXT,
                    source TEXT NOT NULL DEFAULT 'system',
                    is_official INTEGER NOT NULL DEFAULT 0,
                    actor_id TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(person_id, absence_date, subject),
                    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS attendance_schedules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    class_name TEXT NOT NULL,
                    subject TEXT,
                    weekday INTEGER NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    grace_minutes INTEGER NOT NULL DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(class_name, subject, weekday, start_time)
                );
            """)
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
                    attendance_status TEXT DEFAULT 'recorded',
                    absence_type    TEXT,
                    is_official     INTEGER DEFAULT 1,
                    subject         TEXT,
                    schedule_key    TEXT,
                    expected_start TEXT,
                    expected_end   TEXT,
                    early_departure_minutes INTEGER DEFAULT 0,
                    last_seen_at    TEXT,
                    clock_out_source TEXT,
                    UNIQUE(person_id, date),
                    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS absence_records (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id       INTEGER NOT NULL,
                    absence_date    TEXT NOT NULL,
                    subject         TEXT NOT NULL DEFAULT '',
                    status          TEXT NOT NULL DEFAULT 'inferred',
                    reason          TEXT,
                    source          TEXT NOT NULL DEFAULT 'system',
                    is_official     INTEGER NOT NULL DEFAULT 0,
                    actor_id        TEXT,
                    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(person_id, absence_date, subject),
                    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS attendance_schedules (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    class_name      TEXT NOT NULL,
                    subject         TEXT,
                    weekday         INTEGER NOT NULL,
                    start_time      TEXT NOT NULL,
                    end_time        TEXT,
                    grace_minutes   INTEGER NOT NULL DEFAULT 0,
                    active          INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(class_name, subject, weekday, start_time)
                );
                CREATE TABLE IF NOT EXISTS presence_sessions (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id           TEXT NOT NULL,
                    track_id            INTEGER,
                    track_generation    INTEGER DEFAULT 1,
                    person_id           INTEGER,
                    label               TEXT NOT NULL DEFAULT 'UNKNOWN',
                    identity_state      TEXT NOT NULL DEFAULT 'UNRESOLVED',
                    liveness_status     TEXT,
                    camera_id           TEXT NOT NULL DEFAULT 'cam_0',
                    started_at          TEXT NOT NULL,
                    last_seen_at        TEXT NOT NULL,
                    ended_at            TEXT,
                    status              TEXT NOT NULL DEFAULT 'active',
                    confidence          REAL DEFAULT 0.0,
                    first_frame_id      INTEGER,
                    last_frame_id       INTEGER,
                    closed_reason       TEXT,
                    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE SET NULL
                );
                CREATE TABLE IF NOT EXISTS recognition_evidence (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id           TEXT NOT NULL,
                    presence_session_id INTEGER,
                    track_id            INTEGER,
                    person_id           INTEGER,
                    candidate_name      TEXT NOT NULL DEFAULT 'UNKNOWN',
                    decision            TEXT NOT NULL,
                    similarity          REAL DEFAULT 0.0,
                    quality_score       REAL DEFAULT 0.0,
                    quality_ok           INTEGER DEFAULT 0,
                    liveness_status     TEXT,
                    identity_state      TEXT,
                    reason              TEXT,
                    source_frame_id     INTEGER,
                    observed_at         TEXT NOT NULL,
                    details_json        TEXT,
                    FOREIGN KEY(presence_session_id) REFERENCES presence_sessions(id) ON DELETE SET NULL,
                    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE SET NULL
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
                    timestamp       TEXT NOT NULL,
                    source_event_id INTEGER
                );
                CREATE TABLE IF NOT EXISTS incident_evidence (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id     INTEGER NOT NULL,
                    event_id        INTEGER,
                    path            TEXT NOT NULL,
                    evidence_type   TEXT NOT NULL DEFAULT 'snapshot',
                    source_frame_id INTEGER,
                    captured_at     TEXT NOT NULL,
                    checksum        TEXT,
                    status          TEXT NOT NULL DEFAULT 'available',
                    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(incident_id, event_id, path)
                );
                CREATE TABLE IF NOT EXISTS enrollment_operations (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id       INTEGER,
                    person_name     TEXT NOT NULL,
                    operation       TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    sample_count    INTEGER DEFAULT 0,
                    quality_json    TEXT,
                    provenance_json TEXT,
                    actor_id        TEXT,
                    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE SET NULL
                );
                CREATE TABLE IF NOT EXISTS attendance_decisions (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_key            TEXT NOT NULL UNIQUE,
                    person_id               INTEGER,
                    entity_id               TEXT,
                    presence_session_id    INTEGER,
                    recognition_evidence_id INTEGER,
                    decision                TEXT NOT NULL,
                    reason                  TEXT,
                    identity_state          TEXT,
                    liveness_status         TEXT,
                    quality_score           REAL,
                    recognition_confidence REAL,
                    source_frame_id        INTEGER,
                    observed_at            TEXT NOT NULL,
                    details_json           TEXT,
                    created_at             TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE SET NULL,
                    FOREIGN KEY(presence_session_id) REFERENCES presence_sessions(id) ON DELETE SET NULL,
                    FOREIGN KEY(recognition_evidence_id) REFERENCES recognition_evidence(id) ON DELETE SET NULL
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
                CREATE INDEX IF NOT EXISTS idx_attendance_decisions_entity
                    ON attendance_decisions(entity_id, observed_at);
                CREATE INDEX IF NOT EXISTS idx_attendance_decisions_person
                    ON attendance_decisions(person_id, observed_at);
            """)
            self.conn.commit()
            DatabaseMigrationManager(self.conn, self.lock).run()
            self.conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_events_entity
                    ON events(entity_id);
                CREATE INDEX IF NOT EXISTS idx_events_session
                    ON events(presence_session_id);
                CREATE INDEX IF NOT EXISTS idx_presence_entity
                    ON presence_sessions(entity_id, status);
                CREATE INDEX IF NOT EXISTS idx_presence_person
                    ON presence_sessions(person_id, last_seen_at);
                CREATE INDEX IF NOT EXISTS idx_evidence_entity
                    ON recognition_evidence(entity_id, observed_at);
                CREATE INDEX IF NOT EXISTS idx_evidence_person
                    ON recognition_evidence(person_id, observed_at);
            """)
            self.conn.commit()
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
        row = self._fetchone("SELECT id FROM people WHERE lower(name)=lower(?)", (name,))
        return row["id"] if row else None

    def get_active_person_id(self, name: str):
        row = self._fetchone(
            "SELECT id, metadata_json FROM people WHERE lower(name)=lower(?)",
            (name,),
        )
        if not row:
            return None
        metadata = _safe_json_parse(row["metadata_json"])
        if isinstance(metadata, dict) and metadata.get("active") is False:
            return None
        return row["id"]

    def set_person_active(self, person_id: int, active: bool) -> bool:
        """Toggle roster availability without deleting historical records."""
        with self.lock:
            row = self.conn.execute(
                "SELECT metadata_json FROM people WHERE id=?", (int(person_id),)
            ).fetchone()
            if not row:
                return False
            metadata = _safe_json_parse(row["metadata_json"])
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["active"] = bool(active)
            self.conn.execute(
                "UPDATE people SET metadata_json=?, updated_at=? WHERE id=?",
                (json.dumps(metadata), _utc_now(), int(person_id)),
            )
            self.conn.commit()
            return bool(active)

    def delete_person_records(self, person_id: int) -> bool:
        """Permanently remove a roster row after an explicit operator action."""
        with self.lock:
            cur = self.conn.execute("DELETE FROM people WHERE id=?", (int(person_id),))
            self.conn.commit()
            return cur.rowcount > 0

    def merge_person_records(self, source_id: int, target_id: int) -> bool:
        """Move operational history to a target identity after explicit confirmation."""
        source_id, target_id = int(source_id), int(target_id)
        if source_id == target_id:
            return False
        with self.lock:
            source = self.conn.execute("SELECT id FROM people WHERE id=?", (source_id,)).fetchone()
            target = self.conn.execute("SELECT id FROM people WHERE id=?", (target_id,)).fetchone()
            if not source or not target:
                return False
            attendance_collisions = self.conn.execute(
                "SELECT a.id FROM attendance a JOIN attendance t "
                "ON t.person_id=? AND t.date=a.date WHERE a.person_id=?",
                (target_id, source_id),
            ).fetchall()
            for row in attendance_collisions:
                self.conn.execute("DELETE FROM attendance WHERE id=?", (row["id"],))
            absence_collisions = self.conn.execute(
                "SELECT a.id FROM absence_records a JOIN absence_records t "
                "ON t.person_id=? AND t.absence_date=a.absence_date AND t.subject=a.subject "
                "WHERE a.person_id=?",
                (target_id, source_id),
            ).fetchall()
            for row in absence_collisions:
                self.conn.execute("DELETE FROM absence_records WHERE id=?", (row["id"],))
            for table in ("attendance", "absence_records", "presence_sessions", "recognition_evidence", "events", "behavior_profiles"):
                self.conn.execute(f"UPDATE {table} SET person_id=? WHERE person_id=?", (target_id, source_id))
            self.conn.execute("DELETE FROM people WHERE id=?", (source_id,))
            self.conn.commit()
            return True

    def get_known_face_names(self, limit=1000, offset=0):
        return self._fetchall(
            "SELECT id, name, role, thumbnail_path, created_at FROM people "
            "ORDER BY name LIMIT ? OFFSET ?", (limit, offset))

    def get_face_count(self) -> int:
        row = self._fetchone("SELECT COUNT(*) AS c FROM people")
        return row["c"] if row else 0

    #Events
    def log_event(self, event_type, person_id=None, confidence=None, details=None,
                  snapshot_path=None, camera_id="cam_0", location=None, severity=0,
                  entity_id=None, presence_session_id=None, source_frame_id=None,
                  observation_type=None, evidence_path=None):
        details_json = json.dumps(details) if isinstance(details, dict) else json.dumps(
            {"message": str(details)} if details else {})
        with self.lock:
            cursor = self.conn.execute("""
                INSERT INTO events (person_id, event_type, confidence, details_json,
                    snapshot_path, camera_id, location, severity, timestamp,
                    entity_id, presence_session_id, source_frame_id,
                    observation_type, evidence_path, review_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
            """, (person_id, event_type, confidence, details_json,
                  snapshot_path, camera_id, location, severity, _utc_now(),
                  entity_id, presence_session_id, source_frame_id,
                  observation_type or event_type, evidence_path))
            self.conn.execute("""
                INSERT INTO event_stats_daily (date, event_type, count) VALUES (?, ?, 1)
                ON CONFLICT(date, event_type) DO UPDATE SET count = count + 1
            """, (_today_iso(), event_type))
            self.conn.execute("""
                INSERT INTO event_stats_hourly (hour, event_type, count) VALUES (?, ?, 1)
                ON CONFLICT(hour, event_type) DO UPDATE SET count = count + 1
            """, (_utc_datetime().strftime("%Y-%m-%d %H:00"), event_type))
            self.conn.commit()
            return int(cursor.lastrowid)

    def get_recent_events(self, limit=100):
        return self._fetchall("""
            SELECT e.*, p.name AS person_name FROM events e
            LEFT JOIN people p ON e.person_id = p.id
            ORDER BY e.timestamp DESC LIMIT ?""", (limit,))

    def get_event_summary(self, days=7):
        since = (_utc_datetime() - timedelta(days=days)).date().isoformat()
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

    # Correlated operational records. These are deliberately separate from
    # frame-level events so presence, identity evidence, and attendance can
    # be audited independently.
    def upsert_presence_session(self, entity_id, track_id=None, person_id=None,
                                 label="UNKNOWN", identity_state="UNRESOLVED",
                                 liveness_status=None, camera_id="cam_0",
                                 confidence=0.0, source_frame_id=None,
                                 observed_at=None, visible=True, track_generation=1):
        if not entity_id:
            return None
        observed_at = observed_at or _utc_now()
        with self.lock:
            row = self.conn.execute(
                "SELECT id, person_id, label FROM presence_sessions WHERE entity_id=? AND status IN ('active', 'occluded') "
                "ORDER BY id DESC LIMIT 1", (str(entity_id),)).fetchone()
            if row:
                session_id = row["id"]
                existing_person_id = row["person_id"]
                effective_person_id = existing_person_id or person_id
                effective_state = str(identity_state or "UNRESOLVED")
                effective_label = row["label"] if existing_person_id else str(label or "UNKNOWN")
                if (existing_person_id and person_id
                        and int(existing_person_id) != int(person_id)):
                    # A tracker or recognizer disagreement must be visible and
                    # must never silently reassign one presence session.
                    effective_state = "CONTRADICTED"
                if visible:
                    self.conn.execute(
                        """UPDATE presence_sessions
                            SET track_id=?, track_generation=?, person_id=?, label=?, identity_state=?,
                               liveness_status=?, camera_id=?, last_seen_at=?,
                               status='active', confidence=?, last_frame_id=?
                           WHERE id=?""",
                         (track_id, int(track_generation or 1), effective_person_id, effective_label,
                         effective_state, liveness_status,
                         camera_id, observed_at or _utc_now(), float(confidence or 0.0),
                         source_frame_id, session_id),
                    )
                else:
                    self.conn.execute(
                        """UPDATE presence_sessions
                            SET track_id=?, track_generation=?, person_id=?, label=?, identity_state=?,
                               liveness_status=?, camera_id=?, status='occluded',
                               confidence=?, last_frame_id=?
                           WHERE id=?""",
                         (track_id, int(track_generation or 1), effective_person_id, effective_label,
                         effective_state, liveness_status, camera_id,
                         float(confidence or 0.0), source_frame_id, session_id),
                    )
            else:
                cur = self.conn.execute(
                    """INSERT INTO presence_sessions
                        (entity_id, track_id, track_generation, person_id, label, identity_state,
                        liveness_status, camera_id, started_at, last_seen_at,
                        status, confidence, first_frame_id, last_frame_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)""",
                     (str(entity_id), track_id, int(track_generation or 1), person_id, str(label or "UNKNOWN"),
                     str(identity_state or "UNRESOLVED"), liveness_status,
                     camera_id, observed_at, observed_at, float(confidence or 0.0),
                     source_frame_id, source_frame_id),
                )
                session_id = cur.lastrowid
            self.conn.commit()
            return session_id

    def log_enrollment_operation(self, person_name: str, operation: str, status: str,
                                 sample_count: int = 0, quality=None, provenance=None,
                                 actor_id: str = "system") -> int:
        with self.lock:
            cur = self.conn.execute(
                """INSERT INTO enrollment_operations
                   (person_id, person_name, operation, status, sample_count,
                    quality_json, provenance_json, actor_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.get_person_id(person_name), str(person_name), str(operation), str(status),
                 int(sample_count or 0), json.dumps(quality or {}, default=str),
                 json.dumps(provenance or {}, default=str), actor_id),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def close_stale_presence_sessions(self, timeout_sec=5.0, now=None):
        now = _utc_datetime() if now is None else now
        closed = 0
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, last_seen_at FROM presence_sessions WHERE status IN ('active', 'occluded')"
            ).fetchall()
            for row in rows:
                try:
                    age = (now - _parse_timestamp(row["last_seen_at"])).total_seconds()
                except Exception:
                    age = timeout_sec + 1
                if age > float(timeout_sec):
                    self.conn.execute(
                        "UPDATE presence_sessions SET status='closed', ended_at=?, closed_reason=? WHERE id=?",
                        (row["last_seen_at"] or _utc_now(), "timeout", row["id"]),
                    )
                    closed += 1
            if closed:
                self.conn.commit()
        return closed

    def close_open_presence_sessions(self, reason="runtime_shutdown") -> int:
        """Close sessions left open by a stopped or restarted edge runtime."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, last_seen_at FROM presence_sessions WHERE status IN ('active', 'occluded')"
            ).fetchall()
            for row in rows:
                self.conn.execute(
                    """UPDATE presence_sessions
                       SET status='closed', ended_at=?, closed_reason=?
                       WHERE id=? AND status IN ('active', 'occluded')""",
                    (row["last_seen_at"] or _utc_now(), str(reason), row["id"]),
                )
            if rows:
                self.conn.commit()
            return len(rows)

    def record_recognition_evidence(
        self, entity_id, candidate_name="UNKNOWN", decision="unresolved",
        similarity=0.0, quality_score=0.0, quality_ok=False,
        liveness_status=None, identity_state=None, reason=None,
        source_frame_id=None, observed_at=None, track_id=None,
        presence_session_id=None, person_id=None, details=None,
    ):
        if not entity_id:
            return None
        with self.lock:
            cur = self.conn.execute(
                """INSERT INTO recognition_evidence
                   (entity_id, presence_session_id, track_id, person_id,
                    candidate_name, decision, similarity, quality_score,
                    quality_ok, liveness_status, identity_state, reason,
                    source_frame_id, observed_at, details_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(entity_id), presence_session_id, track_id, person_id,
                 str(candidate_name or "UNKNOWN"), str(decision or "unresolved"),
                 float(similarity or 0.0), float(quality_score or 0.0),
                 1 if quality_ok else 0, liveness_status, identity_state,
                 reason, source_frame_id, observed_at or _utc_now(),
                 json.dumps(details or {}, default=str)),
            )
            self.conn.commit()
            return cur.lastrowid

    def record_attendance_decision(
        self, decision_key, decision, reason=None, person_id=None,
        entity_id=None, presence_session_id=None, recognition_evidence_id=None,
        identity_state=None, liveness_status=None, quality_score=None,
        recognition_confidence=None, source_frame_id=None, observed_at=None,
        details=None,
    ):
        """Persist one idempotent attendance eligibility decision.

        Rejected decisions are retained as well as accepted ones. This keeps
        the automatic gate explainable without turning frame detections into
        attendance records.
        """
        if not decision_key:
            return None
        observed_at = observed_at or _utc_now()
        with self.lock:
            self.conn.execute(
                """INSERT OR IGNORE INTO attendance_decisions
                   (decision_key, person_id, entity_id, presence_session_id,
                    recognition_evidence_id, decision, reason, identity_state,
                    liveness_status, quality_score, recognition_confidence,
                    source_frame_id, observed_at, details_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(decision_key), person_id, entity_id, presence_session_id,
                 recognition_evidence_id, str(decision or "rejected"),
                 str(reason or ""), identity_state, liveness_status,
                 quality_score, recognition_confidence, source_frame_id,
                 observed_at, json.dumps(details or {}, default=str)),
            )
            row = self.conn.execute(
                "SELECT id FROM attendance_decisions WHERE decision_key=?",
                (str(decision_key),),
            ).fetchone()
            self.conn.commit()
            return int(row["id"]) if row else None

    def _schedule_for_person(self, person_id: int, local_now: dt_datetime) -> Optional[dict]:
        """Resolve an optional class schedule, falling back to global policy."""
        row = self._fetchone("SELECT metadata_json FROM people WHERE id=?", (person_id,))
        metadata = _safe_json_parse(row["metadata_json"] if row else None)
        class_name = metadata.get("class") or metadata.get("class_name") if isinstance(metadata, dict) else None
        if not class_name:
            return None
        schedule = self._fetchone(
            """SELECT * FROM attendance_schedules
               WHERE class_name=? AND weekday=? AND active=1
               ORDER BY CASE WHEN subject IS NULL OR subject='' THEN 0 ELSE 1 END, id
               LIMIT 1""",
            (str(class_name), local_now.weekday()),
        )
        if not schedule:
            return None
        try:
            start_clock = dt_datetime.strptime(str(schedule["start_time"]), "%H:%M").time()
            scheduled = local_now.replace(hour=start_clock.hour, minute=start_clock.minute, second=0, microsecond=0)
        except (TypeError, ValueError):
            return None
        return {
            "schedule_key": f"{class_name}:{local_now.weekday()}:{schedule['start_time']}",
            "subject": schedule["subject"],
            "expected_start": str(schedule["start_time"]),
            "expected_end": schedule["end_time"],
            "scheduled": scheduled,
            "grace_minutes": int(schedule["grace_minutes"] or 0),
        }

    def search_events(self, query, days=7, limit=30):
        since = (_utc_datetime() - timedelta(days=days)).isoformat()
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
                            location: str = None, confidence: float = None,
                            presence_session_id=None, recognition_evidence_id=None,
                            source_frame_id=None, identity_state="CONFIRMED",
                            liveness_status="real", decision_source="automatic") -> dict:
        today = _today_iso()
        now = _utc_now()
        existing = self._fetchone(
            "SELECT * FROM attendance WHERE person_id=? AND date=?", (person_id, today))
        if existing and existing["clock_in"]:
            return {"already_clocked_in": True, "clock_in": existing["clock_in"]}
        now_local = _local_datetime()
        if str(decision_source or "").lower().startswith("automatic") and not _is_school_day(now_local):
            return {
                "not_school_day": True,
                "date": today,
                "reason": "Automatic attendance is disabled for this school day.",
            }
        schedule = self._schedule_for_person(person_id, now_local)
        if schedule:
            scheduled = schedule["scheduled"]
            grace = schedule["grace_minutes"]
        else:
            work_start = CONFIG["ATTENDANCE"]["WORK_START_HOUR"]
            grace = CONFIG["ATTENDANCE"]["LATE_GRACE_MIN"]
            scheduled = now_local.replace(hour=work_start, minute=0, second=0, microsecond=0)
        late = max(0, int((now_local - scheduled).total_seconds() / 60) - grace)
        with self.lock:
            self.conn.execute("""
                INSERT INTO attendance
                    (person_id, date, clock_in, late_minutes, camera_id, location,
                     presence_session_id, recognition_evidence_id, decision_source,
                     identity_state, liveness_status, source_frame_id,
                     recognition_confidence, attendance_status, is_official,
                     subject, schedule_key, expected_start, expected_end,
                     last_seen_at, clock_out_source)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'recorded', 1, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(person_id, date) DO UPDATE SET
                    clock_in      = COALESCE(attendance.clock_in, excluded.clock_in),
                    late_minutes  = excluded.late_minutes,
                    camera_id     = excluded.camera_id,
                    location      = excluded.location,
                    presence_session_id = COALESCE(attendance.presence_session_id, excluded.presence_session_id),
                    recognition_evidence_id = COALESCE(attendance.recognition_evidence_id, excluded.recognition_evidence_id),
                    decision_source = COALESCE(attendance.decision_source, excluded.decision_source),
                    identity_state = COALESCE(attendance.identity_state, excluded.identity_state),
                    liveness_status = COALESCE(attendance.liveness_status, excluded.liveness_status),
                    source_frame_id = COALESCE(attendance.source_frame_id, excluded.source_frame_id),
                    recognition_confidence = COALESCE(attendance.recognition_confidence, excluded.recognition_confidence),
                    subject = COALESCE(attendance.subject, excluded.subject),
                    schedule_key = COALESCE(attendance.schedule_key, excluded.schedule_key),
                    expected_start = COALESCE(attendance.expected_start, excluded.expected_start),
                    expected_end = COALESCE(attendance.expected_end, excluded.expected_end),
                    last_seen_at = excluded.last_seen_at
             """, (person_id, today, now, late, camera_id, location,
                  presence_session_id, recognition_evidence_id, decision_source,
                  identity_state, liveness_status, source_frame_id, confidence,
                   schedule.get("subject") if schedule else None,
                   schedule.get("schedule_key") if schedule else None,
                   schedule.get("expected_start") if schedule else None,
                   schedule.get("expected_end") if schedule else None,
                   now))
            self.conn.commit()
        return {"clocked_in_at": now, "late_minutes": late}

    def touch_attendance_presence(self, person_id: int, observed_at=None,
                                  presence_session_id=None) -> bool:
        """Refresh the last valid presence used by automatic clock-out."""
        observed_at = observed_at or _utc_now()
        with self.lock:
            cur = self.conn.execute(
                """UPDATE attendance SET last_seen_at=?,
                          presence_session_id=COALESCE(presence_session_id, ?)
                   WHERE person_id=? AND date=? AND clock_in IS NOT NULL AND clock_out IS NULL""",
                (observed_at, presence_session_id, person_id, _today_iso()),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def attendance_clock_out(self, person_id: int, clock_out_at=None,
                             source="manual", reason=None) -> dict:
        today = _today_iso()
        existing = self._fetchone(
            "SELECT * FROM attendance WHERE person_id=? AND date=?", (person_id, today))
        if not existing or not existing["clock_in"]:
            return {"error": "Not clocked in today"}
        now = clock_out_at or _utc_now()
        try:
            ci = _parse_timestamp(existing["clock_in"])
            co = _parse_timestamp(now)
            work_min = max(0, int((co - ci).total_seconds() / 60))
        except Exception:
            work_min = 0
        early_departure = 0
        try:
            if existing["expected_end"]:
                local_out = _parse_timestamp(now).astimezone(_APP_TIMEZONE)
                end_clock = dt_datetime.strptime(str(existing["expected_end"]), "%H:%M").time()
                expected_end = local_out.replace(
                    hour=end_clock.hour, minute=end_clock.minute, second=0, microsecond=0)
                early_departure = max(0, int((expected_end - local_out).total_seconds() / 60))
        except (TypeError, ValueError, KeyError):
            early_departure = 0
        status = "early_departure" if early_departure else "completed"
        note = str(reason or "").strip() or None
        with self.lock:
            self.conn.execute(
                """UPDATE attendance SET clock_out=?, work_minutes=?,
                          early_departure_minutes=?, attendance_status=?,
                          clock_out_source=?, notes=COALESCE(?, notes),
                          last_seen_at=COALESCE(last_seen_at, ?)
                   WHERE person_id=? AND date=? AND clock_out IS NULL""",
                (now, work_min, early_departure, status, source, note, now,
                 person_id, today),
            )
            self.conn.commit()
        return {"clocked_out_at": now, "work_minutes": work_min,
                "early_departure_minutes": early_departure,
                "status": status, "source": source}

    def close_stale_attendance(self, timeout_min=15, now=None) -> list[dict]:
        """Close today's open rows from their last valid observed presence."""
        now_dt = _utc_datetime() if now is None else now
        timeout_sec = max(1.0, float(timeout_min) * 60.0)
        rows = self._fetchall(
            """SELECT a.person_id, a.clock_in, a.last_seen_at,
                      a.presence_session_id, p.name,
                      ps.last_seen_at AS session_last_seen
               FROM attendance a JOIN people p ON p.id=a.person_id
               LEFT JOIN presence_sessions ps ON ps.id=a.presence_session_id
               WHERE a.date=? AND a.clock_in IS NOT NULL AND a.clock_out IS NULL""",
            (_today_iso(),),
        )
        closed = []
        for row in rows:
            last_seen = row["last_seen_at"] or row["session_last_seen"] or row["clock_in"]
            try:
                age = (now_dt - _parse_timestamp(last_seen).astimezone(dt_timezone.utc)).total_seconds()
            except Exception:
                age = timeout_sec + 1.0
            if age < timeout_sec:
                continue
            result = self.attendance_clock_out(
                int(row["person_id"]), clock_out_at=last_seen,
                source="automatic_timeout", reason="No valid presence detected.")
            if "clocked_out_at" in result:
                closed.append({"person_id": row["person_id"], "name": row["name"], **result})
        return closed

    def close_open_attendance_before(self, before_date=None) -> list[dict]:
        """Reconcile rows left open by a previous runtime/day."""
        before_date = before_date or _today_iso()
        rows = self._fetchall(
            """SELECT person_id, date, clock_in, last_seen_at
               FROM attendance WHERE date < ? AND clock_in IS NOT NULL AND clock_out IS NULL""",
            (before_date,),
        )
        closed = []
        with self.lock:
            for row in rows:
                end = row["last_seen_at"] or row["clock_in"]
                try:
                    work_min = max(0, int((_parse_timestamp(end) - _parse_timestamp(row["clock_in"])).total_seconds() / 60))
                except Exception:
                    work_min = 0
                self.conn.execute(
                    """UPDATE attendance SET clock_out=?, work_minutes=?,
                              attendance_status='completed', clock_out_source='day_rollover'
                       WHERE person_id=? AND date=? AND clock_out IS NULL""",
                    (end, work_min, row["person_id"], row["date"]),
                )
                closed.append({"person_id": row["person_id"], "date": row["date"], "clocked_out_at": end})
            if closed:
                self.conn.commit()
        return closed

    def close_open_attendance_for_shutdown(self) -> list[dict]:
        """Close current open rows when the edge agent is intentionally stopped."""
        rows = self._fetchall(
            """SELECT person_id, name, last_seen_at, clock_in
               FROM attendance a JOIN people p ON p.id=a.person_id
               WHERE a.date=? AND a.clock_in IS NOT NULL AND a.clock_out IS NULL""",
            (_today_iso(),),
        )
        closed = []
        for row in rows:
            end = row["last_seen_at"] or row["clock_in"] or _utc_now()
            result = self.attendance_clock_out(
                int(row["person_id"]), clock_out_at=end,
                source="runtime_shutdown", reason="Runtime stopped.")
            if "clocked_out_at" in result:
                closed.append({"person_id": row["person_id"], "name": row["name"], **result})
        return closed

    def attendance_report(self, days=7, person_id=None):
        since = (_local_datetime().date() - timedelta(days=days)).isoformat()
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

    def log_alert(self, channel, event_type, target=None, status="sent", error=None,
                  source_event_id=None):
        with self.lock:
            self.conn.execute("""INSERT INTO alert_log (channel, event_type, target,
                                 status, error, timestamp, source_event_id)
                              VALUES (?, ?, ?, ?, ?, ?, ?)""",
                              (channel, event_type, target, status, error, _utc_now(),
                               source_event_id))
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

        # Environment values are the preferred secret source. Merge only the
        # operational destination fields before validation so overrides cannot
        # bypass the outbound-network policy.
        email_config = dict(self.config.get("email", {}) or {})
        for env_name, config_name in (
            ("OPTIVOX_SMTP_SERVER", "smtp_server"),
            ("OPTIVOX_SMTP_USER", "smtp_user"),
            ("OPTIVOX_SMTP_PASS", "smtp_pass"),
            ("OPTIVOX_SMTP_FROM", "from"),
            ("OPTIVOX_SMTP_TO", "to"),
        ):
            if os.environ.get(env_name):
                email_config[config_name] = os.environ[env_name]
        if email_config:
            self.config["email"] = email_config
        webhook_config = dict(self.config.get("webhook", {}) or {})
        if os.environ.get("OPTIVOX_WEBHOOK_URL"):
            webhook_config["url"] = os.environ["OPTIVOX_WEBHOOK_URL"]
            webhook_config["enabled"] = True
        if webhook_config:
            self.config["webhook"] = webhook_config

        alert_issues = validate_alert_config(
            self.config, os.environ.get("OPTIVOX_RUNTIME_MODE", "development"))
        if alert_issues:
            message = "Alert configuration rejected: " + "; ".join(alert_issues)
            if is_strict_mode(os.environ.get("OPTIVOX_RUNTIME_MODE", "development")):
                raise DeploymentSecurityError(message)
            print(f"[ALERT] {message}. Unsafe channels disabled.")

        self.enabled = self.config.get("enabled", True)

        email = self.config.get("email", {})
        self.email_enabled = email.get("enabled", False)
        self.smtp_server   = os.environ.get("OPTIVOX_SMTP_SERVER", email.get("smtp_server", "smtp.gmail.com"))
        self.smtp_port     = email.get("smtp_port", 587)
        self.smtp_user     = os.environ.get("OPTIVOX_SMTP_USER", email.get("smtp_user", ""))
        self.smtp_pass     = os.environ.get("OPTIVOX_SMTP_PASS", email.get("smtp_pass", ""))
        self.email_from    = os.environ.get("OPTIVOX_SMTP_FROM", email.get("from", self.smtp_user))
        self.email_to      = os.environ.get("OPTIVOX_SMTP_TO", "") or email.get("to", [])
        if isinstance(self.email_to, str): self.email_to = [self.email_to]

        tg = self.config.get("telegram", {})
        self.telegram_enabled = tg.get("enabled", False) and REQUESTS_AVAILABLE
        self.tg_bot_token = os.environ.get("OPTIVOX_TELEGRAM_BOT_TOKEN", tg.get("bot_token", ""))
        self.tg_chat_id   = os.environ.get("OPTIVOX_TELEGRAM_CHAT_ID", tg.get("chat_id", ""))

        wh = self.config.get("webhook", {})
        self.webhook_enabled = wh.get("enabled", False) and REQUESTS_AVAILABLE and not alert_issues
        self.webhook_url = os.environ.get("OPTIVOX_WEBHOOK_URL", wh.get("url", ""))
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
                                      files={"photo": f}, timeout=(3, 15),
                                      allow_redirects=False,
                                      headers={"User-Agent": "OptiVox-Edge-Alert/1"})
            else:
                url = f"https://api.telegram.org/bot{self.tg_bot_token}/sendMessage"
                r = requests.post(url, data={"chat_id": self.tg_chat_id,
                                             "text": text[:4000]}, timeout=(3, 15),
                                  allow_redirects=False,
                                  headers={"User-Agent": "OptiVox-Edge-Alert/1"})
            if r.status_code == 200: return True, "ok"
            return False, f"http {r.status_code}: {r.text[:200]}"
        except Exception as e:
            return False, str(e)

    def _send_webhook(self, payload: dict) -> Tuple[bool, str]:
        if not self.webhook_enabled: return False, "disabled"
        try:
            validate_webhook_url(
                self.webhook_url,
                os.environ.get("OPTIVOX_RUNTIME_MODE", "development"),
                os.environ.get("OPTIVOX_ALLOWED_WEBHOOK_HOSTS", "").split(","),
            )
            r = requests.post(
                self.webhook_url, json=payload, timeout=(3, 10), allow_redirects=False,
                headers={"User-Agent": "OptiVox-Edge-Alert/1"},
            )
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

    def check_and_alert(self, event_type, name, confidence, details, snapshot_path,
                        source_event_id=None):
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
                                  None if ok else msg, source_event_id=source_event_id)
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
        roi = hand.get("roi")
        if roi and len(roi) == 4:
            rx1, ry1, rx2, ry2 = (int(value) for value in roi)
            rw = max(1, rx2 - rx1)
            rh = max(1, ry2 - ry1)
        else:
            rx1, ry1, rw, rh = 0, 0, w, h
        for p in lm.landmark:
            pts.append((int(rx1 + p.x * rw), int(ry1 + p.y * rh)))

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
    def __init__(self, cfg=None):
        cfg = cfg or CONFIG
        face_root = os.path.expanduser(os.environ.get("OPTIVOX_INSIGHTFACE_ROOT", "~/.insightface"))
        if is_strict_mode(os.environ.get("OPTIVOX_RUNTIME_MODE", "development")):
            verify_face_model_cache(required=True)
        self.app = FaceAnalysis(name="buffalo_l", root=face_root, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.enable_age_gender = bool(cfg.get("ENABLE_AGE_GENDER_INFERENCE", False))

    def detect(self, frame):
        """Detect and landmark faces without running the recognition model."""
        bboxes, kpss = self.app.det_model.detect(
            frame, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return []
        faces = []
        for index in range(len(bboxes)):
            bbox = bboxes[index, 0:4]
            det_score = bboxes[index, 4]
            kps = kpss[index] if kpss is not None else None
            face = InsightFace(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.app.models.items():
                if taskname in {"detection", "recognition"}:
                    continue
                if taskname == "genderage" and not self.enable_age_gender:
                    continue
                model.get(frame, face)
            faces.append(face)
        return faces

    def get_embedding(self, face, frame=None):
        """Run recognition only when the caller explicitly requests it."""
        if frame is not None:
            recognition_model = self.app.models.get("recognition")
            if recognition_model is not None:
                recognition_model.get(frame, face)
        return getattr(face, "embedding", None)

    def scale_face(self, face, scale_x=1.0, scale_y=1.0):
        """Map a processing-frame face onto the original capture frame."""
        if abs(float(scale_x) - 1.0) < 1e-6 and abs(float(scale_y) - 1.0) < 1e-6:
            return face

        mapped = {}
        for key, value in face.items():
            if key == "embedding":
                # Embeddings belong to a specific image crop and must not be
                # carried into a new coordinate system.
                continue
            if key == "bbox":
                arr = np.asarray(value, dtype=np.float32).copy()
                if arr.size >= 4:
                    arr[0] *= float(scale_x)
                    arr[2] *= float(scale_x)
                    arr[1] *= float(scale_y)
                    arr[3] *= float(scale_y)
                mapped[key] = arr
                continue
            if key in {"kps", "landmark_2d_106", "landmark_3d_68"}:
                arr = np.asarray(value).copy()
                if arr.ndim >= 2 and arr.shape[-1] >= 2:
                    arr[..., 0] *= float(scale_x)
                    arr[..., 1] *= float(scale_y)
                elif arr.ndim == 1 and arr.size >= 2:
                    arr[0] *= float(scale_x)
                    arr[1] *= float(scale_y)
                mapped[key] = arr
                continue
            mapped[key] = value
        return InsightFace(mapped)

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
    def get_pose_keypoints(self, face):
        """Return InsightFace's stable 5-point eye/nose/mouth landmarks."""
        for attr in ("kps", "landmark_2d_106"):
            kps = getattr(face, attr, None)
            if kps is not None:
                kps = np.asarray(kps, dtype=np.float32)
                if kps.ndim == 1:
                    kps = kps.reshape(-1, 2)
                if len(kps) >= 5:
                    return kps[:, :2]
        return None
    def get_age(self, face): return int(face.age) if getattr(face, "age", None) else None
    def get_gender(self, face): 
        g = getattr(face, "gender", None)
        return "M" if g == 1 else ("F" if g == 0 else "?")


#7.5: Hand detector
class HandDetector:
    def __init__(self, cfg=None):
        self.cfg = cfg or CONFIG
        self.enabled = MEDIAPIPE_AVAILABLE and hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
        if MEDIAPIPE_AVAILABLE and not self.enabled:
            print("[WARN] MediaPipe legacy solutions API unavailable - hand detection disabled.")
        if self.enabled:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.cfg["HAND_MAX_NUM"],
                min_detection_confidence=self.cfg["HAND_MIN_DETECTION"],
                min_tracking_confidence=self.cfg["HAND_MIN_TRACKING"])
            self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame, roi=None):
        if not self.enabled: return []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        out = []
        if res.multi_hand_landmarks:
            for idx, lm in enumerate(res.multi_hand_landmarks):
                handedness = "Right"
                if res.multi_handedness and idx < len(res.multi_handedness):
                    handedness = res.multi_handedness[idx].classification[0].label
                out.append({"landmarks": lm, "handedness": handedness, "roi": roi})
        return out

    def draw(self, frame, hands):
        if not self.enabled or not self.cfg.get("SHOW_HAND_LANDMARKS", False):
            return frame
        for h in hands:
            roi = h.get("roi")
            if roi and len(roi) == 4:
                rx1, ry1, rx2, ry2 = (int(value) for value in roi)
                rw = max(1, rx2 - rx1)
                rh = max(1, ry2 - ry1)
                points = [
                    (int(rx1 + point.x * rw), int(ry1 + point.y * rh))
                    for point in h["landmarks"].landmark
                ]
                for start, end in self.mp_hands.HAND_CONNECTIONS:
                    if start < len(points) and end < len(points):
                        cv2.line(frame, points[start], points[end], (255, 0, 255), 2)
                for point in points:
                    cv2.circle(frame, point, 2, (0, 255, 0), -1)
            else:
                self.mp_draw.draw_landmarks(
                    frame, h["landmarks"], self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2))
        return frame


#7.6: Pose detection
class PoseDetector:
    def __init__(self, cfg=None):
        self.cfg = cfg or CONFIG
        self.enabled = MEDIAPIPE_AVAILABLE and hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
        if MEDIAPIPE_AVAILABLE and not self.enabled:
            print("[WARN] MediaPipe legacy solutions API unavailable - pose detection disabled.")
        if self.enabled:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=self.cfg["POSE_DETECTION_CONF"],
                min_tracking_confidence=0.5)
            self.mp_draw = mp.solutions.drawing_utils

    def analyze(self, frame, roi=None) -> dict:
        """Returns dict with keys: pose (landmarks), is_fallen, hands_raised."""
        if not self.enabled: return {"pose": None, "is_fallen": False, "hands_raised": False}
        source = frame
        if roi and len(roi) == 4:
            x1, y1, x2, y2 = (int(value) for value in roi)
            source = frame[max(0, y1):min(frame.shape[0], y2),
                           max(0, x1):min(frame.shape[1], x2)]
        else:
            x1, y1 = 0, 0
        if source.size == 0:
            return {"pose": None, "is_fallen": False, "hands_raised": False, "roi": roi}
        rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return {"pose": None, "is_fallen": False, "hands_raised": False, "roi": roi}
        lm = res.pose_landmarks.landmark
        h, w = source.shape[:2]
        def pt(i): return (x1 + lm[i].x * w, y1 + lm[i].y * h, lm[i].visibility)
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
                "roi": roi,
                 "landmarks_px": {"nose": nose, "l_sh": l_sh, "r_sh": r_sh,
                                 "l_hip": l_hip, "r_hip": r_hip}}

    def draw(self, frame, result):
        if not self.enabled or not result or not result.get("pose"): return frame
        if not self.cfg.get("SHOW_POSE_LANDMARKS", False): return frame
        roi = result.get("roi")
        if roi and len(roi) == 4:
            x1, y1, x2, y2 = (int(value) for value in roi)
            rw = max(1, x2 - x1)
            rh = max(1, y2 - y1)
            points = [
                (int(x1 + landmark.x * rw), int(y1 + landmark.y * rh))
                for landmark in result["pose"].landmark
            ]
            for start, end in self.mp_pose.POSE_CONNECTIONS:
                if start < len(points) and end < len(points):
                    cv2.line(frame, points[start], points[end], (255, 100, 0), 2)
            for point in points:
                cv2.circle(frame, point, 2, (0, 255, 255), -1)
        else:
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
            if not os.path.isabs(mp_):
                mp_ = os.path.join(_BASE_DIR, mp_)
            if not os.path.exists(mp_):
                if is_strict_mode(os.environ.get("OPTIVOX_RUNTIME_MODE", "development")):
                    print(f"[ERROR] Model {mp_} is missing; production never downloads models silently.")
                    return
                print(f"[INFO] YOLO model {mp_} not present; local mode may download it.")
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

    @staticmethod
    def estimate_yaw(keypoints):
        """Estimate horizontal head turn from five InsightFace keypoints.

        The result is normalized by eye distance: near zero is frontal,
        negative/positive values are opposite head-turn directions.
        """
        if keypoints is None or len(keypoints) < 5:
            return None
        pts = np.asarray(keypoints, dtype=np.float32)
        left_eye, right_eye = pts[0], pts[1]
        eye_distance = float(np.linalg.norm(right_eye - left_eye))
        if eye_distance < 1.0:
            return None
        eye_mid_x = float((left_eye[0] + right_eye[0]) * 0.5)
        nose_x = float(pts[2][0])
        mouth_mid_x = float((pts[3][0] + pts[4][0]) * 0.5)
        nose_offset = (nose_x - eye_mid_x) / eye_distance
        mouth_offset = (mouth_mid_x - eye_mid_x) / eye_distance
        return float(0.75 * nose_offset + 0.25 * mouth_offset)


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
        # A stable apparent depth has low variance. High variance indicates
        # unstable scale/depth evidence and must not pass this check.
        return var <= self.cfg.get("DEPTH_VARIANCE_THRESHOLD", 0.01)


class AntiSpoofDetector:
    REAL = "REAL"
    UNCERTAIN = "UNCERTAIN"
    SUSPECT = "SUSPECT"

    def __init__(self, cfg=None):
        self._root_cfg = cfg or CONFIG
        self.cfg = self._root_cfg.get("ANTI_SPOOFING", {})
        self.enabled = self.cfg.get("ENABLED", True)
        self.blink = BlinkDetector(self._root_cfg)
        self.pose = HeadPoseEstimator(self._root_cfg)
        self.depth = DepthEstimator(self._root_cfg)
        self._frames_seen = 0
        self._global_state = {"frames_seen": 0}
        self._track_states: OrderedDict[int, dict] = OrderedDict()

    def _state_for_track(self, track_id):
        if track_id is None or int(track_id) <= 0:
            return self._global_state, self.blink, self.pose, self.depth
        track_id = int(track_id)
        state = self._track_states.get(track_id)
        if state is None:
            state = {
                "frames_seen": 0,
                "blink": BlinkDetector(self._root_cfg),
                "pose": HeadPoseEstimator(self._root_cfg),
                "depth": DepthEstimator(self._root_cfg),
            }
            self._track_states[track_id] = state
        self._track_states.move_to_end(track_id)
        while len(self._track_states) > 256:
            self._track_states.popitem(last=False)
        return state, state["blink"], state["pose"], state["depth"]

    def forget_track(self, track_id) -> None:
        """Drop liveness history when a tracker identity is closed or reused."""
        try:
            self._track_states.pop(int(track_id), None)
        except (TypeError, ValueError):
            return

    def analyze(self, face_crop, frame_shape, landmarks=None, bbox=None,
                pose_keypoints=None, track_id=None) -> Tuple[str, dict]:
        if not self.enabled:
            return self.REAL, {"method": "disabled", "status": self.REAL}

        state, blink, pose, depth = self._state_for_track(track_id)
        state["frames_seen"] += 1
        details = {"frames_seen": state["frames_seen"], "track_id": track_id}

        yaw = pose.estimate_yaw(pose_keypoints)
        if yaw is not None:
            details["yaw"] = yaw

        if landmarks is not None:
            pose.update(landmarks)
            blink.update(landmarks, time.time())
        if bbox is not None:
            depth.update(bbox)

        if state["frames_seen"] < int(self.cfg.get("WARMUP_FRAMES", 12)):
            details.update({"reason": "anti-spoof warmup", "status": self.UNCERTAIN})
            return self.UNCERTAIN, details

        if landmarks is None or len(landmarks) < 30:
            details.update({"reason": "insufficient landmarks", "status": self.UNCERTAIN})
            return self.UNCERTAIN, details

        pose_var = pose.variance()
        blink_n = blink.get_blink_count()
        depth_ok = depth.is_consistent()

        strictness = self.cfg.get("STRICTNESS", "normal").lower()
        movement_thr = {"low": 0.20, "normal": 0.35, "high": 0.65}.get(strictness, 0.35)
        required = int(self.cfg.get("REQUIRED_CHECKS_PASSED", 2))

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
                    "zone_entered_at": None,
                    "loitering_emitted": False,
                    "running_since": None,
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
                             ("SCANNING", self._scanning),
                             ("LOITERING", self._loitering),
                             ("RUNNING", self._running)):
                r = fn(h)
                if r and (self.current_time - self.last_behavior_flag[oid][name]) >= cooldown:
                    events.append((name, f"ID_{oid}", 1.0, r, {"track_id": oid}))
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
                               f"ID_{oid} in unusual location.", {"track_id": oid}))
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
        self._track_motion: Dict[int, dict] = {}
        self._evacuation_started = None
        self._last_evacuation_alert = 0.0

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

        # Evacuation is a sustained crowd-motion signal, not a single fast
        # track. Keep the calculation local so center attendance cannot pause
        # the independent safety pipeline.
        now = time.monotonic()
        speeds = []
        for track_id, center in tracked.items():
            current = (float(center[0]), float(center[1]))
            previous = self._track_motion.get(int(track_id))
            if previous:
                elapsed = max(1e-6, now - previous["at"])
                speeds.append(math.hypot(current[0] - previous["center"][0],
                                         current[1] - previous["center"][1]) / elapsed)
            self._track_motion[int(track_id)] = {"center": current, "at": now}
        for track_id in list(self._track_motion):
            if track_id not in tracked and now - self._track_motion[track_id]["at"] > 10.0:
                self._track_motion.pop(track_id, None)
        evac_min = int(self.cfg.get("EVAC_MIN_PEOPLE", 4))
        evac_threshold = float(self.cfg.get("EVAC_AVG_SPEED_THRESHOLD", 25.0))
        average_speed = sum(speeds) / len(speeds) if speeds else 0.0
        if len(tracked) >= evac_min and average_speed >= evac_threshold:
            self._evacuation_started = self._evacuation_started or now
            confirm_seconds = float(self.cfg.get("EVAC_CONFIRM_SECONDS", 1.0))
            cooldown = float(self.cfg.get("EVAC_COOLDOWN_SECONDS", 30.0))
            if (now - self._evacuation_started >= confirm_seconds
                    and now - self._last_evacuation_alert >= cooldown):
                self._last_evacuation_alert = now
                events.append(("EVACUATION_ALERT", "SYSTEM", 0.9,
                               f"{len(tracked)} people moving at {average_speed:.1f}px/s",
                               {"people": len(tracked), "average_speed_px_sec": round(average_speed, 2)}))
        else:
            self._evacuation_started = None
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
        self.face_analyzer    = FaceAnalyzer(self.cfg)
        self.hand_detector    = HandDetector(self.cfg)
        self.object_detector  = ObjectDetector(self.cfg) if YOLO_AVAILABLE else None
        self.danger_detector = DangerDetector(self.cfg) if YOLO_AVAILABLE and self.cfg.get("DANGER_DETECTION", {}).get("ENABLED", False) else None
        self.pose_detector    = PoseDetector(self.cfg)
        self.anti_spoof       = AntiSpoofDetector(self.cfg)
        self.custom_objects = CustomObjectManager(self.cfg)
        self.quality_scorer   = ImageQualityScorer(self.cfg)
        self.behavior         = BehaviorAnalyzer(self.cfg)
        self.crowd_intel      = CrowdIntelligence(self.cfg)
        self.security_signals = SecuritySignalEngine(self.cfg)
        self.supplementary    = SupplementaryDetector(self.cfg)
        self.person_tracker   = CentroidTracker(self.cfg["TRACKER_MAX_DISAPPEARED"], self.cfg["TRACKER_MAX_DISTANCE"])
        self.face_indexer     = FAISSIndexer(dim=512)
        self.face_db: Dict[str, Dict[str, Any]] = {}
        # Disabled roster identities remain in the local biometric store for
        # historical traceability, but must not be returned as live matches.
        self._disabled_identities: set[str] = set()
        self._last_recognition_details: Dict[str, float] = {}
        self._face_import_stats = {
            "source_images": 0,
            "source_embeddings": 0,
            "augmented_embeddings": 0,
            "augmentation_enabled": False,
        }
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
        self._last_face_observation_frame_id = None
        self._last_face_observation_at = None
        self._last_face_observation_wallclock = None
        self._last_object_detections = []
        self._last_object_observation_frame_id = None
        self._last_object_observation_at = None
        self._last_object_observation_wallclock = None
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
        self._last_hand_dets = []
        self._last_pose_result = {}
        self._last_pose_observation_frame_id = None
        self._last_pose_observation_at = None
        self._last_pose_observation_wallclock = None
        correlation_cfg = self.cfg.get("CORRELATION_CORE", {})
        self.correlation = CorrelationCore(
            max_entities=correlation_cfg.get("MAX_ENTITIES", 128),
            max_observations_per_type=correlation_cfg.get(
                "MAX_OBSERVATIONS_PER_TYPE", 12),
            close_after_sec=correlation_cfg.get("ENTITY_CLOSE_AFTER_SEC", 10.0),
            ttl_overrides_ms=correlation_cfg.get("TTL_MS"),
            face_visible_after_sec=correlation_cfg.get("FACE_VISIBLE_AFTER_SEC", 1.0),
            quality_valid_after_sec=correlation_cfg.get("QUALITY_VALID_AFTER_SEC", 1.5),
            liveness_valid_after_sec=correlation_cfg.get("LIVENESS_VALID_AFTER_SEC", 2.0),
            identity_confirmation_observations=self.cfg.get("PERFORMANCE", {}).get(
                "IDENTITY_CONFIRMATION_OBSERVATIONS",
                self.cfg.get("PERFORMANCE", {}).get("FACE_RECOG_STABLE_FRAMES", 3),
            ),
            track_switch_distance=max(
                220.0,
                float(self.cfg.get("TRACKER_MAX_DISTANCE", 150)) * 1.75,
            ),
        )
        self._last_correlation_state = {}
        perf_cfg = self.cfg.get("PERFORMANCE", {})
        scheduler_config = {
            "enabled": bool(perf_cfg.get("ADAPTIVE_SCHEDULING", True)),
            "target_total_ms": float(perf_cfg.get("ADAPTIVE_TARGET_TOTAL_MS", 120.0)),
        }
        for key, value in perf_cfg.items():
            if key.endswith("_ACTIVE_EVERY_N_FRAMES") or key.endswith("_IDLE_EVERY_N_FRAMES"):
                scheduler_config[key.lower()] = value
        self._scheduler = AdaptiveInferenceScheduler(scheduler_config)
        self._recognition_cache_hits = 0
        self._recognition_cache_misses = 0
        self._recognition_attempts = 0
        self._recognition_confirmations = 0
        self._embeddings_skipped_due_to_cache = 0
        self._embeddings_skipped_due_to_quality = 0
        self._face_quality_rejects = 0
        self._roi_inference_runs = 0
        self._model_metrics = ModelCallProfiler(
            window_size=perf_cfg.get("PROFILER_WINDOW_FRAMES", 120))
        self._identity_started_at: Dict[int, float] = {}
        self._identity_timing: Dict[int, dict] = {}
        self._identity_timing_samples: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=120))
        self._identity_confirmation_times: deque = deque(maxlen=120)
        self._last_evidence_at: Dict[int, float] = {}
        self._web_enrollment = None
        self._pending_web_enrollment = None
        self._last_enrollment_metadata = {}

    def performance_snapshot(self) -> dict:
        return {
            "recognition_cache_hits": self._recognition_cache_hits,
            "recognition_cache_misses": self._recognition_cache_misses,
            "recognition_attempts": self._recognition_attempts,
            "recognition_confirmations": self._recognition_confirmations,
            "embeddings_skipped_due_to_cache": self._embeddings_skipped_due_to_cache,
            "embeddings_skipped_due_to_quality": self._embeddings_skipped_due_to_quality,
            "face_quality_rejects": self._face_quality_rejects,
            "roi_inference_runs": self._roi_inference_runs,
            "face_import": dict(self._face_import_stats),
            "models": self._model_metrics.snapshot(),
            "execution_provider": {
                "face": "CPUExecutionProvider",
                "yolo": "NOT MEASURED",
            },
            "identity_timing": {
                "confirmed_count": len(self._identity_confirmation_times),
                "mean_time_to_confirm_sec": round(
                    sum(self._identity_confirmation_times) /
                    len(self._identity_confirmation_times), 3
                ) if self._identity_confirmation_times else None,
                "last_time_to_confirm_sec": round(
                    self._identity_confirmation_times[-1], 3
                ) if self._identity_confirmation_times else None,
                "mean_time_to_first_usable_face_sec": self._mean_timing(
                    "first_usable_face"),
                "mean_time_to_first_embedding_sec": self._mean_timing(
                    "first_embedding"),
                "mean_time_to_candidate_sec": self._mean_timing("candidate"),
            },
            "scheduler": self._scheduler.snapshot(),
            "correlation": self.correlation.snapshot(),
            "security_capabilities": self.security_signals.capability_state(),
        }

    def _mean_timing(self, name):
        values = self._identity_timing_samples.get(name, ())
        return round(sum(values) / len(values), 3) if values else 0.0

    def _profile_call(self, name, callback, *args, **kwargs):
        started = time.perf_counter()
        try:
            return callback(*args, **kwargs)
        finally:
            self._model_metrics.record(
                name, (time.perf_counter() - started) * 1000.0)

    def _face_quality(self, frame, bbox):
        if not self.cfg.get("PERFORMANCE", {}).get("FACE_QUALITY_GATE_ENABLED", True):
            return 100.0, {}, True
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(frame.shape[1], int(x1)))
        y1 = max(0, min(frame.shape[0], int(y1)))
        x2 = max(0, min(frame.shape[1], int(x2)))
        y2 = max(0, min(frame.shape[0], int(y2)))
        if x2 <= x1 or y2 <= y1:
            return 0.0, {}, False
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0, {}, False
        score, metrics = self._profile_call(
            "face_quality", self.quality_scorer.score, crop)
        perf_cfg = self.cfg.get("PERFORMANCE", {})
        accepted = (
            score >= float(perf_cfg.get("FACE_QUALITY_MIN_SCORE", 50.0))
            and metrics.get("blur", 0.0) >= float(
                perf_cfg.get("FACE_QUALITY_MIN_BLUR", 45.0))
        )
        return float(score), metrics, bool(accepted)

    @staticmethod
    def _bbox_motion(previous_bbox, current_bbox):
        if not previous_bbox or not current_bbox:
            return float("inf")
        px = (float(previous_bbox[0]) + float(previous_bbox[2])) * 0.5
        py = (float(previous_bbox[1]) + float(previous_bbox[3])) * 0.5
        cx = (float(current_bbox[0]) + float(current_bbox[2])) * 0.5
        cy = (float(current_bbox[1]) + float(current_bbox[3])) * 0.5
        return math.hypot(cx - px, cy - py)

    def _relevant_roi(self, frame, object_detections=None, face_objects=None, tracked=None):
        """Return a padded person-focused ROI, or None when it is not useful."""
        if not self.cfg.get("PERFORMANCE", {}).get("ROI_INFERENCE_ENABLED", True):
            return None
        height, width = frame.shape[:2]
        boxes = []
        for detection in object_detections or []:
            if str(detection.get("class_name", "")).lower() == "person":
                boxes.append(detection.get("bbox"))
        for face in face_objects or []:
            try:
                boxes.append(self.face_analyzer.get_bbox(face))
            except Exception:
                continue
        if not boxes and tracked:
            for center in tracked.values():
                cx, cy = center
                boxes.append((cx - 120, cy - 260, cx + 120, cy + 260))
        boxes = [box for box in boxes if box and len(box) == 4]
        if not boxes:
            return None
        x1 = min(float(box[0]) for box in boxes)
        y1 = min(float(box[1]) for box in boxes)
        x2 = max(float(box[2]) for box in boxes)
        y2 = max(float(box[3]) for box in boxes)
        padding = float(self.cfg.get("PERFORMANCE", {}).get("ROI_PADDING_RATIO", 0.12))
        pad_x = max(24.0, (x2 - x1) * padding)
        pad_y = max(24.0, (y2 - y1) * padding)
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(width, int(x2 + pad_x))
        y2 = min(height, int(y2 + pad_y))
        roi_area = max(0, x2 - x1) * max(0, y2 - y1)
        if roi_area <= 0 or roi_area >= width * height * float(
                self.cfg.get("PERFORMANCE", {}).get("ROI_MAX_FRAME_RATIO", 0.88)):
            return None
        return x1, y1, x2, y2

    def import_known_faces_folder(self, db: EventDatabase = None):
        root = self.cfg.get("KNOWN_FACES_DIR")
        if not root or not os.path.isdir(root):
            print(f"[FACES] Known faces folder not found: {root}")
            self.sync_roster_state(db)
            return

        imported_people = 0
        imported_embeddings = 0
        augmented_embeddings = 0
        source_image_count = 0
        quality = self.quality_scorer
        perf = self.cfg.get("PERFORMANCE", {})
        augment_enabled = bool(perf.get("AUTO_AUGMENT_SPARSE_DATASET", True))
        sparse_limit = max(0, int(perf.get("AUGMENT_WHEN_SOURCE_COUNT_BELOW", 3)))
        variants_per_source = max(0, int(perf.get("AUGMENTATIONS_PER_SOURCE_IMAGE", 3)))
        max_augmented = max(0, int(perf.get("MAX_AUGMENTED_EMBEDDINGS_PER_PERSON", 8)))

        def source_fingerprint(path):
            try:
                with open(path, "rb") as handle:
                    return hashlib.sha256(handle.read()).hexdigest()
            except OSError:
                return None

        def accept_image(person_name, img, provenance, reference_embedding=None):
            detected = self.face_analyzer.detect(img)
            if len(detected) != 1:
                return False, None
            x1, y1, x2, y2 = self.face_analyzer.get_bbox(detected[0])
            crop = img[max(0, y1):min(img.shape[0], y2),
                       max(0, x1):min(img.shape[1], x2)]
            quality_score = 0.0
            if crop.size > 0:
                ok, score = quality.is_acceptable(crop)
                quality_score = float(score)
                if not ok:
                    return False, None
            embedding = self.face_analyzer.get_embedding(detected[0], img)
            if embedding is None:
                return False, None
            if reference_embedding is not None and _cosine(
                    embedding, reference_embedding) < float(perf.get(
                        "AUGMENT_MIN_SOURCE_SIMILARITY", 0.86)):
                return False, embedding
            return self._append_face_embedding(
                person_name, embedding,
                {**provenance, "quality_score": quality_score}), embedding

        for person_name in sorted(os.listdir(root)):
            person_dir = os.path.join(root, person_name)

            if not os.path.isdir(person_dir):
                continue

            image_paths = sorted(
                os.path.join(person_dir, fname)
                for fname in os.listdir(person_dir)
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            )
            if not image_paths:
                continue
            source_image_count += len(image_paths)
            record = self.face_db.setdefault(
                person_name,
                {"embeddings": [], "threshold": self.cfg["FACE_RECOG_THRESHOLD"]},
            )
            record.setdefault("embeddings", [])
            fingerprints = set(record.get("source_fingerprints") or [])
            provenance = record.get("embedding_provenance")
            if not isinstance(provenance, list):
                provenance = []
                record["embedding_provenance"] = provenance
            # Legacy pickles may not have one provenance entry per embedding.
            # Keep the embeddings, but only use aligned entries for deduplication.
            if len(provenance) > len(record["embeddings"]):
                if record["embeddings"]:
                    del provenance[:-len(record["embeddings"])]
                else:
                    provenance.clear()
            while len(provenance) < len(record["embeddings"]):
                provenance.insert(0, {"kind": "legacy"})
            added_for_person = 0
            augmented_for_person = 0

            for path in image_paths:
                fingerprint = source_fingerprint(path)
                if fingerprint and fingerprint in fingerprints:
                    continue
                img = cv2.imread(path)
                if img is None:
                    print(f"[FACES] Could not read {path}")
                    continue

                rel_path = os.path.relpath(path, root)
                source_accepted, source_embedding = accept_image(person_name, img, {
                        "kind": "source", "source_path": rel_path,
                        "source_fingerprint": fingerprint,
                        "transform": "identity",
                })
                if source_accepted:
                    added_for_person += 1
                    imported_embeddings += 1
                    if fingerprint:
                        fingerprints.add(fingerprint)

                if (augment_enabled and source_embedding is not None
                        and len(image_paths) < sparse_limit
                        and augmented_for_person < max_augmented):
                    remaining = min(
                        variants_per_source,
                        max_augmented - augmented_for_person,
                    )
                    for transform, variant in generate_variants(img, remaining):
                        if augmented_for_person >= max_augmented:
                            break
                        variant_accepted, _ = accept_image(person_name, variant, {
                                "kind": "augmented", "source_path": rel_path,
                                "source_fingerprint": fingerprint,
                                "transform": transform,
                                "augmentation_version": perf.get(
                                    "AUGMENTATION_VERSION", 1),
                        }, reference_embedding=source_embedding)
                        if variant_accepted:
                            augmented_for_person += 1
                            augmented_embeddings += 1

            record["source_fingerprints"] = sorted(fingerprints)
            if len(record["embeddings"]) >= 2:
                mean_variance, std_variance = _compute_intra_class_variance(
                    record["embeddings"])
                record["threshold"] = _auto_threshold(mean_variance, std_variance)

            if added_for_person or augmented_for_person:
                imported_people += 1
                if db is not None:
                    try:
                        # Folder import may run every startup. Do not overwrite
                        # an existing role or active/inactive roster decision.
                        if db.get_person_id(person_name) is None:
                            db.upsert_person(person_name)
                    except Exception as e:
                        print(f"[FACES] DB upsert failed for {person_name}: {e}")
                print(
                    f"[FACES] Imported {added_for_person} source + "
                    f"{augmented_for_person} derived sample(s) for {person_name}"
                )

        if imported_embeddings or augmented_embeddings:
            self._rebuild_index()
            self.save_face_db()

        self.sync_roster_state(db)

        print(
            f"[FACES] Import complete: {imported_people} people, "
            f"{imported_embeddings} source + {augmented_embeddings} derived embeddings"
        )
        self._face_import_stats = {
            "source_images": source_image_count,
            "source_embeddings": imported_embeddings,
            "augmented_embeddings": augmented_embeddings,
            "augmentation_enabled": augment_enabled,
        }

    def sync_roster_state(self, db: EventDatabase = None) -> None:
        """Keep live biometric matches aligned with the active roster."""
        if db is None:
            return
        disabled = set()
        roster_roles = {}
        try:
            roster_roles = {
                str(row["name"]): str(row["role"] or "")
                for row in db.get_known_face_names(limit=2000)
            }
            self.security_signals.set_roster_roles(roster_roles)
        except Exception:
            pass
        for name in self.face_db:
            try:
                if db.get_active_person_id(name) is None:
                    disabled.add(str(name).casefold())
            except Exception:
                # A transient database failure must not disable every identity.
                continue
        self._disabled_identities = disabled

    def _is_identity_disabled(self, person_name: object) -> bool:
        return str(person_name or "").strip().casefold() in self._disabled_identities

    def disable_identity(self, person_name: str) -> None:
        """Disable live matching while retaining biometric history on disk."""
        self._disabled_identities.add(str(person_name or "").strip().casefold())
        self.invalidate_identity(person_name)

    def enable_identity(self, person_name: str) -> None:
        self._disabled_identities.discard(str(person_name or "").strip().casefold())
        self.invalidate_identity(person_name)

    def _load_face_db(self):
        path = self.cfg["FACE_DB_FILE"]
        try:
            self.face_db = load_face_database(path)
            print(f"[INFO] Face DB loaded: {len(self.face_db)} person(s)")
            self._rebuild_index()
        except BiometricStorageError as e:
            # Fail closed for recognition instead of executing an untrusted
            # object stream or continuing with partially decoded embeddings.
            print(f"[SECURITY] Face DB rejected: {e}")
            self.face_db = {}

    def save_face_db(self):
        path = self.cfg["FACE_DB_FILE"]
        try:
            return save_face_database(path, self.face_db)
        except (OSError, BiometricStorageError) as e:
            print(f"[WARN] Face DB save failed: {e}")
            return False

    def _rebuild_index(self):
        self.face_indexer.reset()
        names, embs, ths = [], [], []
        for name, data in self.face_db.items():
            for embedding in data.get("embeddings", []):
                if embedding is None:
                    continue
                names.append(name)
                embs.append(np.asarray(embedding, dtype=np.float32))
                ths.append(data.get("threshold", self.cfg["FACE_RECOG_THRESHOLD"]))
        if names:
            self.face_indexer.add_embeddings(names, embs, ths)
            print(
                f"[INFO] FAISS index rebuilt: {len(names)} embedding(s) "
                f"for {len(set(names))} person(s)"
            )

    def recognize_face(self, embedding) -> Tuple[str, float, str]:
        self._last_recognition_details = {}
        if embedding is None:
            return "UNKNOWN", 0.0, "No embedding"

        # Search individual enrolled samples. Pooling one vector per person can
        # erase the pose that best matches a live face, while a bounded index
        # search avoids a Python loop over the entire roster on every refresh.
        person_matches = defaultdict(list)
        person_thresholds = {}
        search_k = max(10, len(self.face_db) * 3)
        names, similarities, thresholds = self.face_indexer.search(
            embedding, k=search_k)
        for person_name, similarity, threshold in zip(
                names, similarities, thresholds):
            person_matches[person_name].append(float(similarity))
            person_thresholds[person_name] = float(threshold)

        candidates = []
        for person_name, scores in person_matches.items():
            scores.sort(reverse=True)
            # Average the strongest samples so one accidental near-match does
            # not identify a stranger, while pose changes still have a chance.
            candidates.append((
                float(np.mean(scores[:min(3, len(scores))])),
                person_name,
                person_thresholds[person_name],
            ))

        if not candidates:
            return "UNKNOWN", 0.0, "No enrollments"
        candidates.sort(reverse=True)
        best_sim, best_name, stored_threshold = candidates[0]
        second_sim = candidates[1][0] if len(candidates) > 1 else -1.0
        required = max(
            float(self.cfg.get("FACE_RECOG_ACCEPT_THRESHOLD", 0.62)),
            float(stored_threshold),
        )
        margin = best_sim - second_sim if second_sim >= 0 else 1.0
        min_margin = float(self.cfg.get("FACE_RECOG_MIN_MARGIN", 0.08))
        self._last_recognition_details = {
            "best_score": float(best_sim),
            "second_score": float(second_sim),
            "margin": float(margin),
            "required": float(required),
            "min_margin": float(min_margin),
        }
        if best_sim >= required and margin >= min_margin:
            return best_name, float(np.clip(best_sim, 0, 1)), (
                f"score={best_sim:.3f}>={required:.3f} margin={margin:.3f}")
        return "UNKNOWN", float(np.clip(best_sim, 0, 1)), (
            f"uncertain score={best_sim:.3f}/{required:.3f} margin={margin:.3f}/{min_margin:.3f}")

    def _smooth_recognize(self, oid, embedding):
        name, conf, reason = self._profile_call(
            "identity_matching", self.recognize_face, embedding)
        self._face_recog_history[oid].append(name)
        if name not in ("UNKNOWN", "SPOOF") and not name.startswith("STRANGER_"):
            return name, conf, reason

        # Keep a recently confirmed identity during short detector/landmark
        # flicker instead of immediately converting the person to a stranger.
        known = [n for n in self._face_recog_history[oid]
                 if n not in ("UNKNOWN", "SPOOF") and not n.startswith("STRANGER_")]
        if len(known) >= 2:
            counts = defaultdict(int)
            for n in known:
                counts[n] += 1
            stable_name = max(counts.items(), key=lambda kv: kv[1])[0]
            return stable_name, conf, f"held_recent_match ({stable_name})"
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
        added = self._append_face_embedding(
            person_name, embedding,
            {"kind": "runtime", "source": "visible_stranger_enrollment"},
        )
        if not added:
            print(f"[ENROLL] {person_name}: duplicate or invalid embedding.")
            return False
        embs = self.face_db[person_name]["embeddings"]
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

        duplicates = self.find_duplicate_identities(
            [embedding], exclude_name=person_name)
        if duplicates:
            print(
                f"[ENROLL] Duplicate guard: visible face is similar to "
                f"{duplicates[0]['name']} ({duplicates[0]['similarity']:.3f})."
            )
            return False

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
            "identity_name": person_name,
            "identity_state": "CONFIRMED",
            "conf": 1.0,
            "cached_identity_confidence": 1.0,
            "current_observation_similarity": 1.0,
            "reason": "manual_visible_enrollment",
            "embedding": embedding,
            "stable_frames": self.cfg.get("PERFORMANCE", {}).get(
                "FACE_RECOG_STABLE_FRAMES", 3),
            "candidate_hits": self.cfg.get("PERFORMANCE", {}).get(
                "FACE_RECOG_STABLE_FRAMES", 3),
            "last_attempt_at": time.time(),
            "last_observation_at": time.time(),
            "last_verified_at": time.time(),
            "ts": time.time(),
        }

        self._face_recog_history[oid].clear()
        self._face_recog_history[oid].append(person_name)

        print(f"[ENROLL] {old_name} on track {oid} is now enrolled as {person_name}.")
        return True

    def begin_web_enrollment(self, person_name: str, min_embeddings=5,
                             max_embeddings=10, metadata=None,
                             replace_existing: bool = False) -> dict:
        """Start non-blocking enrollment driven by the live runtime buffer."""
        person_name = str(person_name or "").strip()
        if not person_name:
            return {"ok": False, "message": "A name is required."}
        if self._web_enrollment and self._web_enrollment.get("active"):
            return {"ok": False, "message": "Another enrollment is already in progress."}
        try:
            min_embeddings = max(1, int(min_embeddings))
            max_embeddings = max(min_embeddings, int(max_embeddings))
        except (TypeError, ValueError):
            return {"ok": False, "message": "Enrollment sample limits must be valid numbers."}
        if person_name in self.face_db and not replace_existing:
            return {"ok": False, "message": f"{person_name} is already enrolled. Use a new name or retraining workflow."}
        self._pending_web_enrollment = None
        self._web_enrollment = {
            "active": True,
            "person_name": person_name,
            "min_embeddings": min_embeddings,
            "max_embeddings": max_embeddings,
            "embeddings": [],
            "last_capture_at": 0.0,
            "started_at": time.time(),
            "last_score": 0.0,
            "metadata": dict(metadata or {}),
            "replace_existing": bool(replace_existing),
            "message": "Keep exactly one clear face visible.",
            "quality_scores": [],
            "quality_metrics": [],
            "pose_yaws": [],
            "rejected_samples": 0,
        }
        return {"ok": True, "stage": "capturing", "person_name": person_name,
                "accepted": 0, "minimum": self._web_enrollment["min_embeddings"],
                "maximum": self._web_enrollment["max_embeddings"]}

    def cancel_web_enrollment(self) -> dict:
        current = self._web_enrollment or {}
        self._web_enrollment = None
        self._pending_web_enrollment = None
        return {"ok": True, "stage": "cancelled",
                "person_name": current.get("person_name")}

    def collect_web_enrollment_frame(self, frame) -> dict:
        """Collect one quality-gated sample without blocking camera processing."""
        state = self._web_enrollment
        if not state or not state.get("active"):
            return {"active": False, "stage": "idle"}
        if frame is None or getattr(frame, "size", 0) == 0:
            return {"active": True, "stage": "capturing", "accepted": len(state["embeddings"]),
                    "message": "Waiting for a camera frame."}

        try:
            faces = self.face_analyzer.detect(frame)
        except Exception as exc:
            state["message"] = f"Face detector unavailable: {exc}"
            return {"active": True, "stage": "capturing", "accepted": len(state["embeddings"]),
                    "message": state["message"]}

        accepted = len(state["embeddings"])
        if len(faces) != 1:
            state["rejected_samples"] += 1
            state["message"] = ("Keep exactly one face visible."
                                 if len(faces) > 1 else "Move one clear face into view.")
            return {"active": True, "stage": "capturing", "accepted": accepted,
                    "minimum": state["min_embeddings"], "maximum": state["max_embeddings"],
                    "message": state["message"]}

        x1, y1, x2, y2 = self.face_analyzer.get_bbox(faces[0])
        crop = frame[max(0, y1):min(frame.shape[0], y2),
                     max(0, x1):min(frame.shape[1], x2)]
        if crop.size == 0:
            state["message"] = "Face crop is unavailable."
            return {"active": True, "stage": "capturing", "accepted": accepted,
                    "message": state["message"]}
        quality_ok, quality_score = self.quality_scorer.is_acceptable(crop)
        state["last_score"] = float(quality_score)
        if not quality_ok:
            state["rejected_samples"] += 1
            state["quality_metrics"].append(dict(getattr(self.quality_scorer, "last_metrics", {}) or {}))
            state["message"] = f"Improve lighting or sharpness. Quality {quality_score:.0f}."
            return {"active": True, "stage": "capturing", "accepted": accepted,
                    "minimum": state["min_embeddings"], "maximum": state["max_embeddings"],
                    "quality_score": quality_score, "message": state["message"]}

        now = time.time()
        if now - state["last_capture_at"] < 0.35:
            return {"active": True, "stage": "capturing", "accepted": accepted,
                    "minimum": state["min_embeddings"], "maximum": state["max_embeddings"],
                    "quality_score": quality_score, "message": "Sample spacing active."}
        embedding = self.face_analyzer.get_embedding(faces[0], frame)
        if embedding is None:
            state["message"] = "Embedding unavailable. Hold still and try again."
            return {"active": True, "stage": "capturing", "accepted": accepted,
                    "message": state["message"]}

        duplicate_threshold = float(
            self.cfg.get("PERFORMANCE", {}).get(
                "ENROLLMENT_DUPLICATE_SIMILARITY", 0.9995))
        if any(_cosine(embedding, previous) >= duplicate_threshold
               for previous in state["embeddings"]):
            state["rejected_samples"] += 1
            state["message"] = "Sample too similar. Turn slightly or change expression."
            return {"active": True, "stage": "capturing", "accepted": accepted,
                    "minimum": state["min_embeddings"],
                    "maximum": state["max_embeddings"],
                    "quality_score": quality_score, "message": state["message"]}

        state["embeddings"].append(np.asarray(embedding, dtype=np.float32))
        state["quality_scores"].append(float(quality_score))
        state["quality_metrics"].append({"score": float(quality_score)})
        state["pose_yaws"].append(self.anti_spoof.pose.estimate_yaw(
            self.face_analyzer.get_pose_keypoints(faces[0])))
        state["last_capture_at"] = now
        accepted = len(state["embeddings"])
        state["message"] = f"Accepted sample {accepted}/{state['min_embeddings']}."
        if accepted < state["max_embeddings"]:
            return {"active": True, "stage": "capturing", "accepted": accepted,
                    "minimum": state["min_embeddings"], "maximum": state["max_embeddings"],
                    "quality_score": quality_score, "message": state["message"]}

        completed = {
            "active": False,
            "stage": "completed",
            "person_name": state["person_name"],
            "accepted": accepted,
            "minimum": state["min_embeddings"],
            "maximum": state["max_embeddings"],
            "quality_score": quality_score,
            "quality_scores": list(state.get("quality_scores", [])),
            "pose_yaws": list(state.get("pose_yaws", [])),
            "rejected_samples": int(state.get("rejected_samples", 0)),
            "message": f"Collected {accepted} quality-gated samples.",
            "metadata": dict(state.get("metadata") or {}),
            "replace_existing": bool(state.get("replace_existing")),
            "embeddings": list(state["embeddings"]),
        }
        completed["duplicate_candidates"] = self.find_duplicate_identities(
            completed["embeddings"], exclude_name=completed["person_name"])
        self._pending_web_enrollment = completed
        self._last_enrollment_metadata = dict(state.get("metadata") or {})
        self._web_enrollment = None
        if completed["duplicate_candidates"] and not completed["replace_existing"]:
            completed["stage"] = "duplicate_warning"
            completed["message"] = "A similar enrolled identity was found. Confirm to override or cancel."
        return completed

    def find_duplicate_identities(self, embeddings, exclude_name: Optional[str] = None) -> list[dict]:
        """Return existing identities that are too close to new samples."""
        threshold = float(self.cfg.get("PERFORMANCE", {}).get(
            "ENROLLMENT_DUPLICATE_IDENTITY_THRESHOLD", 0.78))
        candidates = []
        vectors = [np.asarray(value, dtype=np.float32) for value in embeddings or [] if value is not None]
        if not vectors:
            return candidates
        for name, record in self.face_db.items():
            if exclude_name and str(name).casefold() == str(exclude_name).casefold():
                continue
            stored = [np.asarray(value, dtype=np.float32) for value in record.get("embeddings", []) if value is not None]
            if not stored:
                continue
            score = max(_cosine(vector, old) for vector in vectors for old in stored)
            if score >= threshold:
                candidates.append({"name": name, "similarity": round(float(score), 4), "threshold": threshold})
        return sorted(candidates, key=lambda item: item["similarity"], reverse=True)

    def enrollment_quality_report(self, person_name: str) -> dict:
        record = self.face_db.get(person_name) or {}
        embeddings = record.get("embeddings", [])
        provenance = record.get("embedding_provenance") or []
        mean_variance, std_variance = _compute_intra_class_variance(embeddings) if len(embeddings) >= 2 else (0.0, 0.0)
        source_count = sum(1 for item in provenance if item.get("kind") == "source")
        augmented_count = sum(1 for item in provenance if item.get("kind") == "augmented")
        quality_scores = [float(item["quality_score"]) for item in provenance
                          if item.get("quality_score") is not None]
        yaw_values = [float(item["yaw"]) for item in provenance
                      if item.get("yaw") is not None]
        pose_buckets = {
            "left" if yaw < -0.08 else "right" if yaw > 0.08 else "forward"
            for yaw in yaw_values
        }
        return {
            "person_name": person_name,
            "sample_count": len(embeddings),
            "source_count": source_count,
            "augmented_count": augmented_count,
            "mean_variance": round(float(mean_variance), 6),
            "std_variance": round(float(std_variance), 6),
            "threshold": float(record.get("threshold", self.cfg.get("FACE_RECOG_THRESHOLD", 0.6))),
            "provenance_count": len(provenance),
            "quality_range": {
                "min": round(min(quality_scores), 2) if quality_scores else None,
                "max": round(max(quality_scores), 2) if quality_scores else None,
                "mean": round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else None,
            },
            "pose_coverage": {
                "observed_buckets": sorted(pose_buckets),
                "distinct_buckets": len(pose_buckets),
            },
            "rejected_samples": int(record.get("rejected_samples", 0)),
        }

    def commit_web_enrollment(self, override_duplicate: bool = False) -> dict:
        pending = self._pending_web_enrollment
        if not pending:
            return {"ok": False, "message": "No completed enrollment is awaiting confirmation."}
        duplicates = pending.get("duplicate_candidates") or []
        if duplicates and not override_duplicate:
            return {"ok": False, "stage": "duplicate_warning", "duplicate_candidates": duplicates,
                    "message": "Confirm duplicate override before saving this identity."}
        name = str(pending["person_name"])
        previous_face_db = self.face_db
        candidate_db = {
            person: {
                **record,
                "embeddings": list(record.get("embeddings", [])),
                "embedding_provenance": list(record.get("embedding_provenance", []) or []),
            }
            for person, record in self.face_db.items()
        }
        if pending.get("replace_existing"):
            candidate_db[name] = {
                "embeddings": [],
                "threshold": self.cfg["FACE_RECOG_THRESHOLD"],
                "embedding_provenance": [],
            }
        self.face_db = candidate_db
        try:
            quality_scores = pending.get("quality_scores") or []
            pose_yaws = pending.get("pose_yaws") or []
            for index, embedding in enumerate(pending.get("embeddings", [])):
                self._append_face_embedding(name, embedding, {
                    "kind": "web_guided",
                    "sample_index": index,
                    "quality_score": quality_scores[index] if index < len(quality_scores) else None,
                    "yaw": pose_yaws[index] if index < len(pose_yaws) else None,
                    "embedding_version": 1,
                })
            record = self.face_db.get(name) or {}
            record["rejected_samples"] = int(pending.get("rejected_samples", 0))
            if len(record.get("embeddings", [])) >= 2:
                mean_variance, std_variance = _compute_intra_class_variance(record["embeddings"])
                record["threshold"] = _auto_threshold(mean_variance, std_variance)
            self._rebuild_index()
            if not self.save_face_db():
                raise RuntimeError("Face database persistence failed; previous identity set restored.")
            report = self.enrollment_quality_report(name)
        except Exception:
            self.face_db = previous_face_db
            self._rebuild_index()
            raise
        report.update({"ok": True, "stage": "completed", "person_name": name,
                       "accepted": len(record.get("embeddings", [])),
                       "rejected_samples": pending.get("rejected_samples", 0),
                       "duplicate_override": bool(duplicates and override_duplicate)})
        self._pending_web_enrollment = None
        return report

    def delete_face_identity(self, person_name: str) -> dict:
        if person_name not in self.face_db:
            return {"ok": False, "message": "Identity is not present in the local face database."}
        self.face_db.pop(person_name, None)
        self._rebuild_index()
        self.save_face_db()
        return {"ok": True, "person_name": person_name, "stage": "deleted"}

    def invalidate_identity(self, person_name: str) -> None:
        """Invalidate cached recognition for a disabled roster identity."""
        target = str(person_name or "").casefold()
        for oid, cache in list(self._recognition_cache.items()):
            cached_name = str(cache.get("identity_name") or cache.get("name") or "").casefold()
            if cached_name == target:
                self._recognition_cache.pop(oid, None)
                self._face_recog_history[oid].clear()
                self._identity_started_at.pop(oid, None)
                self._identity_timing.pop(oid, None)

    def active_identity_names(self) -> list[str]:
        return sorted(str(name) for name in self.face_db)

    def merge_face_identities(self, source_name: str, target_name: str) -> dict:
        source = self.face_db.get(source_name)
        target = self.face_db.get(target_name)
        if not source or not target:
            return {"ok": False, "message": "Both source and target identities must be enrolled."}
        merged = 0
        provenance_items = source.get("embedding_provenance", []) or []
        for index, embedding in enumerate(source.get("embeddings", [])):
            provenance = provenance_items[index] if index < len(provenance_items) else {"kind": "legacy"}
            if self._append_face_embedding(target_name, embedding, {**provenance, "merged_from": source_name}):
                merged += 1
        self.face_db.pop(source_name, None)
        self._rebuild_index()
        self.save_face_db()
        return {"ok": True, "stage": "merged", "source_name": source_name,
                "target_name": target_name, "embeddings_added": merged,
                "quality": self.enrollment_quality_report(target_name)}

    def _append_face_embedding(self, person_name: str, embedding: np.ndarray,
                               provenance: Optional[dict] = None) -> bool:
        """Append a normalized enrollment sample unless it is effectively a duplicate."""
        if embedding is None:
            return False
        vector = np.asarray(embedding, dtype=np.float32).flatten()
        if vector.size == 0 or not np.isfinite(vector).all():
            return False
        if person_name not in self.face_db:
            self.face_db[person_name] = {"embeddings": [],
                                          "threshold": self.cfg["FACE_RECOG_THRESHOLD"]}
        record = self.face_db[person_name]
        embs = record.setdefault("embeddings", [])
        metadata = record.get("embedding_provenance")
        if not isinstance(metadata, list):
            metadata = []
            record["embedding_provenance"] = metadata
        while len(metadata) < len(embs):
            metadata.insert(0, {"kind": "legacy"})
        duplicate_threshold = float(
            self.cfg.get("PERFORMANCE", {}).get(
                "ENROLLMENT_DUPLICATE_SIMILARITY", 0.9995))
        if any(_cosine(vector, sample) >= duplicate_threshold for sample in embs):
            return False
        max_embeddings = max(1, int(self.cfg.get("MAX_ENROLLMENT_EMBEDDINGS", 10)))
        if len(embs) >= max_embeddings:
            embs.pop(0)
            if metadata:
                metadata.pop(0)
        embs.append(vector)
        metadata.append(dict(provenance or {"kind": "runtime"}))
        return True

    def add_embedding_for_name(self, person_name: str, embedding: np.ndarray,
                               rebuild: bool = True, persist: bool = True) -> bool:
        if not self._append_face_embedding(person_name, embedding):
            return False
        embs = self.face_db[person_name]["embeddings"]
        if len(embs) >= 2:
            mv, sv = _compute_intra_class_variance(embs)
            self.face_db[person_name]["threshold"] = _auto_threshold(mv, sv)
        if rebuild:
            self._rebuild_index()
        if persist:
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
    def _draw_face_box(frame, x1, y1, x2, y2, name, conf, oid, is_real,
                       identity_state=None, confidence_label="similarity"):
        if not is_real: color = (0, 0, 255)
        elif name == "UNKNOWN" or name.startswith("STRANGER_"): color = (0, 165, 255)
        else: color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} ({confidence_label}={conf:.2f}) ID:{oid}"
        if identity_state and identity_state != "CONFIRMED":
            label += f" [{identity_state}]"
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

    def _recognize_and_draw(self, frame, face_objects, tracked,
                            recognition_frame=None, recognition_scale=None):
        recognition_frame = recognition_frame if recognition_frame is not None else frame
        if recognition_scale is None:
            processing_h, processing_w = frame.shape[:2]
            source_h, source_w = recognition_frame.shape[:2]
            recognition_scale = (
                float(source_w) / max(1, processing_w),
                float(source_h) / max(1, processing_h),
            )
        info = []
        for fobj in face_objects:
            x1, y1, x2, y2 = self.face_analyzer.get_bbox(fobj)
            recognition_fobj = self.face_analyzer.scale_face(
                fobj, recognition_scale[0], recognition_scale[1])
            rx1, ry1, rx2, ry2 = self.face_analyzer.get_bbox(recognition_fobj)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            oid = -1; best_d = float("inf")
            for t_oid, t_c in tracked.items():
                d = math.hypot(cx - t_c[0], cy - t_c[1])
                if d < best_d and d < 150: best_d, oid = d, t_oid

            quality_score, quality_metrics, quality_ok = self._face_quality(
                recognition_frame, (rx1, ry1, rx2, ry2))
            if not quality_ok:
                self._face_quality_rejects += 1
            # Embedding generation is deliberately deferred until the track
            # policy below proves that fresh biometric evidence is needed.
            embedding = None

            spoof_status = AntiSpoofDetector.REAL
            spoof_details = {}
            spoof_confirmed = False
            landmarks = self.face_analyzer.get_landmarks(recognition_fobj)
            pose_keypoints = self.face_analyzer.get_pose_keypoints(recognition_fobj)
            yaw = self.anti_spoof.pose.estimate_yaw(pose_keypoints)

            if (self.anti_spoof.enabled and
                    self.cfg.get("ENABLE_LIVENESS_INFERENCE", True)):
                crop = recognition_frame[max(0, ry1):min(recognition_frame.shape[0], ry2),
                                          max(0, rx1):min(recognition_frame.shape[1], rx2)]
                if quality_ok:
                    spoof_status, spoof_details = self._profile_call(
                        "liveness", self.anti_spoof.analyze,
                        crop, recognition_frame.shape, landmarks=landmarks,
                        bbox=(rx1, ry1, rx2, ry2), pose_keypoints=pose_keypoints,
                        track_id=oid if oid > 0 else None)
                else:
                    spoof_status = AntiSpoofDetector.UNCERTAIN
                    spoof_details = {
                        "reason": "low_face_quality",
                        "quality_score": round(quality_score, 2),
                    }
            if yaw is not None:
                spoof_details["yaw"] = yaw

            if oid > 0:
                self._spoof_history[oid].append(spoof_status)
                needed = self.cfg["ANTI_SPOOFING"].get("SUSPECT_CONFIRM_FRAMES", 4)
                recent_suspects = sum(
                    1 for s in self._spoof_history[oid]
                    if s == AntiSpoofDetector.SUSPECT
                )
                spoof_confirmed = recent_suspects >= needed

            uncertain_blocks_attendance = bool(
                self.cfg["ANTI_SPOOFING"].get("UNCERTAIN_BLOCKS_ATTENDANCE", True)
            )
            liveness_blocks_attendance = uncertain_blocks_attendance and spoof_status in {
                AntiSpoofDetector.UNCERTAIN,
                AntiSpoofDetector.SUSPECT,
            }
            is_real = not spoof_confirmed
            attendance_eligible = (
                not spoof_confirmed
                and not liveness_blocks_attendance
                and quality_ok
            )

            if spoof_confirmed:
                self._draw_face_box(frame, x1, y1, x2, y2, "SPOOF", 0.0, oid, False)
                info.append({"oid": oid, "name": "SPOOF", "confidence": 0.0,
                             "bbox": (x1, y1, x2, y2), "is_real": False,
                             "attendance_eligible": False,
                             "spoof_status": spoof_status,
                             "spoof_details": spoof_details,
                             "yaw": yaw,
                             "identity_state": "SPOOF_SUSPECT",
                             "quality_score": quality_score,
                             "quality_ok": quality_ok,
                             "evidence_due": oid > 0})
                continue

            perf = self.cfg.get("PERFORMANCE", {})
            cache_ttl = float(perf.get("RECOGNITION_CACHE_TTL_SEC", 4.0))
            refresh_sec = float(perf.get("FACE_RECOG_REFRESH_SEC", 2.5))
            movement_limit = float(perf.get("FACE_RECOG_REFRESH_MOVEMENT_PX", 36.0))
            stable_required = max(1, int(perf.get("FACE_RECOG_STABLE_FRAMES", 3)))
            unresolved_every = float(perf.get("RECOGNITION_UNRESOLVED_EVERY_SEC", 0.20))
            candidate_every = float(perf.get("RECOGNITION_CANDIDATE_EVERY_SEC", 0.30))
            contradiction_limit = max(1, int(
                perf.get("RECOGNITION_CONTRADICTION_CONFIRMATIONS", 2)))
            max_attendance_age = float(
                perf.get("MAX_ATTENDANCE_IDENTITY_AGE_SEC", refresh_sec))
            cached = self._recognition_cache.get(oid) if oid > 0 else None
            now = time.time()
            cache_similarity = None
            second_score = float((cached or {}).get("second_score", 0.0))
            margin = float((cached or {}).get("margin", 0.0))
            evidence_interval = float(perf.get(
                "RECOGNITION_EVIDENCE_MIN_INTERVAL_SEC", 0.75))
            timing = None
            if oid > 0:
                self._identity_started_at.setdefault(oid, now)
                timing = self._identity_timing.setdefault(oid, {"started_at": now})
                if quality_ok and "first_usable_at" not in timing:
                    timing["first_usable_at"] = now
                    self._identity_timing_samples["first_usable_face"].append(
                        max(0.0, now - timing["started_at"]))
            cache_age = now - float((cached or {}).get("last_observation_at",
                                                        (cached or {}).get("ts", 0.0)))
            cached_name = str((cached or {}).get("identity_name") or
                              (cached or {}).get("name") or "UNKNOWN")
            identity_state = normalize_identity_state(
                (cached or {}).get("identity_state") or "UNRESOLVED")
            previous_state = identity_state
            cached_known = (
                cached is not None
                and cached_name not in ("UNKNOWN", "SPOOF")
                and not cached_name.startswith("STRANGER_")
                and not self._is_identity_disabled(cached_name)
            )
            stable_frames = int((cached or {}).get("stable_frames", 0))
            candidate_hits = int((cached or {}).get("candidate_hits", 0))
            unknown_streak = int((cached or {}).get("unknown_streak", 0))
            identity_name = cached_name if cached_known else None
            last_verified_at = float((cached or {}).get("last_verified_at", 0.0))
            verification_age = now - last_verified_at if last_verified_at else float("inf")
            movement = self._bbox_motion((cached or {}).get("bbox"),
                                          (x1, y1, x2, y2))
            stable_cache_valid = (
                cached_known
                and identity_state == "CONFIRMED"
                and stable_frames >= stable_required
                and cache_age <= min(cache_ttl, refresh_sec)
                and verification_age <= refresh_sec
                and movement <= movement_limit
            )
            quality_hold_valid = (
                cached_known
                and identity_state in {"CONFIRMED", "OCCLUDED", "CANDIDATE"}
                and cache_age <= cache_ttl
            )
            identity_age_valid = (
                cached_known and last_verified_at > 0
                and verification_age <= max_attendance_age
            )
            attempt_interval = (
                refresh_sec if identity_state == "CONFIRMED" else
                candidate_every if identity_state == "CANDIDATE" else
                unresolved_every
            )
            last_attempt_at = float((cached or {}).get("last_attempt_at", 0.0))
            attempt_due = (
                not last_attempt_at
                or now - last_attempt_at >= max(0.01, attempt_interval)
            )
            recognition_due = (
                cached is None
                or not cached_known
                or cache_age > cache_ttl
                or verification_age > refresh_sec
                or movement > movement_limit
                or attempt_due
            )

            if stable_cache_valid and quality_ok:
                name = cached_name
                conf = float(cached.get("cached_identity_confidence",
                                      cached.get("conf", 0.0)))
                embedding = cached.get("embedding")
                reason = "stable_track_cache" if quality_ok else "quality_hold"
                self._recognition_cache_hits += 1
                self._embeddings_skipped_due_to_cache += 1
            elif not quality_ok:
                if quality_hold_valid:
                    name, conf = "UNKNOWN", 0.0
                    reason = "WAITING_FOR_GOOD_FACE"
                    identity_state = "OCCLUDED" if identity_state == "CONFIRMED" else identity_state
                else:
                    name, conf, reason = "UNKNOWN", 0.0, (
                        f"WAITING_FOR_GOOD_FACE quality={quality_score:.1f}")
                    identity_state = "UNRESOLVED"
                self._embeddings_skipped_due_to_quality += 1
                self._recognition_cache_misses += 1
            elif recognition_due:
                if timing is not None and "first_embedding_at" not in timing:
                    timing["first_embedding_at"] = now
                    self._identity_timing_samples["first_embedding"].append(
                        max(0.0, now - timing["started_at"]))
                embedding = self._profile_call(
                    "face_embedding", self.face_analyzer.get_embedding,
                    recognition_fobj, recognition_frame)
                self._recognition_attempts += 1
                previous_identity = cached_name if cached_known else None
                if cached_known and identity_state == "CONFIRMED":
                    name, conf, reason = self._profile_call(
                        "identity_matching", self.recognize_face, embedding)
                else:
                    name, conf, reason = self._profile_call(
                        "identity_smoothing", self._smooth_recognize, oid, embedding)

                match_details = self._last_recognition_details or {}
                second_score = float(match_details.get("second_score", second_score))
                margin = float(match_details.get("margin", margin))

                cached_embedding = (cached or {}).get("embedding")
                if cached_known and cached_embedding is not None:
                    cache_similarity = _cosine(embedding, cached_embedding)
                track_consistent = (
                    not cached_known
                    or cache_similarity is None
                    or cache_similarity >= float(perf.get(
                        "RECOGNITION_CACHE_MIN_SIMILARITY", 0.45))
                )
                fresh_known = (
                    name not in ("UNKNOWN", "SPOOF")
                    and not str(name).startswith("STRANGER_")
                    and track_consistent
                    and not self._is_identity_disabled(name)
                )
                if self._is_identity_disabled(name):
                    name, conf, reason = "UNKNOWN", 0.0, "identity_disabled"
                if cached_known and not track_consistent:
                    name, conf, reason = "UNKNOWN", 0.0, "track_identity_changed"
                    identity_state = "CONTRADICTED"
                    identity_name = None
                    unknown_streak = contradiction_limit
                if fresh_known:
                    unknown_streak = 0
                    candidate_hits = candidate_hits + 1 if name == previous_identity else 1
                    stable_frames = stable_frames + 1 if name == previous_identity else 1
                    identity_state = (
                        "CONFIRMED" if candidate_hits >= stable_required else "CANDIDATE")
                    identity_name = name
                    if (identity_state == "CANDIDATE" and timing is not None
                            and "candidate_at" not in timing):
                        timing["candidate_at"] = now
                        self._identity_timing_samples["candidate"].append(
                            max(0.0, now - timing["started_at"]))
                    accepted_at = now if identity_state == "CONFIRMED" else last_verified_at
                    if identity_state == "CONFIRMED" and previous_state != "CONFIRMED":
                        self._recognition_confirmations += 1
                        started_at = self._identity_started_at.pop(oid, now)
                        self._identity_confirmation_times.append(max(0.0, now - started_at))
                        if timing is not None:
                            self._identity_timing.pop(oid, None)
                    if identity_state == "CONFIRMED":
                        last_verified_at = accepted_at
                else:
                    candidate_hits = 0
                    stable_frames = 0
                    unknown_streak = unknown_streak + 1 if cached_known else 0
                    if cached_known and unknown_streak < contradiction_limit:
                        identity_state = "OCCLUDED"
                        identity_name = cached_name
                    else:
                        identity_state = "CONTRADICTED" if cached_known else "UNRESOLVED"
                        identity_name = None
                        name = "UNKNOWN"
                        conf = 0.0

                if oid > 0:
                    self._recognition_cache[oid] = {
                        "name": name,
                        "identity_name": identity_name,
                        "identity_state": identity_state,
                        "conf": conf,
                        "cached_identity_confidence": conf if fresh_known else float(
                            (cached or {}).get("cached_identity_confidence", 0.0)),
                        "current_observation_similarity": conf if fresh_known else 0.0,
                        "second_score": second_score,
                        "margin": margin,
                        "cache_similarity": cache_similarity,
                        "reason": reason,
                        "embedding": embedding,
                        "bbox": (x1, y1, x2, y2),
                        "stable_frames": stable_frames,
                        "candidate_hits": candidate_hits,
                        "unknown_streak": unknown_streak,
                        "quality_score": quality_score,
                        "last_attempt_at": now,
                        "last_observation_at": now,
                        "last_verified_at": last_verified_at if identity_state == "CONFIRMED" else (
                            last_verified_at if cached_known else 0.0),
                        "ts": now,
                    }
                self._recognition_cache_misses += 1
            elif cached:
                name = cached.get("name", "UNKNOWN")
                conf = float(cached.get("cached_identity_confidence", 0.0))
                reason = cached.get("reason", "cached_identity")
                if self._is_identity_disabled(name):
                    name, conf, reason = "UNKNOWN", 0.0, "identity_disabled"
                embedding = cached.get("embedding")
                second_score = float(cached.get("second_score", second_score))
                margin = float(cached.get("margin", margin))
                self._recognition_cache_hits += 1
                self._embeddings_skipped_due_to_cache += 1
            else:
                name, conf, reason = "UNKNOWN", 0.0, "no_recognition_cache"
                self._recognition_cache_misses += 1

            if name == "UNKNOWN" and embedding is not None and self.cfg.get("STRANGER_TRACKING_ENABLED", True):
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
            last_verified_at = float((self._recognition_cache.get(oid) or {}).get(
                "last_verified_at", last_verified_at))
            identity_age_valid = (
                identity_state == "CONFIRMED"
                and last_verified_at > 0
                and time.time() - last_verified_at <= max_attendance_age
            )
            attendance_eligible = (
                attendance_eligible
                and identity_state == "CONFIRMED"
                and identity_age_valid
            )
            evidence_due = bool(
                oid > 0 and (
                    identity_state != previous_state
                    or reason not in {"stable_track_cache", "cached_identity"}
                ) and now - self._last_evidence_at.get(oid, 0.0) >= evidence_interval
            )
            if evidence_due:
                self._last_evidence_at[oid] = now
            self._current_face_labels[oid] = (name, conf, embedding)
            self._draw_face_box(
                frame, x1, y1, x2, y2, name, conf, oid, is_real,
                identity_state=identity_state,
                confidence_label=("last" if reason in {
                    "stable_track_cache", "quality_hold"} else "similarity"))
            age = None
            gender = None
            if (self.cfg.get("SHOW_AGE_GENDER", False)
                    and self.cfg.get("ENABLE_AGE_GENDER_INFERENCE", False)):
                age, gender = self._profile_call(
                    "age_gender",
                    lambda: (self.face_analyzer.get_age(recognition_fobj) or "?",
                             self.face_analyzer.get_gender(recognition_fobj)))
                emotion = "neutral"
                if oid > 0:
                    self._draw_person_info(frame, fobj, x1, y2, oid, age, gender, emotion)

            info.append({"oid": oid, "name": name, "confidence": conf,
                         "bbox": (x1, y1, x2, y2), "is_real": is_real,
                         "attendance_eligible": attendance_eligible,
                         "quality_score": quality_score,
                         "quality_ok": quality_ok,
                         "quality_metrics": quality_metrics,
                         "reason": reason, "age": age, "gender": gender,
                         "yaw": yaw, "liveness_status": spoof_status,
                         "liveness_details": spoof_details,
                         "identity_state": identity_state,
                         "cached_identity_confidence": float(
                             (self._recognition_cache.get(oid) or {}).get(
                                 "cached_identity_confidence", conf)),
                          "current_observation_similarity": 0.0 if reason in {
                              "stable_track_cache", "quality_hold"} else float(
                                  (self._recognition_cache.get(oid) or {}).get(
                                      "current_observation_similarity", conf)),
                           "second_score": float((self._recognition_cache.get(oid) or {}).get(
                               "second_score", second_score)),
                           "margin": float((self._recognition_cache.get(oid) or {}).get(
                               "margin", margin)),
                           "candidate_hits": candidate_hits,
                           "stable_frames": stable_frames,
                           "last_verified_at": last_verified_at,
                          "evidence_due": evidence_due})
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
                                   f"ID_{oid} near {d['class_name']}",
                                   {"track_id": oid, "object_class": d["class_name"]}))
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

    def process(self, frame, original_frame=None, source_frame_id=None,
                camera_id="cam_0"):
        t0 = time.perf_counter()
        stage_started = t0
        stage_timings = {}

        def mark_stage(name):
            nonlocal stage_started
            now = time.perf_counter()
            stage_timings[name] = round((now - stage_started) * 1000.0, 3)
            stage_started = now

        self._last_stage_timings = {}
        events: List[tuple] = []
        h, w = frame.shape[:2]
        self.behavior.set_frame_size(h, w)
        # Detectors only read their input; keep one analysis buffer and one
        # drawing buffer instead of copying the frame twice every iteration.
        analysis = frame
        display = frame.copy()
        self._frame_count += 1
        frame_observed_at = time.monotonic()
        frame_observed_wallclock = _runtime_now()
        perf = self.cfg.get("PERFORMANCE", {})
        face_every = max(1, int(perf.get("FACE_DETECT_EVERY_N_FRAMES", 2)))
        face_due = self._scheduler.should_run(
            "face_detect", self._frame_count, face_every,
            has_signal=bool(self._last_face_objects),
            force=not bool(self._last_face_objects),
        )
        if face_due:
            try:
                face_objects = self._profile_call(
                    "face_detection", self.face_analyzer.detect, analysis)
                self._last_face_objects = face_objects
                self._last_face_observation_frame_id = source_frame_id
                self._last_face_observation_at = frame_observed_at
                self._last_face_observation_wallclock = frame_observed_wallclock
            except Exception as e:
                print(f"[ERROR] face detect: {e}")
                face_objects = self._last_face_objects or []
            self._scheduler.complete(
                "face_detect", self._frame_count, face_every,
                has_signal=bool(face_objects),
            )
        else:
            face_objects = self._last_face_objects
        mark_stage("face_detection")

        object_detections = []
        yolo_every = max(1, int(perf.get("YOLO_EVERY_N_FRAMES", 3)))

        if self.object_detector is not None and self.cfg.get(
                "ENABLE_OBJECT_DETECTION", True):
            yolo_due = self._scheduler.should_run(
                "yolo", self._frame_count, yolo_every,
                has_signal=bool(self._last_object_detections),
                force=not bool(self._last_object_detections),
            )
            if yolo_due:
                object_detections = self._profile_call(
                    "yolo", self.object_detector.detect, analysis)
                self._last_object_detections = object_detections
                self._last_object_observation_frame_id = source_frame_id
                self._last_object_observation_at = frame_observed_at
                self._last_object_observation_wallclock = frame_observed_wallclock
                self._scheduler.complete(
                    "yolo", self._frame_count, yolo_every,
                    has_signal=bool(object_detections),
                )
            else:
                object_detections = list(self._last_object_detections)

        supplementary_due = self._scheduler.should_run(
            "supplementary", self._frame_count, 3,
            has_signal=False,
            force=self._frame_count == 1,
        )
        if self.cfg.get("ENABLE_FIRE_SMOKE_HEURISTICS", False) and supplementary_due:
            object_detections.extend(self.supplementary.detect_all(analysis))
            self._scheduler.complete("supplementary", self._frame_count, 3,
                                     has_signal=bool(object_detections))

        if (self.danger_detector is not None
                and self.cfg.get("ENABLE_DANGER_INFERENCE", True)
                and self.cfg.get("DANGER_DETECTION", {}).get("ENABLED", False)):
            danger_cfg = self.cfg.get("DANGER_DETECTION", {})
            danger_every = max(1, int(danger_cfg.get("EVERY_N_FRAMES", 10)))
            allowed = {c.lower() for c in danger_cfg.get("ALLOWED_CLASSES", set())}
            min_conf_by_class = {
                str(k).lower(): float(v)
                for k, v in danger_cfg.get("MIN_CONF_BY_CLASS", {}).items()
            }

            danger_due = self._scheduler.should_run(
                "danger", self._frame_count, danger_every,
                has_signal=bool(self._last_danger_detections),
                force=not bool(self._last_danger_detections),
            )
            if danger_due:
                raw_danger_dets = self._profile_call(
                    "danger", self.danger_detector.detect, analysis)
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
                self._last_object_observation_frame_id = source_frame_id
                self._last_object_observation_at = frame_observed_at
                self._last_object_observation_wallclock = frame_observed_wallclock
                self._scheduler.complete(
                    "danger", self._frame_count, danger_every,
                    has_signal=bool(danger_dets),
                )
            else:
                danger_dets = list(self._last_danger_detections)

            object_detections.extend(danger_dets)

        mark_stage("objects_and_danger")

        roi = self._relevant_roi(analysis, object_detections, face_objects)
        hand_every = max(1, int(perf.get("HAND_EVERY_N_FRAMES", 4)))
        hand_enabled = bool(self.cfg.get("ENABLE_HAND_INFERENCE", True))
        hand_due = hand_enabled and self._scheduler.should_run(
            "hand", self._frame_count, hand_every,
            has_signal=bool(self._last_hand_dets),
            force=not bool(self._last_hand_dets),
        )
        if hand_due:
            hand_input = analysis
            if roi is not None:
                x1, y1, x2, y2 = roi
                hand_input = analysis[y1:y2, x1:x2]
                self._roi_inference_runs += 1
            self._last_hand_dets = self._profile_call(
                "hands", self.hand_detector.detect, hand_input, roi=roi)
            self._scheduler.complete(
                "hand", self._frame_count, hand_every,
                has_signal=bool(self._last_hand_dets),
            )
        if not hand_enabled:
            self._last_hand_dets = []
        hand_dets = self._last_hand_dets

        if self.cfg.get("SHOW_HAND_LANDMARKS", False):
            self.hand_detector.draw(display, hand_dets)
        mark_stage("hands_and_overlays")

        if self.cfg.get("SHOW_OBJECT_BOXES", True) and self.object_detector is not None:
            self.object_detector.draw_detections(display, object_detections, skip_person=True)
        
        if (self.custom_objects is not None
                and self.custom_objects.enabled
                and hand_enabled
                and self.cfg.get("ENABLE_CUSTOM_OBJECT_INFERENCE", True)):
            custom_cfg = self.cfg.get("CUSTOM_OBJECTS", {})
            custom_every = max(1, int(custom_cfg.get("MATCH_EVERY_N_FRAMES", 5)))

            custom_due = self._scheduler.should_run(
                "custom", self._frame_count, custom_every,
                has_signal=bool(self._last_custom_object_detections),
                force=not bool(self._last_custom_object_detections),
            )
            if custom_due:
                custom_dets = self._profile_call(
                    "custom_objects",
                    self.custom_objects.detect_from_hands,
                    analysis, hand_dets)
                self._last_custom_object_detections = custom_dets
                self._scheduler.complete(
                    "custom", self._frame_count, custom_every,
                    has_signal=bool(custom_dets),
                )
            else:
                custom_dets = list(self._last_custom_object_detections)

            object_detections.extend(custom_dets)

        person_rects = [d["bbox"] for d in object_detections if d["class_name"] == "person"]
        tracked = self.person_tracker.update(person_rects)

        self._extract_shirt_colors(analysis, object_detections, tracked)

        events.extend(self.behavior.update(tracked))
        events.extend(self.crowd_intel.update(tracked, frame_size=(h, w)))
        # SecuritySignalEngine owns debounced operational versions of these
        # signals. Behavior/Crowd still update their internal analytics, but
        # their duplicate tuples must not reach the persistence/alert path.
        events = [event for event in events if str(event[0]) not in {
            "LOITERING", "RUNNING", "EVACUATION_ALERT",
        }]
        mark_stage("tracking_and_behavior")

        pose_result = self._last_pose_result
        try:
            pose_enabled = bool(self.cfg.get("ENABLE_POSE_INFERENCE", True))
            if not pose_enabled:
                self._last_pose_result = {}
                self._last_pose_observation_frame_id = None
                self._last_pose_observation_at = None
                self._last_pose_observation_wallclock = None
            else:
                pose_every = max(1, int(perf.get("POSE_EVERY_N_FRAMES", 4)))
                pose_due = self._scheduler.should_run(
                    "pose", self._frame_count, pose_every,
                    has_signal=bool(self._last_pose_result.get("pose")),
                    force=not bool(self._last_pose_result),
                )
                if pose_due:
                    pose_roi = self._relevant_roi(
                        analysis, object_detections, face_objects, tracked)
                    if pose_roi is not None:
                        self._roi_inference_runs += 1
                    self._last_pose_result = self._profile_call(
                        "pose", self.pose_detector.analyze, analysis, roi=pose_roi)
                    self._last_pose_observation_frame_id = source_frame_id
                    self._last_pose_observation_at = frame_observed_at
                    self._last_pose_observation_wallclock = frame_observed_wallclock
                    self._scheduler.complete(
                        "pose", self._frame_count, pose_every,
                        has_signal=bool(self._last_pose_result.get("pose")),
                    )
                pose_result = self._last_pose_result
                if pose_result.get("is_fallen"):
                    events.append(("FALL_DETECTED", "PERSON", 0.85,
                                   "Possible fall: torso horizontal"))
                if pose_result.get("hands_raised"):
                    events.append(("HANDS_RAISED", "PERSON", 0.8,
                                   "Both hands raised above head"))
                if self.cfg.get("SHOW_POSE_LANDMARKS", False):
                    self.pose_detector.draw(display, pose_result)
        except Exception as e:
            print(f"[WARN] pose analysis failed: {e}")
        mark_stage("pose")

        recognition_frame = original_frame if original_frame is not None else frame
        processing_h, processing_w = frame.shape[:2]
        recognition_h, recognition_w = recognition_frame.shape[:2]
        recognition_scale = (
            float(recognition_w) / max(1, processing_w),
            float(recognition_h) / max(1, processing_h),
        )
        faces_info = self._recognize_and_draw(
            display, face_objects, tracked,
            recognition_frame=recognition_frame,
            recognition_scale=recognition_scale)
        mark_stage("recognition_and_liveness")

        correlation_state = {}
        if self.cfg.get("CORRELATION_CORE", {}).get("ENABLED", True):
            correlation_state = self.correlation.update(
                tracked=tracked,
                faces_info=faces_info,
                object_detections=object_detections,
                pose_result=pose_result,
                source_frame_id=source_frame_id,
                camera_id=camera_id,
                observed_at_monotonic=frame_observed_at,
                observed_at_wallclock=frame_observed_wallclock,
                provenance={
                    "faces": {
                        "frame_id": self._last_face_observation_frame_id,
                        "monotonic": self._last_face_observation_at,
                        "wallclock": self._last_face_observation_wallclock,
                    },
                    "objects": {
                        "frame_id": self._last_object_observation_frame_id,
                        "monotonic": self._last_object_observation_at,
                        "wallclock": self._last_object_observation_wallclock,
                    },
                    "pose": {
                        "frame_id": self._last_pose_observation_frame_id,
                        "monotonic": self._last_pose_observation_at,
                        "wallclock": self._last_pose_observation_wallclock,
                    },
                },
                # Behavior, crowd, pose, and other pre-correlation signals are
                # reduced first. Zone policy is evaluated in a second stage
                # after canonical identity state is available.
                security_events=events,
            )
            events = list(correlation_state.get("security_event_tuples") or [])
        self._last_correlation_state = correlation_state
        entity_by_track = {
            int(entity["track_id"]): entity
            for entity in correlation_state.get("entities", [])
            if entity.get("track_id") is not None
        }

        # The correlation reducer is the decision boundary. Model output can
        # still be shown for diagnostics, but downstream attendance must use
        # the entity's canonical identity/liveness/quality state.
        for fi in faces_info:
            entity = entity_by_track.get(int(fi.get("oid", -1)), {})
            identity = entity.get("identity", {}) or {}
            face_state = normalize_identity_state(identity.get("state"))
            canonical_name = str(identity.get("confirmed_name") or "").strip()
            if face_state == "CONFIRMED" and canonical_name:
                fi["name"] = canonical_name
            elif face_state == "SPOOF_SUSPECT":
                fi["name"] = "SPOOF"
            else:
                fi["name"] = "UNKNOWN"
            fi["identity_state"] = face_state
            fi["attendance_eligible"] = bool(entity.get("attendance_eligibility", False))
            fi["liveness_status"] = (entity.get("liveness", {}) or {}).get(
                "state", fi.get("liveness_status", "NOT_EVALUATED"))
            fi["quality_ok"] = bool((entity.get("face", {}) or {}).get(
                "quality_ok", fi.get("quality_ok", False)))
            fi["correlation_reason"] = (
                "authoritative_entity_state" if entity else "entity_not_found")

        # Security policy consumes canonical entity state. This prevents a
        # stale raw recognition label from authorizing a restricted-zone
        # presence before the correlation reducer has checked continuity,
        # quality, liveness, and contradiction state.
        security_events = self.security_signals.update(
            tracked,
            faces_info=faces_info,
            object_detections=object_detections,
            now=frame_observed_at,
            wallclock=frame_observed_wallclock,
        )
        if self.cfg.get("CORRELATION_CORE", {}).get("ENABLED", True):
            correlated_security = self.correlation.correlate_security_events(
                security_events,
                source_frame_id=source_frame_id,
                camera_id=camera_id,
                observed_at_monotonic=frame_observed_at,
                observed_at_wallclock=frame_observed_wallclock,
            )
            events.extend(correlated_security.get("security_event_tuples") or [])
            correlation_state.setdefault("security_events", []).extend(
                correlated_security.get("security_events") or [])
            correlation_state["stats"] = self.correlation.stats()
        else:
            events.extend(security_events)

        self._last_faces_seen = []
        for fi in faces_info:
            oid = fi.get("oid", -1)
            entity = entity_by_track.get(oid, {})
            fi["entity_id"] = entity.get("entity_id")
            fi["identity_age_ms"] = entity.get("identity", {}).get("identity_age_ms")
            fi["source_frame_id"] = source_frame_id
            fi["observed_at"] = frame_observed_wallclock
            shirt_color = None
            if oid in self._person_colors:
                r, g, b = self._person_colors[oid]
                shirt_color = {"rgb": [int(r), int(g), int(b)]}

            self._last_faces_seen.append({
                "entity_id": entity.get("entity_id"),
                "track_id": oid,
                "name": fi.get("name"),
                "confidence": float(fi.get("confidence", 0.0)),
                "is_real": bool(fi.get("is_real", True)),
                "attendance_eligible": bool(fi.get("attendance_eligible", fi.get("is_real", True))),
                "yaw": fi.get("yaw"),
                "liveness_status": fi.get("liveness_status"),
                "quality_score": float(fi.get("quality_score", 0.0)),
                "quality_ok": bool(fi.get("quality_ok", False)),
                "identity_state": fi.get("identity_state", "UNRESOLVED"),
                "correlation_reason": fi.get("correlation_reason"),
                "cached_identity_confidence": float(
                    fi.get("cached_identity_confidence", 0.0)),
                "current_observation_similarity": float(
                    fi.get("current_observation_similarity", 0.0)),
                "second_score": float(fi.get("second_score", 0.0)),
                "margin": float(fi.get("margin", 0.0)),
                "candidate_hits": int(fi.get("candidate_hits", 0)),
                "stable_frames": int(fi.get("stable_frames", 0)),
                "last_verified_at": fi.get("last_verified_at"),
                "identity_age_ms": fi.get("identity_age_ms"),
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
        "correlation": correlation_state,
        "security_capabilities": self.security_signals.capability_state(),
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
        for oid in list(self._recognition_cache.keys()):
            if oid > 0 and oid not in active_ids:
                self.anti_spoof.forget_track(oid)
                self._recognition_cache.pop(oid, None)
                self._identity_started_at.pop(oid, None)
                self._identity_timing.pop(oid, None)
                self._last_evidence_at.pop(oid, None)

        self._draw_ui(display, events, tracked, object_detections)
        mark_stage("rendering")
        stage_timings["total"] = round((time.perf_counter() - t0) * 1000.0, 3)
        self._last_stage_timings = stage_timings
        dt = time.perf_counter() - t0
        self._scheduler.record_total_ms(stage_timings["total"])
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
        self._unregistered_logged: set = set()
        self._announce = self.cfg.get("ANNOUNCE_ARRIVAL", True)
        self._date = _today_iso()
        self._last_reconcile = 0.0

    def _maybe_rollover(self):
        today = _today_iso()
        if today != self._date:
            try:
                self.db.close_open_attendance_before(today)
                self.db.close_open_presence_sessions("day_rollover")
            except Exception as exc:
                print(f"[ATTENDANCE] Rollover reconciliation failed: {exc}")
            self._today_clocked_in.clear()
            self._unregistered_logged.clear()
            self._date = today

    def handle_recognition(self, person_name: str, camera_id: str = "cam_0", location: str = None,
                           confidence: float = 0.0, presence_session_id=None,
                           recognition_evidence_id=None, source_frame_id=None,
                           liveness_status=None, identity_state="UNRESOLVED",
                           attendance_eligible=False, quality_ok=False,
                           authoritative=True):
        if not self.enabled: return
        if person_name in ("UNKNOWN", "SPOOF") or person_name.startswith("STRANGER_"):
            return
        if (normalize_identity_state(identity_state) != "CONFIRMED"
                or str(liveness_status or "").upper() != "REAL"
                or not attendance_eligible or not quality_ok):
            return
        self._maybe_rollover()
        if not _is_school_day(_local_datetime()):
            return
        now = time.time()
        pid = self.db.get_active_person_id(person_name)
        if pid is None:
            if person_name not in self._unregistered_logged:
                self.db.log_audit(
                    "ATTENDANCE_REJECTED_UNREGISTERED",
                    person_name,
                    {"reason": "recognized label is not an active roster person"},
                )
                self._unregistered_logged.add(person_name)
            return
        self.db.touch_attendance_presence(pid, _utc_now(), presence_session_id)
        if person_name in self._today_clocked_in:
            return
        if authoritative:
            result = self.db.attendance_clock_in(
                pid, camera_id, location, confidence=confidence,
                presence_session_id=presence_session_id,
                recognition_evidence_id=recognition_evidence_id,
                source_frame_id=source_frame_id,
                identity_state="CONFIRMED",
                liveness_status=liveness_status or "REAL",
                decision_source="automatic_correlated")
            if "clocked_in_at" in result or result.get("already_clocked_in"):
                self._today_clocked_in.add(person_name)
            if "clocked_in_at" in result:
                late = result.get("late_minutes", 0)
                if self._announce:
                    message = f"Welcome {person_name}."
                    if late > 0:
                        message = f"Welcome {person_name}. You are {late} minutes late."
                    voice(message, "INFO", dedup_key=f"clockin:{person_name}")
                self.db.log_event(
                    "ATTENDANCE_CLOCKIN", person_id=pid, confidence=confidence,
                    details={"late_min": late, "source": "correlation_core"},
                    camera_id=camera_id, location=location, severity=0,
                    presence_session_id=presence_session_id,
                    source_frame_id=source_frame_id,
                    observation_type="ATTENDANCE_DECISION")
            return
        self._recognitions[person_name].append(now)
        # purge old
        win = self.cfg.get("RECOGNITION_WINDOW_SEC", 5.0)
        while (self._recognitions[person_name]
               and now - self._recognitions[person_name][0] > win):
            self._recognitions[person_name].popleft()
        min_frames = self.cfg.get("MIN_RECOGNITION_FRAMES", 8)
        if (len(self._recognitions[person_name]) >= min_frames
                and person_name not in self._today_clocked_in):
            result = self.db.attendance_clock_in(
                pid, camera_id, location, confidence=confidence,
                presence_session_id=presence_session_id,
                recognition_evidence_id=recognition_evidence_id,
                source_frame_id=source_frame_id,
                identity_state="CONFIRMED",
                liveness_status=liveness_status or "REAL",
                decision_source="automatic")
            if "clocked_in_at" in result or result.get("already_clocked_in"):
                self._today_clocked_in.add(person_name)
            if "clocked_in_at" in result:
                late = result.get("late_minutes", 0)
                msg = f"Welcome {person_name}."
                if late > 0: msg = f"Welcome {person_name}. You are {late} minutes late."
                if self._announce: voice(msg, "INFO", dedup_key=f"clockin:{person_name}")
                self.db.log_event("ATTENDANCE_CLOCKIN", person_id=pid, confidence=1.0,
                                  details={"late_min": late}, camera_id=camera_id,
                                  location=location, severity=0,
                                  presence_session_id=presence_session_id,
                                  source_frame_id=source_frame_id,
                                  observation_type="ATTENDANCE_DECISION")

    def clock_in_verified(self, person_name: str, camera_id: str = "cam_0",
                          location: str = None, confidence: float = 1.0,
                          method: str = "verified", presence_session_id=None,
                          recognition_evidence_id=None, source_frame_id=None,
                          liveness_status="REAL", identity_state="CONFIRMED",
                          attendance_eligible=True, quality_ok=True) -> dict:
        """Clock in after a deliberate verification flow such as center mode."""
        if not self.enabled:
            return {"error": "Attendance tracking is disabled"}
        if person_name in ("UNKNOWN", "SPOOF") or person_name.startswith("STRANGER_"):
            return {"error": "Only a recognized enrolled person can clock in"}
        if (normalize_identity_state(identity_state) != "CONFIRMED"
                or str(liveness_status or "").upper() != "REAL"
                or not attendance_eligible or not quality_ok):
            return {"error": "Identity, quality, and liveness verification did not pass"}
        self._maybe_rollover()
        if not _is_school_day(_local_datetime()):
            return {"error": "Automatic attendance is disabled for this school day."}
        pid = self.db.get_active_person_id(person_name)
        if pid is None:
            return {"error": f"No database person found for {person_name}"}
        result = self.db.attendance_clock_in(
            pid, camera_id, location, confidence=confidence,
            presence_session_id=presence_session_id,
            recognition_evidence_id=recognition_evidence_id,
            source_frame_id=source_frame_id,
            identity_state="CONFIRMED",
            liveness_status=liveness_status,
            decision_source=method)
        if "clocked_in_at" in result or result.get("already_clocked_in"):
            self._today_clocked_in.add(person_name)
        self._recognitions[person_name].clear()
        if "clocked_in_at" in result:
            late = result.get("late_minutes", 0)
            self.db.log_event(
                "ATTENDANCE_CLOCKIN", person_id=pid,
                confidence=float(confidence),
                details={"late_min": late, "method": method},
                camera_id=camera_id, location=location, severity=0,
                presence_session_id=presence_session_id,
                source_frame_id=source_frame_id,
                observation_type="ATTENDANCE_DECISION")
            msg = f"Welcome {person_name}."
            if late > 0:
                msg = f"Welcome {person_name}. You are {late} minutes late."
            if self._announce:
                voice(msg, "INFO", dedup_key=f"clockin:{person_name}")
        return result

    def reconcile(self, force=False) -> list[dict]:
        """Close stale automatic attendance without blocking inference."""
        if not self.enabled:
            return []
        now = time.time()
        if not force and now - self._last_reconcile < 1.0:
            return []
        self._last_reconcile = now
        self._maybe_rollover()
        timeout = self.cfg.get("AUTO_CLOCKOUT_TIMEOUT_MIN", 15)
        closed = self.db.close_stale_attendance(timeout)
        for row in closed:
            self._today_clocked_in.discard(row.get("name"))
            self.db.log_event(
                "ATTENDANCE_CLOCKOUT",
                person_id=row.get("person_id"),
                confidence=1.0,
                details={"work_min": row.get("work_minutes", 0), "source": "automatic_timeout"},
                severity=0,
                observation_type="ATTENDANCE_DECISION",
            )
            if self._announce and row.get("name"):
                voice(f"Goodbye {row['name']}.", "INFO", dedup_key=f"clockout:{row['name']}")
        return closed

    def close_for_shutdown(self) -> list[dict]:
        if not self.enabled or not self.cfg.get("AUTO_CLOCKOUT_ON_SHUTDOWN", True):
            return []
        closed = self.db.close_open_attendance_for_shutdown()
        for row in closed:
            self._today_clocked_in.discard(row.get("name"))
            self.db.log_event(
                "ATTENDANCE_CLOCKOUT",
                person_id=row.get("person_id"),
                confidence=1.0,
                details={"work_min": row.get("work_minutes", 0), "source": "runtime_shutdown"},
                severity=0,
                observation_type="ATTENDANCE_DECISION",
            )
        return closed

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
        f"Current UTC time: {_utc_datetime().strftime('%Y-%m-%d %H:%M')}\n"
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
    validated = validate_camera_source(
        source, _BASE_DIR, os.environ.get("OPTIVOX_RUNTIME_MODE", "development"))
    if validated.get("kind") == "file":
        source = validated["value"]
    elif validated.get("kind") == "index":
        source = validated["value"]
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
                    embedding = vision.face_analyzer.get_embedding(faces[0], display)
                    if embedding is not None:
                        captured_embeddings.append(np.asarray(embedding, dtype=np.float32))
                        captured += 1
                        last_capture_ts = now
                        status = f"Accepted sample {captured}"
                        status_color = (0, 255, 0)
                        print(f"[ENROLL] Accepted {captured}/{min_embeddings} quality={score:.0f}")
                        if captured >= max_embeddings:
                            status = f"Collected {captured} samples. Saving..."
                            cv2.putText(display, status, (20, 115),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                                        cv2.LINE_AA)
                            cv2.imshow("Enrollment Wizard", display)
                            cv2.waitKey(250)
                            break
        elif len(faces) > 1:
            status = "Multiple faces detected"
            status_color = (0, 0, 255)
        else:
            status = "No face detected"
            status_color = (0, 0, 255)

        cv2.putText(display, status, (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
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

    accepted_embeddings = []
    for emb in captured_embeddings[:max_embeddings]:
        if vision.add_embedding_for_name(
                person_name, emb, rebuild=False, persist=False):
            accepted_embeddings.append(emb)
    if len(accepted_embeddings) < min_embeddings:
        print("[ENROLL] Samples were not sufficiently distinct; enrollment rejected.")
        return False
    vision._rebuild_index()
    vision.save_face_db()

    db.upsert_person(person_name)
    db.log_audit("ENROLL_PERSON", person_name, {
        "embeddings": len(accepted_embeddings), "source": "multi_angle",
    })

    voice(f"Enrollment complete. Welcome, {person_name}.", "INFO")
    print(f"[ENROLL] SUCCESS: {person_name} enrolled with {len(accepted_embeddings)} embeddings.")
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
    "FALL_DETECTED", "HANDS_RAISED", "CONGESTION", "ZONE_INTRUSION",
    "LOITERING", "RUNNING", "PPE_VIOLATION",
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
    "LOITERING": 1,
    "RUNNING": 1,
    "ZONE_ENTRY": 0,
    "ZONE_EXIT": 0,
    "ZONE_INTRUSION": 2,
    "PPE_VIOLATION": 2,
    "OBJECT_INTERACTION": 0,
}


def _panel_wrap(value, width=42, limit=4):
    lines = []
    for paragraph in str(value or "").splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines[:limit]


def _draw_center_attendance_guide(display, ui, cfg):
    """Draw the focused attendance guide without changing the vision frame."""
    state = ui.get("center_attendance", {})
    if not state.get("enabled"):
        return display

    height, width = display.shape[:2]
    attendance_cfg = cfg.get("ATTENDANCE", {})
    x1 = int(width * attendance_cfg.get("CENTER_MODE_X_MIN", 0.34))
    x2 = int(width * attendance_cfg.get("CENTER_MODE_X_MAX", 0.66))
    y1 = int(height * attendance_cfg.get("CENTER_MODE_Y_MIN", 0.12))
    y2 = int(height * attendance_cfg.get("CENTER_MODE_Y_MAX", 0.82))
    phase = state.get("phase", "READY")
    color = {
        "READY": (42, 193, 245),
        "WAITING": (42, 193, 245),
        "VERIFYING": (0, 220, 255),
        "COMPLETE": (60, 220, 120),
        "BLOCKED": (0, 120, 255),
    }.get(phase, (42, 193, 245))
    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.putText(display, "CENTER ATTENDANCE", (x1, max(24, y1 - 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(display, str(state.get("status", "Stand in the center"))[:58],
                (x1, min(height - 18, y2 + 25)), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, color, 2, cv2.LINE_AA)
    if phase == "VERIFYING":
        progress = float(state.get("progress", 0.0))
        bar_w = max(1, x2 - x1)
        cv2.rectangle(display, (x1, min(height - 11, y2 + 34)),
                      (x1 + int(bar_w * max(0.0, min(1.0, progress))),
                       min(height - 3, y2 + 42)), color, -1)
    return display


def _draw_command_panel(display, faces_info, events, vision, db, assistant, ui,
                        cam_id, cam_location, performance=None):
    """Render live controls and system details beside the annotated camera frame."""
    height, width = display.shape[:2]
    panel_width = 370
    canvas = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
    canvas[:, :width] = display
    canvas[:, width:] = (5, 10, 20)
    px = width
    ui["frame_width"] = width
    ui["hitboxes"] = []

    def line(value, y, color=(220, 230, 235), scale=0.45, thickness=1):
        text = str(value)[:52]
        origin = (px + 15, y)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    def button(label, x, y, w=160, h=31, color=(45, 212, 191)):
        box = (px + x, y, px + x + w, y + h, label)
        ui["hitboxes"].append(box)
        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), (9, 19, 34), -1)
        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, 1, cv2.LINE_AA)
        shortcut = {
            "ENROLL PERSON": "[E] ", "ASK ASSISTANT": "[A] ",
            "CLOCK IN": "[I] ", "CLOCK OUT": "[C] ",
        }.get(label, "")
        if label.startswith("CENTER ATTENDANCE"):
            shortcut = "[K] "
        button_text = shortcut + label
        origin = (box[0] + 8, box[1] + 21)
        button_scale = 0.43 if len(button_text) <= 18 else 0.34
        cv2.putText(canvas, button_text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    button_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, button_text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    button_scale, (245, 250, 250), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (px, 0), (px + panel_width, height), (5, 10, 20), -1)
    cv2.line(canvas, (px, 0), (px, height), (18, 70, 105), 1, cv2.LINE_AA)
    line("OPTIVOX CONTROL CENTER", 25, (235, 248, 255), 0.57, 2)
    line("LOCAL-FIRST COMPUTER VISION", 47, (42, 193, 245), 0.37, 1)
    performance = performance or {}
    fps = float(performance.get("inference_fps", getattr(vision, "_fps", 0.0)) or 0.0)
    frame_age = float((performance.get("latency_ms") or {}).get("latest_frame_age", 0.0) or 0.0)
    line(f"CAMERA: ONLINE   {cam_id} / {cam_location}", 71, (225, 239, 248), 0.39, 1)
    line(f"AI {fps:.1f} FPS   AGE {frame_age:.0f}ms   FACES {len(faces_info)}", 91, (225, 239, 248))
    line(f"DB {db.get_face_count()}   BUF {performance.get('queue_depths', {}).get('side_effects', 0)}", 108, (170, 190, 205), 0.37)

    button("ENROLL PERSON", 14, 105, color=(42, 193, 245))
    button("ASK ASSISTANT", 184, 105, color=(42, 145, 235))
    button("CLOCK IN", 14, 142, color=(42, 193, 245))
    button("CLOCK OUT", 184, 142, color=(42, 145, 235))
    button("DETAILS: ON" if ui.get("show_details", True) else "DETAILS: OFF", 14, 179, color=(105, 135, 160))
    button("CLEAR STATUS", 184, 179, color=(105, 135, 160))
    center_state = ui.get("center_attendance", {})
    center_label = f"CENTER ATTENDANCE: {'ON' if center_state.get('enabled') else 'OFF'}"
    button(center_label, 14, 216, w=330,
           color=(60, 220, 120) if center_state.get("enabled") else (42, 145, 235))

    line("LIVE TOGGLES  (click buttons)", 260, (42, 193, 245), 0.43, 2)
    toggle_items = [
        ("[H] HEATMAP", "heatmap", "h"),
        ("[O] OBJECT BOXES", "objects", "o"),
        ("[P] POSE", "pose", "p"),
        ("[D] DANGER", "danger", "d"),
        ("[G] AGE/GENDER", "age", "g"),
        ("[Z] ZONES/GRID", "zones", "z"),
        ("[L] COUNT LINE", "count_line", "l"),
        ("[M] FACE MESH", "face_mesh", "m"),
        ("[N] HAND LANDMARKS", "hands", "n"),
        ("[V] DENSITY", "density", "v"),
    ]
    for index, (label, key, _shortcut) in enumerate(toggle_items):
        col, row = index % 2, index // 2
        enabled = bool(ui["toggles"].get(key, False))
        button(f"{label}: {'ON' if enabled else 'OFF'}", 14 + col * 170, 272 + row * 32,
               color=(42, 193, 245) if enabled else (75, 105, 130))

    state_y = 451
    line("CURRENT STATE", state_y, (42, 193, 245), 0.42, 2)
    for index, text in enumerate(_panel_wrap(ui.get("message", "Ready"), limit=3)):
        line(text, state_y + 21 + index * 16, (235, 235, 220), 0.40)

    if ui.get("assistant_answer"):
        line("ASSISTANT", 519, (42, 145, 235), 0.42, 2)
        for index, text in enumerate(_panel_wrap(ui["assistant_answer"], limit=4)):
            line(text, 539 + index * 16, (205, 220, 240), 0.38)

    if ui.get("show_details", True):
        event_y = 610
        line("RECENT EVENTS", event_y, (42, 193, 245), 0.42, 2)
        for index, event in enumerate(events[-4:]):
            line(f"{event[0]}: {event[1]}", event_y + 20 + index * 16, (200, 210, 220), 0.36)

    if ui.get("mode") == "text":
        line(f"{ui.get('prompt', 'INPUT')}: {ui.get('text', '')}_", height - 47, (255, 255, 255), 0.42, 2)
        line("TYPE HERE | ENTER CONFIRMS | ESC CANCELS", height - 23, (170, 185, 198), 0.37)
    elif ui.get("clock_mode"):
        roster = ui.get("roster", [])
        roster_y = 585
        line(f"SELECT PERSON TO CLOCK {ui['clock_mode'].upper()}", roster_y, (245, 158, 11), 0.39, 2)
        for index, row in enumerate(roster[:6]):
            y = roster_y + 17 + index * 17
            name = row["name"] if isinstance(row, dict) else row["name"]
            status = "IN" if row.get("clock_in") else "OUT"
            line(f"{name}  [{status}]", y, (225, 235, 235), 0.36)
            ui["hitboxes"].append((px + 10, y - 13, px + panel_width - 10, y + 3,
                                    f"PERSON:{row['id']}:{ui['clock_mode']}"))

    line("Q QUIT | CLICK | K CENTER ATTENDANCE | E ENROLL", height - 5, (170, 185, 198), 0.32)
    return canvas

# Runtime bridge
RUNTIME_VERSION = "2.1.0"
RUNTIME_ID = os.environ.get("OPTIVOX_RUNTIME_ID", "edge-local-01")


def _runtime_dir() -> str:
    path = os.path.join(_BASE_DIR, "runtime")
    os.makedirs(path, exist_ok=True)
    return path


def _atomic_json_write(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
    os.replace(tmp, path)


def _runtime_now() -> str:
    return dt_datetime.now().astimezone().isoformat(timespec="milliseconds")


def _runtime_capabilities(vision=None) -> dict:
    danger_enabled = bool(CONFIG.get("DANGER_DETECTION", {}).get("ENABLED", False))
    capabilities = {
        "face_detection": bool(INSIGHTFACE_AVAILABLE),
        "face_recognition": bool(INSIGHTFACE_AVAILABLE),
        "attendance": bool(CONFIG.get("ATTENDANCE", {}).get("ENABLED", True)),
        "guided_liveness": "EXPERIMENTAL" if CONFIG.get("ATTENDANCE", {}).get("CENTER_MODE_REQUIRE_LIVENESS", True) else "DISABLED",
        "object_detection": bool(YOLO_AVAILABLE),
        "pose_detection": bool(MEDIAPIPE_AVAILABLE),
        "hand_detection": bool(MEDIAPIPE_AVAILABLE),
        "danger_detection": "AVAILABLE" if danger_enabled else "DISABLED",
        "web_enrollment": vision is not None,
        "voice_assistant": bool(STT_AVAILABLE),
        "runtime_bridge": True,
    }
    if vision is not None and getattr(vision, "security_signals", None) is not None:
        security = vision.security_signals.capability_state()
        capabilities.update({f"security_{key}": value for key, value in security.items()})
    capabilities["ml_anti_spoof"] = (
        "AVAILABLE" if CONFIG.get("ANTI_SPOOFING", {}).get("ENABLE_ML_CLASSIFIER", False)
        else "NOT_CONFIGURED"
    )
    return capabilities


def _publish_runtime_capabilities(runtime_dir: str, started_at: str, vision=None):
    _atomic_json_write(os.path.join(runtime_dir, "capability.json"), {
        "schema_version": 1,
        "runtime_id": RUNTIME_ID,
        "runtime_version": RUNTIME_VERSION,
        "pid": os.getpid(),
        "process": process_identity(),
        "started_at": started_at,
        "generated_at": _runtime_now(),
        "capabilities": _runtime_capabilities(vision),
    })


def _publish_runtime_heartbeat(
    runtime_dir: str,
    engine_status: str,
    camera_status: str,
    fps: float,
    started_at: str,
    last_error=None,
    performance=None,
    frame_id=None,
    frame_age_ms=None,
):
    timestamp = _runtime_now()
    _atomic_json_write(os.path.join(runtime_dir, "heartbeat.json"), {
        "schema_version": 1,
        "runtime_id": RUNTIME_ID,
        "runtime_version": RUNTIME_VERSION,
        "pid": os.getpid(),
        "process": process_identity(),
        "watchdog": watchdog_status(os.getpid()),
        "started_at": started_at,
        "timestamp": timestamp,
        "last_heartbeat": timestamp,
        "engine_status": str(engine_status).upper(),
        "camera_status": str(camera_status).upper(),
        "fps": round(float(fps or 0), 2),
        "last_error": last_error,
        "frame_id": frame_id,
        "frame_age_ms": round(float(frame_age_ms or 0), 2),
        "performance": performance or {},
    })


def _publish_runtime_state(
    runtime_dir: str,
    vision,
    display,
    cam_id: str,
    location: str,
    fps: float,
    started_epoch: float,
    started_at: str,
    events: list,
    last_error=None,
    source_result=None,
    performance=None,
):
    faces = getattr(vision, "_last_faces_seen", []) or []
    registered = []
    unknown = []
    for face in faces:
        name = str(face.get("name") or "UNKNOWN")
        item = {
            "entity_id": face.get("entity_id"),
            "track_id": face.get("track_id"),
            "name": name,
            "temporary_name": name,
            "confidence": float(face.get("confidence") or 0),
            "attendance_status": "present",
            "visible_seconds": 0,
            "spoof_status": str(face.get("liveness_status") or (
                "passed" if face.get("is_real", True) else "suspect")).lower(),
            "quality_score": float(face.get("quality_score") or 0),
            "quality_ok": bool(face.get("quality_ok", True)),
            "identity_state": str(face.get("identity_state") or "UNRESOLVED"),
            "cached_identity_confidence": float(
                face.get("cached_identity_confidence") or 0),
            "current_observation_similarity": float(
                face.get("current_observation_similarity") or 0),
            "last_verified_at": face.get("last_verified_at"),
            "identity_age_ms": face.get("identity_age_ms"),
            "bbox": face.get("bbox"),
        }
        if name in ("UNKNOWN", "SPOOF") or name.startswith("STRANGER_"):
            unknown.append(item)
        else:
            registered.append(item)

    object_counts = {}
    for obj in getattr(vision, "_last_objects_seen", []) or []:
        name = str(obj.get("class_name") or "object")
        object_counts.setdefault(name, {
            "class_name": name,
            "count": 0,
            "confidence": 0.0,
            "category": obj.get("category"),
        })
        object_counts[name]["count"] += 1
        object_counts[name]["confidence"] = max(
            object_counts[name]["confidence"],
            float(obj.get("confidence") or 0),
        )

    active_events = []
    for index, evt in enumerate((events or [])[-10:]):
        try:
            active_events.append({
                "id": f"live_{int(time.time() * 1000)}_{index}",
                "time": _runtime_now(),
                "type": evt[0],
                "severity": "Critical" if evt[0] in _SEVERE_EVENT_TYPES else "Warning",
                "person": evt[1] if len(evt) > 1 else "System",
                "location": location,
                "confidence": float(evt[2]) if len(evt) > 2 else 0,
                "details": evt[3] if len(evt) > 3 else "",
            })
        except Exception:
            continue

    height, width = display.shape[:2] if display is not None else (None, None)
    security_level = (
        "critical" if any(e.get("severity") == "Critical" for e in active_events)
        else ("attention" if unknown or active_events else "normal")
    )
    _atomic_json_write(os.path.join(runtime_dir, "live_state.json"), {
        "schema_version": 1,
        "runtime_id": RUNTIME_ID,
        "runtime_version": RUNTIME_VERSION,
        "timestamp": _runtime_now(),
        "source_frame_id": source_result.get("frame_id") if source_result else None,
        "source_capture_timestamp": source_result.get("captured_at") if source_result else None,
        "inference_completed_at": source_result.get("completed_at") if source_result else None,
        "frame_age_ms": round(float((source_result or {}).get("frame_age_ms") or 0), 2),
        "performance": performance or {},
        "engine": {
            "status": "online",
            "uptime_seconds": round(time.time() - started_epoch, 1),
            "fps": round(float(fps or 0), 2),
            "frame_width": width,
            "frame_height": height,
            "last_error": last_error,
        },
        "camera": {
            "status": "connected",
            "id": cam_id,
            "name": cam_id,
            "location": location,
        },
        "security": {
            "level": security_level,
            "message": "Unknown person detected" if unknown else (
                "Security event active" if active_events else "No active warning"
            ),
            "active_event_count": len(active_events),
        },
        "presence": {"registered": registered, "unknown": unknown},
        "correlation": getattr(vision, "_last_correlation_state", {}),
        "capabilities": _runtime_capabilities(vision),
        "objects": list(object_counts.values()),
        "recent_events": active_events,
    })


def _publish_runtime_frame(runtime_dir: str, display, quality: int = 78):
    if display is None:
        return
    tmp = os.path.join(runtime_dir, "latest_frame.tmp.jpg")
    final = os.path.join(runtime_dir, "latest_frame.jpg")
    if cv2.imwrite(tmp, display, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]):
        os.replace(tmp, final)


def _publish_performance_summary(runtime_dir, profiler, vision, started_at,
                                 operations=None, frame_buffer=None):
    """Persist one compact benchmark snapshot at shutdown for repeatable review."""
    performance = profiler.snapshot(
        queue_depths=operations.queue_depths() if operations else None,
        latest_frame_age_ms=(frame_buffer.metrics().get("latest_frame_age_ms", 0.0)
                             if frame_buffer else 0.0),
        current_source_frame_id=(frame_buffer.metrics().get("frame_id")
                                 if frame_buffer else None),
    )
    performance["vision"] = vision.performance_snapshot() if vision else {}
    performance["benchmark"] = {
        "started_at": started_at,
        "ended_at": _runtime_now(),
        "duration_sec": performance.get("uptime_sec", 0.0),
    }
    _atomic_json_write(os.path.join(runtime_dir, "performance_summary.json"),
                       performance)


def _read_pending_runtime_commands(runtime_dir: str) -> list:
    path = os.path.join(runtime_dir, "commands.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        commands = data.get("commands", []) if isinstance(data, dict) else data
        return [cmd for cmd in commands if cmd.get("status") == "pending"]
    except Exception:
        return []


def _write_runtime_command_result(runtime_dir: str, result: dict):
    path = os.path.join(runtime_dir, "command_results.json")
    results = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            results = data.get("results", []) if isinstance(data, dict) else data
        except Exception:
            results = []
    results = [item for item in results if item.get("id") != result.get("id")][-100:]
    results.append(result)
    _atomic_json_write(path, {"results": results})


def _mark_runtime_commands_completed(runtime_dir: str, handled_ids: set):
    path = os.path.join(runtime_dir, "commands.json")
    if not handled_ids or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        commands = data.get("commands", []) if isinstance(data, dict) else data
        for command in commands:
            if command.get("id") in handled_ids:
                command["status"] = "completed"
        _atomic_json_write(path, {"commands": commands})
    except Exception:
        pass


def _set_enrollment_status(runtime_dir: str, stage: str, message: str, extra=None):
    data = {"stage": stage, "message": message, "updated_at": _runtime_now()}
    if extra:
        data.update(extra)
    _atomic_json_write(os.path.join(runtime_dir, "enrollment_status.json"), data)


def _process_runtime_commands(
    runtime_dir: str,
    db,
    attendance,
    alert_mgr,
    vision,
    last_raw_frame,
    snapshot_dir: str,
    cam_id: str,
    cam_location: str,
):
    if last_raw_frame is not None and getattr(vision, "_web_enrollment", None):
        try:
            progress = vision.collect_web_enrollment_frame(last_raw_frame)
            if progress.get("stage") in {"completed", "duplicate_warning"}:
                name = progress["person_name"]
                _set_enrollment_status(
                    runtime_dir,
                    progress.get("stage"),
                    progress.get("message", "Enrollment samples are ready for confirmation."),
                    {"mode": "web_guided", "person_name": name,
                     "accepted": progress.get("accepted", 0),
                     "minimum": progress.get("minimum", 5),
                     "maximum": progress.get("maximum", 10),
                     "quality_scores": progress.get("quality_scores", []),
                     "rejected_samples": progress.get("rejected_samples", 0),
                     "duplicate_candidates": progress.get("duplicate_candidates", [])},
                )
            elif progress.get("active"):
                _set_enrollment_status(
                    runtime_dir,
                    "capturing",
                    progress.get("message", "Collecting enrollment samples."),
                    {"mode": "web_guided",
                     "person_name": (getattr(vision, "_web_enrollment", {}) or {}).get("person_name"),
                     "accepted": progress.get("accepted", 0),
                     "minimum": progress.get("minimum", 5),
                     "maximum": progress.get("maximum", 10),
                     "quality_score": progress.get("quality_score"),
                     "rejected_samples": progress.get("rejected_samples", 0)},
                )
        except Exception as exc:
            print(f"[ERROR] Web enrollment progress failed: {exc}")
            _set_enrollment_status(runtime_dir, "failed", str(exc))
            vision.cancel_web_enrollment()

    handled = set()
    for command in _read_pending_runtime_commands(runtime_dir):
        command_id = command.get("id")
        command_type = command.get("type")
        payload = command.get("payload") or {}
        actor_id = str(payload.get("actor_id") or "system")
        result = {
            "id": command_id,
            "type": command_type,
            "status": "completed",
            "completed_at": _runtime_now(),
            "result": {},
            "error": None,
        }
        try:
            if command_type == "save_snapshot":
                if last_raw_frame is None:
                    raise RuntimeError("No frame is available for a snapshot.")
                path = _save_snapshot(snapshot_dir, "MANUAL", "web", last_raw_frame)
                result["result"] = {"message": "Snapshot saved successfully.", "snapshot_path": path}
            elif command_type == "test_alert":
                result["result"] = {"message": "Test alert requested.", "channels": alert_mgr.test_alert()}
            elif command_type == "manual_clock_in":
                person_id = payload.get("person_id")
                if not person_id:
                    raise RuntimeError("person_id is required.")
                person = db._fetchone("SELECT id FROM people WHERE id=?", (int(person_id),))
                if not person:
                    raise RuntimeError("The selected person is not in the roster.")
                result["result"] = db.attendance_clock_in(int(person_id), cam_id, cam_location)
            elif command_type == "manual_clock_out":
                person_id = payload.get("person_id")
                name = str(payload.get("name") or "").strip()
                if not name and person_id:
                    person = db._fetchone("SELECT name FROM people WHERE id=?", (int(person_id),))
                    name = str(person["name"]) if person else ""
                if not name:
                    raise RuntimeError("name or person_id is required.")
                result["result"] = attendance.manual_clock_out(name)
            elif command_type == "start_enrollment":
                name = str(payload.get("name") or "").strip()
                if not name:
                    raise RuntimeError("name is required.")
                started = vision.begin_web_enrollment(
                    name,
                    min_embeddings=int(payload.get("min_embeddings", 5)),
                    max_embeddings=int(payload.get("max_embeddings", 10)),
                    replace_existing=bool(payload.get("replace_existing", False)),
                    metadata={
                        "role": str(payload.get("role") or "").strip(),
                        "consent_confirmed": bool(payload.get("consent_confirmed")),
                    },
                )
                if not started.get("ok"):
                    raise RuntimeError(started.get("message", "Could not start enrollment."))
                _set_enrollment_status(
                    runtime_dir,
                    "capturing",
                    f"Collecting samples for {name}. Keep one face visible.",
                    {"mode": "web_guided", "person_name": name,
                     "accepted": 0, "minimum": started.get("minimum", 5),
                     "maximum": started.get("maximum", 10)},
                )
                result["result"] = {
                    "message": f"Guided enrollment started for {name}.",
                    "mode": "web_guided",
                }
            elif command_type == "confirm_enrollment":
                committed = vision.commit_web_enrollment(
                    override_duplicate=bool(payload.get("override_duplicate", False)))
                if not committed.get("ok"):
                    _set_enrollment_status(
                        runtime_dir,
                        committed.get("stage", "failed"),
                        committed.get("message", "Enrollment confirmation failed."),
                        {"duplicate_candidates": committed.get("duplicate_candidates", [])},
                    )
                    raise RuntimeError(committed.get("message", "Enrollment confirmation failed."))
                name = committed["person_name"]
                vision.enable_identity(name)
                metadata = dict((getattr(vision, "_last_enrollment_metadata", {}) or {}))
                metadata.update({
                    "enrollment_quality": committed,
                    "active": True,
                })
                db.upsert_person(name, role=metadata.get("role") or None, metadata=metadata)
                db.log_audit("ENROLL_PERSON", name, {
                    "embeddings": committed.get("accepted", 0),
                    "mode": "web_guided",
                    "quality": committed,
                    "duplicate_override": committed.get("duplicate_override", False),
                })
                db.log_enrollment_operation(
                    name,
                    "retrain" if (metadata.get("retrain") or False) else "guided_enrollment",
                    "committed",
                    sample_count=committed.get("accepted", 0),
                    quality=committed,
                    provenance={"mode": "web_guided", "augmented": committed.get("augmented_count", 0)},
                    actor_id=actor_id,
                )
                _set_enrollment_status(
                    runtime_dir,
                    "completed",
                    f"{name} enrolled with {committed.get('accepted', 0)} samples.",
                    {"mode": "web_guided", "person_name": name,
                     "accepted": committed.get("accepted", 0),
                     "quality": committed},
                )
                result["result"] = committed
            elif command_type == "disable_person":
                name = str(payload.get("name") or "").strip()
                if not name:
                    raise RuntimeError("name is required.")
                person_id = db.get_person_id(name)
                if not person_id:
                    raise RuntimeError(f"No roster person found for {name}.")
                disabled = db.set_person_active(person_id, False)
                vision.disable_identity(name)
                db.log_audit("DISABLE_PERSON", name, {"source": "web_command", "actor_id": actor_id})
                db.log_enrollment_operation(name, "disable", "completed", actor_id=actor_id)
                result["result"] = {"message": f"{name} disabled.", "person_id": person_id, "active": disabled}
            elif command_type == "retrain_person":
                name = str(payload.get("name") or "").strip()
                if not name:
                    raise RuntimeError("name is required.")
                started = vision.begin_web_enrollment(
                    name,
                    min_embeddings=int(payload.get("min_embeddings", 5)),
                    max_embeddings=int(payload.get("max_embeddings", 10)),
                    replace_existing=True,
                    metadata={"retrain": True, "consent_confirmed": True},
                )
                if not started.get("ok"):
                    raise RuntimeError(started.get("message", "Could not start retraining."))
                _set_enrollment_status(
                    runtime_dir, "capturing", f"Retraining {name}. Keep exactly one face visible.",
                    {"mode": "retrain", "person_name": name, "accepted": 0,
                     "minimum": started.get("minimum", 5), "maximum": started.get("maximum", 10)},
                )
                result["result"] = {"message": f"Retraining started for {name}.", "mode": "retrain"}
            elif command_type == "merge_people":
                source_name = str(payload.get("source_name") or "").strip()
                target_name = str(payload.get("target_name") or "").strip()
                merged = vision.merge_face_identities(source_name, target_name)
                if not merged.get("ok"):
                    raise RuntimeError(merged.get("message", "Could not merge identities."))
                source_id = db.get_person_id(source_name)
                target_id = db.get_person_id(target_name)
                if source_id and target_id:
                    db.merge_person_records(source_id, target_id)
                db.log_audit("MERGE_PERSON", target_name, {"source": source_name, "target": target_name, "actor_id": actor_id})
                db.log_enrollment_operation(target_name, "merge", "completed",
                                             provenance={"source_name": source_name}, actor_id=actor_id)
                result["result"] = merged
            elif command_type == "delete_person":
                if not bool(payload.get("confirm")):
                    raise RuntimeError("Permanent deletion requires confirm=true.")
                name = str(payload.get("name") or "").strip()
                person_id = db.get_person_id(name)
                if not person_id:
                    raise RuntimeError(f"No roster person found for {name}.")
                deleted = vision.delete_face_identity(name)
                if not deleted.get("ok"):
                    raise RuntimeError(deleted.get("message", "Could not delete biometric identity."))
                db.log_enrollment_operation(
                    name, "delete", "completed", actor_id=actor_id)
                db.delete_person_records(person_id)
                db.log_audit("DELETE_PERSON", name, {"source": "web_command", "confirmed": True, "actor_id": actor_id})
                result["result"] = deleted
            elif command_type == "register_visible_unknown":
                name = str(payload.get("name") or "").strip()
                if not name:
                    raise RuntimeError("name is required.")
                _set_enrollment_status(
                    runtime_dir,
                    "capturing",
                    f"Registering the visible face as {name}.",
                    {"mode": "visible_face", "person_name": name},
                )
                ok = vision.enroll_best_visible_face(name)
                if not ok:
                    _set_enrollment_status(runtime_dir, "failed", "No visible unknown face was available.")
                    raise RuntimeError("No visible unknown face was available.")
                db.upsert_person(name, role=str(payload.get("role") or "").strip())
                db.log_audit("ENROLL_VISIBLE_FACE_WEB", name, payload)
                db.log_enrollment_operation(
                    name, "visible_face", "committed", sample_count=1,
                    provenance={"mode": "visible_face"}, actor_id=actor_id)
                _set_enrollment_status(
                    runtime_dir,
                    "completed",
                    f"{name} enrolled from the visible face.",
                    {"mode": "visible_face", "person_name": name, "accepted": 1},
                )
                result["result"] = {"message": f"{name} enrolled.", "mode": "visible_face"}
            elif command_type == "cancel_enrollment":
                vision.cancel_web_enrollment()
                _set_enrollment_status(runtime_dir, "cancelled", "Enrollment cancelled.")
                result["result"] = {"message": "Enrollment cancelled."}
            elif command_type == "reset_demo_data":
                result["result"] = {"message": "Demo reset is intentionally non-destructive."}
            else:
                raise RuntimeError(f"Unsupported command: {command_type}")
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
        handled.add(command_id)
        _write_runtime_command_result(runtime_dir, result)
    _mark_runtime_commands_completed(runtime_dir, handled)


class LatestFrameCaptureProxy:
    """VideoCapture-compatible reader for workflows using the latest buffer."""

    def __init__(self, frame_buffer):
        self.frame_buffer = frame_buffer
        self._last_frame_id = 0

    def read(self):
        packet = self.frame_buffer.wait_for_latest(self._last_frame_id, timeout=0.5)
        if not packet:
            return False, None
        self._last_frame_id = packet["frame_id"]
        return True, packet["frame"].copy()

    def isOpened(self):
        return not self.frame_buffer.metrics().get("closed", False)

    def release(self):
        return None

    def _loitering(self, h):
        """Detect sustained dwell in the configured legacy loitering zone."""
        # Configured polygon/rect security zones are handled by the
        # attribution-aware SecuritySignalEngine below.
        if self.cfg.get("SECURITY", {}).get("ZONES"):
            return None
        zone = self.cfg.get("LOITERING_ZONE")
        if not zone or len(zone) < 4 or not h["centroids"]:
            return None
        x1, y1, x2, y2 = zone[:4]
        x, y = h["centroids"][-1]
        inside = min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)
        if not inside:
            h["zone_entered_at"] = None
            h["loitering_emitted"] = False
            return None
        if h["zone_entered_at"] is None:
            h["zone_entered_at"] = self.current_time
            h["loitering_emitted"] = False
        dwell = self.current_time - h["zone_entered_at"]
        threshold = float(self.cfg.get("LOITERING_DURATION_SEC", 15.0))
        if dwell >= threshold and not h["loitering_emitted"]:
            h["loitering_emitted"] = True
            return f"Dwell {dwell:.1f}s in loitering zone"
        return None

    def _running(self, h):
        """Detect sustained movement after smoothing short tracker spikes."""
        if len(h["speeds"]) < 3:
            h["running_since"] = None
            return None
        threshold = float(self.cfg.get("RUNNING_SPEED_THRESHOLD", 280.0))
        recent = list(h["speeds"])[-8:]
        average = float(np.mean(recent))
        if average < threshold:
            h["running_since"] = None
            return None
        if h["running_since"] is None:
            h["running_since"] = self.current_time
            return None
        duration = self.current_time - h["running_since"]
        if duration >= float(self.cfg.get("RUNNING_MIN_SEC", 0.6)):
            return f"Sustained movement {average:.1f}px/s for {duration:.1f}s"
        return None


def _compact_inference_result(result):
    """Drop the annotated image before putting a result on side-effect queues."""
    return {
        "frame_id": result.get("frame_id"),
        "captured_at": result.get("captured_at"),
        "completed_at": result.get("completed_at"),
        "faces_info": list(result.get("faces_info") or []),
        "events": list(result.get("events") or []),
        "correlation": result.get("correlation") or {},
        "evidence_frame": result.get("evidence_frame"),
    }


def _process_runtime_side_effects(
    task,
    db,
    attendance,
    alert_mgr,
    snapshot_dir,
    cam_id,
    cam_location,
    attendance_lock,
    center_mode_getter,
):
    """Persist attendance/events and dispatch alerts outside model inference."""
    correlation = task.get("correlation") or {}
    sessions_by_entity = {}
    entities = correlation.get("entities") or []
    entities_by_id = {
        str(entity.get("entity_id")): entity
        for entity in entities if entity.get("entity_id")
    }
    for entity in entities:
        entity_id = entity.get("entity_id")
        if not entity_id:
            continue
        identity = entity.get("identity") or {}
        liveness = entity.get("liveness") or {}
        lifecycle_state = str(entity.get("lifecycle_state") or "ACTIVE").upper()
        visible = lifecycle_state in {"NEW", "ACTIVE"}
        identity_state = normalize_identity_state(identity.get("state"))
        confirmed_name = str(identity.get("confirmed_name") or "").strip()
        label = confirmed_name if identity_state == "CONFIRMED" and confirmed_name else "UNKNOWN"
        person_id = db.get_active_person_id(confirmed_name) if label != "UNKNOWN" else None
        try:
            session_id = db.upsert_presence_session(
                entity_id=entity_id,
                track_id=entity.get("track_id"),
                person_id=person_id,
                label=label,
                identity_state=identity_state,
                liveness_status=liveness.get("state"),
                camera_id=cam_id,
                confidence=identity.get("best_score") or 0.0,
                source_frame_id=task.get("frame_id"),
                observed_at=_utc_now() if visible else None,
                visible=visible,
                track_generation=entity.get("track_generation", 1),
            )
            sessions_by_entity[str(entity_id)] = session_id
        except Exception as exc:
            print(f"[ERROR] Presence session update failed: {exc}")
    try:
        db.close_stale_presence_sessions(
            CONFIG.get("ATTENDANCE", {}).get("PRESENCE_SESSION_CLOSE_AFTER_SEC", 5.0))
    except Exception as exc:
        print(f"[ERROR] Presence session cleanup failed: {exc}")

    evidence_by_entity = {}
    for face in task.get("faces_info") or []:
        if not face.get("evidence_due"):
            continue
        entity_id = face.get("entity_id")
        if not entity_id:
            continue
        correlated_entity = entities_by_id.get(str(entity_id), {})
        correlated_identity = correlated_entity.get("identity") or {}
        name = str(face.get("name") or "UNKNOWN")
        identity_state = normalize_identity_state(
            correlated_identity.get("state")
            or face.get("identity_state")
            or "UNRESOLVED"
        )
        if identity_state == "CONFIRMED" and correlated_identity.get("confirmed_name"):
            name = str(correlated_identity["confirmed_name"])
        if identity_state == "SPOOF_SUSPECT" or name == "SPOOF" or str(face.get("liveness_status")) in {"SUSPECT", "UNCERTAIN"}:
            decision = "spoof_or_uncertain"
        elif identity_state == "CONFIRMED" and name not in {"UNKNOWN", "SPOOF"} and not name.startswith("STRANGER_"):
            decision = "confirmed"
        elif identity_state in {"CONTRADICTED", "OCCLUDED", "EXPIRED"}:
            decision = "contradicted"
        elif str(face.get("reason")) == "WAITING_FOR_GOOD_FACE":
            decision = "quality_rejected"
        else:
            decision = "unresolved"
        person_id = db.get_active_person_id(name) if decision == "confirmed" else None
        try:
            evidence_by_entity[str(entity_id)] = db.record_recognition_evidence(
                entity_id=entity_id,
                presence_session_id=sessions_by_entity.get(str(entity_id)),
                track_id=face.get("oid"),
                person_id=person_id,
                candidate_name=name,
                decision=decision,
                similarity=face.get("current_observation_similarity", face.get("confidence", 0.0)),
                quality_score=face.get("quality_score", 0.0),
                quality_ok=face.get("quality_ok", False),
                liveness_status=face.get("liveness_status"),
                identity_state=identity_state,
                reason=face.get("reason"),
                source_frame_id=face.get("source_frame_id") or task.get("frame_id"),
                observed_at=face.get("observed_at") or task.get("completed_at") or _utc_now(),
                details={
                    "cached_identity_confidence": face.get("cached_identity_confidence"),
                    "liveness_details": face.get("liveness_details", {}),
                },
            )
            active_person_id = db.get_active_person_id(name) if identity_state == "CONFIRMED" else None
            attendance_reasons = []
            if identity_state != "CONFIRMED":
                attendance_reasons.append(f"identity_{identity_state.lower()}")
            if not bool(correlated_entity.get("attendance_eligibility")):
                attendance_reasons.append("entity_gate_rejected")
            if not bool(face.get("quality_ok", False)):
                attendance_reasons.append("face_quality_rejected")
            correlated_liveness = correlated_entity.get("liveness") or {}
            if str(face.get("liveness_status") or correlated_liveness.get("state") or "NOT_EVALUATED").upper() != "REAL":
                attendance_reasons.append("liveness_not_real")
            if active_person_id is None:
                attendance_reasons.append("person_not_active_in_roster")
            attendance_decision = "eligible" if not attendance_reasons else "rejected"
            decision_frame_id = face.get("source_frame_id") or task.get("frame_id")
            decision_key = f"{entity_id}:{decision_frame_id}:{attendance_decision}"
            db.record_attendance_decision(
                decision_key=decision_key,
                decision=attendance_decision,
                reason="eligible" if not attendance_reasons else ",".join(dict.fromkeys(attendance_reasons)),
                person_id=active_person_id,
                entity_id=entity_id,
                presence_session_id=sessions_by_entity.get(str(entity_id)),
                recognition_evidence_id=evidence_by_entity[str(entity_id)],
                identity_state=identity_state,
                liveness_status=face.get("liveness_status") or correlated_liveness.get("state"),
                quality_score=face.get("quality_score"),
                recognition_confidence=face.get("current_observation_similarity", face.get("confidence", 0.0)),
                source_frame_id=decision_frame_id,
                observed_at=face.get("observed_at") or task.get("completed_at") or _utc_now(),
                details={"method": "automatic_gate", "gate_reasons": list(dict.fromkeys(attendance_reasons))},
            )
        except Exception as exc:
            print(f"[ERROR] Recognition evidence failed: {exc}")

    if not center_mode_getter():
        for face in task.get("faces_info") or []:
            name = str(face.get("name") or "UNKNOWN")
            eligible = face.get("attendance_eligible", face.get("is_real", True))
            entity = entities_by_id.get(str(face.get("entity_id")), {})
            identity = entity.get("identity") or {}
            correlated_name = str(identity.get("confirmed_name") or "")
            correlation_eligible = bool(entity.get("attendance_eligibility"))
            active_person_id = db.get_active_person_id(name)
            confirmed_by_correlation = (
                identity.get("state") == "CONFIRMED"
                and correlated_name
                and correlated_name == name
                and correlation_eligible
                and active_person_id is not None
            )
            if (eligible and confirmed_by_correlation
                    and name not in ("UNKNOWN", "SPOOF")
                    and not name.startswith("STRANGER_")):
                try:
                    evidence_id = evidence_by_entity.get(str(face.get("entity_id")))
                    if evidence_id is None:
                        existing_attendance = db._fetchone(
                            "SELECT clock_in FROM attendance WHERE person_id=? AND date=?",
                            (active_person_id, _today_iso()),
                        )
                        if not existing_attendance or not existing_attendance["clock_in"]:
                            evidence_id = db.record_recognition_evidence(
                                entity_id=face.get("entity_id"),
                                presence_session_id=sessions_by_entity.get(str(face.get("entity_id"))),
                                track_id=face.get("oid"),
                                person_id=active_person_id,
                                candidate_name=name,
                                decision="confirmed",
                                similarity=face.get("current_observation_similarity", face.get("confidence", 0.0)),
                                quality_score=face.get("quality_score", 0.0),
                                quality_ok=True,
                                liveness_status="REAL",
                                identity_state="CONFIRMED",
                                reason="attendance_decision",
                                source_frame_id=face.get("source_frame_id") or task.get("frame_id"),
                                observed_at=face.get("observed_at") or task.get("completed_at") or _utc_now(),
                            )
                    with attendance_lock:
                        attendance.handle_recognition(
                            name, cam_id, cam_location,
                            confidence=float(face.get("confidence") or 0.0),
                            presence_session_id=sessions_by_entity.get(str(face.get("entity_id"))),
                            recognition_evidence_id=evidence_id,
                            source_frame_id=face.get("source_frame_id") or task.get("frame_id"),
                            liveness_status=face.get("liveness_status"),
                            identity_state=identity.get("state", "UNRESOLVED"),
                            attendance_eligible=correlation_eligible,
                            quality_ok=bool(face.get("quality_ok", False)),
                            authoritative=True,
                        )
                except Exception as exc:
                    print(f"[ERROR] Attendance update failed: {exc}")

    for evt in task.get("events") or []:
        try:
            event_type = evt[0]
            target = evt[1] if len(evt) > 1 else "SYSTEM"
            confidence = float(evt[2]) if len(evt) > 2 else 0.0
            details = evt[3] if len(evt) > 3 else ""
            metadata = dict(evt[4]) if len(evt) > 4 and isinstance(evt[4], dict) else {}
        except (IndexError, TypeError, ValueError):
            continue

        snapshot_path = None
        evidence_frame = task.get("evidence_frame")
        if event_type in _SEVERE_EVENT_TYPES and evidence_frame is not None:
            snapshot_path = _save_snapshot(
                snapshot_dir, event_type, target, evidence_frame)

        pid = None
        if (target not in ("SYSTEM", "UNKNOWN", "PERSON")
                and not str(target).startswith(("STRANGER_", "ID_", "AREA_", "ZONE_"))):
            pid = db.get_person_id(target)

        event_entity_id = None
        event_session_id = None
        track_id = metadata.get("track_id")
        for face in task.get("faces_info") or []:
            face_name = str(face.get("name") or "")
            if ((track_id is not None and int(face.get("oid", -1)) == int(track_id))
                    or target == "PERSON" or target == face_name):
                event_entity_id = face.get("entity_id")
                event_session_id = sessions_by_entity.get(str(event_entity_id))
                break
        if event_entity_id is None and track_id is not None:
            for entity in entities:
                if str(entity.get("track_id")) == str(track_id):
                    event_entity_id = entity.get("entity_id")
                    event_session_id = sessions_by_entity.get(str(event_entity_id))
                    break
        if event_entity_id is None and len(entities) == 1:
            event_entity_id = entities[0].get("entity_id")
            event_session_id = sessions_by_entity.get(str(event_entity_id))

        event_id = None
        try:
            event_details = {
                "message": str(details),
                "security_metadata": metadata,
            } if metadata else details
            event_id = db.log_event(
                event_type=event_type,
                person_id=pid,
                confidence=confidence,
                details=event_details,
                snapshot_path=snapshot_path,
                camera_id=cam_id,
                location=cam_location,
                # Zone policy may raise severity above the generic event
                # default. Preserve that policy value in the durable record.
                severity=max(
                    _SEVERITY_MAP.get(event_type, 0),
                    int(metadata.get("severity", 0) or 0),
                ),
                entity_id=event_entity_id,
                presence_session_id=event_session_id,
                source_frame_id=metadata.get("source_frame_id") or task.get("frame_id"),
                observation_type=event_type,
                evidence_path=snapshot_path,
            )
        except Exception as exc:
            print(f"[ERROR] log_event failed: {exc}")

        if event_type in _SEVERE_EVENT_TYPES:
            try:
                alert_mgr.check_and_alert(
                    event_type=event_type,
                    name=str(target),
                    confidence=confidence,
                    details=details,
                    snapshot_path=snapshot_path,
                    source_event_id=event_id,
                )
            except Exception as exc:
                print(f"[ERROR] Alert dispatch failed: {exc}")

        if event_type == "DANGEROUS_OBJECT":
            voice(f"Warning. Dangerous object detected: {target}.",
                  "CRITICAL", dedup_key=f"danger:{target}")
        elif event_type == "WEAPON_DETECTED":
            voice(f"Warning. Weapon detected: {target}.",
                  "CRITICAL", dedup_key=f"weapon:{target}")
        elif event_type == "FIRE_DETECTED":
            voice("Warning. Fire or flame detected.", "CRITICAL", dedup_key="fire")
        elif event_type == "SMOKE_DETECTED":
            voice("Warning. Smoke detected.", "CRITICAL", dedup_key="smoke")
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


class CameraCaptureWorker(threading.Thread):
    def __init__(self, capture, frame_buffer, profiler, stop_event,
                 target_size=(1280, 720), runtime_dir=None, camera_id="cam_0"):
        super().__init__(name="optivox-camera", daemon=True)
        self.capture = capture
        self.frame_buffer = frame_buffer
        self.profiler = profiler
        self.stop_event = stop_event
        self.target_size = target_size
        self.runtime_dir = runtime_dir
        self.camera_id = str(camera_id)
        self.health_monitor = CameraHealthMonitor(
            freeze_after_sec=float(os.environ.get("OPTIVOX_CAMERA_FREEZE_SECONDS", "5")),
            event_cooldown_sec=10.0,
        )
        self.health = self.health_monitor.observe(False, False, now=time.monotonic())
        self.last_error = None
        self.frames_read = 0

    def health_status(self):
        return dict(self.health)

    def _record_health(self, capture_open, read_ok, frame):
        self.health = self.health_monitor.observe(capture_open, read_ok, frame)
        event = self.health.get("event")
        if event:
            event = {**event, "camera_id": self.camera_id, "process": process_identity()}
            print(f"[CAMERA] {event['state']}: camera health transition")
            if self.runtime_dir:
                try:
                    append_runtime_health_event(self.runtime_dir, event)
                except Exception as exc:
                    print(f"[CAMERA] health event write failed: {exc}")

    def run(self):
        try:
            while not self.stop_event.is_set():
                try:
                    capture_open = bool(self.capture.isOpened())
                except Exception:
                    capture_open = False
                ret, frame = self.capture.read()
                captured_at = time.time()
                self._record_health(capture_open, ret, frame)
                if not ret or frame is None:
                    self.last_error = "Camera read failed."
                    time.sleep(0.05)
                    continue
                width, height = self.target_size
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                frame_id = self.frame_buffer.publish(frame, captured_at)
                buffer_metrics = self.frame_buffer.metrics()
                self.profiler.record_capture(
                    replaced=bool(buffer_metrics.get("last_publish_replaced")),
                    frame_id=frame_id)
                self.frames_read += 1
                self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)
            self._record_health(False, False, None)
            print(f"[CAPTURE] Worker stopped: {exc}")
        finally:
            try:
                self.capture.release()
            except Exception:
                pass


class InferenceWorker(threading.Thread):
    def __init__(self, vision, frame_buffer, inference_state, profiler,
                 side_effect_worker, stop_event, vision_lock, max_age_ms):
        super().__init__(name="optivox-inference", daemon=True)
        self.vision = vision
        self.frame_buffer = frame_buffer
        self.inference_state = inference_state
        self.profiler = profiler
        self.side_effect_worker = side_effect_worker
        self.stop_event = stop_event
        self.vision_lock = vision_lock
        self.max_age_ms = max(1, int(max_age_ms))
        self.cam_id = "cam_0"
        self.last_error = None
        self._last_frame_id = 0
        self.processing_size = None

    def set_processing_size(self, size):
        if not size:
            self.processing_size = None
            return
        width, height = (int(value) for value in size)
        self.processing_size = (max(1, width), max(1, height))

    def run(self):
        try:
            while not self.stop_event.is_set():
                packet = self.frame_buffer.wait_for_latest(self._last_frame_id, timeout=0.2)
                if packet is None:
                    continue
                self._last_frame_id = packet["frame_id"]
                self.profiler.record_frame_consumed(
                    packet["frame_id"], packet["age_ms"],
                    skipped=packet.get("frames_skipped", 0))
                if packet["age_ms"] > self.max_age_ms:
                    self.profiler.record_stale_drop()
                    continue
                started = time.perf_counter()
                try:
                    source_frame = packet["frame"]
                    processing_frame = source_frame
                    if self.processing_size:
                        width, height = self.processing_size
                        if (processing_frame.shape[1], processing_frame.shape[0]) != self.processing_size:
                            processing_frame = cv2.resize(
                                processing_frame, self.processing_size,
                                interpolation=cv2.INTER_LINEAR)
                    with self.vision_lock:
                        processed = self.vision.process(
                            processing_frame,
                            original_frame=source_frame,
                            source_frame_id=packet["frame_id"],
                            camera_id=self.cam_id,
                        )
                    (display, faces_info, hand_dets, object_detections,
                     events, tracked_count, dt) = processed
                    completed_at = time.time()
                    result = {
                        "frame_id": packet["frame_id"],
                        "source_frame_id": packet["frame_id"],
                        "captured_at": packet["captured_at"],
                        "completed_at": completed_at,
                        "frame_age_ms": max(0.0, (completed_at - packet["captured_at"]) * 1000.0),
                        "display": display,
                        "faces_info": faces_info,
                        "hand_dets": hand_dets,
                        "object_detections": object_detections,
                        "events": events,
                        "correlation": getattr(self.vision, "_last_correlation_state", {}),
                        "tracked_count": tracked_count,
                        "dt": dt,
                        "inference_ms": (time.perf_counter() - started) * 1000.0,
                        "evidence_frame": display.copy() if any(
                            event[0] in _SEVERE_EVENT_TYPES for event in events
                            if isinstance(event, (tuple, list)) and event
                        ) else None,
                    }
                    self.vision._last_frame_id = packet["frame_id"]
                    self.vision._last_frame_capture_timestamp = packet["captured_at"]
                    self.profiler.record_inference(
                        packet["frame_id"], result["frame_age_ms"],
                        result["inference_ms"],
                        getattr(self.vision, "_last_stage_timings", {}),
                        frame_age_start_ms=packet["age_ms"],
                        frame_age_end_ms=result["frame_age_ms"],
                    )
                    self.inference_state.publish(result)
                    self.side_effect_worker.submit(_compact_inference_result(result))
                    self.last_error = None
                except Exception as exc:
                    self.profiler.record_inference_error()
                    self.last_error = str(exc)
                    print(f"[ERROR] Vision pipeline failed (worker continues): {exc}")
                    traceback.print_exc()
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[INFERENCE] Worker stopped: {exc}")


class OperationalWorker(threading.Thread):
    def __init__(self, db, attendance, alert_mgr, snapshot_dir, cam_id,
                 cam_location, attendance_lock, center_mode_getter, profiler,
                 regular_size=256, critical_size=64):
        super().__init__(name="optivox-operations", daemon=True)
        self.db = db
        self.attendance = attendance
        self.alert_mgr = alert_mgr
        self.snapshot_dir = snapshot_dir
        self.cam_id = cam_id
        self.cam_location = cam_location
        self.attendance_lock = attendance_lock
        self.center_mode_getter = center_mode_getter
        self.profiler = profiler
        self.regular_queue = queue.Queue(maxsize=max(8, int(regular_size)))
        self.critical_queue = queue.Queue(maxsize=max(4, int(critical_size)))
        self.stop_event = threading.Event()
        self.last_error = None

    def _is_critical(self, task):
        return any(
            isinstance(event, (tuple, list)) and event and event[0] in _SEVERE_EVENT_TYPES
            for event in task.get("events") or []
        )

    def submit(self, task):
        correlation_entities = bool(
            (task.get("correlation") or {}).get("entities"))
        useful_face = any(
            face.get("attendance_eligible", face.get("is_real", True))
            and str(face.get("name") or "UNKNOWN") not in ("UNKNOWN", "SPOOF")
            and not str(face.get("name") or "").startswith("STRANGER_")
            for face in task.get("faces_info") or []
        )
        if not useful_face and not task.get("events") and not correlation_entities:
            return
        critical = self._is_critical(task)
        target = self.critical_queue if critical else self.regular_queue
        try:
            target.put_nowait(task)
        except queue.Full:
            self.profiler.record_queue_drop(critical=critical)
            print(f"[OPERATIONS] {'Critical' if critical else 'Regular'} queue full; "
                  f"frame {task.get('frame_id')} was not enqueued.")

    def queue_depths(self):
        return {
            "side_effects": self.regular_queue.qsize(),
            "critical_side_effects": self.critical_queue.qsize(),
        }

    def run(self):
        last_maintenance = 0.0
        while not self.stop_event.is_set() or not self.critical_queue.empty() or not self.regular_queue.empty():
            now = time.time()
            if now - last_maintenance >= 1.0:
                try:
                    self.attendance.reconcile()
                except Exception as exc:
                    self.last_error = str(exc)
                    print(f"[OPERATIONS] Attendance reconciliation failed: {exc}")
                last_maintenance = now
            task = None
            try:
                task = self.critical_queue.get_nowait()
            except queue.Empty:
                try:
                    task = self.regular_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
            try:
                _process_runtime_side_effects(
                    task, self.db, self.attendance, self.alert_mgr,
                    self.snapshot_dir, self.cam_id, self.cam_location,
                    self.attendance_lock, self.center_mode_getter,
                )
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
                print(f"[OPERATIONS] Side-effect task failed: {exc}")
            finally:
                if self.critical_queue.empty() and self.regular_queue.empty():
                    time.sleep(0.001)

    def stop(self):
        self.stop_event.set()


class RuntimeWorker(threading.Thread):
    def __init__(self, runtime_dir, vision, db, attendance, alert_mgr,
                 frame_buffer, inference_state, profiler, operations,
                 vision_lock, stop_event, started_epoch, started_at,
                 cam_id, cam_location, snapshot_dir, capture_worker,
                 inference_worker):
        super().__init__(name="optivox-runtime", daemon=True)
        self.runtime_dir = runtime_dir
        self.vision = vision
        self.db = db
        self.attendance = attendance
        self.alert_mgr = alert_mgr
        self.frame_buffer = frame_buffer
        self.inference_state = inference_state
        self.profiler = profiler
        self.operations = operations
        self.vision_lock = vision_lock
        self.stop_event = stop_event
        self.started_epoch = started_epoch
        self.started_at = started_at
        self.cam_id = cam_id
        self.cam_location = cam_location
        self.snapshot_dir = snapshot_dir
        self.capture_worker = capture_worker
        self.inference_worker = inference_worker
        self.last_error = None
        self._last_state = 0.0
        self._last_frame = 0.0
        self._last_commands = 0.0

    def run(self):
        while not self.stop_event.is_set():
            now = time.time()
            result = self.inference_state.snapshot()
            packet = self.frame_buffer.snapshot()
            display = result.get("display") if result else (packet.get("frame") if packet else None)
            display_for_output = display
            if display is not None:
                display_size = (
                    int(CONFIG.get("DISPLAY_WIDTH", display.shape[1])),
                    int(CONFIG.get("DISPLAY_HEIGHT", display.shape[0])),
                )
                if (display.shape[1], display.shape[0]) != display_size:
                    display_for_output = cv2.resize(
                        display, display_size, interpolation=cv2.INTER_LINEAR)
            events = result.get("events", []) if result else []
            latest_age = packet.get("age_ms", 0.0) if packet else 0.0
            performance = self.profiler.snapshot(
                queue_depths=self.operations.queue_depths(),
                latest_frame_age_ms=latest_age,
                current_source_frame_id=packet.get("frame_id") if packet else None,
                current_inference_frame_id=result.get("frame_id") if result else None,
            )
            with self.vision_lock:
                performance["vision"] = self.vision.performance_snapshot()
            fps = performance.get("inference_fps", 0.0)
            worker_error = self.inference_worker.last_error or self.operations.last_error
            if self.capture_worker.last_error:
                worker_error = self.capture_worker.last_error
            camera_health = self.capture_worker.health_status()
            try:
                if now - self._last_state >= 1.0:
                    _publish_runtime_heartbeat(
                        self.runtime_dir,
                        "online" if camera_health.get("state") == "HEALTHY" else "degraded",
                        str(camera_health.get("state", "UNKNOWN")).lower(), fps,
                        self.started_at, worker_error, performance=performance,
                        frame_id=result.get("frame_id") if result else None,
                        frame_age_ms=result.get("frame_age_ms") if result else latest_age,
                    )
                    with self.vision_lock:
                        _publish_runtime_state(
                            self.runtime_dir, self.vision, display_for_output,
                            self.cam_id, self.cam_location, fps,
                            self.started_epoch, self.started_at, events,
                            worker_error, source_result=result,
                            performance=performance,
                        )
                    self._last_state = now
                if display_for_output is not None and now - self._last_frame >= 0.16:
                    _publish_runtime_frame(self.runtime_dir, display_for_output)
                    self._last_frame = now
                if now - self._last_commands >= 0.5:
                    raw = packet.get("frame").copy() if packet and packet.get("frame") is not None else None
                    with self.vision_lock:
                        _process_runtime_commands(
                            self.runtime_dir, self.db, self.attendance,
                            self.alert_mgr, self.vision, raw,
                            self.snapshot_dir, self.cam_id, self.cam_location,
                        )
                    self._last_commands = now
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
                print(f"[RUNTIME] Worker update failed: {exc}")
            time.sleep(0.05)

# Main function
def main():
    print("=" * 70)
    print(f"  INTELLIGENT SECURITY & ATTENDANCE SYSTEM  v{RUNTIME_VERSION}")
    print("=" * 70)
    ensure_dirs()
    runtime_dir = _runtime_dir()
    started_at_epoch = time.time()
    started_at = _runtime_now()
    runtime_mode = os.environ.get("OPTIVOX_RUNTIME_MODE", "development").strip().lower()
    edge_validation = validate_edge_configuration(
        CONFIG, _BASE_DIR, runtime_mode, manifest_path=CONFIG["MODEL_MANIFEST_PATH"])
    if edge_validation.get("issues"):
        print("[CONFIG] " + "; ".join(edge_validation["issues"]))
        if is_strict_mode(runtime_mode):
            _publish_runtime_heartbeat(
                runtime_dir, "offline", "configuration_error", 0.0,
                started_at, "Unsafe edge configuration.")
            try:
                append_runtime_health_event(runtime_dir, {
                    "type": "CONFIGURATION_REJECTED",
                    "issues": edge_validation["issues"],
                    "process": process_identity(),
                })
            except Exception:
                pass
            return
    _publish_runtime_capabilities(runtime_dir, started_at)
    _publish_runtime_heartbeat(
        runtime_dir,
        "starting",
        "disconnected",
        0.0,
        started_at,
    )

    global VOICE
    VOICE = VoiceManager(CONFIG)
    if VOICE.enabled:
        print("[INFO] Voice manager ready.")
    else:
        print("[WARN] Voice manager disabled or pyttsx3 unavailable.")

    db = EventDatabase()
    db.setup_database()
    print("[INFO] Database ready.")
    recovered_sessions = db.close_open_presence_sessions("startup_recovery")
    if recovered_sessions:
        print(f"[DB] Closed {recovered_sessions} stale presence session(s) from a previous run.")

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
    _publish_runtime_capabilities(runtime_dir, started_at, vision)

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
        _publish_runtime_heartbeat(
            runtime_dir,
            "offline",
            "disconnected",
            0.0,
            started_at,
            "Camera could not be opened.",
        )
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.get("CAPTURE_WIDTH", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.get("CAPTURE_HEIGHT", 720))
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera active at {actual_w}x{actual_h}.")

    snapshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    _set_enrollment_status(runtime_dir, "idle", "No enrollment in progress.")

    print()
    print("=" * 70)
    print("  CONTROLS")
    print("    q  -- quit")
    print("    k  -- center attendance mode")
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

    ui = {
        "mode": "normal",
        "prompt": "",
        "text": "",
        "message": "System ready. Use the right-side panel for controls.",
        "assistant_answer": "",
        "clock_mode": None,
        "center_attendance": {
            "enabled": False,
            "phase": "READY",
            "status": "Stand one person inside the center guide.",
            "candidate": None,
            "stable_frames": 0,
            "started_at": None,
            "progress": 0.0,
            "completed_name": None,
            "liveness_phase": "CENTER",
            "liveness_started_at": None,
            "liveness_hold_started_at": None,
            "liveness_passed": False,
            "liveness_first_turn": None,
            "liveness_yaw": None,
            "last_gate_key": None,
        },
        "show_details": True,
        "roster": [],
        "hitboxes": [],
        "toggles": {
            "heatmap": bool(CONFIG.get("SHOW_HEATMAP", False)),
            "objects": bool(CONFIG.get("SHOW_OBJECT_BOXES", True)),
            "pose": bool(CONFIG.get("SHOW_POSE_LANDMARKS", False)),
            "danger": bool(CONFIG.get("DANGER_DETECTION", {}).get("ENABLED", False)),
            "age": bool(CONFIG.get("SHOW_AGE_GENDER", False)),
            "zones": bool(CONFIG.get("SHOW_ZONES_GRID", False)),
            "count_line": bool(CONFIG.get("SHOW_COUNT_LINE", False)),
            "face_mesh": bool(CONFIG.get("SHOW_FACE_MESH", False)),
            "hands": bool(CONFIG.get("SHOW_HAND_LANDMARKS", False)),
            "density": bool(CONFIG.get("CROWD_INTELLIGENCE", {}).get("SHOW_DENSITY_HEATMAP", False)),
        },
    }

    vision_lock = threading.RLock()
    attendance_lock = threading.RLock()
    liveness_challenge = LivenessChallenge(CONFIG.get("ATTENDANCE", {}))

    def refresh_roster():
        try:
            roster = []
            for row in db.get_known_face_names(limit=200):
                name = row["name"]
                pid = db.get_person_id(name)
                today_rows = db.attendance_report(days=1, person_id=pid) if pid else []
                today = today_rows[0] if today_rows else {}
                clock_in = today["clock_in"] if today and "clock_in" in today.keys() else None
                clock_out = today["clock_out"] if today and "clock_out" in today.keys() else None
                roster.append({"id": pid, "name": name, "clock_in": clock_in, "clock_out": clock_out})
            ui["roster"] = roster
        except Exception:
            ui["roster"] = []

    def toggle_center_attendance():
        state = ui["center_attendance"]
        state["enabled"] = not state.get("enabled", False)
        liveness_challenge.reset_all()
        state.update({
            "phase": "WAITING" if state["enabled"] else "READY",
            "status": ("Stand alone in the center guide. Hold still for a few seconds."
                       if state["enabled"] else "Center attendance is OFF."),
            "candidate": None,
            "stable_frames": 0,
            "started_at": None,
            "progress": 0.0,
            "completed_name": None,
            "liveness_phase": "CENTER",
            "liveness_started_at": None,
            "liveness_hold_started_at": None,
            "liveness_passed": False,
            "liveness_first_turn": None,
            "liveness_yaw": None,
        })
        ui["message"] = ("Center Attendance ON: one person, centered, hold still."
                          if state["enabled"] else "Center Attendance OFF.")

    def update_center_attendance(faces_info):
        """Advance focused attendance while all other vision work stays live."""
        state = ui["center_attendance"]
        if not state.get("enabled"):
            return

        now = time.time()
        cfg = CONFIG["ATTENDANCE"]
        total_faces = len(faces_info)
        width = float(CONFIG.get("PROCESSING_WIDTH", 1280))
        height = float(CONFIG.get("PROCESSING_HEIGHT", 720))
        valid = []
        for face in faces_info:
            if not face.get("quality_ok", False):
                continue
            name = str(face.get("name", "UNKNOWN"))
            if (name in ("UNKNOWN", "SPOOF") or name.startswith("STRANGER_")
                    or normalize_identity_state(face.get("identity_state")) != "CONFIRMED"
                    or str(face.get("liveness_status") or "").upper() in {"SUSPECT", "SPOOF_SUSPECT"}):
                continue
            x1, y1, x2, y2 = face.get("bbox", (0, 0, 0, 0))
            cx = ((x1 + x2) * 0.5) / width
            cy = ((y1 + y2) * 0.5) / height
            if (cfg.get("CENTER_MODE_X_MIN", 0.34) <= cx <= cfg.get("CENTER_MODE_X_MAX", 0.66)
                    and cfg.get("CENTER_MODE_Y_MIN", 0.12) <= cy <= cfg.get("CENTER_MODE_Y_MAX", 0.82)):
                valid.append(face)

        # Focus mode is deliberately singular: do not guess if another face
        # is visible, even if one of the faces is known.
        if total_faces != 1 or len(valid) != 1:
            liveness_challenge.reset_all()
            state.update({
                "phase": "BLOCKED" if total_faces > 1 else "WAITING",
                "status": ("Only one person may be visible."
                           if total_faces > 1 else "Move one person into the center guide."),
                "candidate": None,
                "stable_frames": 0,
                "started_at": None,
                "progress": 0.0,
                "liveness_phase": "CENTER",
                "liveness_started_at": None,
                "liveness_hold_started_at": None,
                "liveness_passed": False,
                "liveness_first_turn": None,
                "liveness_yaw": None,
                "last_gate_key": None,
            })
            ui["message"] = state["status"]
            return

        face = valid[0]
        name = str(face.get("name"))
        confidence = float(face.get("confidence", 0.0))
        if state.get("completed_name") and state["completed_name"] == name:
            state.update({"phase": "COMPLETE", "status": f"{name} is clocked in. Step away for the next person."})
            return
        if state.get("completed_name") and state["completed_name"] != name:
            state["completed_name"] = None

        if state.get("candidate") != name:
            liveness_challenge.reset_all()
            state.update({
                "candidate": name,
                "stable_frames": 1,
                "started_at": now,
                "liveness_phase": "CENTER",
                "liveness_started_at": now,
                "liveness_hold_started_at": None,
                "liveness_passed": not cfg.get("CENTER_MODE_REQUIRE_LIVENESS", True),
                "liveness_first_turn": None,
                "last_gate_key": None,
            })
        else:
            state["stable_frames"] += 1

        require_liveness = bool(cfg.get("CENTER_MODE_REQUIRE_LIVENESS", True))
        challenge_key = face.get("entity_id") or f"track:{face.get('oid')}"
        challenge = liveness_challenge.update(
            challenge_key, face.get("yaw"), now, enabled=require_liveness)
        state.update({
            "liveness_phase": challenge.get("phase", "CENTER"),
            "liveness_started_at": challenge.get("started_at"),
            "liveness_hold_started_at": challenge.get("hold_started_at"),
            "liveness_passed": bool(challenge.get("passed")),
            "liveness_first_turn": challenge.get("first_turn_sign"),
            "liveness_yaw": challenge.get("yaw"),
        })
        if require_liveness and not challenge.get("passed"):
            state["status"] = challenge.get("message", "Complete the liveness challenge.")
            state["progress"] = min(0.9, float(challenge.get("progress", 0.0)))
            ui["message"] = state["status"]
            return

        dwell = float(cfg.get("CENTER_MODE_DWELL_SEC", 3.0))
        stable_needed = int(cfg.get("CENTER_MODE_STABLE_FRAMES", 6))
        elapsed = max(0.0, now - float(state.get("started_at") or now))
        progress = min(1.0, elapsed / max(dwell, 0.1))
        state["progress"] = progress
        state["phase"] = "VERIFYING"
        if state.get("liveness_passed"):
            state["status"] = f"Verified {name}: {elapsed:.1f}/{dwell:.1f}s"

        if elapsed < dwell or state["stable_frames"] < stable_needed:
            ui["message"] = state["status"]
            return

        person_id = db.get_active_person_id(name)
        correlation_gate = vision.correlation.attendance_decision(
            int(face.get("oid", -1)),
            expected_name=name,
            active_roster=person_id is not None,
        )
        session_id = None
        evidence_id = None
        entity_id = face.get("entity_id")
        track_id = face.get("oid")
        entity_state = vision.correlation.entities.get(int(track_id or -1))
        track_generation = entity_state.track_generation if entity_state else 1
        observed_at = face.get("observed_at") or _utc_now()
        gate_eligible = bool(correlation_gate.get("eligible"))
        gate_reason = str(correlation_gate.get("reason") or (
            "eligible" if gate_eligible else "not_eligible"))
        gate_key = f"center:{entity_id or f'track:{track_id}'}:{'eligible' if gate_eligible else gate_reason}"

        # Center mode has its own focused path, so persist the same evidence
        # chain as normal automatic attendance without writing one row per
        # display frame. A new row is created only when the gate state/reason
        # changes for the current focused entity.
        if entity_id and state.get("last_gate_key") != gate_key:
            canonical_person_id = person_id if correlation_gate.get("identity_state") == "CONFIRMED" else None
            canonical_label = name if canonical_person_id is not None else "UNKNOWN"
            session_id = db.upsert_presence_session(
                entity_id=entity_id,
                track_id=track_id,
                person_id=canonical_person_id,
                label=canonical_label,
                identity_state=correlation_gate.get("identity_state", "CONFIRMED"),
                liveness_status=correlation_gate.get("liveness_state", "REAL"),
                camera_id=cam_id,
                confidence=confidence,
                source_frame_id=face.get("source_frame_id"),
                observed_at=observed_at,
                track_generation=track_generation,
            )
            evidence_decision = (
                "confirmed" if gate_eligible else
                "spoof_or_uncertain" if str(correlation_gate.get("liveness_state") or "").upper() != "REAL"
                else "unresolved"
            )
            evidence_id = db.record_recognition_evidence(
                entity_id=entity_id,
                presence_session_id=session_id,
                track_id=track_id,
                person_id=canonical_person_id,
                candidate_name=name if canonical_person_id is not None else "UNKNOWN",
                decision=evidence_decision,
                similarity=face.get("current_observation_similarity", confidence),
                quality_score=face.get("quality_score", 0.0),
                quality_ok=face.get("quality_ok", False),
                liveness_status=correlation_gate.get("liveness_state", "REAL"),
                identity_state=correlation_gate.get("identity_state", "CONFIRMED"),
                reason=("center_attendance_challenge_passed" if gate_eligible
                        else f"center_gate_rejected:{gate_reason}"),
                source_frame_id=face.get("source_frame_id"),
                observed_at=observed_at,
            )
            db.record_attendance_decision(
                decision_key=gate_key,
                decision="eligible" if gate_eligible else "rejected",
                reason=gate_reason,
                person_id=canonical_person_id,
                entity_id=entity_id,
                presence_session_id=session_id,
                recognition_evidence_id=evidence_id,
                identity_state=correlation_gate.get("identity_state"),
                liveness_status=correlation_gate.get("liveness_state"),
                quality_score=face.get("quality_score"),
                recognition_confidence=face.get("current_observation_similarity", confidence),
                source_frame_id=face.get("source_frame_id"),
                observed_at=observed_at,
                details={
                    "method": "center_attendance",
                    "challenge_phase": state.get("liveness_phase"),
                    "track_generation": track_generation,
                    "gate": correlation_gate,
                },
            )
            state["last_gate_key"] = gate_key

        if not gate_eligible:
            state.update({
                "phase": "VERIFYING",
                "status": f"Verification paused: {gate_reason}.",
            })
            ui["message"] = state["status"]
            return

        # A policy rejection (for example a non-school day) is separate from
        # the recognition gate. Preserve that outcome in the same audit trail.
        result = attendance.clock_in_verified(
            name, cam_id, cam_location, confidence=confidence,
            method="center_attendance", presence_session_id=session_id,
            recognition_evidence_id=evidence_id,
            source_frame_id=face.get("source_frame_id"),
            liveness_status=correlation_gate.get("liveness_state", "REAL"),
            identity_state=correlation_gate.get("identity_state", "CONFIRMED"),
            attendance_eligible=True,
            quality_ok=bool(correlation_gate.get("quality_ok")))
        if "error" in result:
            if entity_id:
                db.record_attendance_decision(
                    decision_key=f"{gate_key}:policy",
                    decision="rejected",
                    reason=f"attendance_policy:{result['error']}",
                    person_id=person_id,
                    entity_id=entity_id,
                    presence_session_id=session_id,
                    recognition_evidence_id=evidence_id,
                    identity_state=correlation_gate.get("identity_state"),
                    liveness_status=correlation_gate.get("liveness_state"),
                    quality_score=face.get("quality_score"),
                    recognition_confidence=face.get("current_observation_similarity", confidence),
                    source_frame_id=face.get("source_frame_id"),
                    observed_at=observed_at,
                    details={"method": "center_attendance", "policy_result": result},
                )
            state.update({"phase": "BLOCKED", "status": result["error"], "progress": 0.0})
            return
        state.update({
            "phase": "COMPLETE",
            "status": (f"{name} attendance complete. Step away for the next person."
                       if "clocked_in_at" in result else f"{name} is already clocked in."),
            "progress": 1.0,
            "completed_name": name,
            "candidate": None,
            "stable_frames": 0,
            "started_at": None,
        })
        ui["message"] = state["status"]
        refresh_roster()

    def _set_feature_unlocked(feature):
        ui["toggles"][feature] = not ui["toggles"].get(feature, False)
        if feature == "heatmap":
            CONFIG["SHOW_HEATMAP"] = ui["toggles"][feature]
            CONFIG.setdefault("CROWD_INTELLIGENCE", {})["SHOW_DENSITY_HEATMAP"] = ui["toggles"][feature]
        elif feature == "objects":
            CONFIG["SHOW_OBJECT_BOXES"] = ui["toggles"][feature]
        elif feature == "pose":
            CONFIG["SHOW_POSE_LANDMARKS"] = ui["toggles"][feature]
        elif feature == "age":
            CONFIG["SHOW_AGE_GENDER"] = ui["toggles"][feature]
        elif feature == "zones":
            CONFIG["SHOW_ZONES_GRID"] = ui["toggles"][feature]
        elif feature == "count_line":
            CONFIG["SHOW_COUNT_LINE"] = ui["toggles"][feature]
        elif feature == "face_mesh":
            CONFIG["SHOW_FACE_MESH"] = ui["toggles"][feature]
        elif feature == "hands":
            CONFIG["SHOW_HAND_LANDMARKS"] = ui["toggles"][feature]
        elif feature == "density":
            CONFIG.setdefault("CROWD_INTELLIGENCE", {})["SHOW_DENSITY_HEATMAP"] = ui["toggles"][feature]
        elif feature == "danger":
            danger_cfg = CONFIG.setdefault("DANGER_DETECTION", {})
            danger_cfg["ENABLED"] = ui["toggles"][feature]
            if danger_cfg["ENABLED"] and getattr(vision, "danger_detector", None) is None:
                try:
                    vision.danger_detector = DangerDetector(CONFIG)
                except Exception as exc:
                    ui["toggles"][feature] = False
                    danger_cfg["ENABLED"] = False
                    ui["message"] = f"Danger detector could not start: {exc}"
            if getattr(vision, "danger_detector", None) is not None:
                vision.danger_detector.enabled = danger_cfg["ENABLED"]
        ui["message"] = f"{feature.replace('_', ' ').title()} = {'ON' if ui['toggles'][feature] else 'OFF'}"

    def set_feature(feature):
        with vision_lock:
            _set_feature_unlocked(feature)

    def handle_panel_click(event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for x1, y1, x2, y2, label in ui.get("hitboxes", []):
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                continue
            if label == "ENROLL PERSON":
                ui.update({"mode": "text", "prompt": "Enrollment name", "text": "", "clock_mode": None, "message": "Type a name in the panel, press ENTER, then follow the enrollment window."})
            elif label == "ASK ASSISTANT":
                ui.update({"mode": "text", "prompt": "Assistant question", "text": "", "clock_mode": None, "message": "Type a question in the panel, then press ENTER."})
            elif label.startswith("CENTER ATTENDANCE"):
                toggle_center_attendance()
            elif label == "CLOCK IN":
                refresh_roster()
                ui.update({"mode": "normal", "clock_mode": "in", "message": "Click a person in the roster to clock in."})
            elif label == "CLOCK OUT":
                refresh_roster()
                ui.update({"mode": "normal", "clock_mode": "out", "message": "Click a person in the roster to clock out."})
            elif label.startswith("DETAILS:"):
                ui["show_details"] = not ui.get("show_details", True)
            elif label == "CLEAR STATUS":
                ui["message"] = "System ready."
                ui["assistant_answer"] = ""
            elif "HEATMAP" in label:
                set_feature("heatmap")
            elif "OBJECT BOXES" in label:
                set_feature("objects")
            elif "POSE" in label:
                set_feature("pose")
            elif "DANGER" in label:
                set_feature("danger")
            elif "AGE/GENDER" in label:
                set_feature("age")
            elif "ZONES/GRID" in label:
                set_feature("zones")
            elif "COUNT LINE" in label:
                set_feature("count_line")
            elif "FACE MESH" in label:
                set_feature("face_mesh")
            elif "HAND LANDMARKS" in label:
                set_feature("hands")
            elif "DENSITY" in label:
                set_feature("density")
            elif label.startswith("PERSON:"):
                _, pid, mode = label.split(":", 2)
                try:
                    if mode == "out":
                        result = attendance.manual_clock_out(next(row["name"] for row in ui["roster"] if str(row["id"]) == pid))
                    else:
                        result = db.attendance_clock_in(int(pid), cam_id, cam_location)
                    ui["message"] = f"Attendance result: {result}"
                    refresh_roster()
                except Exception as exc:
                    ui["message"] = f"Attendance action failed: {exc}"
                ui["clock_mode"] = None
            return

    cv2.namedWindow("Security & Attendance Feed", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Security & Attendance Feed", handle_panel_click)

    performance_cfg = CONFIG.get("PERFORMANCE", {})
    profiler = PerformanceProfiler(
        window_size=performance_cfg.get("PROFILER_WINDOW_FRAMES", 120),
        enabled=performance_cfg.get("ENABLE_PROFILING", True),
    )
    frame_buffer = LatestFrameBuffer(
        max_age_ms=performance_cfg.get("MAX_INFERENCE_FRAME_AGE_MS", 250),
    )
    capture_proxy = LatestFrameCaptureProxy(frame_buffer)
    inference_state = LatestInferenceState()
    stop_event = threading.Event()
    capture_worker = CameraCaptureWorker(
        cap, frame_buffer, profiler, stop_event,
        target_size=(
            CONFIG.get("CAPTURE_WIDTH", 1280),
            CONFIG.get("CAPTURE_HEIGHT", 720),
        ), runtime_dir=runtime_dir, camera_id=cam_id)
    operations = OperationalWorker(
        db, attendance, alert_mgr, snapshot_dir, cam_id, cam_location,
        attendance_lock, lambda: bool(ui["center_attendance"].get("enabled")),
        profiler,
        regular_size=performance_cfg.get("SIDE_EFFECT_QUEUE_SIZE", 256),
        critical_size=performance_cfg.get("CRITICAL_QUEUE_SIZE", 64),
    )
    inference_worker = InferenceWorker(
        vision, frame_buffer, inference_state, profiler, operations,
        stop_event, vision_lock,
        max_age_ms=performance_cfg.get("MAX_INFERENCE_FRAME_AGE_MS", 250),
    )
    inference_worker.cam_id = cam_id
    inference_worker.set_processing_size((
        CONFIG.get("PROCESSING_WIDTH", CONFIG.get("CAPTURE_WIDTH", 1280)),
        CONFIG.get("PROCESSING_HEIGHT", CONFIG.get("CAPTURE_HEIGHT", 720)),
    ))
    runtime_worker = RuntimeWorker(
        runtime_dir, vision, db, attendance, alert_mgr,
        frame_buffer, inference_state, profiler, operations,
        vision_lock, stop_event, started_at_epoch, started_at,
        cam_id, cam_location, snapshot_dir, capture_worker,
        inference_worker,
    )
    operations.start()
    capture_worker.start()
    inference_worker.start()
    runtime_worker.start()
    print("[PERF] Capture, inference, operations, and runtime workers started.")

    frame_count = 0
    start_time = started_at_epoch
    last_raw_frame = None
    latest_result = None
    latest_result_id = 0

    try:
        while True:
            packet = frame_buffer.snapshot()
            if packet and packet.get("frame") is not None:
                last_raw_frame = packet["frame"].copy()

            result = inference_state.snapshot()
            if result and result.get("frame_id") != latest_result_id:
                latest_result = result
                latest_result_id = result.get("frame_id") or latest_result_id
                with attendance_lock:
                    update_center_attendance(result.get("faces_info") or [])

            if latest_result:
                display = latest_result["display"].copy()
                faces_info = latest_result.get("faces_info") or []
                events = latest_result.get("events") or []
                display_frame_id = latest_result.get("frame_id")
                display_captured_at = latest_result.get("captured_at")
            elif packet and packet.get("frame") is not None:
                display = packet["frame"].copy()
                faces_info = []
                events = []
                display_frame_id = packet.get("frame_id")
                display_captured_at = packet.get("captured_at")
            else:
                display = np.zeros((720, 1280, 3), dtype=np.uint8)
                faces_info = []
                events = []
                display_frame_id = None
                display_captured_at = None

            display_size = (
                int(CONFIG.get("DISPLAY_WIDTH", display.shape[1])),
                int(CONFIG.get("DISPLAY_HEIGHT", display.shape[0])),
            )
            if (display.shape[1], display.shape[0]) != display_size:
                display = cv2.resize(display, display_size,
                                     interpolation=cv2.INTER_LINEAR)

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            h_disp = display.shape[0]
            _draw_center_attendance_guide(display, ui, CONFIG)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            ai_color = (0, 255, 255) if assistant.is_listening() else (140, 140, 140)
            ai_text = "AI: LIVE" if assistant.is_listening() else "AI: idle"
            cv2.putText(display, ai_text, (10, h_disp - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, ai_color, 2)
            cv2.putText(display, f"Cam: {cam_id} ({cam_location})",
                        (220, h_disp - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if frame_count % 15 == 0 or not ui.get("roster"):
                refresh_roster()

            performance_snapshot = profiler.snapshot(
                queue_depths=operations.queue_depths(),
                latest_frame_age_ms=(packet or {}).get("age_ms", 0.0),
                current_source_frame_id=(packet or {}).get("frame_id"),
                current_inference_frame_id=(latest_result or {}).get("frame_id"),
            )
            cv2.imshow("Security & Attendance Feed",
                       _draw_command_panel(display, faces_info, events, vision, db, assistant, ui,
                                           cam_id, cam_location,
                                           performance_snapshot))
            display_age_ms = (
                max(0.0, (time.time() - float(display_captured_at)) * 1000.0)
                if display_captured_at else None
            )
            profiler.record_display(display_frame_id, display_age_ms)

            #Keyboard controls
            display_delay = max(1, int(1000 / max(1, int(
                performance_cfg.get("DISPLAY_LOOP_FPS", 30)))))
            key = cv2.waitKey(display_delay) & 0xFF

            if ui["mode"] == "text":
                if key == 27:
                    ui.update({"mode": "normal", "text": "", "prompt": "", "message": "Input cancelled."})
                elif key in (13, 10):
                    entered = ui.get("text", "").strip()
                    prompt = ui.get("prompt", "")
                    if not entered:
                        ui.update({"mode": "normal", "message": "Nothing entered; action cancelled."})
                    elif prompt == "Enrollment name":
                        ui.update({"mode": "normal", "text": "", "prompt": "", "message": f"Starting enrollment for {entered}."})
                        _set_enrollment_status(
                            runtime_dir,
                            "capturing",
                            f"Capturing enrollment samples for {entered}.",
                            {"mode": "multi_angle", "person_name": entered},
                        )
                        try:
                            # Enrollment mutates the face index, so it takes
                            # the same lock as inference for this explicit action.
                            with vision_lock:
                                ok = _multi_angle_enrollment(
                                    vision, db, capture_proxy, entered,
                                    min_embeddings=CONFIG.get("MIN_ENROLLMENT_EMBEDDINGS", 5),
                                    max_embeddings=CONFIG.get("MAX_ENROLLMENT_EMBEDDINGS", 10),
                                )
                            ui["message"] = f"{entered} enrollment {'completed' if ok else 'cancelled or failed'}."
                            _set_enrollment_status(
                                runtime_dir,
                                "completed" if ok else "cancelled",
                                f"{entered} enrollment {'completed' if ok else 'cancelled or failed'}.",
                                {"mode": "multi_angle", "person_name": entered},
                            )
                            refresh_roster()
                        except Exception as exc:
                            ui["message"] = f"Enrollment failed: {exc}"
                            _set_enrollment_status(
                                runtime_dir,
                                "failed",
                                f"Enrollment failed: {exc}",
                                {"mode": "multi_angle", "person_name": entered},
                            )
                    else:
                        ui.update({"mode": "normal", "text": "", "prompt": "", "message": "Local-only Assistant input received."})
                        ui["assistant_answer"] = "External AI is disabled in this panel. Approve OpenAI data sharing before enabling live Assistant answers."
                elif key in (8, 127):
                    ui["text"] = ui.get("text", "")[:-1]
                elif 32 <= key <= 126:
                    ui["text"] = (ui.get("text", "") + chr(key))[:100]
                continue

            if key == ord('q'):
                break

            elif key == ord('k'):
                toggle_center_attendance()

            elif key == ord('a'):
                ui.update({"mode": "text", "prompt": "Assistant question", "text": "", "clock_mode": None,
                           "message": "Type a question in the panel and press ENTER."})

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
                ui.update({"mode": "text", "prompt": "Enrollment name", "text": "", "clock_mode": None,
                           "message": "Type a name in the panel and press ENTER."})

            elif key == ord('e') and False:
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
                                vision, db, capture_proxy, name,
                                min_embeddings=CONFIG.get("MIN_ENROLLMENT_EMBEDDINGS", 5),
                                max_embeddings=CONFIG.get("MAX_ENROLLMENT_EMBEDDINGS", 10),
                            )
                    except Exception as e:
                        print(f"[ENROLL] Failed: {e}")

            elif key == ord('c'):
                refresh_roster()
                ui.update({"mode": "normal", "clock_mode": "out",
                           "message": "Click a person in the roster to clock out."})

            elif key == ord('c') and False:
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
                refresh_roster()
                ui.update({"mode": "normal", "clock_mode": "in",
                           "message": "Click a person in the roster to clock in."})

            elif key == ord('i') and False:
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
                        with vision_lock:
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
        try:
            stop_event.set()
            operations.stop()
            frame_buffer.close()
            for worker in (capture_worker, inference_worker, runtime_worker, operations):
                worker.join(timeout=5.0)
            print("[PERF] Workers stopped cleanly.")
        except Exception as exc:
            print(f"[PERF] Worker shutdown warning: {exc}")
        try:
            with attendance_lock:
                closed = attendance.close_for_shutdown()
            if closed:
                print(f"[ATTENDANCE] Closed {len(closed)} open record(s) at shutdown.")
        except Exception as exc:
            print(f"[ATTENDANCE] Shutdown reconciliation failed: {exc}")
        try:
            closed_sessions = db.close_open_presence_sessions("runtime_shutdown")
            if closed_sessions:
                print(f"[DB] Closed {closed_sessions} open presence session(s) at shutdown.")
        except Exception as exc:
            print(f"[DB] Presence-session shutdown reconciliation failed: {exc}")
        try:
            _publish_performance_summary(
                runtime_dir, profiler, vision, started_at,
                operations=operations, frame_buffer=frame_buffer)
            print(f"[PERF] Summary written to {os.path.join(runtime_dir, 'performance_summary.json')}")
        except Exception as exc:
            print(f"[PERF] Summary write failed: {exc}")
        try:
            _publish_runtime_heartbeat(
                runtime_dir,
                "offline",
                "disconnected",
                0.0,
                started_at,
                "Runtime stopped.",
            )
        except Exception as exc:
            print(f"[RUNTIME] Shutdown heartbeat failed: {exc}")

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


if __name__ == "__main__":
    main()
