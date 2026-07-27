import os
import time
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import pickle
import mediapipe as mp
from collections import OrderedDict, deque, defaultdict
from insightface.app import FaceAnalysis
import faiss
from scipy.spatial.distance import cosine

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not found. Crowd dynamics disabled.")
    print("  pip install scikit-learn")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not found. Object detection disabled.")
    print("  pip install ultralytics")

CONFIG = {
    "MODEL_PATH": "yolov8n.pt",
    "DATABASE_FILE": os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "face_database.pkl"),
    "UNKNOWN_DB_FILE": os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "unknown_strangers.pkl"),

    "NMS_CONF_THRESHOLD": 0.4,
    "NMS_IOU_THRESHOLD": 0.5,
    "YOLO_CONF": 0.45,
    "HAND_DETECTION_CONF": 0.5,

    "FACE_RECOG_THRESHOLD": 0.35,
    "RECOGNITION_HISTORY_LENGTH": 10,
    "PROFILE_CONFIDENCE_THRESHOLD": 5,

    "COUNT_LINE_Y": 300,
    "LOITERING_ZONE": (400, 100, 800, 400),

    "BLUR_THRESHOLD": 100.0,
    "BRIGHTNESS_MIN": 50,
    "BRIGHTNESS_MAX": 200,
    "CONTRAST_THRESHOLD": 50,

    "SUSPICION_DECAY_RATE": 0.85,
    "SUSPICION_DECAY_INTERVAL": 1.0,
    "SUSPICION_POINTS": {
        "HESITATION": 3, "PACING": 5, "SCANNING": 2,
        "SPATIAL_ANOMALY": 2, "LOITERING": 4, "RUNNING": 3,
        "OBJECT_INTERACTION": 4, "CROWD_FORMING": 2,
    },
    "HESITATION_SPEED_THRESHOLD": 2.0,
    "HESITATION_STOP_TIME_SEC": 5.0,
    "PACING_WINDOW_SEC": 10.0,
    "PACING_DIRECTION_CHANGES": 6,
    "SCANNING_VAR_THRESHOLD": 250.0,
    "SCANNING_DISP_THRESHOLD": 50.0,
    "CROWD_MIN_SIZE": 4,
    "CROWD_RADIUS": 100,
    "INTERACTION_IOU_THRESHOLD": 0.1,
    "SPATIAL_GRID_SIZE": (20, 20),
    "SPATIAL_ANOMALY_THRESHOLD": 0.02,
    "HEATMAP_UPDATE_INTERVAL": 5.0,

    "STRESS_THRESHOLDS": {"LOW": 20, "MEDIUM": 50},

    "SUSPICION_MIN_TRACK_FRAMES": 30,
    "SUSPICION_BEHAVIOR_COOLDOWN": 10.0,

    "ANTI_SPOOFING": {
        "ENABLED": True,
        "ENABLE_ML_CLASSIFIER": False,
        "ML_CLASSIFIER_THRESHOLD": 0.7,
        "EAR_HISTORY_FRAMES": 30,
        "BLINK_EAR_THRESHOLD": 0.2,
        "MIN_TIME_BETWEEN_BLINKS_SEC": 0.5,
        "MIN_BLINKS_FOR_LIVENESS": 2,
        "LIVENESS_TIME_WINDOW_SEC": 15.0,
        "POSE_HISTORY_FRAMES": 30,
        "DEPTH_VARIANCE_THRESHOLD": 0.01,
        "ANTI_SPOOF_MODEL_PATH": "antispoof_model.bin",
    },

    "DANGEROUS_OBJECTS": {
        "knife", "scissors", "gun", "pistol", "rifle", "weapon",
        "bat", "axe", "hammer", "baseball bat", "cleaver",
        "machete", "sword", "explosive", "bomb", "fire",
    },

    "HAND_MAX_NUM": 2,
    "HAND_MIN_DETECTION": 0.5,
    "HAND_MIN_TRACKING": 0.5,
    "SHOW_HAND_LANDMARKS": False,
    "SHOW_HAND_CONNECTIONS": False,

    "SHOW_OBJECT_BOXES": True,
    "OBJECT_DISPLAY_CATEGORIES": {
        "person": (0, 255, 0),
        "vehicle": (255, 165, 0),
        "electronics": (255, 0, 255),
        "furniture": (0, 165, 255),
        "animal": (0, 255, 255),
        "food": (0, 128, 0),
        "utensil": (128, 0, 128),
        "sports": (255, 200, 0),
        "accessory": (180, 130, 255),
        "clothing": (100, 200, 200),
        "household": (200, 150, 100),
        "infrastructure": (150, 150, 150),
        "misc": (180, 180, 180),
        "toy": (255, 150, 200),
        "default": (200, 200, 200),
    },

    "CROWD_INTELLIGENCE": {
        "ENABLED": True,
        "HEATMAP_GRID": (40, 30),
        "HEATMAP_DECAY": 0.998,
        "HEATMAP_GAUSSIAN_RADIUS": 2.5,
        "HEATMAP_GAUSSIAN_STRENGTH": 1.0,
        "SHOW_DENSITY_HEATMAP": False,
        "HEATMAP_OPACITY": 0.35,
        "FLOW_LINE_Y": 300,
        "FLOW_STATS_WINDOW_SEC": 60,
        "SHOW_FLOW_STATS": True,
        "CONGESTION_GRID": (3, 3),
        "CONGESTION_THRESHOLD": 4,
        "CONGESTION_WARNING": 2,
        "SHOW_CONGESTION_ZONES": False,
        "SHOW_FLOW_VECTORS": False,
        "FLOW_VECTOR_SCALE": 4.0,
        "FLOW_VECTOR_MAX_LEN": 50,
        "SHOW_ZONE_METRICS": False,
        "ZONE_METRICS_WINDOW_SEC": 120,
        "EVAC_AVG_SPEED_THRESHOLD": 20.0,
        "EVAC_DIRECTION_CONSENSUS": 0.55,
        "EVAC_CHECK_WINDOW_SEC": 3.0,
        "EVAC_MIN_PEOPLE": 3,
    },

    "SHOW_COUNT_LINE": False,
    "SHOW_HEATMAP": False,
    "SHOW_FACE_MESH": False,
    "DISPLAY_FPS": True,

    "MULTI_EMBEDDING_POOLING": "max",
    "MIN_ENROLLMENT_EMBEDDINGS": 3,
    "MAX_ENROLLMENT_EMBEDDINGS": 10,
    "CALIBRATION_SIGMOID_A": 12.0,
    "CALIBRATION_SIGMOID_B": -4.0,
    "AUTO_THRESHOLD_MARGIN": 0.15,
    "ENROLLMENT_QUALITY_THRESHOLD": 70,
    "ENROLLMENT_QUALITY_BLUR_MIN": 80.0,

    "STRANGER_TRACKING_ENABLED": True,
    "STRANGER_FRAMES_THRESHOLD": 15,
    "STRANGER_REID_THRESHOLD": 0.40,
}

_YOLO_CATEGORY_MAP = {
    0: "person",
    1: "vehicle", 2: "vehicle", 3: "vehicle", 4: "vehicle",
    5: "vehicle", 6: "vehicle", 7: "vehicle", 8: "vehicle",
    9: "infrastructure", 10: "infrastructure", 11: "infrastructure", 12: "infrastructure",
    13: "furniture",
    14: "animal", 15: "animal", 16: "animal", 17: "animal", 18: "animal",
    19: "animal", 20: "animal", 21: "animal", 22: "animal", 23: "animal",
    24: "accessory", 25: "accessory", 26: "accessory",
    27: "clothing", 28: "accessory",
    29: "sports", 30: "sports", 31: "sports", 32: "sports",
    33: "sports", 34: "sports", 35: "sports", 36: "sports",
    37: "sports", 38: "sports",
    39: "household",
    40: "utensil", 41: "utensil",
    42: "utensil", 43: "utensil", 44: "utensil", 45: "utensil",
    46: "food", 47: "food", 48: "food", 49: "food",
    50: "food", 51: "food", 52: "food", 53: "food", 54: "food", 55: "food",
    56: "furniture", 57: "furniture", 58: "furniture",
    59: "furniture", 60: "furniture", 61: "furniture",
    62: "electronics", 63: "electronics", 64: "electronics",
    65: "electronics", 66: "electronics", 67: "electronics",
    68: "electronics", 69: "electronics", 70: "electronics",
    71: "household", 72: "electronics",
    73: "misc", 74: "misc", 75: "misc",
    76: "utensil",
    77: "toy", 78: "electronics", 79: "misc",
}


# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def _compute_intra_class_variance(embeddings):
    """Compute mean and std of pairwise cosine distances within embeddings."""
    if len(embeddings) < 2:
        return 0.0, 0.0
    distances = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            d = cosine(embeddings[i], embeddings[j])
            distances.append(d)
    return np.mean(distances), np.std(distances)


def _auto_threshold(mean_var, std_var):
    """Compute an automatic recognition threshold from intra-class variance."""
    margin = CONFIG.get("AUTO_THRESHOLD_MARGIN", 0.15)
    return max(0.1, mean_var + 2 * std_var + margin)


def get_dominant_color(image, k=3):
    """Extract dominant color from an image using k-means clustering on pixels."""
    if image is None or image.size == 0:
        return None
    try:
        small = cv2.resize(image, (50, 50))
        data = small.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(labels.flatten())
        dominant = centers[np.argmax(counts)]
        return (int(dominant[2]), int(dominant[1]), int(dominant[0]))  # BGR -> RGB-ish for display
    except Exception:
        return None


# ===========================================================================
# FAISS INDEXER
# ===========================================================================

class FAISSIndexer:
    """FAISS-based fast embedding index for face recognition."""

    def __init__(self, dim=512):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
        self.id_to_name = []
        self.id_to_threshold = []
        self._next_id = 0

    def add_embeddings(self, names, embeddings, thresholds=None):
        """Add face embeddings to the index."""
        if embeddings is None or len(embeddings) == 0:
            return
        emb_array = np.array(embeddings, dtype=np.float32)
        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        emb_array = emb_array / norms
        self.index.add(emb_array)
        for i, name in enumerate(names):
            self.id_to_name.append(name)
            th = thresholds[i] if thresholds and i < len(thresholds) else CONFIG["FACE_RECOG_THRESHOLD"]
            self.id_to_threshold.append(th)
        self._next_id += len(names)

    def search(self, query_embedding, k=5):
        """Search for the nearest neighbors of a query embedding."""
        if self.index.ntotal == 0:
            return [], [], []
        q = np.array([query_embedding], dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return [], [], []
        q = q / norm
        distances, indices = self.index.search(q, min(k, self.index.ntotal))
        names = []
        dists = []
        ths = []
        for d, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.id_to_name):
                names.append(self.id_to_name[idx])
                dists.append(float(d))
                ths.append(self.id_to_threshold[idx])
        return names, dists, ths

    def reset(self):
        """Clear the index."""
        self.index.reset()
        self.id_to_name = []
        self.id_to_threshold = []
        self._next_id = 0


# ===========================================================================
# IMAGE QUALITY SCORER
# ===========================================================================

class ImageQualityScorer:
    """Assess image quality (blur, brightness, contrast) for enrollment."""

    def __init__(self, config=None):
        self.config = config or CONFIG

    def score(self, image):
        """Return a quality score 0-100 and individual metrics."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray))
        contrast = float(gray.std())

        blur_score = min(blur_val / 200.0, 1.0) * 40
        bright_score = 20.0
        if self.config["BRIGHTNESS_MIN"] <= brightness <= self.config["BRIGHTNESS_MAX"]:
            bright_score = 20.0
        else:
            dist = min(abs(brightness - self.config["BRIGHTNESS_MIN"]),
                       abs(brightness - self.config["BRIGHTNESS_MAX"]))
            bright_score = max(0, 20.0 - dist * 0.2)
        contrast_score = min(contrast / 80.0, 1.0) * 20
        sharpness_score = min(blur_val / 300.0, 1.0) * 20

        total = blur_score + bright_score + contrast_score + sharpness_score
        return total, {
            "blur": blur_val,
            "brightness": brightness,
            "contrast": contrast,
            "scores": {
                "blur": blur_score,
                "brightness": bright_score,
                "contrast": contrast_score,
                "sharpness": sharpness_score,
            },
        }

    def is_acceptable(self, image):
        """Check if image meets minimum quality for enrollment."""
        score, metrics = self.score(image)
        min_score = self.config.get("ENROLLMENT_QUALITY_THRESHOLD", 70)
        min_blur = self.config.get("ENROLLMENT_QUALITY_BLUR_MIN", 80.0)
        return score >= min_score and metrics["blur"] >= min_blur, score


# ===========================================================================
# CENTROID TRACKER
# ===========================================================================

class CentroidTracker:
    """Track objects using centroid distance with max distance and max disappeared."""

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
        del self.objects[oid]
        del self.disappeared[oid]

    def update(self, rects):
        """Update tracker with new bounding boxes [(x1,y1,x2,y2), ...]."""
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects.copy()

        input_centroids = np.zeros((len(rects), 2), dtype=np.int32)
        for i, (x1, y1, x2, y2) in enumerate(rects):
            input_centroids[i] = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = np.array(object_centroids).reshape(-1, 2)
            I = np.array(input_centroids).reshape(-1, 2)
            dists = np.linalg.norm(D[:, None] - I[None, :], axis=2)

            rows = dists.min(axis=1).argsort()
            cols = dists.argmin(axis=0)
            used_rows = set()
            used_cols = set()

            for r in rows:
                if r in used_rows:
                    continue
                c = cols[r]
                if c in used_cols:
                    continue
                if dists[r, c] > self.max_distance:
                    continue
                object_ids[r]  # unused but keeps variable alive
                oid = list(self.objects.keys())[r]
                self.objects[oid] = tuple(input_centroids[c])
                self.disappeared[oid] = 0
                used_rows.add(r)
                used_cols.add(c)

            for c in range(len(input_centroids)):
                if c not in used_cols:
                    self.register(tuple(input_centroids[c]))

            for oid in list(self.disappeared.keys()):
                if oid not in [list(self.objects.keys())[r] for r in used_rows]:
                    self.disappeared[oid] += 1
                    if self.disappeared[oid] > self.max_disappeared:
                        self.deregister(oid)

        return self.objects.copy()


# ===========================================================================
# FACE ANALYZER
# ===========================================================================

class FaceAnalyzer:
    """Wrapper around InsightFace for face detection, landmarks, and embedding extraction."""

    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def detect(self, frame):
        """Detect faces in a frame. Returns list of face objects."""
        faces = self.app.get(frame)
        return faces

    def get_embedding(self, face):
        """Get the 512-d embedding from a detected face."""
        return face.embedding

    def get_bbox(self, face):
        """Get bounding box as (x1, y1, x2, y2)."""
        bbox = face.bbox.astype(int)
        return (bbox[0], bbox[1], bbox[2], bbox[3])

    def get_landmarks(self, face):
        """Get facial landmarks."""
        return face.landmark_2d_106 if hasattr(face, "landmark_2d_106") else face.kps


# ===========================================================================
# HAND DETECTOR (MediaPipe)
# ===========================================================================

class HandDetector:
    """MediaPipe hand detection and landmark tracking."""

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config["HAND_MAX_NUM"],
            min_detection_confidence=self.config["HAND_MIN_DETECTION"],
            min_tracking_confidence=self.config["HAND_MIN_TRACKING"],
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        """Detect hands in frame. Returns list of hand results."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        hands = []
        if results.multi_hand_landmarks:
            for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                handedness = "Right"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[idx].classification[0].label
                hands.append({
                    "landmarks": hand_lms,
                    "handedness": handedness,
                })
        return hands

    def draw(self, frame, hands):
        """Draw hand landmarks on frame."""
        for hand in hands:
            if self.config.get("SHOW_HAND_LANDMARKS", False):
                self.mp_draw.draw_landmarks(
                    frame, hand["landmarks"], self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2),
                )
        return frame


# ===========================================================================
# OBJECT DETECTOR (YOLO)
# ===========================================================================

class ObjectDetector:
    """YOLOv8-based object detection."""

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.model = None
        if YOLO_AVAILABLE:
            try:
                model_path = self.config["MODEL_PATH"]
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                    print(f"[INFO] YOLO model loaded: {model_path}")
                else:
                    print(f"[WARNING] YOLO model not found at {model_path}. Downloading...")
                    self.model = YOLO("yolov8n.pt")
                    print("[INFO] YOLO model downloaded.")
            except Exception as e:
                print(f"[ERROR] YOLO init failed: {e}")
                self.model = None

    def detect(self, frame):
        """Run object detection. Returns list of detection dicts."""
        if self.model is None:
            return []
        try:
            results = self.model(frame, conf=self.config["YOLO_CONF"],
                                 verbose=False, classes=None)
            detections = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = _YOLO_CATEGORY_MAP.get(cls_id, "default")
                        detections.append({
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": conf,
                            "bbox": (x1, y1, x2, y2),
                            "category": cls_name,
                            "color": self.config["OBJECT_DISPLAY_CATEGORIES"].get(
                                cls_name, self.config["OBJECT_DISPLAY_CATEGORIES"]["default"]),
                        })
            return detections
        except Exception as e:
            print(f"[ERROR] YOLO detection failed: {e}")
            return []

    def draw_detections(self, frame, detections, skip_person=False):
        """Draw bounding boxes for detections."""
        for det in detections:
            if skip_person and det["class_name"] == "person":
                continue
            x1, y1, x2, y2 = det["bbox"]
            color = det.get("color", (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame


# ===========================================================================
# FACE MESH DRAWER
# ===========================================================================

class FaceMeshDrawer:
    """Draw MediaPipe face mesh tesselation."""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame):
        """Detect face mesh and draw tesselation."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_draw.draw_tesselation(
                    frame, face_landmarks,
                    self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_draw.DrawingSpec(
                        color=(0, 255, 0), thickness=1),
                )
        return frame


# ===========================================================================
# SUSPICION SCORER
# ===========================================================================

class SuspicionScorer:
    """Track and decay suspicion scores per object."""

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.scores = {}
        self.last_decay = time.time()

    def add_event(self, oid, points):
        """Add suspicion points to an object."""
        if oid not in self.scores:
            self.scores[oid] = 0.0
        self.scores[oid] += points

    def decay(self, active_ids):
        """Apply decay to all scores and remove inactive."""
        now = time.time()
        if now - self.last_decay < self.config["SUSPICION_DECAY_INTERVAL"]:
            return
        self.last_decay = now
        rate = self.config["SUSPICION_DECAY_RATE"]
        for oid in list(self.scores.keys()):
            self.scores[oid] *= rate
            if oid not in active_ids:
                self.scores.pop(oid, None)

    def get_score(self, oid):
        return self.scores.get(oid, 0.0)

    def get_stress_level(self, oid):
        """Map suspicion score to stress level."""
        s = self.get_score(oid)
        thresholds = self.config["STRESS_THRESHOLDS"]
        if s >= thresholds["MEDIUM"]:
            return "High"
        elif s >= thresholds["LOW"]:
            return "Medium"
        return "Low"


# ===========================================================================
# BEHAVIOR ANALYZER
# ===========================================================================

class BehaviorAnalyzer:
    """Analyze object behavior: hesitation, pacing, scanning, spatial anomaly."""

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.object_history = {}
        self.suspicion_scorer = SuspicionScorer(config)
        self.heatmap = np.zeros(self.config.get("SPATIAL_GRID_SIZE", (20, 20)), dtype=np.float32)
        self.heatmap_last_update = 0.0
        self.object_track_start = {}  # {oid: first_seen_time}
        self.last_behavior_flag = defaultdict(lambda: defaultdict(float))  # {oid: {behavior: last_flag_time}}

    def update(self, tracked_objects):
        """Analyze behavior for all tracked objects. Returns events list."""
        self.current_time = time.time()

        # Track when each object was first seen
        for oid in tracked_objects:
            if oid not in self.object_track_start:
                self.object_track_start[oid] = self.current_time

        events = []

        for oid, centroid in tracked_objects.items():
            if oid not in self.object_history:
                self.object_history[oid] = {
                    "centroids": deque(maxlen=300),
                    "speeds": deque(maxlen=300),
                    "directions": deque(maxlen=300),
                    "timestamps": deque(maxlen=300),
                    "behaviors": set(),
                    "suspicion_history": deque(maxlen=100),
                    "profile": {
                        "avg_speed": 0.0,
                        "speed_variance": 0.0,
                        "typical_zone": None,
                        "visit_count": 0,
                        "first_seen": self.current_time,
                        "last_seen": self.current_time,
                        "total_time": 0.0,
                    },
                }
            h = self.object_history[oid]
            h["centroids"].append(centroid)
            h["timestamps"].append(self.current_time)

            # Compute speed
            if len(h["centroids"]) >= 2:
                prev = h["centroids"][-2]
                dt = self.current_time - h["timestamps"][-2]
                if dt > 0:
                    dist = math.sqrt((centroid[0] - prev[0])**2 + (centroid[1] - prev[1])**2)
                    speed = dist / dt
                    h["speeds"].append(speed)
                    # Direction
                    dx = centroid[0] - prev[0]
                    dy = centroid[1] - prev[1]
                    angle = math.atan2(dy, dx)
                    h["directions"].append(angle)

            h["last_seen"] = self.current_time
            h["profile"]["visit_count"] += 1

            # Update heatmap
            self._update_heatmap(centroid)

            # Update suspicion profile
            self._update_profile(h)

        # Behavior analysis with min track frames and cooldown
        min_track_frames = self.config.get("SUSPICION_MIN_TRACK_FRAMES", 30)
        behavior_cooldown = self.config.get("SUSPICION_BEHAVIOR_COOLDOWN", 10.0)

        for oid in list(self.object_history):
            if oid not in tracked_objects:
                continue

            # Skip objects that haven't been tracked long enough
            h = self.object_history[oid]
            if len(h["centroids"]) < min_track_frames:
                continue

            h["behaviors"].clear()

            # HESITATION
            r = self._analyze_hesitation(h)
            if r:
                last = self.last_behavior_flag[oid]["HESITATION"]
                if (self.current_time - last) >= behavior_cooldown:
                    events.append(("HESITATION", f"ID_{oid}", 1.0, r))
                    h["behaviors"].add("HESITATION")
                    self.suspicion_scorer.add_event(oid, self.config["SUSPICION_POINTS"].get("HESITATION", 3))
                    self.last_behavior_flag[oid]["HESITATION"] = self.current_time

            # PACING
            r = self._analyze_pacing(h)
            if r:
                last = self.last_behavior_flag[oid]["PACING"]
                if (self.current_time - last) >= behavior_cooldown:
                    events.append(("PACING", f"ID_{oid}", 1.0, r))
                    h["behaviors"].add("PACING")
                    self.suspicion_scorer.add_event(oid, self.config["SUSPICION_POINTS"].get("PACING", 5))
                    self.last_behavior_flag[oid]["PACING"] = self.current_time

            # SCANNING
            r = self._analyze_scanning(h)
            if r:
                last = self.last_behavior_flag[oid]["SCANNING"]
                if (self.current_time - last) >= behavior_cooldown:
                    events.append(("SCANNING", f"ID_{oid}", 1.0, r))
                    h["behaviors"].add("SCANNING")
                    self.suspicion_scorer.add_event(oid, self.config["SUSPICION_POINTS"].get("SCANNING", 2))
                    self.last_behavior_flag[oid]["SCANNING"] = self.current_time

            self._update_psychological_profile(h)

        # Spatial anomaly check (per tracked object)
        for oid, cent in tracked_objects.items():
            if len(self.object_history.get(oid, {}).get("centroids", [])) < min_track_frames:
                continue
            if self._analyze_spatial_anomaly(cent):
                last = self.last_behavior_flag[oid]["SPATIAL_ANOMALY"]
                if (self.current_time - last) >= behavior_cooldown:
                    events.append(("SPATIAL_ANOMALY", f"ID_{oid}", 1.0, f"ID_{oid} in unusual location."))
                    self.suspicion_scorer.add_event(oid, self.config["SUSPICION_POINTS"].get("SPATIAL_ANOMALY", 2))
                    self.last_behavior_flag[oid]["SPATIAL_ANOMALY"] = self.current_time

        # Decay suspicion scores
        self.suspicion_scorer.decay(set(tracked_objects.keys()))

        return events

    def get_object_state(self, oid):
        """Get the current analysis state for an object."""
        if oid not in self.object_history:
            return None
        h = self.object_history[oid]
        return {
            "suspicion_score": self.suspicion_scorer.get_score(oid),
            "stress_level": self.suspicion_scorer.get_stress_level(oid),
            "active_behaviors": list(h.get("behaviors", set())),
            "avg_speed": h["profile"]["avg_speed"],
            "speed_variance": h["profile"]["speed_variance"],
            "visit_count": h["profile"]["visit_count"],
        }

    def cleanup(self, active_ids):
        """Remove history for objects no longer tracked."""
        for oid in set(self.object_history) - set(active_ids):
            del self.object_history[oid]
            self.suspicion_scorer.scores.pop(oid, None)
            self.object_track_start.pop(oid, None)
            self.last_behavior_flag.pop(oid, None)

    def _update_heatmap(self, centroid):
        """Update spatial heatmap with a new centroid observation."""
        fh, fw = self.config.get("FRAME_SIZE", (720, 1280))
        gh, gw = self.heatmap.shape
        gx = int(np.clip((centroid[0] / fw) * gw, 0, gw - 1))
        gy = int(np.clip((centroid[1] / fh) * gh, 0, gh - 1))
        self.heatmap[gy, gx] += 1.0
        # Apply decay periodically
        now = time.time()
        if now - self.heatmap_last_update > self.config.get("HEATMAP_UPDATE_INTERVAL", 5.0):
            self.heatmap *= self.config.get("CROWD_INTELLIGENCE", {}).get("HEATMAP_DECAY", 0.998)
            self.heatmap_last_update = now

    def _analyze_hesitation(self, h):
        """Detect hesitation: very low speed for an extended period."""
        if len(h["speeds"]) < 10:
            return None
        recent_speeds = list(h["speeds"])[-30:]
        avg_speed = np.mean(recent_speeds)
        stop_time = self.config["HESITATION_STOP_TIME_SEC"]
        speed_thresh = self.config["HESITATION_SPEED_THRESHOLD"]

        if avg_speed < speed_thresh:
            # Check how long they've been slow
            timestamps = list(h["timestamps"])
            speeds_list = list(h["speeds"])
            slow_duration = 0.0
            for i in range(len(speeds_list) - 1, -1, -1):
                if speeds_list[i] < speed_thresh:
                    if i > 0:
                        slow_duration += timestamps[i] - timestamps[i - 1]
                else:
                    break
            if slow_duration >= stop_time:
                return f"Stationary for {slow_duration:.1f}s (avg speed: {avg_speed:.1f})"
        return None

    def _analyze_pacing(self, h):
        """Detect pacing: frequent direction changes within a time window."""
        if len(h["directions"]) < 10:
            return None
        window = self.config["PACING_WINDOW_SEC"]
        min_changes = self.config["PACING_DIRECTION_CHANGES"]

        timestamps = list(h["timestamps"])
        directions = list(h["directions"])

        # Get recent entries within the window
        cutoff = self.current_time - window
        recent_dirs = []
        recent_ts = []
        for ts, d in zip(timestamps, directions):
            if ts >= cutoff:
                recent_dirs.append(d)
                recent_ts.append(ts)

        if len(recent_dirs) < 5:
            return None

        # Count direction changes (sign changes in angle)
        changes = 0
        for i in range(1, len(recent_dirs)):
            diff = recent_dirs[i] - recent_dirs[i - 1]
            # Normalize to [-pi, pi]
            while diff > math.pi:
                diff -= 2 * math.pi
            while diff < -math.pi:
                diff += 2 * math.pi
            if abs(diff) > math.pi / 4:  # > 45 degree change
                changes += 1

        if changes >= min_changes:
            return f"Direction changes: {changes} in {window:.0f}s window"
        return None

    def _analyze_scanning(self, h):
        """Detect scanning: high head/body movement variance (looking around)."""
        if len(h["centroids"]) < 15:
            return None
        recent_centroids = list(h["centroids"])[-60:]

        # Compute variance of positions
        positions = np.array(recent_centroids)
        var = np.var(positions, axis=0)
        total_var = np.sum(var)

        # Compute displacement (total distance traveled)
        displacements = np.diff(positions, axis=0)
        total_disp = np.sum(np.sqrt(np.sum(displacements**2, axis=1)))

        var_thresh = self.config["SCANNING_VAR_THRESHOLD"]
        disp_thresh = self.config["SCANNING_DISP_THRESHOLD"]

        if total_var > var_thresh and total_disp > disp_thresh:
            return f"Movement variance: {total_var:.1f}, displacement: {total_disp:.1f}"
        return None

    def _analyze_spatial_anomaly(self, centroid):
        """Check if an object is in an unusual location based on the heatmap."""
        total = np.sum(self.heatmap)
        if total < 5.0:  # Heatmap doesn't have enough data yet
            return False
        fh, fw = self.config.get("FRAME_SIZE", (720, 1280))
        gh, gw = self.heatmap.shape
        gx = int(np.clip((centroid[0] / fw) * gw, 0, gw - 1))
        gy = int(np.clip((centroid[1] / fh) * gh, 0, gh - 1))
        normalized = self.heatmap / np.max(self.heatmap)
        if normalized[gy, gx] < self.config.get("SPATIAL_ANOMALY_THRESHOLD", 0.02):
            return True
        return False

    def _update_profile(self, h):
        """Update the behavioral profile for an object."""
        if len(h["speeds"]) > 0:
            speeds = list(h["speeds"])
            h["profile"]["avg_speed"] = float(np.mean(speeds))
            h["profile"]["speed_variance"] = float(np.var(speeds))
        if h["timestamps"]:
            h["profile"]["total_time"] = h["timestamps"][-1] - h["timestamps"][0]

    def _update_psychological_profile(self, h):
        """Update psychological analysis based on observed behaviors."""
        score = self.suspicion_scorer.get_score(next(
            (oid for oid, hist in self.object_history.items() if hist is h), -1))
        h["suspicion_history"].append(score)


# ===========================================================================
# CROWD INTELLIGENCE
# ===========================================================================

class CrowdIntelligence:
    """Crowd dynamics: density heatmap, flow analysis, congestion, evacuation detection."""

    def __init__(self, config=None):
        self.config = (config or CONFIG).get("CROWD_INTELLIGENCE", {})
        self.enabled = self.config.get("ENABLED", True) and SKLEARN_AVAILABLE
        self.heatmap_grid = self.config.get("HEATMAP_GRID", (40, 30))
        self.density_heatmap = np.zeros(self.heatmap_grid, dtype=np.float32)
        self.flow_history = deque(maxlen=300)
        self.count_in = 0
        self.count_out = 0
        self.evacuation_events = []

    def update(self, tracked_objects, frame_size=(720, 1280)):
        """Update crowd analysis with current tracked objects."""
        if not self.enabled:
            return []

        events = []
        gh, gw = self.heatmap_grid
        fw, fh = frame_size  # Note: frame_size is (width, height) typically

        # Decay heatmap
        self.density_heatmap *= self.config.get("HEATMAP_DECAY", 0.998)

        # Add Gaussian splat for each tracked person
        radius = self.config.get("HEATMAP_GAUSSIAN_RADIUS", 2.5)
        strength = self.config.get("HEATMAP_GAUSSIAN_STRENGTH", 1.0)
        for oid, cent in tracked_objects.items():
            gx = int(np.clip((cent[0] / fw) * gw, 0, gw - 1))
            gy = int(np.clip((cent[1] / fh) * gh, 0, gh - 1))
            # Simple Gaussian splat
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < gh and 0 <= nx < gw:
                        dist2 = dx * dx + dy * dy
                        self.density_heatmap[ny, nx] += strength * math.exp(-dist2 / (2 * radius * radius))

        # Check for crowd forming
        min_size = CONFIG.get("CROWD_MIN_SIZE", 4)
        crowd_radius = CONFIG.get("CROWD_RADIUS", 100)
        positions = list(tracked_objects.values())
        if len(positions) >= min_size:
            for i, p1 in enumerate(positions):
                nearby = sum(1 for p2 in positions
                             if math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) < crowd_radius)
                if nearby >= min_size:
                    events.append(("CROWD_FORMING", f"AREA_{i}", 1.0,
                                   f"{nearby} people within {crowd_radius}px"))
                    break  # Only report one crowd event per frame

        # Congestion check
        cong_grid = self.config.get("CONGESTION_GRID", (3, 3))
        cong_thresh = self.config.get("CONGESTION_THRESHOLD", 4)
        if len(positions) >= cong_thresh:
            cell_w = fw / cong_grid[1]
            cell_h = fh / cong_grid[0]
            for row in range(cong_grid[0]):
                for col in range(cong_grid[1]):
                    cx = cell_w * (col + 0.5)
                    cy = cell_h * (row + 0.5)
                    count = sum(1 for p in positions
                                if abs(p[0] - cx) < cell_w / 2 and abs(p[1] - cy) < cell_h / 2)
                    if count >= cong_thresh:
                        events.append(("CONGESTION", f"ZONE_{row}_{col}", 1.0,
                                       f"{count} people in zone"))

        return events

    def get_density_overlay(self, frame):
        """Generate a colored density heatmap overlay."""
        if not self.enabled or not self.config.get("SHOW_DENSITY_HEATMAP", False):
            return None
        h, w = frame.shape[:2]
        gh, gw = self.heatmap_grid
        resized = cv2.resize(self.density_heatmap, (w, h))
        normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        opacity = self.config.get("HEATMAP_OPACITY", 0.35)
        overlay = cv2.addWeighted(frame, 1 - opacity, colored, opacity, 0)
        return overlay


# ===========================================================================
# BLINK DETECTOR
# ===========================================================================

class BlinkDetector:
    """Detect eye blinks using Eye Aspect Ratio (EAR) from face landmarks."""

    def __init__(self, config=None):
        self.config = (config or CONFIG).get("ANTI_SPOOFING", {})
        self.ear_history = deque(maxlen=self.config.get("EAR_HISTORY_FRAMES", 30))
        self.blink_timestamps = []
        self.last_blink_time = 0.0

    def compute_ear(self, landmarks, eye_indices):
        """Compute Eye Aspect Ratio from landmarks."""
        if landmarks is None or len(landmarks) < max(eye_indices) + 1:
            return None
        pts = np.array([landmarks[i] for i in eye_indices], dtype=np.float32)
        # Vertical distances
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        # Horizontal distance
        h = np.linalg.norm(pts[0] - pts[3])
        if h < 1e-6:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def update(self, ear, current_time):
        """Update with new EAR value. Returns True if a blink was detected."""
        self.ear_history.append(ear)
        if len(self.ear_history) < 3:
            return False

        threshold = self.config.get("BLINK_EAR_THRESHOLD", 0.2)
        min_gap = self.config.get("MIN_TIME_BETWEEN_BLINKS_SEC", 0.5)

        # Blink = EAR drops below threshold then recovers
        if (self.ear_history[-2] < threshold and
                self.ear_history[-1] >= self.ear_history[-2] and
                (current_time - self.last_blink_time) >= min_gap):
            self.last_blink_time = current_time
            self.blink_timestamps.append(current_time)
            # Keep only recent blinks
            window = self.config.get("LIVENESS_TIME_WINDOW_SEC", 15.0)
            cutoff = current_time - window
            self.blink_timestamps = [t for t in self.blink_timestamps if t >= cutoff]
            return True
        return False

    def get_blink_count(self, within_sec=None):
        """Get number of blinks in a time window."""
        if within_sec is None:
            within_sec = self.config.get("LIVENESS_TIME_WINDOW_SEC", 15.0)
        cutoff = time.time() - within_sec
        return sum(1 for t in self.blink_timestamps if t >= cutoff)

    def is_alive(self):
        """Check if enough blinks have been detected for liveness."""
        return self.get_blink_count() >= self.config.get("MIN_BLINKS_FOR_LIVENESS", 2)


# ===========================================================================
# HEAD POSE ESTIMATOR
# ===========================================================================

class HeadPoseEstimator:
    """Estimate head pose (pitch, yaw, roll) from face landmarks."""

    def __init__(self, config=None):
        self.config = (config or CONFIG).get("ANTI_SPOOFING", {})
        self.history = deque(maxlen=self.config.get("POSE_HISTORY_FRAMES", 30))
        # Simple 3D model points for head pose
        self.model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye corner
            (225.0, 170.0, -135.0),   # Right eye corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0),  # Right mouth corner
        ], dtype=np.float64)
        self.focal_length = 800.0

    def estimate(self, landmarks, image_size):
        # SAFETY CHECK: Ensure we have at least 11 landmarks (index 10 is required)
        # Previously it was < 6, but we are being safer.
        if landmarks is None or len(landmarks) < 11:
            return None

        image_points = np.array([
            landmarks[0],
            landmarks[4],
            landmarks[2],
            landmarks[6],
            landmarks[8],
            landmarks[10],
        ], dtype=np.float64)

        h, w = image_size[:2]
        camera_matrix = np.array([
            [self.focal_length, 0, w / 2],
            [0, self.focal_length, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rvec, tvec = cv2.solvePnP(
                self.model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                return None
            rmat, _ = cv2.Rodrigues(rvec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            pitch, yaw, roll = angles
            self.history.append((pitch, yaw, roll))
            return (pitch, yaw, roll)
        except Exception:
            return None

    def get_variance(self):
        """Get variance of recent head poses (for liveness checking)."""
        if len(self.history) < 5:
            return float("inf")
        poses = np.array(self.history)
        return float(np.mean(np.var(poses, axis=0)))


# ===========================================================================
# DEPTH ESTIMATOR
# ===========================================================================

class DepthEstimator:
    """Simple depth estimation from face size (larger face = closer)."""

    def __init__(self, config=None):
        self.config = (config or CONFIG).get("ANTI_SPOOFING", {})
        self.history = deque(maxlen=self.config.get("DEPTH_HISTORY_FRAMES", 30))
        self.reference_face_size = None
        self.reference_depth = 0.5  # meters

    def estimate(self, bbox):
        """Estimate depth from face bounding box size."""
        x1, y1, x2, y2 = bbox
        face_size = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if face_size < 1:
            return None

        if self.reference_face_size is None:
            self.reference_face_size = face_size
            return self.reference_depth

        # Inverse relationship: bigger face = closer
        depth = self.reference_depth * (self.reference_face_size / face_size)
        depth = max(0.1, min(depth, 5.0))  # Clamp
        self.history.append(depth)
        return depth

    def get_variance(self):
        """Get variance of depth estimates (low variance = possibly a photo)."""
        if len(self.history) < 10:
            return float("inf")
        depths = np.array(self.history)
        return float(np.var(depths))

    def is_consistent(self):
        """Check if depth is too consistent (might be a static photo)."""
        if len(self.history) < 10:
            return True
        var = self.get_variance()
        thresh = self.config.get("DEPTH_VARIANCE_THRESHOLD", 0.01)
        return var >= thresh


# ===========================================================================
# SPOOF CLASSIFIER CNN
# ===========================================================================

class SpoofClassifierCNN(nn.Module):
    """Simple CNN for anti-spoofing classification (optional ML-based)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ===========================================================================
# SPOOF CLASSIFIER
# ===========================================================================

class SpoofClassifier:
    """ML-based spoof detector using CNN."""

    def __init__(self, model_path=None):
        self.model = SpoofClassifierCNN()
        self.model.eval()
        self.loaded = False
        if model_path and os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location="cpu", weights_only=False)
                self.model.load_state_dict(state)
                self.loaded = True
                print(f"[INFO] Anti-spoof ML model loaded from {model_path}")
            except Exception as e:
                print(f"[WARNING] Failed to load anti-spoof model: {e}")

    def predict(self, face_crop):
        """Predict if face is real or spoof. Returns (is_real, confidence)."""
        if not self.loaded:
            return True, 0.5
        try:
            img = cv2.resize(face_crop, (64, 64))
            img = img.astype(np.float32) / 255.0
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                output = self.model(tensor)
                prob = torch.softmax(output, dim=1)
                real_prob = prob[0, 1].item()
            threshold = CONFIG.get("ANTI_SPOOFING", {}).get("ML_CLASSIFIER_THRESHOLD", 0.7)
            return real_prob >= threshold, real_prob
        except Exception:
            return True, 0.5


# ===========================================================================
# ANTI-SPOOF DETECTOR
# ===========================================================================

class AntiSpoofDetector:
    """Combined anti-spoofing: blink + head pose + depth + optional ML."""

    def __init__(self, config=None):
        self.config = (config or CONFIG).get("ANTI_SPOOFING", {})
        self.enabled = self.config.get("ENABLED", True)
        self.blink_detector = BlinkDetector(config)
        self.head_pose = HeadPoseEstimator(config)
        self.depth_estimator = DepthEstimator(config)

        self.ml_classifier = None
        if self.config.get("ENABLE_ML_CLASSIFIER", False):
            model_path = self.config.get("ANTI_SPOOF_MODEL_PATH", "antispoof_model.bin")
            self.ml_classifier = SpoofClassifier(model_path)

    def analyze(self, face, frame_size):
        """Run anti-spoof analysis. Returns (is_real, details_dict)."""
        if not self.enabled:
            return True, {"method": "disabled"}

        details = {}

        # Head pose variance (real person moves head)
        pose_var = self.head_pose.get_variance()
        details["head_pose_variance"] = pose_var
        has_movement = pose_var > 1.0  # Some head movement expected

        # Depth variance (real person's depth changes)
        depth_consistent = self.depth_estimator.is_consistent()
        details["depth_consistent"] = depth_consistent

        # Blink detection
        blink_count = self.blink_detector.get_blink_count()
        details["blink_count"] = blink_count
        has_blinks = blink_count >= self.config.get("MIN_BLINKS_FOR_LIVENESS", 2)

        # Combine: at least 2 out of 3 checks pass
        checks_passed = sum([has_movement, depth_consistent, has_blinks])

        # ML classifier if available
        ml_real = True
        ml_conf = 0.5
        if self.ml_classifier and face is not None:
            ml_real, ml_conf = self.ml_classifier.predict(face)
            details["ml_real"] = ml_real
            details["ml_confidence"] = ml_conf

        is_real = checks_passed >= 2
        if self.ml_classifier:
            is_real = is_real and ml_real

        details["is_real"] = is_real
        details["checks_passed"] = checks_passed
        return is_real, details


# ===========================================================================
# SUPPLEMENTARY DETECTOR (Fire, etc.)
# ===========================================================================

class SupplementaryDetector:
    """
    Detects objects that YOLO (COCO dataset) cannot detect:
    - Fire/flames (HSV color-based detection)
    """
    def __init__(self, config=None):
        if config is None:
            config = CONFIG
        self.config = config
        self.enabled = True
        # UPDATED: Tighter HSV ranges to reduce false positives
        # Focus on Red/Orange (0-25), ignoring Yellow/White/Yellowish light
        self.fire_lower = np.array([0, 120, 150], dtype=np.uint8) 
        self.fire_upper = np.array([25, 255, 255], dtype=np.uint8)
        self.min_fire_area = 2000  # Increased to 2000 to ignore small noise
        self.prev_fire_mask = None

    def detect_fire(self, frame):
        """Detect fire/flames using HSV color analysis."""
        if not self.enabled:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for fire-like colors (strictly red/orange)
        mask = cv2.inRange(hsv, self.fire_lower, self.fire_upper)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        h, w = frame.shape[:2]
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter small detections
            if area < self.min_fire_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 15 or bh < 15:
                continue

            # Flickering check (optional, but good)
            is_flickering = True
            if self.prev_fire_mask is not None:
                overlap = cv2.bitwise_and(mask, self.prev_fire_mask)
                overlap_ratio = np.sum(overlap > 0) / max(np.sum(mask > 0), 1)
                # Fire flickers (0.3 to 0.95 overlap ratio)
                is_flickering = 0.3 < overlap_ratio < 0.95

            self.prev_fire_mask = mask.copy()

            if is_flickering:
                detections.append({
                    "class_id": -1,
                    "class_name": "fire",
                    "confidence": min(area / 10000.0, 0.95),
                    "bbox": (x, y, x + bw, y + bh),
                    "category": "dangerous",
                    "color": (0, 0, 255),
                })

        return detections

    def detect_all(self, frame):
        """Run all supplementary detections."""
        results = []
        fire_dets = self.detect_fire(frame)
        results.extend(fire_dets)
        return results


# ===========================================================================
# VISION SYSTEM (Main orchestrator)
# ===========================================================================

class VisionSystem:
    """Main computer vision pipeline: face recognition, object detection,
    hand tracking, behavior analysis, anti-spoofing, crowd intelligence."""

    def __init__(self, config=None):
        self.config = config or CONFIG
        print("[INFO] Initializing VisionSystem...")

        self._init_components()
        self._init_state()

        print("[INFO] VisionSystem ready.")

    def _init_components(self):
        """Initialize all sub-components."""
        # Face analysis
        print("[INFO] Loading InsightFace...")
        self.face_analyzer = FaceAnalyzer()
        print("[INFO] InsightFace ready.")

        # Hand detection
        self.hand_detector = HandDetector(self.config)

        # Object detection
        self.object_detector = None
        if YOLO_AVAILABLE:
            self.object_detector = ObjectDetector(self.config)
            print("[INFO] Object detector initialized (YOLO).")
        else:
            print("[WARNING] Object detection disabled (ultralytics not installed).")

        # Face mesh
        self.face_mesh_drawer = None
        if self.config.get("SHOW_FACE_MESH", False):
            self.face_mesh_drawer = FaceMeshDrawer()

        # Anti-spoofing
        self.anti_spoof = AntiSpoofDetector(self.config)

        # Image quality
        self.quality_scorer = ImageQualityScorer(self.config)

        # Behavior analysis
        self.behavior_analyzer = BehaviorAnalyzer(self.config)

        # Crowd intelligence
        self.crowd_intel = CrowdIntelligence(CONFIG)
        self.supplementary_detector = SupplementaryDetector(CONFIG)
        print("[INFO] Supplementary detector initialized (fire detection).")

        # Centroid tracker for persons (from YOLO)
        self.person_tracker = CentroidTracker(max_disappeared=30, max_distance=120)

        # Face indexing
        self.face_indexer = FAISSIndexer(dim=512)

        # Load existing face database
        self.face_db = {}
        self._load_database()

        # Stranger tracking
        self.stranger_buffer = {}  # {track_oid: {embedding, bbox, first_seen, last_seen}}
        self.stranger_counter = 0

        # Enrollment
        self._max_embeddings_per_person = self.config.get("MAX_ENROLLMENT_EMBEDDINGS", 10)
        self._last_enrollment_time = 0

    def _init_state(self):
        """Initialize runtime state variables."""
        self._current_face_labels = {}   # {oid: (name, confidence, embedding)}
        self._last_analysis_frame = None
        self._last_person_colors = {}    # {oid: (r, g, b)}
        self._frame_count = 0
        self._fps = 0.0
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # DATABASE
    # ------------------------------------------------------------------

    def _load_database(self):
        """Load face database from pickle file."""
        db_path = self.config.get("DATABASE_FILE")
        if db_path and os.path.exists(db_path):
            try:
                with open(db_path, "rb") as f:
                    self.face_db = pickle.load(f)
                print(f"[INFO] Face database loaded: {len(self.face_db)} person(s) from {db_path}")
                self._rebuild_faiss_index()
            except Exception as e:
                print(f"[WARNING] Failed to load face database: {e}")
                self.face_db = {}

    def save_database(self):
        """Save face database to pickle file."""
        db_path = self.config.get("DATABASE_FILE")
        if not db_path:
            return
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        try:
            with open(db_path, "wb") as f:
                pickle.dump(self.face_db, f)
        except Exception as e:
            print(f"[WARNING] Failed to save face database: {e}")

    def _rebuild_faiss_index(self):
        """Rebuild the FAISS index from face_db."""
        self.face_indexer.reset()
        names = []
        embeddings = []
        thresholds = []
        pooling = self.config.get("MULTI_EMBEDDING_POOLING", "max")

        for name, data in self.face_db.items():
            embs = data.get("embeddings", [])
            if not embs:
                continue
            embs_arr = np.array(embs, dtype=np.float32)
            if len(embs_arr) > 1:
                if pooling == "mean":
                    pooled = np.mean(embs_arr, axis=0)
                else:  # max
                    pooled = np.max(embs_arr, axis=0)
            else:
                pooled = embs_arr[0]
            names.append(name)
            embeddings.append(pooled)
            thresholds.append(data.get("threshold", self.config["FACE_RECOG_THRESHOLD"]))

        if names:
            self.face_indexer.add_embeddings(names, embeddings, thresholds)
            print(f"[INFO] FAISS index rebuilt: {len(names)} person(s)")

    # ------------------------------------------------------------------
    # FACE RECOGNITION
    # ------------------------------------------------------------------

    def recognize_face(self, track_oid, embedding):
        """Recognize a face embedding. Returns (name, confidence, reason)."""
        if embedding is None:
            return "UNKNOWN", 0.0, "No embedding"

        names, dists, ths = self.face_indexer.search(embedding, k=3)

        if not names:
            return "UNKNOWN", 0.0, "No faces enrolled"

        # History-based smoothing would go here
        for i, (name, dist, thresh) in enumerate(zip(names, dists, ths)):
            if dist >= thresh:
                # Convert inner-product similarity to confidence (0-1)
                conf = float(np.clip(dist, 0, 1))
                return name, conf, f"Match (similarity={dist:.3f}, threshold={thresh:.3f})"

        return "UNKNOWN", 0.0, f"Below threshold (best={dists[0]:.3f})"

    # ------------------------------------------------------------------
    # ENROLLMENT
    # ------------------------------------------------------------------

    def enroll_person(self, frame=None, name=None, skip_quality_check=False):
        """Enroll a new face. If frame is provided, extract embedding from it."""
        if name is None:
            return False, "No name provided"

        if frame is not None and not skip_quality_check:
            acceptable, score = self.quality_scorer.is_acceptable(frame)
            if not acceptable:
                return False, f"Image quality too low (score={score:.1f})"

        cooldown = 3.0
        if time.time() - self._last_enrollment_time < cooldown and not skip_quality_check:
            return False, "Enrollment cooldown active"

        if name not in self.face_db:
            self.face_db[name] = {
                "embeddings": [],
                "threshold": self.config["FACE_RECOG_THRESHOLD"],
            }

        return True, "Ready for embedding"

    def get_current_unknowns(self):
        """
        Return list of currently tracked unknown/stranger faces.
        Returns: [(oid, label, confidence), ...]
        """
        results = []
        for oid, (name, conf, emb) in self._current_face_labels.items():
            if name == "UNKNOWN" or name.startswith("STRANGER_"):
                results.append((oid, name, conf))
        return results

    def enroll_unknown_face(self, oid, person_name):
        """
        Enroll a currently tracked unknown face with the given name.
        Uses the stored embedding from face detection.
        Returns True on success.
        """
        if oid not in self._current_face_labels:
            print(f"[ENROLL] Track ID {oid} not found in current faces.")
            return False

        _, _, embedding = self._current_face_labels[oid]
        if embedding is None:
            print(f"[ENROLL] No embedding stored for track ID {oid}.")
            return False

        # Remove from stranger tracking if applicable
        label = self._current_face_labels[oid][0]
        if label.startswith("STRANGER_"):
            print(f"[ENROLL] Removing {label} from stranger database.")
            self.stranger_buffer.pop(oid, None)

        # Enroll using the existing method
        self._last_enrollment_time = 0  # Skip cooldown for manual enrollment
        result = self.enroll_person(
            frame=None,  # We don't need the frame since we have the embedding
            name=person_name,
            skip_quality_check=True,
        )

        # The enroll_person method extracts embedding from frame, but we already have it.
        # We need to directly add it.
        if person_name not in self.face_db:
            self.face_db[person_name] = {
                "embeddings": [],
                "threshold": CONFIG["FACE_RECOG_THRESHOLD"],
            }

        max_embs = self._max_embeddings_per_person
        if len(self.face_db[person_name]["embeddings"]) >= max_embs:
            self.face_db[person_name]["embeddings"].pop(0)
            self._rebuild_faiss_index()

        self.face_db[person_name]["embeddings"].append(embedding)

        embs = self.face_db[person_name]["embeddings"]
        if len(embs) >= 2:
            mean_var, std_var = _compute_intra_class_variance(embs)
            auto_thresh = _auto_threshold(mean_var, std_var)
            self.face_db[person_name]["threshold"] = auto_thresh

        self.face_indexer.add_embeddings(
            [person_name], np.array([embedding]),
            [self.face_db[person_name]["threshold"]],
        )

        self.save_database()
        self._last_enrollment_time = time.time()

        n = len(self.face_db[person_name]["embeddings"])
        print(f"[SUCCESS] Enrolled {person_name} (embeddings: {n}/{max_embs}).")
        return True

    # ------------------------------------------------------------------
    # SHIRT COLOR EXTRACTION
    # ------------------------------------------------------------------

    def _extract_shirt_colors(self, frame, obj_detections, tracked):
        """Extract dominant shirt/torso color for each tracked person."""
        person_dets = [d for d in obj_detections if d["class_name"] == "person"]
        if not person_dets:
            return

        for oid, cent in tracked.items():
            # Find closest YOLO person detection
            best_det = None
            best_dist = float("inf")
            for pd in person_dets:
                bx1, by1, bx2, by2 = pd["bbox"]
                pc = ((bx1 + bx2) / 2, (by1 + by2) / 2)
                d = math.sqrt((cent[0] - pc[0])**2 + (cent[1] - pc[1])**2)
                if d < best_dist and d < 150:
                    best_dist = d
                    best_det = pd

            if best_det:
                x1, y1, x2, y2 = best_det["bbox"]
                h = y2 - y1
                w = x2 - x1
                # Torso region: roughly from 30% to 65% of body height
                ty1 = y1 + int(h * 0.30)
                ty2 = y1 + int(h * 0.65)
                tx1 = x1 + int(w * 0.15)
                tx2 = x2 - int(w * 0.15)
                # Clamp to frame
                ty1 = max(0, ty1)
                ty2 = min(frame.shape[0], ty2)
                tx1 = max(0, tx1)
                tx2 = min(frame.shape[1], tx2)
                if ty2 > ty1 and tx2 > tx1:
                    torso = frame[ty1:ty2, tx1:tx2]
                    color = get_dominant_color(torso, k=3)
                    if color:
                        self._last_person_colors[oid] = color

    @staticmethod
    def _draw_face_box(frame, x1, y1, x2, y2, name, conf, oid, is_real):
        # Color based on recognition status
        if not is_real:
            color = (0, 0, 255)  # Red = spoof
        elif name == "UNKNOWN" or name.startswith("STRANGER_"):
            color = (0, 165, 255)  # Orange = unknown
        else:
            color = (0, 255, 0)  # Green = recognized
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} ({conf:.2f}) ID:{oid}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def _draw_person_info(self, frame, fobj, x1, y2, oid, age, gender, emotion):
        st = self.behavior_analyzer.get_object_state(oid)
        if not st:
            return

        y_offset = y2 + 20

        # Shirt color swatch
        shirt_color = self._last_person_colors.get(oid)
        if shirt_color:
            r, g, b = shirt_color
            cv2.rectangle(frame, (x1, y_offset - 12), (x1 + 14, y_offset + 2), (int(b), int(g), int(r)), -1)
            cv2.rectangle(frame, (x1, y_offset - 12), (x1 + 14, y_offset + 2), (255, 255, 255), 1)
            cv2.putText(frame, "Shirt", (x1 + 18, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 20

        # Attributes
        attr_text = f"Age:{age} G:{gender} E:{emotion}"
        cv2.putText(frame, attr_text, (x1, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y_offset += 20

        sus = st.get("suspicion_score", 0)
        if sus > 5:
            cv2.putText(frame, f"Suspicion: {sus:.1f}", (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += 20

        sl = st.get("stress_level", "Low")
        sc = {"Low": (0, 255, 0), "Medium": (0, 255, 255), "High": (0, 0, 255)}.get(sl, (0, 255, 0))
        cv2.putText(frame, f"Stress: {sl}", (x1, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, sc, 1)
        y_offset += 20

        beh = st.get("active_behaviors", [])
        if beh:
            cv2.putText(frame, ", ".join(beh), (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)

    def _recognize_and_draw(self, frame, face_objects, tracked):
        """Recognize faces and draw boxes. Returns list of face info dicts."""
        faces_info = []
        dangerous_objects = self.config.get("DANGEROUS_OBJECTS", set())

        for fobj in face_objects:
            bbox = self.face_analyzer.get_bbox(fobj)
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            embedding = self.face_analyzer.get_embedding(fobj)

            # Find matching tracked object
            oid = None
            min_dist = float("inf")
            for t_oid, t_cent in tracked.items():
                d = math.sqrt((cx - t_cent[0])**2 + (cy - t_cent[1])**2)
                if d < min_dist and d < 150:
                    min_dist = d
                    oid = t_oid

            if oid is None:
                oid = -1  # Untracked face

            # Anti-spoof check
            is_real = True
            if self.anti_spoof.enabled:
                face_crop = frame[max(0, y1):min(frame.shape[0], y2),
                                  max(0, x1):min(frame.shape[1], x2)]
                is_real, spoof_details = self.anti_spoof.analyze(face_crop, frame.shape)

                if not is_real:
                    faces_info.append({
                        "oid": oid, "name": "SPOOF", "confidence": 0.0,
                        "bbox": bbox, "is_real": False,
                    })
                    self._draw_face_box(frame, x1, y1, x2, y2, "SPOOF", 0.0, oid, False)
                    continue

            # Recognize
            name, conf, reason = self.recognize_face(oid, embedding)

            # Stranger tracking
            if name == "UNKNOWN" and self.config.get("STRANGER_TRACKING_ENABLED", False):
                if oid not in self.stranger_buffer:
                    self.stranger_counter += 1
                    self.stranger_buffer[oid] = {
                        "embedding": embedding,
                        "bbox": bbox,
                        "first_seen": time.time(),
                        "last_seen": time.time(),
                        "frame_count": 1,
                    }
                    name = f"STRANGER_{self.stranger_counter}"
                else:
                    sb = self.stranger_buffer[oid]
                    sb["last_seen"] = time.time()
                    sb["frame_count"] += 1
                    sb["bbox"] = bbox
                    name = f"STRANGER_{self.stranger_counter}"

            # Store current label for enrollment UI
            self._current_face_labels[oid] = (name, conf, embedding)

            # Draw face box
            self._draw_face_box(frame, x1, y1, x2, y2, name, conf, oid, is_real)

            # Estimate age/gender/emotion from face (simplified)
            age = "~25"
            gender = "?"
            emotion = "neutral"
            if hasattr(fobj, "age"):
                age = str(int(fobj.age)) if fobj.age else "~25"
            if hasattr(fobj, "gender"):
                gender = "M" if fobj.gender == 1 else "F"
            if hasattr(fobj, "emotion"):
                emotion = str(fobj.emotion) if fobj.emotion else "neutral"

            # Draw person info
            if oid > 0:
                self._draw_person_info(frame, fobj, x1, y2, oid, age, gender, emotion)

            faces_info.append({
                "oid": oid, "name": name, "confidence": conf,
                "bbox": bbox, "is_real": is_real,
                "reason": reason,
            })

        return faces_info

    def _draw_ui(self, frame, events, tracked, obj_detections, dt):
        """Draw UI overlays: count, events, heatmap, etc."""
        h, w = frame.shape[:2]

        # Store frame size for behavior analyzer
        self.behavior_analyzer.config["FRAME_SIZE"] = (h, w)

        # Show count line
        if self.config.get("SHOW_COUNT_LINE", False):
            line_y = self.config.get("COUNT_LINE_Y", 300)
            cv2.line(frame, (0, line_y), (w, line_y), (255, 255, 0), 2)
            cv2.putText(frame, "COUNT LINE", (10, line_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Show loitering zone
        if self.config.get("LOITERING_ZONE"):
            lx1, ly1, lx2, ly2 = self.config["LOITERING_ZONE"]
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 0, 255), 2)

        # Display events on frame
        y_pos = 30
        for event in events[-5:]:  # Show last 5 events
            evt_type = event[0]
            evt_name = event[1] if len(event) > 1 else ""
            evt_detail = event[3] if len(event) > 3 else ""

            if evt_type == "DANGEROUS_OBJECT":
                color = (0, 0, 255)
            elif evt_type in ("SPOOF_DETECTED", "EVACUATION_ALERT"):
                color = (0, 0, 255)
            elif evt_type in ("HESITATION", "PACING", "SCANNING", "SPATIAL_ANOMALY",
                              "LOITERING", "CROWD_FORMING"):
                color = (0, 165, 255)
            elif evt_type in ("RECOGNITION",):
                color = (0, 255, 0)
            else:
                color = (200, 200, 200)

            text = f"[{evt_type}] {evt_name}"
            if evt_detail:
                text += f" - {evt_detail[:60]}"
            cv2.putText(frame, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_pos += 20

        # Show heatmap overlay
        if self.config.get("SHOW_HEATMAP", False):
            overlay = self.crowd_intel.get_density_overlay(frame)
            if overlay is not None:
                frame[:] = overlay

        # Show tracked count
        cv2.putText(frame, f"Tracked: {len(tracked)}", (w - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show dangerous objects
        dangerous_objects = self.config.get("DANGEROUS_OBJECTS", set())
        for det in obj_detections:
            if det["class_name"].lower() in dangerous_objects:
                bx1, by1, bx2, by2 = det["bbox"]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
                cv2.putText(frame, f"DANGER: {det['class_name'].upper()}",
                            (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # FPS display
        if self.config.get("DISPLAY_FPS", True):
            self._frame_count += 1
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                self._fps = self._frame_count / elapsed
            cv2.putText(frame, f"FPS: {self._fps:.1f}", (w - 120, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    # ------------------------------------------------------------------
    # MAIN PROCESS PIPELINE
    # ------------------------------------------------------------------

    def process(self, frame):
        """Run full vision pipeline on a frame.
        Returns: (display_frame, faces_info, hand_dets, obj_dets, events, tracked_count, dt)
        """
        t0 = time.time()
        events = []

        # Make a copy for analysis (smaller for speed)
        h, w = frame.shape[:2]
        analysis = frame.copy()
        display = frame.copy()

        # ---- Face detection ----
        face_objects = self.face_analyzer.detect(analysis)

        # ---- Hand detection ----
        hand_dets = self.hand_detector.detect(analysis)
        if self.config.get("SHOW_HAND_LANDMARKS", False):
            self.hand_detector.draw(display, hand_dets)

        # ---- Object detection ----
        object_detections = []
        if self.object_detector is not None:
            object_detections = self.object_detector.detect(analysis)

        # Supplementary detections (fire, etc.)
        supp_detections = self.supplementary_detector.detect_all(analysis)
        object_detections.extend(supp_detections)

        # ---- Draw object detections ----
        if self.config.get("SHOW_OBJECT_BOXES", True):
            self.object_detector.draw_detections(display, object_detections, skip_person=True)

        # Extract shirt colors for tracked persons
        self._extract_shirt_colors(analysis, object_detections, tracked={})

        # ---- Person tracking (YOLO persons) ----
        person_rects = [(d["bbox"]) for d in object_detections if d["class_name"] == "person"]
        tracked = self.person_tracker.update(person_rects)

        # Re-extract shirt colors with proper tracked objects
        self._extract_shirt_colors(analysis, object_detections, tracked)

        # ---- Behavior analysis ----
        behavior_events = self.behavior_analyzer.update(tracked)
        events.extend(behavior_events)

        # ---- Crowd intelligence ----
        crowd_events = self.crowd_intel.update(tracked, frame_size=(w, h))
        events.extend(crowd_events)

        # ---- Face recognition & drawing ----
        faces_info = self._recognize_and_draw(display, face_objects, tracked)

        # ---- Check for dangerous objects ----
        dangerous_objects = self.config.get("DANGEROUS_OBJECTS", set())
        for det in object_detections:
            if det["class_name"].lower() in dangerous_objects:
                events.append(("DANGEROUS_OBJECT", det["class_name"],
                               det["confidence"],
                               f"Detected: {det['class_name']} at {det['bbox']}"))

        # ---- Object interaction check ----
        for det in object_detections:
            if det["class_name"] != "person":
                for oid, cent in tracked.items():
                    bx1, by1, bx2, by2 = det["bbox"]
                    cx_obj = (bx1 + bx2) / 2
                    cy_obj = (by1 + by2) / 2
                    dist = math.sqrt((cent[0] - cx_obj)**2 + (cent[1] - cy_obj)**2)
                    if dist < 80:
                        events.append(("OBJECT_INTERACTION", f"ID_{oid}", 1.0,
                                       f"ID_{oid} near {det['class_name']}"))

        # ---- Cleanup stale tracked objects ----
        active_ids = set(tracked.keys())
        # Also keep IDs from faces that are still visible
        for fi in faces_info:
            if fi["oid"] > 0:
                active_ids.add(fi["oid"])
        self.behavior_analyzer.cleanup(active_ids)

        # ---- Draw UI ----
        dt = time.time() - t0
        self._draw_ui(display, events, tracked, object_detections, dt)

        # ---- Face mesh overlay ----
        if self.face_mesh_drawer and self.config.get("SHOW_FACE_MESH", False):
            display = self.face_mesh_drawer.process(display)

        return (display, faces_info, hand_dets, object_detections,
                events, len(tracked), dt)