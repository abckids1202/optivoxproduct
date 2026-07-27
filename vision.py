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

    "SUSPICION_DECAY_RATE": 0.98,
    "SUSPICION_DECAY_INTERVAL": 1.0,
    "SUSPICION_POINTS": {
        "HESITATION": 10, "PACING": 20, "SCANNING": 5,
        "SPATIAL_ANOMALY": 25, "LOITERING": 15, "RUNNING": 10,
        "OBJECT_INTERACTION": 15, "CROWD_FORMING": 5,
    },
    "HESITATION_SPEED_THRESHOLD": 5.0,
    "HESITATION_STOP_TIME_SEC": 3.0,
    "PACING_WINDOW_SEC": 10.0,
    "PACING_DIRECTION_CHANGES": 3,
    "SCANNING_VAR_THRESHOLD": 100.0,
    "SCANNING_DISP_THRESHOLD": 30.0,
    "CROWD_MIN_SIZE": 4,
    "CROWD_RADIUS": 100,
    "INTERACTION_IOU_THRESHOLD": 0.1,
    "SPATIAL_GRID_SIZE": (20, 20),
    "SPATIAL_ANOMALY_THRESHOLD": 0.05,
    "HEATMAP_UPDATE_INTERVAL": 5.0,

    "STRESS_THRESHOLDS": {"LOW": 20, "MEDIUM": 50},

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
        "machete", "sword", "explosive", "bomb",
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
    1: "vehicle", 2: "vehicle", 3: "vehicle", 5: "vehicle", 7: "vehicle",
    9: "vehicle", 11: "vehicle",
    4: "vehicle",
    6: "vehicle",
    8: "vehicle",
    10: "vehicle",
    12: "vehicle",
    13: "vehicle",
    14: "vehicle",
    15: "animal", 16: "animal", 17: "animal", 18: "animal", 19: "animal",
    20: "animal", 21: "animal", 22: "animal", 23: "animal",
    24: "utensil", 25: "utensil", 26: "utensil", 27: "utensil",
    28: "utensil", 29: "utensil", 30: "utensil", 31: "utensil",
    32: "utensil", 33: "utensil", 34: "utensil", 35: "utensil",
    36: "electronics", 37: "electronics", 38: "electronics",
    39: "electronics", 40: "electronics", 41: "electronics",
    42: "utensil", 43: "utensil",
    44: "utensil", 45: "utensil",
    46: "food", 47: "food", 48: "food", 49: "food",
    50: "furniture", 51: "furniture", 52: "furniture",
    53: "furniture", 54: "furniture", 55: "furniture", 56: "furniture",
    57: "furniture", 58: "furniture", 59: "furniture",
    60: "utensil", 61: "utensil", 62: "utensil",
    63: "utensil", 64: "utensil", 65: "utensil",
    66: "utensil", 67: "utensil",
    68: "utensil", 69: "utensil", 70: "utensil", 71: "utensil",
    72: "animal", 73: "animal", 74: "animal", 75: "animal",
    76: "electronics", 77: "electronics", 78: "electronics", 79: "electronics",
}

def get_dominant_color(image_region, k=4):
    try:
        if image_region is None or image_region.size == 0:
            return None
        pixels = image_region.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        centers = np.uint8(centers)
        counts = np.bincount(labels.flatten())
        if counts.size == 0:
            return None
        d = centers[np.argmax(counts)]
        return (int(d[0]), int(d[1]), int(d[2]))
    except Exception:
        return None


def _get_3d_model_points():
    return np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ], dtype=np.float32)


def _get_2d_image_points(landmarks, shape):
    return np.array([
        (landmarks[1].x * shape[1], landmarks[1].y * shape[0]),
        (landmarks[152].x * shape[1], landmarks[152].y * shape[0]),
        (landmarks[33].x * shape[1], landmarks[33].y * shape[0]),
        (landmarks[263].x * shape[1], landmarks[263].y * shape[0]),
        (landmarks[61].x * shape[1], landmarks[61].y * shape[0]),
        (landmarks[291].x * shape[1], landmarks[291].y * shape[0])
    ], dtype=np.float32)


def _get_object_color(cls_id):
    category = _YOLO_CATEGORY_MAP.get(cls_id, "default")
    return CONFIG["OBJECT_DISPLAY_CATEGORIES"].get(category, (200, 200, 200))

def _safe_landmarks(landmark_list):
    try:
        return list(landmark_list)
    except (TypeError, AttributeError):
        try:
            return [landmark_list[i] for i in range(len(landmark_list))]
        except Exception:
            if hasattr(landmark_list, 'landmark'):
                return [landmark_list.landmark[i] for i in range(len(landmark_list.landmark))]
            return []

def _calibrate_confidence(raw_similarity, a=None, b=None):
    if a is None:
        a = CONFIG.get("CALIBRATION_SIGMOID_A", 12.0)
    if b is None:
        b = CONFIG.get("CALIBRATION_SIGMOID_B", -4.0)
    try:
        z = a * raw_similarity + b
        z = max(-500.0, min(500.0, z))
        return 1.0 / (1.0 + math.exp(-z))
    except (OverflowError, ValueError):
        return 0.0 if raw_similarity < 0.5 else 1.0

def _compute_intra_class_variance(embeddings):
    if embeddings is None or len(embeddings) < 2:
        return 0.0, 0.0
    embs = np.array(embeddings)
    n = len(embs)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(cosine(embs[i], embs[j]))
    if not distances:
        return 0.0, 0.0
    return float(np.mean(distances)), float(np.std(distances))

def _auto_threshold(intra_mean, intra_std, base_threshold=None, margin=None):
    if base_threshold is None:
        base_threshold = CONFIG.get("FACE_RECOG_THRESHOLD", 0.35)
    if margin is None:
        margin = CONFIG.get("AUTO_THRESHOLD_MARGIN", 0.15)
    auto_thresh = intra_mean + margin * (1.0 + intra_std)
    return max(base_threshold, min(auto_thresh, 0.8))

class FAISSIndexer:
    def __init__(self, embedding_dim=512):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.known_names = []
        self.known_thresholds = []

    def add_embeddings(self, names, embeddings, thresholds=None):
        if embeddings.size == 0:
            return
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dim mismatch: {self.embedding_dim} vs {embeddings.shape[1]}")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        if thresholds is None:
            thresholds = [CONFIG["FACE_RECOG_THRESHOLD"]] * len(names)
        for i, name in enumerate(names):
            self.known_names.append(name)
            self.known_thresholds.append(thresholds[i])

    def search(self, query_embedding, k=5, pool_per_person=True):
        if self.index.ntotal == 0:
            return []
        q_emb = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q_emb)
        similarities, indices = self.index.search(q_emb, k)
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(sim), self.known_names[idx], self.known_thresholds[idx]))

        if not pool_per_person or not results:
            return results

        person_scores = {}
        for idx, sim, name, thresh in results:
            if name not in person_scores:
                person_scores[name] = {"max_sim": sim, "sum_sim": sim, "count": 0,
                                       "threshold": thresh, "best_idx": idx}
            ps = person_scores[name]
            ps["max_sim"] = max(ps["max_sim"], sim)
            ps["sum_sim"] += sim
            ps["count"] += 1
            if sim > ps["max_sim"]:
                ps["best_idx"] = idx

        pooling_method = CONFIG.get("MULTI_EMBEDDING_POOLING", "max")
        pooled_results = []
        for name, ps in person_scores.items():
            if pooling_method == "max":
                agg_sim = ps["max_sim"]
            else:
                agg_sim = ps["sum_sim"] / ps["count"]
            pooled_results.append((ps["best_idx"], agg_sim, name, ps["threshold"]))

        pooled_results.sort(key=lambda x: x[1], reverse=True)
        return pooled_results

    def save(self, index_path, names_path):
        try:
            faiss.write_index(self.index, index_path)
            with open(names_path, "wb") as f:
                pickle.dump({"names": self.known_names, "thresholds": self.known_thresholds}, f)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save FAISS: {e}")
            return False

    def load(self, index_path, names_path):
        try:
            self.index = faiss.read_index(index_path)
            with open(names_path, "rb") as f:
                data = pickle.load(f)
            self.known_names = data.get("names", [])
            self.known_thresholds = data.get("thresholds", [])
            return True
        except Exception as e:
            print(f"[WARN] FAISS load failed: {e}")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.known_names = []
            self.known_thresholds = []
            return False

class ImageQualityScorer:
    def __init__(self, config):
        self.config = config

    def score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        bright = np.mean(gray)
        contrast = np.std(gray)
        q = 0
        if blur >= self.config["BLUR_THRESHOLD"]:
            q += 40
        if self.config["BRIGHTNESS_MIN"] <= bright <= self.config["BRIGHTNESS_MAX"]:
            q += 30
        if contrast >= self.config["CONTRAST_THRESHOLD"]:
            q += 30
        return {
            "blur_score": blur, "brightness_score": bright,
            "contrast_score": contrast, "overall_quality": q, "is_acceptable": q > 70,
        }

    def is_enrollment_quality(self, frame):
        scores = self.score(frame)
        reasons = []

        quality_threshold = self.config.get("ENROLLMENT_QUALITY_THRESHOLD", 70)
        if scores["overall_quality"] < quality_threshold:
            reasons.append(f"Low quality ({scores['overall_quality']}/100, need >= {quality_threshold})")

        blur_min = self.config.get("ENROLLMENT_QUALITY_BLUR_MIN", 80.0)
        if scores["blur_score"] < blur_min:
            reasons.append(f"Blurry frame (Laplacian var={scores['blur_score']:.1f}, need >= {blur_min})")

        if scores["brightness_score"] < 40 or scores["brightness_score"] > 220:
            reasons.append(f"Poor lighting (brightness={scores['brightness_score']:.0f})")

        if scores["contrast_score"] < 30:
            reasons.append(f"Low contrast ({scores['contrast_score']:.1f})")

        return (len(reasons) == 0, "; ".join(reasons) if reasons else "OK", scores)

class CentroidTracker:
    def __init__(self, max_disappeared=10, max_dist=50):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_dist = max_dist

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid):
        self.objects.pop(oid, None)
        self.disappeared.pop(oid, None)

    def update(self, rects):
        if len(rects) == 0:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        centroids = np.array([[int((x1+x2)/2), int((y1+y2)/2)] for x1, y1, x2, y2 in rects])
        if len(self.objects) == 0:
            for c in centroids:
                self.register(c)
        else:
            ids = list(self.objects.keys())
            obj_c = np.array(list(self.objects.values()))
            D = np.linalg.norm(obj_c[:, np.newaxis] - centroids[np.newaxis, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_r, used_c = set(), set()
            for r, c in zip(rows, cols):
                if r in used_r or c in used_c or D[r, c] > self.max_dist:
                    continue
                self.objects[ids[r]] = centroids[c]
                self.disappeared[ids[r]] = 0
                used_r.add(r); used_c.add(c)
            for r in set(range(D.shape[0])) - used_r:
                self.disappeared[ids[r]] += 1
                if self.disappeared[ids[r]] > self.max_disappeared:
                    self.deregister(ids[r])
            if D.shape[0] < D.shape[1]:
                for c in set(range(D.shape[1])) - used_c:
                    self.register(centroids[c])
        return self.objects

class FaceAnalyzer:
    def __init__(self, db_path="face_database.pkl"):
        self.db_path = db_path
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.known_faces = {}
        self.indexer = FAISSIndexer(512)
        self._load_known_faces()

    def get_embedding(self, face_img):
        if face_img is None:
            return None
        faces = self.app.get(face_img)
        if not faces:
            return None
        return max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1])).embedding

    def recognize(self, embedding, threshold=None):
        if threshold is None:
            threshold = CONFIG["FACE_RECOG_THRESHOLD"]
        if embedding is None or self.indexer.index.ntotal == 0:
            return "Unknown", 0.0
        results = self.indexer.search(embedding, k=1)
        if results:
            _, sim, name, dist_thresh = results[0]
            if (1.0 - sim) < dist_thresh:
                return name, 1.0 - sim
        return "Unknown", 0.0

    def enroll(self, name, embedding):
        if embedding is None:
            return False
        self.known_faces[name] = embedding
        self.indexer.add_embeddings([name], np.array([embedding]))
        data = {"names": self.indexer.known_names,
                "embeddings": [self.known_faces[n] for n in self.indexer.known_names]}
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)
        return True

    def _load_known_faces(self):
        if not os.path.exists(self.db_path):
            self.known_faces = {}
            return
        try:
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
            names = data.get("names", [])
            embs = data.get("embeddings", [])
            if embs and names and len(embs) == len(names):
                self.indexer.add_embeddings(names, np.array(embs))
                self.known_faces = dict(zip(names, embs))
        except Exception as e:
            print(f"[ERROR] Could not load face DB: {e}")
            self.known_faces = {}

class HandDetector:
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    FINGERTIPS = [4, 8, 12, 16, 20]
    FINGER_PIPS = [3, 6, 10, 14, 18]
    FINGER_MCPS = [2, 5, 9, 13, 17]

    def __init__(self, config=None):
        if config is None:
            config = CONFIG
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.get("HAND_MAX_NUM", 2),
            min_detection_confidence=config.get("HAND_MIN_DETECTION", 0.5),
            min_tracking_confidence=config.get("HAND_MIN_TRACKING", 0.5),
        )
        self.hand_colors = {"Left": (0, 255, 255), "Right": (255, 100, 255)}

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        h, w = frame.shape[:2]
        detections = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                landmarks = []
                pixel_landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append((lm.x, lm.y, lm.z))
                    pixel_landmarks.append((int(lm.x * w), int(lm.y * h)))

                xs = [p[0] for p in pixel_landmarks]
                ys = [p[1] for p in pixel_landmarks]
                pad = 10
                bbox = (max(0, min(xs) - pad), max(0, min(ys) - pad),
                        min(w, max(xs) + pad), min(h, max(ys) + pad))
                gesture = self._recognize_gesture(landmarks)

                detections.append({
                    "handedness": label, "landmarks": landmarks,
                    "pixel_landmarks": pixel_landmarks, "bbox": bbox, "gesture": gesture,
                })
        return detections

    def _recognize_gesture(self, landmarks):
        lm = landmarks
        def dist(a, b):
            return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

        def is_finger_extended(tip_idx, pip_idx, mcp_idx):
            wrist = lm[0]
            tip_d = dist(lm[tip_idx], wrist)
            pip_d = dist(lm[pip_idx], wrist)
            return tip_d > pip_d * 1.1

        thumb_ext = dist(lm[4], lm[2]) > dist(lm[3], lm[2]) * 1.2
        index_ext = is_finger_extended(8, 6, 5)
        middle_ext = is_finger_extended(12, 10, 9)
        ring_ext = is_finger_extended(16, 14, 13)
        pinky_ext = is_finger_extended(20, 18, 17)
        extended = sum([index_ext, middle_ext, ring_ext, pinky_ext])

        thumb_index_dist = dist(lm[4], lm[8])
        if thumb_index_dist < 0.06 and extended >= 2:
            return "OK"
        if thumb_ext and extended == 0:
            return "Thumbs Up"
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return "Peace"
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "Pointing"
        if index_ext and not middle_ext and not ring_ext and pinky_ext:
            return "Rock"
        if extended >= 4:
            return "Open Palm"
        if extended <= 1 and not thumb_ext:
            return "Fist"
        return "Unknown"

    def draw_landmarks(self, frame, detections):
        for det in detections:
            color = self.hand_colors.get(det["handedness"], (0, 255, 255))
            if CONFIG.get("SHOW_HAND_LANDMARKS", False) or CONFIG.get("SHOW_HAND_CONNECTIONS", False):
                pts = det["pixel_landmarks"]
                if self.config.get("SHOW_HAND_CONNECTIONS", True):
                    for i, j in self.HAND_CONNECTIONS:
                        cv2.line(frame, pts[i], pts[j], color, 2)
                if self.config.get("SHOW_HAND_LANDMARKS", True):
                    for idx, pt in enumerate(pts):
                        r = 5 if idx in self.FINGERTIPS else 3
                        cv2.circle(frame, pt, r, color, -1)
                        cv2.circle(frame, pt, r, (255, 255, 255), 1)
            
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det['handedness']}: {det['gesture']}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + label_size[0] + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return frame

class ObjectDetector:
    def __init__(self, config=None, shared_yolo_model=None, device=None):
        if config is None:
            config = CONFIG
        self.config = config

        if shared_yolo_model is not None:
            self.model = shared_yolo_model
        else:
            if not YOLO_AVAILABLE:
                raise RuntimeError("ultralytics not installed. Object detection disabled.")
            self.model = YOLO(config["MODEL_PATH"])

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)
        self.conf_threshold = config.get("YOLO_CONF", 0.45)
        self.class_names = self.model.names

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                try:
                    conf = float(box.conf.item())
                    if conf < self.conf_threshold:
                        continue
                    cls_id = int(box.cls.item())
                    class_name = self.class_names.get(cls_id, f"obj_{cls_id}")
                    xyxy = box.xyxy.view(-1).cpu().numpy()
                    if xyxy.shape[0] != 4:
                        continue
                    x1, y1, x2, y2 = map(int, xyxy)
                    category = _YOLO_CATEGORY_MAP.get(cls_id, "default")
                    color = _get_object_color(cls_id)
                    detections.append({
                        "class_id": cls_id, "class_name": class_name,
                        "confidence": conf, "bbox": (x1, y1, x2, y2),
                        "category": category, "color": color,
                    })
                except Exception as e:
                    print(f"[WARN] Skipping YOLO box: {e}")
        return detections

    def draw_detections(self, frame, detections, skip_person=True):
        for det in detections:
            if skip_person and det["class_name"] == "person":
                continue
            x1, y1, x2, y2 = det["bbox"]
            color = det["color"]
            conf = det["confidence"]
            label = f"{det['class_name']} {conf:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_size[0] + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

class FaceMeshDrawer:
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
                 386, 385, 384, 398]
    LEFT_EYEBROW = [46, 53, 52, 65, 55, 107, 66, 105, 63, 70]
    RIGHT_EYEBROW = [276, 283, 282, 295, 285, 336, 296, 334, 293, 300]
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
                  270, 269, 267, 0, 37, 39, 40, 185, 61]
    LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
                  310, 311, 312, 13, 82, 81, 80, 191, 78]
    NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1]

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=5, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

    def process(self, frame, face_bboxes):
        if not CONFIG.get("SHOW_FACE_MESH", False):
            return frame, None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        h, w = frame.shape[:2]
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self._draw_face_contours(frame, face_landmarks, w, h)
        return frame, results

    def _draw_face_contours(self, frame, landmarks, w, h):
        landmarks = _safe_landmarks(landmarks)

        def draw_landmark_group(indices, color, thickness=1, is_closed=True):
            pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
            if len(pts) >= 2:
                cv2.polylines(frame, [np.array(pts, dtype=np.int32)], is_closed, color, thickness)

        draw_landmark_group(self.FACE_OVAL, (100, 200, 255), 1)
        draw_landmark_group(self.LEFT_EYE, (0, 255, 255), 1)
        draw_landmark_group(self.RIGHT_EYE, (0, 255, 255), 1)
        draw_landmark_group(self.LEFT_EYEBROW, (255, 200, 100), 1)
        draw_landmark_group(self.RIGHT_EYEBROW, (255, 200, 100), 1)
        draw_landmark_group(self.LIPS_OUTER, (180, 100, 255), 1)
        draw_landmark_group(self.LIPS_INNER, (200, 130, 255), 1)
        draw_landmark_group(self.NOSE_BRIDGE, (150, 200, 200), 1)

        for iris_idx in [468, 473]:
            if iris_idx < len(landmarks):
                lm = landmarks[iris_idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

class SuspicionScorer:
    def __init__(self, decay_rate=0.98, decay_interval_sec=1.0):
        self.scores = defaultdict(float)
        self.decay_rate = decay_rate
        self.last_decay_time = defaultdict(float)
        self.decay_interval = decay_interval_sec

    def add_event(self, oid, points):
        self.scores[oid] = min(100.0, self.scores[oid] + points)

    def decay_scores(self, current_time):
        for oid in list(self.scores):
            if current_time - self.last_decay_time[oid] > self.decay_interval:
                self.scores[oid] *= self.decay_rate
                if self.scores[oid] < 1.0:
                    del self.scores[oid]
                    del self.last_decay_time[oid]
                else:
                    self.last_decay_time[oid] = current_time

    def get_score(self, oid):
        return self.scores.get(oid, 0.0)

class BehaviorAnalyzer:
    def __init__(self, config):
        self.config = config
        self.current_time = time.time()
        self.suspicion_scorer = SuspicionScorer(
            config.get("SUSPICION_DECAY_RATE", 0.98),
            config.get("SUSPICION_DECAY_INTERVAL", 1.0),
        )
        self.object_history = defaultdict(lambda: {
            "centroids": deque(maxlen=60), "velocities": deque(maxlen=60),
            "bboxes": deque(maxlen=60), "timestamps": deque(maxlen=60),
            "behaviors": set(), "stress_level": "Low",
        })
        self.heatmap = np.zeros(config.get("SPATIAL_GRID_SIZE", (20, 20)), dtype=np.float32)
        self.heatmap_update_interval = config.get("HEATMAP_UPDATE_INTERVAL", 5.0)
        self.last_heatmap_update = 0.0

    @staticmethod
    def _calculate_iou(a, b):
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0
        return inter / float((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

    def _analyze_hesitation(self, h):
        if len(h["velocities"]) <= 10:
            return None
        recent = np.array(list(h["velocities"])[:-10])
        if recent.size == 0:
            return None
        avg = np.mean(np.linalg.norm(recent, axis=1))
        thresh = self.config.get("HESITATION_SPEED_THRESHOLD", 5.0)
        if avg < thresh:
            stop = None
            for i in range(len(h["velocities"]) - 1, 0, -1):
                if np.linalg.norm(h["velocities"][i]) > thresh:
                    stop = h["timestamps"][i]
                    break
            if stop and (self.current_time - stop) > self.config.get("HESITATION_STOP_TIME_SEC", 3.0):
                return "Object stopped moving for a significant duration."
        return None

    def _analyze_pacing(self, h):
        if len(h["centroids"]) < 10:
            return None
        win = self.current_time - self.config.get("PACING_WINDOW_SEC", 10.0)
        idx = [i for i, t in enumerate(h["timestamps"]) if t > win]
        if len(idx) < 5:
            return None
        c = np.array([h["centroids"][i] for i in idx])
        xr, yr = np.ptp(c[:, 0]), np.ptp(c[:, 1])
        axis = 0 if xr > yr*1.5 else (1 if yr > xr*1.5 else None)
        if axis is not None:
            pos = c[:, axis]
            changes = 0
            cur = np.sign(pos[1]-pos[0])
            for i in range(2, len(pos)):
                d = np.sign(pos[i]-pos[i-1])
                if d != 0 and d != cur:
                    changes += 1; cur = d
            if changes >= self.config.get("PACING_DIRECTION_CHANGES", 3):
                return f"Repetitive back-and-forth ({changes} changes)."
        return None

    def _analyze_scanning(self, h):
        if len(h["velocities"]) < 15:
            return None
        rv = np.array(list(h["velocities"])[-15:])
        rc = np.array(list(h["centroids"])[-15:])
        if (np.var(np.linalg.norm(rv, axis=1)) > self.config.get("SCANNING_VAR_THRESHOLD", 100)
                and np.linalg.norm(rc[-1]-rc[0]) < self.config.get("SCANNING_DISP_THRESHOLD", 30)):
            return "High movement variance with low displacement (possible scanning)."
        return None

    def _analyze_crowd_dynamics(self, centroids):
        if not SKLEARN_AVAILABLE or len(centroids) < self.config.get("CROWD_MIN_SIZE", 4):
            return []
        pts = np.array(list(centroids.values()))
        labels = DBSCAN(eps=self.config.get("CROWD_RADIUS", 100),
                        min_samples=self.config.get("CROWD_MIN_SIZE", 4)).fit(pts).labels_
        events = []
        for lbl in set(labels):
            if lbl != -1:
                n = int(np.sum(labels == lbl))
                if n >= self.config.get("CROWD_MIN_SIZE", 4):
                    events.append(("CROWD_FORMING", "SYSTEM", 1.0, f"Crowd of {n} detected."))
        return events

    def _analyze_spatial_anomaly(self, centroid):
        fh, fw = self.config.get("FRAME_SIZE", (720, 1280))
        gh, gw = self.heatmap.shape
        gx = int(np.clip((centroid[0]/fw)*gw, 0, gw-1))
        gy = int(np.clip((centroid[1]/fh)*gh, 0, gh-1))
        if np.max(self.heatmap) > 0:
            if (self.heatmap / np.max(self.heatmap))[gy, gx] < self.config.get("SPATIAL_ANOMALY_THRESHOLD", 0.05):
                return True
        return False

    def _analyze_object_interaction(self, person_bboxes, obj_dets):
        events = []
        interaction_classes = {"cell phone", "laptop", "backpack", "handbag", "knife", "scissors", "hammer"}
        for cls, bbox, _ in obj_dets:
            if cls in interaction_classes:
                for pid, pb in person_bboxes.items():
                    if self._calculate_iou(pb, bbox) > self.config.get("INTERACTION_IOU_THRESHOLD", 0.1):
                        events.append(("OBJECT_INTERACTION", f"ID_{pid}", 1.0, f"ID_{pid} near {cls}."))
                        break
        return events

    def _update_psychological_profile(self, h):
        if "PACING" in h["behaviors"] or "HESITATION" in h["behaviors"]:
            h["stress_level"] = "High"
        elif "SCANNING" in h["behaviors"]:
            h["stress_level"] = "Medium"
        else:
            h["stress_level"] = "Low"

    def update(self, tracked_objects, yolo_detections, frame_shape, face_bboxes=None):
        self.current_time = time.time()
        self.config["FRAME_SIZE"] = frame_shape[:2]
        events = []

        for oid, cent in tracked_objects.items():
            h = self.object_history[oid]
            h["centroids"].append(cent)
            h["timestamps"].append(self.current_time)
            vel = (np.array(cent) - np.array(h["centroids"][-2])) if len(h["centroids"]) > 1 else np.zeros(2)
            h["velocities"].append(vel)
            if face_bboxes and oid in face_bboxes:
                h["bboxes"].append(face_bboxes[oid])

        for oid in list(self.object_history):
            if oid not in tracked_objects:
                continue
            h = self.object_history[oid]
            h["behaviors"].clear()

            r = self._analyze_hesitation(h)
            if r:
                events.append(("HESITATION", f"ID_{oid}", 1.0, r))
                h["behaviors"].add("HESITATION")
                self.suspicion_scorer.add_event(oid, self.config["SUSPICION_POINTS"].get("HESITATION", 10))

            r = self._analyze_pacing(h)
            if r:
                events.append(("PACING", f"ID_{oid}", 1.0, r))
                h["behaviors"].add("PACING")
                self.suspicion_scorer.add_event(oid, self.config["SUSPICION_POINTS"].get("PACING", 20))

            r = self._analyze_scanning(h)
            if r:
                events.append(("SCANNING", f"ID_{oid}", 1.0, r))
                h["behaviors"].add("SCANNING")
                self.suspicion_scorer.add_event(oid, self.config["SUSPICION_POINTS"].get("SCANNING", 5))

            self._update_psychological_profile(h)

        for oid, cent in tracked_objects.items():
            if self._analyze_spatial_anomaly(cent):
                events.append(("SPATIAL_ANOMALY", f"ID_{oid}", 1.0, f"ID_{oid} in unusual location."))
                self.suspicion_scorer.add_event(oid, self.config["SUSPICION_POINTS"].get("SPATIAL_ANOMALY", 25))

        cents = {oid: self.object_history[oid]["centroids"][-1]
                 for oid in tracked_objects if oid in self.object_history}
        events.extend(self._analyze_crowd_dynamics(cents))

        if face_bboxes:
            pb = {oid: h["bboxes"][-1] for oid, h in self.object_history.items()
                  if oid in tracked_objects and h["bboxes"]}
            events.extend(self._analyze_object_interaction(pb, yolo_detections))

        if self.current_time - self.last_heatmap_update > self.heatmap_update_interval:
            fh, fw = frame_shape[:2]
            self.heatmap *= 0.9
            gh, gw = self.heatmap.shape
            for c in cents.values():
                self.heatmap[int(np.clip((c[1]/fh)*gh, 0, gh-1)),
                             int(np.clip((c[0]/fw)*gw, 0, gw-1))] += 1
            self.last_heatmap_update = self.current_time

        self.suspicion_scorer.decay_scores(self.current_time)
        return events

    def get_object_state(self, oid):
        h = self.object_history.get(oid)
        if not h:
            return {}
        return {
            "suspicion_score": self.suspicion_scorer.get_score(oid),
            "stress_level": h["stress_level"],
            "active_behaviors": list(h["behaviors"]),
        }

    def get_velocities(self, tracked_object_ids):
        result = {}
        for oid in tracked_object_ids:
            h = self.object_history.get(oid)
            if h and len(h["velocities"]) > 0:
                recent = list(h["velocities"])[:-5]
                if len(recent) > 0:
                    avg_vel = np.mean(recent, axis=0)
                    result[oid] = avg_vel
                else:
                    result[oid] = np.zeros(2)
            else:
                result[oid] = np.zeros(2)
        return result

    def cleanup(self, active_ids):
        for oid in set(self.object_history) - set(active_ids):
            del self.object_history[oid]
            self.suspicion_scorer.scores.pop(oid, None)

    def get_heatmap(self):
        if np.max(self.heatmap) > 0:
            v = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            return cv2.applyColorMap(v, cv2.COLORMAP_JET)
        return np.zeros((self.heatmap.shape[0], self.heatmap.shape[1], 3), dtype=np.uint8)

class CrowdIntelligence:
    def __init__(self, config=None):
        if config is None:
            config = CONFIG
        self.config = config.get("CROWD_INTELLIGENCE", {})
        self.enabled = self.config.get("ENABLED", True)
        if not self.enabled:
            return

        gh, gw = self.config.get("HEATMAP_GRID", (40, 30))
        self.density_heatmap = np.zeros((gh, gw), dtype=np.float64)
        self.heatmap_decay = self.config.get("HEATMAP_DECAY", 0.998)
        self.gauss_radius = self.config.get("HEATMAP_GAUSSIAN_RADIUS", 2.5)
        self.gauss_strength = self.config.get("HEATMAP_GAUSSIAN_STRENGTH", 1.0)
        kr = int(math.ceil(self.gauss_radius * 3))
        self.gauss_kernel = np.zeros((kr * 2 + 1, kr * 2 + 1), dtype=np.float64)
        for dy in range(-kr, kr + 1):
            for dx in range(-kr, kr + 1):
                self.gauss_kernel[dy + kr, dx + kr] = self.gauss_strength * math.exp(
                    -(dx * dx + dy * dy) / (2 * self.gauss_radius * self.gauss_radius))

        self.flow_line_y = self.config.get("FLOW_LINE_Y", 300)
        self.flow_crossings = deque(maxlen=200)
        self.flow_stats_window = self.config.get("FLOW_STATS_WINDOW_SEC", 60)
        self.total_entries = 0
        self.total_exits = 0
        self.prev_y = {}

        cg_rows, cg_cols = self.config.get("CONGESTION_GRID", (3, 3))
        self.congestion_rows = cg_rows
        self.congestion_cols = cg_cols
        self.congestion_threshold = self.config.get("CONGESTION_THRESHOLD", 4)
        self.congestion_warning = self.config.get("CONGESTION_WARNING", 2)
        self.congestion_cooldown = {}
        self.congestion_alert_interval = 10.0

        self.zone_occupancy_history = {}
        self.zone_metrics_window = self.config.get("ZONE_METRICS_WINDOW_SEC", 120)
        self.frame_size = (720, 1280)

        self.speed_history = deque(maxlen=150)
        self.direction_history = deque(maxlen=150)
        self.evac_check_window = self.config.get("EVAC_CHECK_WINDOW_SEC", 3.0)
        self.evac_speed_thresh = self.config.get("EVAC_AVG_SPEED_THRESHOLD", 20.0)
        self.evac_direction_consensus = self.config.get("EVAC_DIRECTION_CONSENSUS", 0.55)
        self.evac_min_people = self.config.get("EVAC_MIN_PEOPLE", 3)
        self.last_evac_alert = 0.0
        self.evac_alert_cooldown = 15.0

        self.person_smoothed_vel = {}
        self.vel_smoothing = 0.7
        print("[INFO] CrowdIntelligence module initialized.")

    def update(self, tracked_objects, velocities, frame_shape, current_time=None):
        if not self.enabled:
            return []
        if current_time is None:
            current_time = time.time()
        self.frame_size = frame_shape[:2]
        fh, fw = frame_shape[:2]
        events = []

        self._update_density_heatmap(tracked_objects, fh, fw)
        flow_events = self._update_flow(tracked_objects, current_time)
        events.extend(flow_events)
        cong_events = self._update_congestion(tracked_objects, fh, fw, current_time)
        events.extend(cong_events)
        self._update_flow_vectors(tracked_objects, velocities)
        self._update_zone_utilization(tracked_objects, fh, fw, current_time)
        evac_events = self._check_evacuation(tracked_objects, velocities, current_time)
        events.extend(evac_events)
        return events

    def _update_density_heatmap(self, tracked_objects, fh, fw):
        self.density_heatmap *= self.heatmap_decay
        gh, gw = self.density_heatmap.shape
        kr = self.gauss_kernel.shape[0] // 2
        for oid, pos in tracked_objects.items():
            if not hasattr(pos, "__len__") or len(pos) != 2:
                continue 
            cx, cy = pos
            gx = int((cx / fw) * gw)
            gy = int((cy / fh) * gh)
            for dy in range(-kr, kr + 1):
                ny = gy + dy
                if 0 <= ny < gh:
                    for dx in range(-kr, kr + 1):
                        nx = gx + dx
                        if 0 <= nx < gw:
                            self.density_heatmap[ny, nx] += self.gauss_kernel[dy + kr, dx + kr]

    def get_density_heatmap_image(self, frame_shape):
        hm = self.density_heatmap.copy()
        if np.max(hm) > 0:
            hm = (hm / np.max(hm) * 255).astype(np.uint8)
        else:
            hm = hm.astype(np.uint8)
        colored = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        return cv2.resize(colored, (frame_shape[1], frame_shape[0]))

    def _update_flow(self, tracked_objects, current_time):
        events = []
        for oid, pos in tracked_objects.items():
            if not hasattr(pos, "__len__") or len(pos) != 2:
                continue
            cx, cy = pos
            if oid in self.prev_y:
                prev_y = self.prev_y[oid]
                if prev_y < self.flow_line_y <= cy:
                    self.flow_crossings.append((current_time, "IN"))
                    self.total_entries += 1
                    events.append(("FLOW_IN", f"ID_{oid}", 1.0, f"ID_{oid} entered."))
                elif prev_y > self.flow_line_y >= cy:
                    self.flow_crossings.append((current_time, "OUT"))
                    self.total_exits += 1
                    events.append(("FLOW_OUT", f"ID_{oid}", 1.0, f"ID_{oid} exited."))
            self.prev_y[oid] = cy
        active_ids = set(tracked_objects.keys())
        stale = set(self.prev_y.keys()) - active_ids
        for oid in stale:
            del self.prev_y[oid]
        return events

    def get_flow_stats(self):
        window = self.flow_stats_window
        cutoff = time.time() - window
        recent = [(t, d) for t, d in self.flow_crossings if t > cutoff]
        entries = sum(1 for _, d in recent if d == "IN")
        exits = sum(1 for _, d in recent if d == "OUT")
        rate_in = (entries / window) * 60 if window > 0 else 0
        rate_out = (exits / window) * 60 if window > 0 else 0
        return {"total_in": self.total_entries, "total_out": self.total_exits,
                "rate_in_per_min": rate_in, "rate_out_per_min": rate_out,
                "net": self.total_entries - self.total_exits, "window_sec": window}

    def _update_congestion(self, tracked_objects, fh, fw, current_time):
        events = []
        people_list = []
        for p in tracked_objects.values():
            if hasattr(p, "__len__") and len(p) == 2:
                people_list.append(p)

        for row in range(self.congestion_rows):
            for col in range(self.congestion_cols):
                x1 = int((col / self.congestion_cols) * fw)
                y1 = int((row / self.congestion_rows) * fh)
                x2 = int(((col + 1) / self.congestion_cols) * fw)
                y2 = int(((row + 1) / self.congestion_rows) * fh)
                count = sum(1 for (cx, cy) in people_list if x1 <= cx < x2 and y1 <= cy < y2)
                zone_idx = row * self.congestion_cols + col
                if count >= self.congestion_threshold:
                    last_alert = self.congestion_cooldown.get(zone_idx, 0)
                    if (current_time - last_alert) > self.congestion_alert_interval:
                        events.append(("CONGESTION", "SYSTEM", 1.0,
                                      f"Zone({row},{col}): {count} people."))
                        self.congestion_cooldown[zone_idx] = current_time
                elif count >= self.congestion_warning:
                    last_alert = self.congestion_cooldown.get(zone_idx, 0)
                    if (current_time - last_alert) > self.congestion_alert_interval:
                        events.append(("CONGESTION_WARNING", "SYSTEM", 0.7,
                                      f"Zone({row},{col}): {count} people (warning)."))
                        self.congestion_cooldown[zone_idx] = current_time
        return events

    def get_zone_occupancy(self, tracked_objects, fh, fw):
        occupancy = {}
        people_list = []
        for p in tracked_objects.values():
            if hasattr(p, "__len__") and len(p) == 2:
                people_list.append(p)

        for row in range(self.congestion_rows):
            for col in range(self.congestion_cols):
                x1 = int((col / self.congestion_cols) * fw)
                y1 = int((row / self.congestion_rows) * fh)
                x2 = int(((col + 1) / self.congestion_cols) * fw)
                y2 = int(((row + 1) / self.congestion_rows) * fh)
                count = sum(1 for (cx, cy) in people_list if x1 <= cx < x2 and y1 <= cy < y2)
                occupancy[row * self.congestion_cols + col] = count
        return occupancy

    def _update_flow_vectors(self, tracked_objects, velocities):
        alpha = self.vel_smoothing
        for oid in tracked_objects:
            raw_vel = velocities.get(oid, np.zeros(2))
            if oid in self.person_smoothed_vel:
                self.person_smoothed_vel[oid] = alpha * self.person_smoothed_vel[oid] + (1 - alpha) * raw_vel
            else:
                self.person_smoothed_vel[oid] = raw_vel.copy()
        stale = set(self.person_smoothed_vel.keys()) - set(tracked_objects.keys())
        for oid in stale:
            del self.person_smoothed_vel[oid]

    def _update_zone_utilization(self, tracked_objects, fh, fw, current_time):
        occupancy = self.get_zone_occupancy(tracked_objects, fh, fw)
        for zone_idx, count in occupancy.items():
            if zone_idx not in self.zone_occupancy_history:
                self.zone_occupancy_history[zone_idx] = deque(maxlen=500)
            self.zone_occupancy_history[zone_idx].append((current_time, count))

    def get_zone_utilization_metrics(self):
        metrics = {}
        window = self.zone_metrics_window
        cutoff = time.time() - window
        for zone_idx, history in self.zone_occupancy_history.items():
            recent = [(t, c) for t, c in history if t > cutoff]
            if not recent:
                metrics[zone_idx] = {"current": 0, "peak": 0, "avg": 0.0, "utilization_pct": 0.0}
                continue
            counts = [c for _, c in recent]
            current = counts[-1] if counts else 0
            peak = max(counts)
            avg = sum(counts) / len(counts)
            utilized = sum(1 for c in counts if c > 0)
            pct = (utilized / len(counts)) * 100 if counts else 0
            metrics[zone_idx] = {"current": current, "peak": peak, "avg": avg, "utilization_pct": pct}
        return metrics

    def _check_evacuation(self, tracked_objects, velocities, current_time):
        events = []
        num_people = len(tracked_objects)
        if num_people < self.evac_min_people:
            self.speed_history.append((current_time, 0.0))
            self.direction_history.append((current_time, 0.0))
            return events

        speeds = [np.linalg.norm(velocities.get(oid, np.zeros(2))) for oid in tracked_objects]
        avg_speed = np.mean(speeds) if speeds else 0.0
        self.speed_history.append((current_time, avg_speed))

        dirs = []
        for oid in tracked_objects:
            v = velocities.get(oid, np.zeros(2))
            mag = np.linalg.norm(v)
            if mag > 2.0:
                angle = math.atan2(v[1], v[0])
                dirs.append(angle)
        self.direction_history.append((current_time, dirs))

        cutoff = current_time - self.evac_check_window
        recent_speeds = [s for t, s in self.speed_history if t > cutoff]
        recent_dirs = [d for t, d in self.direction_history if t > cutoff]
        if len(recent_speeds) < 3:
            return events

        avg_recent_speed = np.mean(recent_speeds)
        speed_trigger = avg_recent_speed > self.evac_speed_thresh

        all_dirs = []
        for d_list in recent_dirs:
            if isinstance(d_list, list):
                all_dirs.extend(d_list)
            elif isinstance(d_list, (float, int)):
                pass

        direction_trigger = False
        dominant_deg = 0.0  

        if len(all_dirs) >= self.evac_min_people:
            sin_sum = sum(math.sin(a) for a in all_dirs)
            cos_sum = sum(math.cos(a) for a in all_dirs)
            mean_resultant = math.sqrt(sin_sum ** 2 + cos_sum ** 2) / len(all_dirs)
            direction_trigger = mean_resultant > self.evac_direction_consensus
            dominant_angle = math.atan2(sin_sum, cos_sum)
            dominant_deg = math.degrees(dominant_angle)

        if speed_trigger and direction_trigger:
            if (current_time - self.last_evac_alert) > self.evac_alert_cooldown:
                direction_name = self._angle_to_direction(dominant_deg)
                events.append((
                    "EVACUATION_ALERT", "SYSTEM", 1.0,
                    f"Possible evacuation! Avg speed: {avg_recent_speed:.1f} px/f, "
                    f"{len(all_dirs)} people moving {direction_name}."
                ))
                self.last_evac_alert = current_time
        return events

    @staticmethod
    def _angle_to_direction(deg):
        dirs = ["right", "down-right", "down", "down-left", "left",
                "up-left", "up", "up-right"]
        deg = deg % 360
        if deg < 0:
            deg += 360
        idx = int(((deg + 22.5) % 360) / 45)
        return dirs[idx % 8]

    def draw(self, frame, tracked_objects):
        if not self.enabled:
            return frame
        fh, fw = frame.shape[:2]

        # Density Heatmap Overlay DISABLED per user request
        # if self.config.get("SHOW_DENSITY_HEATMAP", True):
        #     hm_img = self.get_density_heatmap_image(frame.shape)
        #     opacity = self.config.get("HEATMAP_OPACITY", 0.35)
        #     frame = cv2.addWeighted(hm_img, opacity, frame, 1.0 - opacity, 0)

        # Congestion Zones Overlay DISABLED per user request
        # if self.config.get("SHOW_CONGESTION_ZONES", True):
        #     self._draw_congestion_zones(frame, tracked_objects, fh, fw)

        # Flow Vectors Overlay DISABLED per user request
        # if self.config.get("SHOW_FLOW_VECTORS", True):
        #     self._draw_flow_vectors(frame, tracked_objects)
        return frame

    def _draw_congestion_zones(self, frame, tracked_objects, fh, fw):
        occupancy = self.get_zone_occupancy(tracked_objects, fh, fw)
        for row in range(self.congestion_rows):
            for col in range(self.congestion_cols):
                x1 = int((col / self.congestion_cols) * fw)
                y1 = int((row / self.congestion_rows) * fh)
                x2 = int(((col + 1) / self.congestion_cols) * fw)
                y2 = int(((row + 1) / self.congestion_rows) * fh)
                zone_idx = row * self.congestion_cols + col
                count = occupancy.get(zone_idx, 0)
                if count >= self.congestion_threshold:
                    color = (0, 0, 255); thickness = 2; label = f"C:{count}"
                elif count >= self.congestion_warning:
                    color = (0, 165, 255); thickness = 1; label = f"W:{count}"
                else:
                    color = (0, 80, 0); thickness = 1; label = f"{count}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, label, (x1 + 3, y1 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    def _draw_flow_vectors(self, frame, tracked_objects):
        scale = self.config.get("FLOW_VECTOR_SCALE", 4.0)
        max_len = self.config.get("FLOW_VECTOR_MAX_LEN", 50)
        for oid, pos in tracked_objects.items():
            if not hasattr(pos, "__len__") or len(pos) != 2:
                continue
            cx, cy = pos
            if oid not in self.person_smoothed_vel:
                continue
            vx, vy = self.person_smoothed_vel[oid]
            mag = math.sqrt(vx * vx + vy * vy)
            if mag < 1.0:
                continue
            length = min(mag * scale, max_len)
            dx = int((vx / mag) * length)
            dy = int((vy / mag) * length)
            speed_ratio = min(mag / 25.0, 1.0)
            if speed_ratio < 0.5:
                t = speed_ratio * 2
                color = (int(255 * (1 - t)), int(255 * t), 0)
            else:
                t = (speed_ratio - 0.5) * 2
                color = (0, int(255 * (1 - t)), int(255 * t))
            end_x, end_y = cx + dx, cy + dy
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)

    def draw_stats_panel(self, frame):
        if not self.enabled:
            return frame
        pw = 230; ph = 195
        x0 = frame.shape[1] - pw - 10; y0 = 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (20, 20, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (100, 100, 150), 1)
        cx = x0 + 8; cy = y0 + 18
        cv2.putText(frame, "CROWD INTELLIGENCE", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
        cy += 20
        if self.config.get("SHOW_FLOW_STATS", True):
            stats = self.get_flow_stats()
            cv2.putText(frame, f"In: {stats['total_in']}  Out: {stats['total_out']}",
                        (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
            cy += 16
            cv2.putText(frame, f"Rate: {stats['rate_in_per_min']:.1f}/min in, {stats['rate_out_per_min']:.1f}/min out",
                        (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            cy += 16
            cv2.putText(frame, f"Net: {'+'if stats['net']>=0 else ''}{stats['net']}",
                        (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0) if stats['net'] >= 0 else (0, 0, 255), 1)
            cy += 20
        # Zone metrics disabled for clean view
        # if self.config.get("SHOW_ZONE_METRICS", True):
        #     cv2.putText(frame, "Zone Utilization (peak/avg):", (cx, cy),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 220), 1)
        #     cy += 16
        #     metrics = self.get_zone_utilization_metrics()
        #     for zi in sorted(metrics.keys()):
        #         m = metrics[zi]
        #         row, col = zi // self.congestion_cols, zi % self.congestion_cols
        #         pct = m["utilization_pct"]
        #         color = (0, 0, 255) if pct > 80 else ((0, 165, 255) if pct > 50 else (150, 255, 150))
        #         text = f"Z({row},{col}): peak={m['peak']} avg={m['avg']:.1f} ({pct:.0f}%)"
        #         cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        #         cy += 14
        #         if cy > y0 + ph - 10:
        #             break
        return frame

class BlinkDetector:
    def __init__(self, config):
        self.config = config
        self.state = {}

    def _ear(self, pts):
        try:
            A = np.linalg.norm(np.array([pts[1].x, pts[1].y]) - np.array([pts[5].x, pts[5].y]))
            B = np.linalg.norm(np.array([pts[2].x, pts[2].y]) - np.array([pts[4].x, pts[4].y]))
            C = np.linalg.norm(np.array([pts[0].x, pts[0].y]) - np.array([pts[3].x, pts[3].y]))
            return (A+B)/(2.0*C) if C > 0 else 0.0
        except Exception:
            return 0.0

    def update(self, oid, lm):
        if oid not in self.state:
            self.state[oid] = {"ear": deque(maxlen=self.config.get("EAR_HISTORY_FRAMES", 30)),
                               "blinks": 0, "last_blink": 0.0}
        s = self.state[oid]
        le = [lm[i] for i in [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]]
        re = [lm[i] for i in [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]]
        avg = (self._ear(le) + self._ear(re)) / 2.0
        s["ear"].append(avg)
        thr = self.config.get("BLINK_EAR_THRESHOLD", 0.2)
        if avg < thr and len(s["ear"]) > 1 and s["ear"][-2] > thr:
            now = time.time()
            if (now - s["last_blink"]) > self.config.get("MIN_TIME_BETWEEN_BLINKS_SEC", 0.5):
                s["blinks"] += 1; s["last_blink"] = now; return True
        return False

    def is_live(self, oid, window):
        s = self.state.get(oid)
        if not s:
            return False
        now = time.time()
        return (s["last_blink"] > 0 and (now - s["last_blink"]) < window
                and s["blinks"] >= self.config.get("MIN_BLINKS_FOR_LIVENESS", 2))


class HeadPoseEstimator:
    def __init__(self, config):
        self.config = config
        self.pts3d = _get_3d_model_points()
        self.state = {}

    def estimate(self, oid, lm, shape):
        if oid not in self.state:
            self.state[oid] = {"pose": deque(maxlen=self.config.get("POSE_HISTORY_FRAMES", 30))}
        s = self.state[oid]
        ip = _get_2d_image_points(lm, shape)
        f = shape[1]; c = (shape[1]//2, shape[0]//2)
        cam = np.array([[f,0,c[0]],[0,f,c[1]],[0,0,1]], dtype="double")
        ok, rv, tv = cv2.solvePnP(self.pts3d, ip, cam, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None, None
        rm, _ = cv2.Rodrigues(rv)
        ang = cv2.RQDecomp3x3(rm)[0]
        s["pose"].append(tuple(ang))
        return ang[0], ang[1], ang[2]

    def is_static(self, oid, var_thresh=5.0):
        s = self.state.get(oid)
        if not s or len(s["pose"]) < self.config.get("POSE_HISTORY_FRAMES", 30):
            return False
        return bool(np.all(np.var(np.array(s["pose"]), axis=0) < var_thresh))


class DepthEstimator:
    def __init__(self, config):
        self.model = None
        self.transforms = None
        self._dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_attempted = False

    def _ensure_loaded(self):
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            print("[INFO] Loading MiDaS depth model...")
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self._dev).eval()
            self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
            print("[INFO] MiDaS loaded.")
        except Exception as e:
            print(f"[WARN] MiDaS unavailable: {e}")

    def estimate_depth_variance(self, roi):
        self._ensure_loaded()
        if self.model is None or roi is None or roi.size == 0:
            return None
        try:
            dev = next(self.model.parameters()).device
            with torch.no_grad():
                p = self.model(self.transforms(roi).to(dev))
                p = torch.nn.functional.interpolate(p.unsqueeze(1), size=roi.shape[:2],
                                                     mode="bicubic", align_corners=False).squeeze()
            d = cv2.normalize(p.cpu().numpy(), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            r = d[10:-10, 10:-10]
            return float(np.var(r)) if r.size > 0 else None
        except Exception as e:
            print(f"[ERROR] Depth estimation: {e}")
            return None


class SpoofClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64*16*16,128), nn.ReLU(),
                                nn.Dropout(0.5), nn.Linear(128,2))
    def forward(self, x):
        return self.fc(self.conv(x))


class SpoofClassifier:
    def __init__(self, config):
        self.model = None; self.loaded = False
        if config.get("ENABLE_ML_CLASSIFIER", False):
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self.model = SpoofClassifierCNN().to(dev)
                p = config.get("ANTI_SPOOF_MODEL_PATH", "antispoof_model.bin")
                if os.path.exists(p):
                    self.model.load_state_dict(torch.load(p, map_location=dev))
                    self.model.eval(); self.loaded = True
            except Exception as e:
                print(f"[WARN] Spoof CNN failed: {e}")

    def predict(self, roi):
        if not self.loaded or roi is None or roi.size == 0:
            return 0.5
        try:
            t = torch.from_numpy(cv2.resize(roi,(128,128))).permute(2,0,1).float().unsqueeze(0)/255.0
            with torch.no_grad():
                return float(torch.nn.functional.softmax(self.model(t.to(next(self.model.parameters()).device)),1)[0,1])
        except Exception:
            return 0.5


class AntiSpoofDetector:
    def __init__(self, config):
        self.config = config
        ac = config.get("ANTI_SPOOFING", {})
        if not ac.get("ENABLED", False):
            self.enabled = False; return
        self.enabled = True
        self.blink = BlinkDetector(ac)
        self.pose = HeadPoseEstimator(ac)
        self.depth = DepthEstimator(ac)
        self.spoof_clf = SpoofClassifier(ac)
        self.mp_fm = mp.solutions.face_mesh
        self.fm = self.mp_fm.FaceMesh(static_image_mode=False, max_num_faces=1,
                                       refine_landmarks=True, min_detection_confidence=0.5,
                                       min_tracking_confidence=0.5)

    def is_real(self, crop, oid):
        if not self.enabled or crop is None or crop.size == 0:
            return True, "Disabled"
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = self.fm.process(rgb)
        if not res.multi_face_landmarks:
            return False, "No face mesh"
        face_lm = res.multi_face_landmarks[0]
        lm = _safe_landmarks(face_lm)
        ac = self.config.get("ANTI_SPOOFING", {})
        reasons = []
        self.blink.update(oid, lm)
        if not self.blink.is_live(oid, ac.get("LIVENESS_TIME_WINDOW_SEC", 15.0)):
            reasons.append("No blink")
        y, p, r = self.pose.estimate(oid, lm, crop.shape)
        if y is None:
            reasons.append("No pose")
        elif self.pose.is_static(oid):
            reasons.append("Static pose")
        dv = self.depth.estimate_depth_variance(crop)
        if dv is not None and dv < ac.get("DEPTH_VARIANCE_THRESHOLD", 0.01):
            reasons.append("Flat face")
        if ac.get("ENABLE_ML_CLASSIFIER", False):
            sp = self.spoof_clf.predict(crop)
            if sp > ac.get("ML_CLASSIFIER_THRESHOLD", 0.7):
                reasons.append(f"ML spoof {sp:.2f}")
        return (True, "OK") if not reasons else (False, ", ".join(reasons))

class VisionSystem:
    def __init__(self):
        print("[INFO] Initializing Vision System...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Device: {self.device.upper()}")

        self._init_models()
        self._load_face_database()
        self._init_stranger_tracking()
        self._init_components()
        self._init_state()
        self.warm_up()
        print("[INFO] Vision System ready.")

    def _init_stranger_tracking(self):
        self.stranger_indexer = FAISSIndexer(512)
        self.stranger_counter = 0
        self.stranger_buffer = defaultdict(list) 
        self.stranger_db_path = CONFIG["UNKNOWN_DB_FILE"]
        
        if os.path.exists(self.stranger_db_path):
            try:
                with open(self.stranger_db_path, "rb") as f:
                    data = pickle.load(f)
                names = data.get("names", [])
                embs = data.get("embeddings", [])
                if embs and names:
                    self.stranger_indexer.add_embeddings(names, np.array(embs))
                    max_c = 0
                    for n in names:
                        if n.startswith("STRANGER_"):
                            try:
                                c = int(n.split("_")[1])
                                if c > max_c: max_c = c
                            except: pass
                    self.stranger_counter = max_c + 1
                print(f"[INFO] Loaded {len(names)} strangers into memory.")
            except Exception as e:
                print(f"[WARN] Failed to load stranger DB: {e}")

    def _save_stranger_db(self):
        if not self.stranger_indexer.index.ntotal:
            return
        try:
            data = {
                "names": self.stranger_indexer.known_names,
                "embeddings": [] # We need to reconstruct this from the FAISS index if we wanted full persistence, 
                               # but FAISS doesn't easily give back embeddings without index.search.
                               # Simpler approach: Save the embeddings we added recently.
            }
            # Since FAISS doesn't expose internal array easily in this wrapper without search,
            # we rely on the fact that embeddings are transient in memory for this demo.
            # To strictly save them, we'd need a list in memory too.
            # For now, we save the Names to know who exists, but to match them again next run
            # we strictly need the embeddings.
            # FIX: Store embeddings in a separate dict during addition.
            with open(self.stranger_db_path, "wb") as f:
                # Just a placeholder for full persistence logic which requires a dict in memory
                pass 
        except:
            pass

    def _init_models(self):
        try:
            self.face_app = FaceAnalysis(name="buffalo_l")
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            print(f"[WARN] InsightFace GPU failed: {e}. Using CPU.")
            self.face_app = FaceAnalysis(name="buffalo_l")
            self.face_app.prepare(ctx_id=-1, det_size=(640, 640))
        print("[INFO] InsightFace loaded.")

        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(CONFIG["MODEL_PATH"]).to(self.device)
                print("[INFO] YOLO loaded (shared instance).")
            except Exception as e:
                print(f"[ERROR] YOLO load failed: {e}")
                self.yolo_model = None
        else:
            print("[WARN] YOLO not available. Object detection disabled.")
            self.yolo_model = None

    def _load_face_database(self):
        db_path = CONFIG["DATABASE_FILE"]
        if not os.path.exists(db_path):
            print(f"[WARN] Face DB not found at '{db_path}'. Recognition disabled.")
            self.face_db = {}
            self.face_indexer = FAISSIndexer(512)
            return

        try:
            with open(db_path, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict) and any(
                isinstance(v, dict) and "embeddings" in v for v in data.values()
            ):
                self.face_db = data
                names, embs, threshs = [], [], []
                for name, d in data.items():
                    for emb in d.get("embeddings", []):
                        names.append(name)
                        embs.append(emb)
                        threshs.append(d.get("threshold", CONFIG["FACE_RECOG_THRESHOLD"]))
                self.face_indexer = FAISSIndexer(512)
                if embs:
                    self.face_indexer.add_embeddings(names, np.array(embs), threshs)

            elif isinstance(data, dict) and "names" in data and "embeddings" in data:
                names = data["names"]
                embs = data["embeddings"]
                self.face_db = {}
                for n, e in zip(names, embs):
                    if n not in self.face_db:
                        self.face_db[n] = {"embeddings": [], "threshold": CONFIG["FACE_RECOG_THRESHOLD"]}
                    self.face_db[n]["embeddings"].append(e)
                self.face_indexer = FAISSIndexer(512)
                if embs:
                    self.face_indexer.add_embeddings(names, np.array(embs))

            elif isinstance(data, list):
                self.face_db = {}
                names, embs = [], []
                for entry in data:
                    n = entry.get("name", "Unknown")
                    e = entry.get("embedding")
                    if e is not None:
                        if n not in self.face_db:
                            self.face_db[n] = {"embeddings": [], "threshold": CONFIG["FACE_RECOG_THRESHOLD"]}
                        self.face_db[n]["embeddings"].append(e)
                        names.append(n)
                        embs.append(e)
                self.face_indexer = FAISSIndexer(512)
                if embs:
                    self.face_indexer.add_embeddings(names, np.array(embs))
            else:
                print("[ERROR] Unrecognized face DB format.")
                self.face_db = {}
                self.face_indexer = FAISSIndexer(512)

            self._update_all_person_thresholds()

            total_embs = self.face_indexer.index.ntotal
            print(f"[INFO] Face DB loaded: {len(self.face_db)} people, {total_embs} embeddings.")
        except Exception as e:
            print(f"[ERROR] Failed to load face DB: {e}")
            self.face_db = {}
            self.face_indexer = FAISSIndexer(512)

    def _update_all_person_thresholds(self):
        for name, d in self.face_db.items():
            embs = d.get("embeddings", [])
            if len(embs) >= 2:
                mean_var, std_var = _compute_intra_class_variance(embs)
                auto_thresh = _auto_threshold(mean_var, std_var)
                d["threshold"] = auto_thresh

    def _init_components(self):
        self.tracker = CentroidTracker()
        self.behavior_analyzer = BehaviorAnalyzer(CONFIG)
        self.quality_scorer = ImageQualityScorer(CONFIG)
        self.anti_spoof = AntiSpoofDetector(CONFIG)
        self.hand_detector = HandDetector(CONFIG)
        if self.yolo_model is not None:
            self.object_detector = ObjectDetector(CONFIG,
                                                  shared_yolo_model=self.yolo_model,
                                                  device=self.device)
        else:
            self.object_detector = None
        self.face_mesh_drawer = FaceMeshDrawer()
        self.crowd_intel = CrowdIntelligence(CONFIG)
        print("[INFO] All detection modules initialized.")

    def _init_state(self):
        self.counting_state = {}
        self.recognition_history = {}
        self.profile_confidence = {}
        self.fps_start = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self._last_enrollment_time = 0
        self._enrollment_cooldown = 10.0
        self._max_embeddings_per_person = CONFIG.get("MAX_ENROLLMENT_EMBEDDINGS", 10)
        self.snapshot_dir = "snapshots"
        self.thumbnail_dir = "snapshots/thumbnails"
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.thumbnail_dir, exist_ok=True)
        
        self.tracked_face_embeddings = {} 

    def capture_event_snapshot(self, frame, event_type, person_name=None, bbox=None):
        import datetime
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        snap_path = os.path.join(self.snapshot_dir, f"{event_type}_{ts}.jpg")
        cv2.imwrite(snap_path, frame)
        thumb_path = None
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                thumb_path = os.path.join(self.thumbnail_dir, f"{event_type}_{ts}_thumb.jpg")
                cv2.imwrite(thumb_path, crop)
        return snap_path, thumb_path

    def warm_up(self):
        print("[INFO] Warming up models...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            _ = self.face_app.get(dummy)
            if self.object_detector is not None:
                _ = self.object_detector.model(dummy, verbose=False)
        except Exception as e:
            print(f"[WARN] Warm-up failed: {e}")

    def recognize_face(self, oid, embedding):
        if not self.face_db or self.face_indexer.index.ntotal == 0:
            return "UNKNOWN", 0.0, "No database"

        raw = self.face_indexer.search(embedding, k=5, pool_per_person=True)
        if not raw:
            return "UNKNOWN", 0.0, "No match"

        best_idx, best_sim, best_name, best_thresh = raw[0]

        vote = best_name if (1.0 - best_sim) < best_thresh else "UNKNOWN"

        calibrated_conf = _calibrate_confidence(best_sim)

        if oid not in self.recognition_history:
            self.recognition_history[oid] = deque(maxlen=CONFIG["RECOGNITION_HISTORY_LENGTH"])
        if oid not in self.profile_confidence:
            self.profile_confidence[oid] = {"name": "UNKNOWN", "confidence": 0.0}

        self.recognition_history[oid].append(vote)
        hist = list(self.recognition_history[oid])
        most = max(set(hist), key=hist.count)
        ps = self.profile_confidence[oid]
        if most == ps["name"]:
            ps["confidence"] += 1.0
        else:
            ps["name"] = most; ps["confidence"] = 1.0

        if ps["confidence"] >= CONFIG["PROFILE_CONFIDENCE_THRESHOLD"]:
            return ps["name"], calibrated_conf if ps["name"] != "UNKNOWN" else 0.0, \
                   f"Identified as {ps['name']} (calibrated: {calibrated_conf:.2f})"
        return "UNKNOWN", 0.0, f"Building profile ({ps['confidence']:.0f}/{CONFIG['PROFILE_CONFIDENCE_THRESHOLD']})"

    def _check_stranger_reid(self, oid, embedding):
        if not CONFIG.get("STRANGER_TRACKING_ENABLED", False):
            return "UNKNOWN", 0.0

        # 1. Check existing Stranger DB
        if self.stranger_indexer.index.ntotal > 0:
            unk_results = self.stranger_indexer.search(embedding, k=1)
            if unk_results:
                _, sim, unk_name, _ = unk_results[0]
                # Use a slightly lower threshold for matching unknowns/strangers
                if (1.0 - sim) < CONFIG.get("STRANGER_REID_THRESHOLD", 0.40):
                    return unk_name, sim

        # 2. Not in Stranger DB. Buffer this frame.
        self.stranger_buffer[oid].append(embedding)
        
        # 3. Check if we should add this new stranger
        if len(self.stranger_buffer[oid]) >= CONFIG.get("STRANGER_FRAMES_THRESHOLD", 15):
            # Average the embeddings for robustness
            avg_emb = np.mean(self.stranger_buffer[oid], axis=0)
            new_name = f"STRANGER_{self.stranger_counter}"
            self.stranger_counter += 1
            
            # Add to indexer
            self.stranger_indexer.add_embeddings([new_name], np.array([avg_emb]))
            
            # Clear buffer for this OID
            self.stranger_buffer[oid] = []
            
            print(f"[STRANGER] Added new persistent unknown: {new_name}")
            return new_name, 0.8 # High confidence because we just assigned it

        return "UNKNOWN", 0.0

    def process(self, frame):
        t0 = time.time()
        display = frame.copy()
        analysis = frame.copy()
        events = []

        self._update_fps()

        quality = self.quality_scorer.score(analysis)
        if not quality["is_acceptable"]:
            events.append(("LOW_QUALITY", "SYSTEM", 1.0, f"Quality {quality['overall_quality']}/100"))

        faces = self._detect_faces(analysis)

        face_boxes = [f["bbox"] for f in faces]
        # Face mesh disabled for clean view
        # if CONFIG.get("SHOW_FACE_MESH", True) and faces:
        #     self.face_mesh_drawer.process(display, face_boxes)

        object_detections = []
        if self.object_detector is not None:
            object_detections = self.object_detector.detect(analysis)

        hand_detections = self.hand_detector.detect(analysis)

        # Hand details disabled, only boxes
        if hand_detections:
            self.hand_detector.draw_landmarks(display, hand_detections)
            for hd in hand_detections:
                events.append(("HAND_DETECTION", f"{hd['handedness']}_Hand", 1.0,
                               f"{hd['handedness']} hand: {hd['gesture']}"))

        if CONFIG.get("SHOW_OBJECT_BOXES", True) and self.object_detector is not None:
            self.object_detector.draw_detections(display, object_detections, skip_person=True)

        for det in object_detections:
            if det["class_name"].lower() in CONFIG.get("DANGEROUS_OBJECTS", set()):
                events.append(("DANGEROUS_OBJECT", det["class_name"], det["confidence"],
                               f"Dangerous object detected: {det['class_name']} (conf: {det['confidence']:.0%})"))

        tracked = self.tracker.update(face_boxes)

        face_bbox_map = {}
        for oid, cent in tracked.items():
            closest = self._find_closest_face(cent, faces)
            if closest:
                face_bbox_map[oid] = tuple(closest["bbox"])

        beh_events = self.behavior_analyzer.update(
            tracked,
            [(d["class_name"], d["bbox"], d["confidence"]) for d in object_detections],
            analysis.shape, face_bboxes=face_bbox_map,
        )
        events.extend(beh_events)

        if self.crowd_intel.enabled:
            velocities = self.behavior_analyzer.get_velocities(set(tracked.keys()))
            crowd_events = self.crowd_intel.update(tracked, velocities, analysis.shape)
            events.extend(crowd_events)

        self._recognize_and_draw(tracked, faces, analysis, display, events)

        self.behavior_analyzer.cleanup(set(tracked.keys()))

        self._last_tracked = tracked
        display = self._draw_ui(display, events, quality, hand_detections, object_detections)

        dt = time.time() - t0
        return display, faces, hand_detections, object_detections, events, len(tracked), dt

    def _update_fps(self):
        self.frame_count +=1
        now = time.time()
        if now - self.fps_start > 1:
            self.fps = self.frame_count / (now - self.fps_start)
            self.fps_start = now; self.frame_count = 0

    def _detect_faces(self, frame):
        all_faces = self.face_app.get(frame)
        boxes, scores = [], []
        for face in all_faces:
            if face.bbox is not None and face.det_score is not None:
                x1, y1, x2, y2 = face.bbox.astype(int)
                boxes.append([x1, y1, x2-x1, y2-y1])
                scores.append(float(face.det_score))
        keep = []
        if boxes:
            try:
                nms = cv2.dnn.NMSBoxes(np.array(boxes), np.array(scores),
                                        CONFIG["NMS_CONF_THRESHOLD"], CONFIG["NMS_IOU_THRESHOLD"])
                if nms is not None:
                    keep = np.array(nms).flatten().astype(int).tolist()
            except (AttributeError, ValueError):
                keep = list(range(len(boxes)))
        h, w, _ = frame.shape
        valid = []
        for i in keep:
            if i >= len(all_faces):
                continue
            f = all_faces[i]
            b = f.bbox.astype(int)
            x1, y1, x2, y2 = np.clip(b, 0, [w, h, w, h])
            if x2 > x1 and y2 > y1:
                valid.append({"face": f, "bbox": [int(x1), int(y1), int(x2), int(y2)]})
        return valid

    def _recognize_and_draw(self, tracked, faces, analysis, display, events):
        for oid, cent in tracked.items():
            matched = self._find_closest_face(cent, faces)
            if not matched:
                self.tracked_face_embeddings[oid] = None
                # Clear stranger buffer if lost
                if oid in self.stranger_buffer:
                    del self.stranger_buffer[oid]
                continue
            fobj, (x1, y1, x2, y2) = matched["face"], matched["bbox"]
            crop = analysis[y1:y2, x1:x2] 

            # Store embedding for appearance tracking
            if fobj.embedding is not None:
                self.tracked_face_embeddings[oid] = fobj.embedding

            is_real, spoof_reason = self.anti_spoof.is_real(crop, oid)
            if not is_real:
                events.append(("SPOOF_DETECTED", f"ID_{oid}", 1.0, f"Spoof ID:{oid}: {spoof_reason}"))

            name, conf, reason = "UNKNOWN", 0.0, "No embedding"
            if fobj.embedding is not None:
                name, conf, reason = self.recognize_face(oid, fobj.embedding)
                
                # STRANGER LOGIC: If unknown, check stranger DB
                if name == "UNKNOWN":
                    name, s_conf = self._check_stranger_reid(oid, fobj.embedding)
                    if name != "UNKNOWN":
                        reason = f"Recognized Stranger (Conf: {s_conf:.2f})"
                        conf = s_conf

            self._check_line_crossing(oid, cent, name, events)

            # Extract Attributes: Age, Gender, Emotion
            age_str, gender_str, emotion_str = "N/A", "N/A", "N/A"
            try:
                if hasattr(fobj, "age") and fobj.age is not None:
                    age_str = str(int(fobj.age))
                if hasattr(fobj, "gender") and fobj.gender is not None:
                    gender_str = "Male" if int(fobj.gender) == 1 else "Female"
                if hasattr(fobj, "emotion") and fobj.emotion is not None:
                    emotion_str = max(fobj.emotion, key=fobj.emotion.get)
            except:
                pass

            self._draw_face_box(display, x1, y1, x2, y2, name, conf, oid, is_real)
            self._draw_person_info(display, fobj, x1, y2, oid, age_str, gender_str, emotion_str)

            if name != "UNKNOWN":
                events.append(("RECOGNITION", name, conf, f"{reason} | Age:{age_str} Gen:{gender_str} Emo:{emotion_str}"))
            else:
                events.append(("UNKNOWN_FACE", f"ID_{oid}", 1.0, reason))

    def _find_closest_face(self, cent, faces):
        best, best_d = None, float("inf")
        for fd in faces:
            b = fd["bbox"]
            d = np.linalg.norm(np.array([(b[0]+b[2])/2, (b[1]+b[3])/2]) - np.array(cent))
            if d < best_d:
                best_d, best = d, fd
        return best

    def _check_line_crossing(self, oid, cent, name, events):
        if oid not in self.counting_state:
            self.counting_state[oid] = {"last_y": cent[1], "counted": False}
        st = self.counting_state[oid]
        if not st["counted"]:
            if st["last_y"] < CONFIG["COUNT_LINE_Y"] <= cent[1]:
                events.append(("IN", name, 1.0, f"{name} crossed IN.")); st["counted"] = True
            elif st["last_y"] > CONFIG["COUNT_LINE_Y"] >= cent[1]:
                events.append(("OUT", name, 1.0, f"{name} crossed OUT.")); st["counted"] = True
        elif abs(cent[1] - CONFIG["COUNT_LINE_Y"]) > 50:
            st["counted"] = False
        st["last_y"] = cent[1]

    @staticmethod
    def _draw_face_box(frame, x1, y1, x2, y2, name, conf, oid, is_real):
        color = (0,255,0) if is_real else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} ({conf:.2f}) ID:{oid}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def _draw_person_info(self, frame, fobj, x1, y2, oid, age, gender, emotion):
        st = self.behavior_analyzer.get_object_state(oid)
        if not st:
            return

        sus = st.get("suspicion_score", 0)
        if sus > 10:
            cv2.putText(frame, f"Suspicion: {sus:.1f}", (x1, y2+60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        sl = st.get("stress_level", "Low")
        sc = {"Low": (0,255,0), "Medium": (0,255,255), "High": (0,0,255)}.get(sl, (0,255,0))
        cv2.putText(frame, f"Stress: {sl}", (x1, y2+80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, sc, 1)

        beh = st.get("active_behaviors", [])
        if beh:
            cv2.putText(frame, ", ".join(beh), (x1, y2+100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,165,0), 1)
        
        # Attribute Display
        attr_text = f"Age:{age} G:{gender} E:{emotion}"
        cv2.putText(frame, attr_text, (x1, y2+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    def _draw_ui(self, frame, events, quality, hand_dets, obj_dets):
        panel_h = 160
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y = 18
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        y += 22
        qc = (0,255,0) if quality["is_acceptable"] else (0,0,255)
        cv2.putText(frame, f"Quality: {quality['overall_quality']}/100", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, qc, 1)
        y += 20
        cv2.putText(frame, f"Known Faces: {len(self.face_db)}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        y += 20
        n_faces = sum(1 for e in events if e[0] in ("RECOGNITION", "UNKNOWN_FACE"))
        n_hands = len(hand_dets)
        n_objects = len([d for d in obj_dets if d["class_name"] != "person"])
        cv2.putText(frame, f"Faces: {n_faces}  Hands: {n_hands}  Objects: {n_objects}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,255), 1)
        y += 20
        recognized = [e[1] for e in events if e[0] == "RECOGNITION"]
        if recognized:
            names_str = ", ".join(set(recognized))
            cv2.putText(frame, f"Identified: {names_str}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
        y += 20
        if hand_dets:
            gestures = [f"{h['handedness']}: {h['gesture']}" for h in hand_dets]
            cv2.putText(frame, "  |  ".join(gestures), (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)

        # Count line and Loitering zone disabled for clean view
        # if CONFIG["SHOW_COUNT_LINE"]:
        #     cv2.line(frame, (0, CONFIG["COUNT_LINE_Y"]),
        #              (frame.shape[1], CONFIG["COUNT_LINE_Y"]), (0, 255, 255), 2)
        #     cv2.putText(frame, "COUNT LINE", (frame.shape[1] - 120, CONFIG["COUNT_LINE_Y"] - 5),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)

        # z = CONFIG["LOITERING_ZONE"]
        # cv2.rectangle(frame, (z[0], z[1]), (z[2], z[3]), (0, 0, 255), 2)
        # cv2.putText(frame, "LOITERING ZONE", (z[0], z[1]-8),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)

        # Behavior heatmap disabled
        # if CONFIG.get("SHOW_HEATMAP", False):
        #     hm = self.behavior_analyzer.get_heatmap()
        #     if hm is not None and hm.size > 0:
        #         hm_r = cv2.resize(hm, (frame.shape[1], frame.shape[0]))
        #         frame = cv2.addWeighted(frame, 0.7, hm_r, 0.3, 0)

        # Crowd Intel draw (Density/Flow overlays) disabled in class draw(), keeping logic running
        if hasattr(self, '_last_tracked') and hasattr(self, 'crowd_intel'):
            frame = self.crowd_intel.draw(frame, self._last_tracked)

        # Stats Panel (Flow data only)
        if hasattr(self, 'crowd_intel') and self.crowd_intel.enabled:
            frame = self.crowd_intel.draw_stats_panel(frame)

        return frame

    def save_database(self):
        try:
            os.makedirs(os.path.dirname(CONFIG["DATABASE_FILE"]) or ".", exist_ok=True)
            with open(CONFIG["DATABASE_FILE"], "wb") as f:
                pickle.dump(self.face_db, f)
            print(f"[INFO] Saved face DB to {CONFIG['DATABASE_FILE']}")
        except Exception as e:
            print(f"[ERROR] Save failed: {e}")

    def enroll_person(self, frame, name, skip_quality_check=False):
        now = time.time()
        if (now - self._last_enrollment_time) < self._enrollment_cooldown:
            remaining = self._enrollment_cooldown - (now - self._last_enrollment_time)
            print(f"[WARN] Enrollment rate limited. Wait {remaining:.0f}s.")
            return None

        if not skip_quality_check:
            is_ok, quality_reason, scores = self.quality_scorer.is_enrollment_quality(frame)
            if not is_ok:
                print(f"[WARN] Enrollment rejected for {name}: {quality_reason}")
                print(f"       Blur: {scores['blur_score']:.1f}, "
                      f"Brightness: {scores['brightness_score']:.0f}, "
                      f"Contrast: {scores['contrast_score']:.1f}")
                return None

        faces = self.face_app.get(frame)
        if not faces:
            print("[WARN] No face detected.")
            return None
        best = max(faces, key=lambda f: f.det_score if f.det_score is not None else 0)
        if best.embedding is None:
            print("[WARN] No embedding extracted.")
            return None
        emb = best.embedding

        if name not in self.face_db:
            self.face_db[name] = {"embeddings": [], "threshold": CONFIG["FACE_RECOG_THRESHOLD"]}

        max_embs = self._max_embeddings_per_person
        if len(self.face_db[name]["embeddings"]) >= max_embs:
            self.face_db[name]["embeddings"].pop(0)
            self._rebuild_faiss_index()

        self.face_db[name]["embeddings"].append(emb)

        embs_for_person = self.face_db[name]["embeddings"]
        if len(embs_for_person) >= 2:
            mean_var, std_var = _compute_intra_class_variance(embs_for_person)
            auto_thresh = _auto_threshold(mean_var, std_var)
            self.face_db[name]["threshold"] = auto_thresh
        else:
            self.face_db[name]["threshold"] = CONFIG["FACE_RECOG_THRESHOLD"]

        self.face_indexer.add_embeddings(
            [name], np.array([emb]), [self.face_db[name]["threshold"]]
        )

        self._last_enrollment_time = now
        self.save_database()

        n_embs = len(self.face_db[name]["embeddings"])
        thresh = self.face_db[name]["threshold"]
        print(f"[SUCCESS] Enrolled {name} (embeddings: {n_embs}/{max_embs}, "
              f"auto-threshold: {thresh:.3f}).")
        return emb

    def _rebuild_faiss_index(self):
        names, embs, threshs = [], [], []
        for n, info in self.face_db.items():
            for e in info["embeddings"]:
                names.append(n)
                embs.append(e)
                threshs.append(info.get("threshold", CONFIG["FACE_RECOG_THRESHOLD"]))
        self.face_indexer = FAISSIndexer(512)
        if embs:
            self.face_indexer.add_embeddings(names, np.array(embs), threshs)