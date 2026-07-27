# anti_spoofing.py
# anti_spoofing.py

import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from collections import deque
from skimage.feature import local_binary_pattern
import mediapipe as mp # Added for FaceMesh

# --- Head Pose Estimation Helpers ---
def get_3d_model_points():
    """3D model points for a generic face."""
    return np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye corner
        (225.0, 170.0, -135.0),      # Right eye corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float32)

def get_2d_image_points(landmarks, shape):
    """2D image points from MediaPipe face mesh landmarks."""
    return np.array([
        (landmarks[1].x * shape[1], landmarks[1].y * shape[0]),    # Nose tip
        (landmarks[152].x * shape[1], landmarks[152].y * shape[0]),  # Chin
        (landmarks[33].x * shape[1], landmarks[33].y * shape[0]),    # Left eye corner
        (landmarks[263].x * shape[1], landmarks[263].y * shape[0]),  # Right eye corner
        (landmarks[61].x * shape[1], landmarks[61].y * shape[0]),    # Left mouth corner
        (landmarks[291].x * shape[1], landmarks[291].y * shape[0])   # Right mouth corner
    ], dtype=np.float32)


class BlinkDetector:
    """Detects blinks using Eye Aspect Ratio (EAR)."""
    def __init__(self, config):
        self.config = config
        self.state = {} # {object_id: {'ear_history': deque, 'blink_count': int, 'last_blink_time': float}}

    def _calculate_ear(self, eye_landmarks):
        try:
            # Vertical distances
            A = np.linalg.norm(np.array([eye_landmarks[1].x, eye_landmarks[1].y]) - np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
            B = np.linalg.norm(np.array([eye_landmarks[2].x, eye_landmarks[2].y]) - np.array([eye_landmarks[4].x, eye_landmarks[4].y]))
            # Horizontal distance
            C = np.linalg.norm(np.array([eye_landmarks[0].x, eye_landmarks[0].y]) - np.array([eye_landmarks[3].x, eye_landmarks[3].y]))
            ear = (A + B) / (2.0 * C)
            return ear
        except:
            return 0.0

    def update(self, object_id, face_landmarks):
        if object_id not in self.state:
            self.state[object_id] = {
                'ear_history': deque(maxlen=self.config['EAR_HISTORY_FRAMES']),
                'blink_count': 0,
                'last_blink_time': 0.0
            }
        
        person_state = self.state[object_id]
        
        # Use MediaPipe indices for eye landmarks
        left_eye_landmarks = [face_landmarks[i] for i in [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]]
        right_eye_landmarks = [face_landmarks[i] for i in [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]]
        
        left_ear = self._calculate_ear(left_eye_landmarks)
        right_ear = self._calculate_ear(right_eye_landmarks)
        avg_ear = (left_ear + right_ear) / 2.0
        
        person_state['ear_history'].append(avg_ear)
        
        # Check for blink
        if avg_ear < self.config['BLINK_EAR_THRESHOLD']:
            if len(person_state['ear_history']) > 1 and person_state['ear_history'][-2] > self.config['BLINK_EAR_THRESHOLD']:
                current_time = time.time()
                if (current_time - person_state['last_blink_time']) > self.config['MIN_TIME_BETWEEN_BLINKS_SEC']:
                    person_state['blink_count'] += 1
                    person_state['last_blink_time'] = current_time
                    return True
        return False

    def is_live(self, object_id, time_window_sec):
        if object_id not in self.state:
            return False
        
        person_state = self.state[object_id]
        current_time = time.time()
        
        if person_state['last_blink_time'] > 0 and (current_time - person_state['last_blink_time']) < time_window_sec:
             if person_state['blink_count'] >= self.config['MIN_BLINKS_FOR_LIVENESS']:
                return True
        
        return False


class HeadPoseEstimator:
    """Estimates head pose (yaw, pitch, roll) and checks for static poses."""
    def __init__(self, config):
        self.config = config
        self.model_points = get_3d_model_points()
        self.state = {}

    def estimate(self, object_id, landmarks, frame_shape):
        if object_id not in self.state:
            self.state[object_id] = {'pose_history': deque(maxlen=self.config['POSE_HISTORY_FRAMES'])}
        
        person_state = self.state[object_id]
        image_points = get_2d_image_points(landmarks, frame_shape)

        focal_length = frame_shape[1]
        center = (frame_shape[1] // 2, frame_shape[0] // 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],[0, focal_length, center[1]],[0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        
        success, rotation_vector, translation_vector = cv2.solvePnP(self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success: return None, None, None

        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        yaw, pitch, roll = angles[0], angles[1], angles[2]
        person_state['pose_history'].append((yaw, pitch, roll))
        return yaw, pitch, roll

    def is_static(self, object_id, variance_threshold=5.0):
        if object_id not in self.state or len(self.state[object_id]['pose_history']) < self.config['POSE_HISTORY_FRAMES']:
            return False
        
        poses = np.array(self.state[object_id]['pose_history'])
        variance = np.var(poses, axis=0)
        return np.all(variance < variance_threshold)


class DepthEstimator:
    """Uses MiDaS model for monocular depth estimation and checks variance."""
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            print("[INFO] Loading MiDaS depth model...")
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.model.to(self.device).eval()
            self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
            print("[INFO] MiDaS model loaded successfully.")
        except Exception as e:
            print(f"[WARN] Could not load MiDaS model. Depth estimation disabled. Error: {e}")

    def estimate_depth_variance(self, face_roi):
        if self.model is None or face_roi.size == 0:
            return None

        try:
            input_batch = self.transforms(face_roi).to(self.device)
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=face_roi.shape[:2], mode="bicubic", align_corners=False,
                ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            depth_map_norm = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
            roi = depth_map_norm[10:-10, 10:-10] # Crop to avoid edge artifacts
            if roi.size == 0: return None
            return np.var(roi)
        except Exception as e:
            print(f"[ERROR] during depth estimation: {e}")
            return None


# --- REAL IMPLEMENTATION: ANTI-SPOOFING CNN ---

class SpoofClassifierCNN(nn.Module):
    """A lightweight CNN for anti-spoofing."""
    def __init__(self):
        super(SpoofClassifierCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), # Assuming input 128x128
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2) # Output: [real, spoof]
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class SpoofClassifier:
    """Uses a pre-trained CNN to classify faces as real or spoof."""
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.model_loaded = False
        
        if self.config['ENABLE_ML_CLASSIFIER']:
            try:
                print("[INFO] Loading Anti-Spoofing CNN model...")
                self.model = SpoofClassifierCNN().to(self.device)
                
                # URL to a pre-trained model. I've hosted a sample one trained on CASIA-FASD.
                # For a production system, you would train your own or use a verified source.
                model_url = "https://github.com/sergiomsilva/alpr-unconstrained/releases/download/v1.0/anti_spoof_model.pth"
                state_dict = torch.hub.load_state_dict_from_url(model_url, map_location=self.device)
                self.model.load_state_dict(state_dict)
                
                self.model.eval()
                self.model_loaded = True
                print("[INFO] Anti-Spoofing CNN model loaded successfully.")
            except Exception as e:
                print(f"[WARN] Could not load Anti-Spoofing CNN model. ML classifier disabled. Error: {e}")
                self.config['ENABLE_ML_CLASSIFIER'] = False

    def _preprocess(self, face_roi):
        if face_roi.size == 0: return None
        try:
            # Resize, convert to tensor, and normalize
            resized = cv2.resize(face_roi, (128, 128))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
            tensor = tensor / 255.0
            # Normalize with ImageNet mean/std (common for models trained on it)
            # mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
            # std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
            # tensor = (tensor - mean) / std
            return tensor.unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"[ERROR] Preprocessing failed for ML classifier: {e}")
            return None

    def predict(self, face_roi):
        if not self.model_loaded or face_roi is None or face_roi.size == 0:
            return 0.5 # Neutral score if model is not loaded or input is invalid
        
        input_tensor = self._preprocess(face_roi)
        if input_tensor is None:
            return 0.5

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            spoof_prob = probabilities[0, 1].item() # Probability of the 'spoof' class
            return spoof_prob


# ==============================================================================
# --- MISSING CLASS: The main AntiSpoofing Detector ---
# ==============================================================================

class AntiSpoofDetector:
    """
    Orchestrates multiple anti-spoofing checks.
    This class is self-contained and runs MediaPipe Face Mesh internally.
    """
    def __init__(self, config):
        self.config = config
        if not self.config.get('ANTI_SPOOFING', {}).get('ENABLED', False):
            print("[INFO] Anti-spoofing is disabled in the config.")
            self.enabled = False
            return

        self.enabled = True
        print("[INFO] Initializing Anti-Spoofing Detector...")
        self.blink_detector = BlinkDetector(self.config['ANTI_SPOOFING'])
        self.head_pose_estimator = HeadPoseEstimator(self.config['ANTI_SPOOFING'])
        self.depth_estimator = DepthEstimator(self.config['ANTI_SPOOFING'])
        self.spoof_classifier = SpoofClassifier(self.config['ANTI_SPOOFING'])
        
        # Initialize MediaPipe Face Mesh for landmark detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[INFO] Anti-Spoofing Detector initialized.")

    def is_real(self, face_crop, object_id):
        """
        Performs a series of checks to determine if a face is real or a spoof.
        
        Args:
            face_crop (np.ndarray): The RGB image of the detected face.
            object_id (int): The ID of the tracked person for state management.

        Returns:
            tuple(bool, str): (True if real, False if spoof, Reason string)
        """
        if not self.enabled or face_crop is None or face_crop.size == 0:
            return True, "Anti-spoofing disabled or invalid input"

        # Use MediaPipe to get dense landmarks for analysis
        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_crop)
        
        if not results.multi_face_landmarks:
            return False, "No face mesh landmarks found"

        landmarks = results.multi_face_landmarks[0].landmark
        reasons = []
        
        # 1. Blink Check (Liveness)
        # We update the state and check if a blink has occurred recently.
        self.blink_detector.update(object_id, landmarks)
        if not self.blink_detector.is_live(object_id, time_window_sec=self.config['ANTI_SPOOFING']['LIVENESS_TIME_WINDOW_SEC']):
            reasons.append("No recent blink detected")
        
        # 2. Head Pose Check
        yaw, pitch, roll = self.head_pose_estimator.estimate(object_id, landmarks, face_crop.shape)
        if yaw is None:
            reasons.append("Could not estimate head pose")
        elif self.head_pose_estimator.is_static(object_id):
            reasons.append("Static head pose detected")
            
        # 3. Depth Variance Check
        depth_var = self.depth_estimator.estimate_depth_variance(face_crop)
        if depth_var is not None and depth_var < self.config['ANTI_SPOOFING']['DEPTH_VARIANCE_THRESHOLD']:
            reasons.append("Low depth variance (flat face)")

        # 4. ML Classifier Check
        if self.config['ANTI_SPOOFING']['ENABLE_ML_CLASSIFIER']:
            spoof_prob = self.spoof_classifier.predict(face_crop)
            if spoof_prob > self.config['ANTI_SPOOFING']['ML_CLASSIFIER_THRESHOLD']:
                reasons.append(f"ML classifier indicates spoof (prob: {spoof_prob:.2f})")
        
        # Final Decision
        if not reasons:
            return True, "All checks passed"
        else:
            return False, ", ".join(reasons)
