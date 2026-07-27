
import time
import numpy as np
from collections import defaultdict, deque
import cv2

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not found. Crowd dynamics analysis will be disabled.")
    print("To enable, please run: pip install scikit-learn")


class SuspicionScorer:
    """Manages and decays suspicion scores for each tracked object."""
    def __init__(self, decay_rate=0.98, decay_interval_sec=1.0):
        self.scores = defaultdict(float)
        self.decay_rate = decay_rate
        self.last_decay_time = defaultdict(float)
        self.decay_interval = decay_interval_sec

    def add_event(self, object_id, points):
        """Adds points to an object's suspicion score."""
        self.scores[object_id] += points
        self.scores[object_id] = min(100.0, self.scores[object_id]) # Cap at 100

    def decay_scores(self, current_time):
        """Applies time-based decay to all active scores."""
        for object_id in list(self.scores.keys()):
            if current_time - self.last_decay_time[object_id] > self.decay_interval:
                self.scores[object_id] *= self.decay_rate
                if self.scores[object_id] < 1.0: # Clean up very low scores
                    del self.scores[object_id]
                    del self.last_decay_time[object_id]
                else:
                    self.last_decay_time[object_id] = current_time

    def get_score(self, object_id):
        """Returns the current suspicion score for an object."""
        return self.scores.get(object_id, 0.0)


class BehaviorAnalyzer:
    """Analyzes tracked objects for various suspicious and anomalous behaviors."""
    def __init__(self, config):
        self.config = config
        self.current_time = time.time()
        
        self.suspicion_scorer = SuspicionScorer(
            decay_rate=config.get('SUSPICION_DECAY_RATE', 0.98),
            decay_interval_sec=config.get('SUSPICION_DECAY_INTERVAL', 1.0)
        )

        # This will store the full history for each tracked object
        self.object_history = defaultdict(lambda: {
            'centroids': deque(maxlen=60), # Store last 60 centroids
            'velocities': deque(maxlen=60),
            'bboxes': deque(maxlen=60),
            'timestamps': deque(maxlen=60),
            'behaviors': set(),
            'stress_level': 'Low'
        })

        # --- Spatial Anomaly Detection (Heatmap) ---
        self.heatmap = np.zeros(config.get('SPATIAL_GRID_SIZE', (20, 20)), dtype=np.float32)
        self.heatmap_update_interval = config.get('HEATMAP_UPDATE_INTERVAL', 5.0) # seconds
        self.last_heatmap_update = 0

    def _calculate_iou(self, boxA, boxB):
        """Calculates Intersection over Union for two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0
        
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        return interArea / float(boxAArea + boxBArea - interArea)

    # --- Individual Behavior Analysis Methods ---
    def _analyze_hesitation(self, history):
        """Detects if an object has stopped moving for a significant time."""
        if len(history['velocities']) < 10:
            return None
        
        recent_velocities = np.array(list(history['velocities'])[-10:])
        avg_speed = np.mean(np.linalg.norm(recent_velocities, axis=1))
        
        if avg_speed < self.config.get('HESITATION_SPEED_THRESHOLD', 5.0):
            # Find when the stop started
            stop_started_time = None
            for i in range(len(history['velocities']) - 1, 0, -1):
                v = np.linalg.norm(history['velocities'][i])
                if v > self.config.get('HESITATION_SPEED_THRESHOLD', 5.0):
                    stop_started_time = history['timestamps'][i+1]
                    break
            
            if stop_started_time and (self.current_time - stop_started_time) > self.config.get('HESITATION_STOP_TIME_SEC', 3.0):
                return "Object stopped moving for a significant duration."
        return None

    def _analyze_pacing(self, history):
        """Detects repetitive back-and-forth movement."""
        if len(history['centroids']) < 10:
            return None
            
        window_start_time = self.current_time - self.config.get('PACING_WINDOW_SEC', 10.0)
        window_indices = [i for i, ts in enumerate(history['timestamps']) if ts > window_start_time]
        
        if len(window_indices) < 5:
            return None

        window_centroids = np.array([history['centroids'][i] for i in window_indices])
        x_range = np.max(window_centroids[:, 0]) - np.min(window_centroids[:, 0])
        y_range = np.max(window_centroids[:, 1]) - np.min(window_centroids[:, 1])

        # Determine primary axis of movement
        axis = 0 if x_range > y_range * 1.5 else (1 if y_range > x_range * 1.5 else None)
        
        if axis is not None:
            positions = window_centroids[:, axis]
            direction_changes = 0
            current_direction = np.sign(positions[1] - positions[0])
            
            for i in range(2, len(positions)):
                new_direction = np.sign(positions[i] - positions[i-1])
                if new_direction != 0 and new_direction != current_direction:
                    direction_changes += 1
                    current_direction = new_direction
            
            if direction_changes >= self.config.get('PACING_DIRECTION_CHANGES', 3):
                return f"Detected repetitive back-and-forth movement ({direction_changes} changes)."
        return None

    def _analyze_scanning(self, history):
        """Proxy for scanning: high velocity variance with low overall displacement."""
        if len(history['velocities']) < 15:
            return None
            
        recent_velocities = np.array(list(history['velocities'])[-15:])
        velocity_variance = np.var(np.linalg.norm(recent_velocities, axis=1))
        
        recent_centroids = np.array(list(history['centroids'])[-15:])
        displacement = np.linalg.norm(recent_centroids[-1] - recent_centroids[0])

        if velocity_variance > self.config.get('SCANNING_VAR_THRESHOLD', 100) and displacement < self.config.get('SCANNING_DISP_THRESHOLD', 30):
            return "High movement variance with low displacement (possible scanning)."
        return None

    # --- Crowd, Spatial, and Interaction Analysis Methods ---
    def _analyze_crowd_dynamics(self, current_centroids):
        """Detects crowd formation using clustering."""
        if not SKLEARN_AVAILABLE or len(current_centroids) < self.config.get('CROWD_MIN_SIZE', 4):
            return []
        
        points = np.array(list(current_centroids.values()))
        clustering = DBSCAN(eps=self.config.get('CROWD_RADIUS', 100), min_samples=self.config.get('CROWD_MIN_SIZE', 4)).fit(points)
        labels = clustering.labels_
        
        events = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label != -1: # -1 is noise
                cluster_points = points[labels == label]
                if len(cluster_points) >= self.config.get('CROWD_MIN_SIZE', 4):
                    events.append(("CROWD_FORMING", "SYSTEM", 1.0, f"Detected a crowd of {len(cluster_points)} people."))
        return events

    def _analyze_spatial_anomaly(self, centroid):
        """Checks if a centroid is in a historically low-traffic area."""
        frame_h, frame_w = self.config.get('FRAME_SIZE', (720, 1280))
        grid_h, grid_w = self.heatmap.shape
        
        # Map centroid to grid coordinates
        grid_x = int((centroid[0] / frame_w) * grid_w)
        grid_y = int((centroid[1] / frame_h) * grid_h)
        
        grid_x = np.clip(grid_x, 0, grid_w - 1)
        grid_y = np.clip(grid_y, 0, grid_h - 1)
        
        # Anomaly if the heatmap value for this cell is very low
        # Normalize heatmap to 0-1 for comparison
        if np.max(self.heatmap) > 0:
            normalized_heatmap = self.heatmap / np.max(self.heatmap)
            if normalized_heatmap[grid_y, grid_x] < self.config.get('SPATIAL_ANOMALY_THRESHOLD', 0.05):
                return True
        return False

    def _analyze_object_interaction(self, person_bboxes, object_detections):
        """Checks for proximity between people and specific objects."""
        events = []
        interaction_objects = ['cell phone', 'laptop', 'backpack', 'handbag']
        
        for obj_class, obj_bbox, _ in object_detections:
            if obj_class in interaction_objects:
                for person_id, person_bbox in person_bboxes.items():
                    if self._calculate_iou(person_bbox, obj_bbox) > self.config.get('INTERACTION_IOU_THRESHOLD', 0.1):
                        events.append(("OBJECT_INTERACTION", f"ID_{person_id}", 1.0, f"ID_{person_id} interacting with a {obj_class}."))
                        break
        return events

    def _update_psychological_profile(self, object_id, history):
        """Estimates stress level based on detected behaviors."""
        if 'PACING' in history['behaviors'] or 'HESITATION' in history['behaviors']:
            history['stress_level'] = 'High'
        elif 'SCANNING' in history['behaviors']:
            history['stress_level'] = 'Medium'
        else:
            history['stress_level'] = 'Low'

    # --- Main Update Method ---
    def update(self, tracked_objects, yolo_detections, frame_shape):
        """
        Main update function called from VisionSystem.
        tracked_objects: {objectID: centroid}
        yolo_detections: list of (class_name, bbox, confidence)
        frame_shape: (height, width)
        """
        self.current_time = time.time()
        self.config['FRAME_SIZE'] = (frame_shape[0], frame_shape[1]) # Update for spatial analysis
        events = []

        # --- 1. Update Object History ---
        for object_id, centroid in tracked_objects.items():
            history = self.object_history[object_id]
            history['centroids'].append(centroid)
            history['timestamps'].append(self.current_time)
            
            if len(history['centroids']) > 1:
                prev_centroid = history['centroids'][-2]
                dt = self.current_time - history['timestamps'][-2]
                if dt > 0:
                    velocity = np.array(centroid) - np.array(prev_centroid)
                    history['velocities'].append(velocity)
                else:
                    history['velocities'].append(np.array([0,0]))
            else:
                history['velocities'].append(np.array([0,0]))

        # --- 2. Run Analyzers ---
        # A. Individual Behaviors
        for object_id, history in self.object_history.items():
            if object_id in tracked_objects:
                # Clear old behaviors to avoid re-reporting
                history['behaviors'].clear()

                hesitation = self._analyze_hesitation(history)
                if hesitation:
                    events.append(("HESITATION", f"ID_{object_id}", 1.0, hesitation))
                    history['behaviors'].add('HESITATION')
                    self.suspicion_scorer.add_event(object_id, self.config['SUSPICION_POINTS'].get('HESITATION', 10))

                pacing = self._analyze_pacing(history)
                if pacing:
                    events.append(("PACING", f"ID_{object_id}", 1.0, pacing))
                    history['behaviors'].add('PACING')
                    self.suspicion_scorer.add_event(object_id, self.config['SUSPICION_POINTS'].get('PACING', 20))
                
                scanning = self._analyze_scanning(history)
                if scanning:
                    events.append(("SCANNING", f"ID_{object_id}", 1.0, scanning))
                    history['behaviors'].add('SCANNING')
                    self.suspicion_scorer.add_event(object_id, self.config['SUSPICION_POINTS'].get('SCANNING', 5))

                self._update_psychological_profile(object_id, history)

        # B. Spatial Anomaly Detection
        for object_id, centroid in tracked_objects.items():
            if self._analyze_spatial_anomaly(centroid):
                events.append(("SPATIAL_ANOMALY", f"ID_{object_id}", 1.0, f"ID_{object_id} in an unusual location."))
                self.suspicion_scorer.add_event(object_id, self.config['SUSPICION_POINTS'].get('SPATIAL_ANOMALY', 25))

        # C. Crowd Dynamics
        current_centroids = {oid: hist['centroids'][-1] for oid, hist in self.object_history.items() if oid in tracked_objects}
        crowd_events = self._analyze_crowd_dynamics(current_centroids)
        events.extend(crowd_events)

        # D. Object Interaction
        person_bboxes = {oid: hist['bboxes'][-1] for oid, hist in self.object_history.items() if oid in tracked_objects and hist['bboxes']}
        interaction_events = self._analyze_object_interaction(person_bboxes, yolo_detections)
        events.extend(interaction_events)

        # --- 3. Update Spatial Heatmap (Baseline Learning) ---
        if self.current_time - self.last_heatmap_update > self.heatmap_update_interval:
            frame_h, frame_w = frame_shape
            grid_h, grid_w = self.heatmap.shape
            
            # Decay existing heatmap
            self.heatmap *= 0.9
            
            # Add current centroids
            for centroid in current_centroids.values():
                grid_x = int((centroid[0] / frame_w) * grid_w)
                grid_y = int((centroid[1] / frame_h) * grid_h)
                grid_x = np.clip(grid_x, 0, grid_w - 1)
                grid_y = np.clip(grid_y, 0, grid_h - 1)
                self.heatmap[grid_y, grid_x] += 1
            
            self.last_heatmap_update = self.current_time

        # --- 4. Decay Suspicion Scores ---
        self.suspicion_scorer.decay_scores(self.current_time)

        return events

    def get_object_state(self, object_id):
        """Returns the full state of an object for drawing in the main loop."""
        history = self.object_history.get(object_id)
        if not history:
            return {}
        
        return {
            "suspicion_score": self.suspicion_scorer.get_score(object_id),
            "stress_level": history['stress_level'],
            "active_behaviors": list(history['behaviors'])
        }

    def cleanup(self, active_object_ids):
        """Remove history for objects that are no longer tracked."""
        all_ids = set(self.object_history.keys())
        ids_to_remove = all_ids - set(active_object_ids)
        for object_id in ids_to_remove:
            del self.object_history[object_id]
            if object_id in self.suspicion_scorer.scores:
                del self.suspicion_scorer.scores[object_id]

    def get_heatmap(self):
        """Returns the current spatial heatmap for visualization."""
        if np.max(self.heatmap) > 0:
            heatmap_vis = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            return cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
        return np.zeros((self.heatmap.shape[0], self.heatmap.shape[1], 3), dtype=np.uint8)
