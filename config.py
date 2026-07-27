CONFIG = {
    "DEVICE": "auto",                   
    "FRAME_SKIP": 2,                    
    "ASYNC_PIPELINE": True,             
    "BATCH_SIZE": 4,                  

    "YOLO_MODEL_PATH": "yolov8n.pt",
    "DATABASE_FILE": "data/face_database.pkl",
    "INSIGHTFACE_MODEL": "buffalo_l",

    "NMS_CONF_THRESHOLD": 0.5,
    "NMS_IOU_THRESHOLD": 0.45,
    "YOLO_CONF": 0.5,
    "FACE_DET_SIZE": (640, 640),

    "FACE_RECOG_THRESHOLD": 0.35,      
    "RECOGNITION_HISTORY_LENGTH": 15, 
    "PROFILE_CONFIDENCE_THRESHOLD": 7,  
    "MIN_FACE_QUALITY_SCORE": 0.6,    
    "MAX_EMBEDDINGS_PER_PERSON": 10,    
    "REID_BUFFER_SECONDS": 30,         

    "BLUR_THRESHOLD": 80.0,
    "BRIGHTNESS_MIN": 40,
    "BRIGHTNESS_MAX": 210,
    "CONTRAST_THRESHOLD": 40,

    "COUNT_LINE_Y": 300,
    "COUNT_LINE_RESET_PX": 60,         
    "SHOW_COUNT_LINE": True,

    "ZONES": {
        "LOITERING_ZONE": (400, 100, 800, 400),
        "RESTRICTED_ZONE": (0, 0, 150, 150),
        "ENTRY_ZONE": (200, 250, 500, 350),
    },
    "SHOW_ZONES": True,

    "BEHAVIOR_THRESHOLDS": {
        "LOITERING_TIME_SEC": 10,
        "RUNNING_SPEED_PX_PER_SEC": 150,    
        "HESITATION_SPEED_THRESHOLD": 5.0,  
        "HESITATION_STOP_TIME_SEC": 3.0,
        "PACING_WINDOW_SEC": 12.0,
        "PACING_DIRECTION_CHANGES": 3,
        "SCANNING_VAR_THRESHOLD": 120,
        "SCANNING_DISP_THRESHOLD": 35,
        "OBJECT_INTERACTION_CLASSES": [
            "knife", "scissors", "cell phone",
            "laptop", "backpack", "handbag"
        ],
        "INTERACTION_DISTANCE_PX": 120,
        "INTERACTION_IOU_THRESHOLD": 0.08,
        "BAG_ABANDONMENT_SEC": 20,        
        "PICKUP_DROP_VELOCITY_THRESHOLD": 30,
    },
    "SUSPICION_POINTS": {
        "LOITERING": 1.5,
        "RUNNING": 2.5,
        "HESITATION": 1.0,
        "PACING": 2.0,
        "SCANNING": 0.8,
        "OBJECT_INTERACTION": 2.0,
        "SPATIAL_ANOMALY": 3.0,
        "RESTRICTED_ZONE": 5.0,
        "BAG_ABANDONMENT": 4.0,
    },
    "SUSPICION_DECAY_RATE": 0.97,
    "SUSPICION_DECAY_INTERVAL": 1.0,

    "CROWD_MIN_SIZE": 4,
    "CROWD_RADIUS_PX": 120,
    "SPATIAL_GRID_SIZE": (20, 20),
    "SPATIAL_ANOMALY_THRESHOLD": 0.05,
    "HEATMAP_UPDATE_INTERVAL": 3.0,     
    "SHOW_HEATMAP": False,

    "ANTI_SPOOFING": {
        "ENABLED": True,
        "BLINK_EAR_THRESHOLD": 0.22,
        "EAR_HISTORY_FRAMES": 20,
        "MIN_BLINKS_FOR_LIVENESS": 1,
        "MIN_TIME_BETWEEN_BLINKS_SEC": 0.3,
        "LIVENESS_TIME_WINDOW_SEC": 8.0,
        "POSE_HISTORY_FRAMES": 15,
        "STATIC_POSE_VARIANCE_THRESHOLD": 4.0,
        "DEPTH_VARIANCE_THRESHOLD": 0.002,
        "ENABLE_ML_CLASSIFIER": False,   
        "ML_CLASSIFIER_THRESHOLD": 0.65,
        "TEXTURE_LBP_THRESHOLD": 0.3,   
    },

    "STRESS_THRESHOLDS": {
        "LOW": 15,
        "MEDIUM": 40,
        "HIGH": 70,
    },

    "FONT": "FONT_HERSHEY_SIMPLEX",
    "FONT_SCALE": 0.55,
    "LINE_THICKNESS": 2,

    "LOG_IMPORTANT_EVENTS": {
        "SPOOF_DETECTED", "LOITERING", "RUNNING", "IN", "OUT",
        "RESTRICTED_ZONE", "BAG_ABANDONMENT", "CROWD_FORMING",
    },
    "SNAPSHOT_ON_EVENT": True,
    "SNAPSHOT_DIR": "snapshots",
}
