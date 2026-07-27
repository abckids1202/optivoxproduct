import cv2
import os
import time
import sys
import shutil
import numpy as np
from vision import VisionSystem
from database import EventDatabase
from notifier import AlertManager

LOCAL_APP_DIR = os.path.join(os.path.expanduser("~"), ".detection_system")
os.makedirs(LOCAL_APP_DIR, exist_ok=True)

def setup_insightface_local_models():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_models_path = os.path.join(script_dir, "models", "buffalo_l")
    home_dir = os.path.expanduser("~")
    cache_path = os.path.join(home_dir, ".insightface", "models", "buffalo_l")

    print(f"[INFO] Looking for local models in: {local_models_path}")

    if not os.path.isdir(local_models_path):
        print(f"[ERROR] Local models directory not found at '{local_models_path}'")
        print("Please ensure the 'models/buffalo_l' folder exists next to this script.")
        sys.exit(1)

    if os.path.exists(cache_path) and os.path.exists(os.path.join(cache_path, "buffalo_l.yaml")):
        print(f"[INFO] InsightFace cache already configured at '{cache_path}'.")
        return

    print("[INFO] Configuring InsightFace to use local models...")
    try:
        os.makedirs(cache_path, exist_ok=True)
        files = [
            "1k3d68.onnx", "2d106det.onnx", "det_10g.onnx",
            "genderage.onnx", "w600k_r50.onnx", "buffalo_l.yaml",
        ]
        for fname in files:
            src = os.path.join(local_models_path, fname)
            dst = os.path.join(cache_path, fname)
            if not os.path.exists(src):
                print(f"[ERROR] Model file not found: {src}")
                sys.exit(1)
            if os.path.exists(dst) or os.path.islink(dst):
                os.remove(dst)
            try:
                os.symlink(src, dst)
            except (OSError, NotImplementedError):
                print(f"[WARN] Symlink failed for {fname}. Copying (this may take a moment)...")
                shutil.copy2(src, dst)
        print("[OK] Local models linked/copied successfully.")
    except Exception as e:
        print(f"[ERROR] Model setup failed: {e}")
        sys.exit(1)

def main():
    print("=" * 60)
    print("  INTELLIGENT SECURITY MONITORING SYSTEM")
    print("=" * 60)
    print("[DEBUG] Starting Detection System...")

    setup_insightface_local_models()

    yolo_path = "yolov8n.pt"
    if not os.path.exists(yolo_path):
        print(f"\n[CRITICAL] YOLO model file '{yolo_path}' not found!")
        print("Please download it or place it in the script directory.")
        print("Download: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
        sys.exit(1)

    logs_dir = os.path.join(LOCAL_APP_DIR, "logs")
    snaps_dir = os.path.join(LOCAL_APP_DIR, "snapshots")
    data_dir = os.path.join(LOCAL_APP_DIR, "data")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(snaps_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    print("[DEBUG] Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera (index 0).")
        print("  - Is a webcam connected?")
        print("  - Is another app (Zoom/Teams) using it?")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[FATAL] No camera found. Exiting.")
            return
    print("[DEBUG] Camera opened.")

    db_path = os.path.join(LOCAL_APP_DIR, "security.db")
    db = EventDatabase(db_path=db_path)
    db.setup_database()
    print(f"[INFO] Database initialized at: {db_path}")

    alert_config_path = os.path.join(LOCAL_APP_DIR, "alert_config.json")
    alert_manager = AlertManager(config_path=alert_config_path)

    if not os.path.exists(alert_config_path):
        AlertManager.create_sample_config(alert_config_path)
        print(f"\n[INFO] Sample alert config created at: {alert_config_path}")
        print("  Edit this file to enable email/Telegram/Discord/SMS alerts.")
        print("  Then restart the program.\n")

    print("[DEBUG] Initializing VisionSystem (loading AI models)...")
    print("        (This may take 10-30 seconds depending on your CPU/GPU)...")
    try:
        vision = VisionSystem()
    except Exception as e:
        print(f"[CRITICAL] VisionSystem Init failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        cap.release()
        return

    db.log_audit("SYSTEM_START", details={"version": "2.0", "features": "all"})

    print("\n" + "=" * 60)
    print("  SYSTEM READY")
    print("=" * 60)
    print("  'q' = quit")
    print("  'e' = enroll face")
    print("  's' = save snapshot")
    print("  'r' = apply retention policies")
    print("  'd' = generate daily summary")
    print("  't' = send test alert")
    print("  'a' = view audit log")
    print("=" * 60 + "\n")

    ALERT_EVENTS = {
        "SPOOF_DETECTED", "LOITERING", "RUNNING", "IN", "OUT",
        "CROWD_FORMING", "HESITATION", "PACING", "SCANNING", "SPATIAL_ANOMALY",
        "OBJECT_INTERACTION", "EVACUATION_ALERT", "DANGEROUS_OBJECT",
    }

    last_daily_run = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Frame grab failed. Reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(1)
                if not cap.isOpened():
                    print("[FATAL] No camera. Exiting.")
                    break
                continue

            result = vision.process(frame)
            if len(result) != 7:
                print(f"[WARN] Unexpected process() return: {len(result)} values, expected 7")
                continue
            display, faces, hand_dets, obj_dets, events, tracked_count, dt = result

            cv2.imshow("Security Feed", display)

            now = time.time()
            if now - last_daily_run > 6 * 3600:
                try:
                    results = db.apply_retention_policies()
                    if any(v > 0 for v in results.values()):
                        print(f"[DB] Retention cleanup: {results}")
                    db.generate_daily_summary()
                except Exception as db_err:
                    print(f"[WARN] DB maintenance failed: {db_err}")
                last_daily_run = now

            for etype, name, conf, details in events:
                person_id = db.get_person_id(name) if name not in ("SYSTEM",) else None

                snap_path = None
                thumb_path = None
                severity = 0

                if etype in ("SPOOF_DETECTED", "CROWD_FORMING", "EVACUATION_ALERT", "DANGEROUS_OBJECT"):
                    severity = 3
                elif etype in ("HESITATION", "PACING", "SCANNING", "SPATIAL_ANOMALY", "LOITERING"):
                    severity = 2
                elif etype == "RECOGNITION":
                    severity = 1

                if severity >= 2:
                    try:
                        face_bbox = None
                        for f in faces:
                            if f.get("name") == name:
                                face_bbox = f.get("bbox")
                                break
                        snap_path, thumb_path = vision.capture_event_snapshot(
                            frame, etype, name, face_bbox
                        )
                    except Exception as snap_err:
                        print(f"[WARN] Snapshot failed: {snap_err}")

                try:
                    db.log_event(
                        event_type=etype,
                        person_id=person_id,
                        confidence=conf,
                        details=details,
                        snapshot_path=snap_path,
                        severity=severity,
                    )
                except Exception as log_err:
                    print(f"[WARN] Event logging failed: {log_err}")

                if etype in ALERT_EVENTS:
                    print(f"\n[ALERT] {etype}: {name} | {details}\n")

                try:
                    alert_manager.check_and_alert(etype, name, conf, details)
                except Exception as alert_err:
                    print(f"[WARN] Alert failed: {alert_err}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting.")
                break

            elif key == ord("e"):
                print("\n--- ENROLLMENT MODE ---")
                name = input("Enter name to enroll: ").strip()
                if not name:
                    print("Invalid name.")
                    continue
                print(f"Enrolling {name}... look at camera.")
                print("  (Tip: try different angles/lighting for better multi-embedding recognition)")
                time.sleep(2)

                emb = vision.enroll_person(frame, name)
                if emb is not None:
                    person_data = vision.face_db.get(name, {})
                    n_embs = len(person_data.get("embeddings", []))
                    threshold = person_data.get("threshold", 0.0)

                    db.enroll_person(name, emb)
                    db.log_audit("ENROLL_PERSON", target=name,
                                 details={
                                     "embedding_dim": len(emb) if hasattr(emb, "__len__") else "N/A",
                                     "total_embeddings": n_embs,
                                     "auto_threshold": round(threshold, 4),
                                 })
                    print(f"[OK] {name} enrolled and saved to DB.")
                    print(f"     Embeddings: {n_embs}, Auto-threshold: {threshold:.3f}")

                    min_embs = 3  
                    if n_embs < min_embs:
                        print(f"     [TIP] Enroll {min_embs - n_embs} more time(s) at different angles")
                        print(f"           for reliable multi-embedding recognition.")
                else:
                    print(f"[FAIL] Enrollment rejected for {name}.")
                print("--- Resuming Detection ---\n")

            elif key == ord("s"):
                ts = int(time.time())
                snap_path = os.path.join(snaps_dir, f"manual_snap_{ts}.jpg")
                cv2.imwrite(snap_path, display)
                print(f"[INFO] Manual snapshot saved: {snap_path}")

            elif key == ord("r"):
                try:
                    results = db.apply_retention_policies()
                    print(f"[DB] Retention results: {results}")
                except Exception as re_err:
                    print(f"[ERROR] Retention failed: {re_err}")

            elif key == ord("d"):
                try:
                    db.generate_daily_summary()
                    print("[DB] Daily summary generated.")
                except Exception as ds_err:
                    print(f"[ERROR] Summary failed: {ds_err}")

            elif key == ord("t"):
                alert_manager.test_alert()

            elif key == ord("a"):
                print("\n--- AUDIT LOG (last 20) ---")
                for entry in db.get_audit_log(limit=20):
                    print(f"  [{entry['timestamp']}] {entry['action']}: "
                          f"{entry.get('target', '')} {entry.get('details_json', '')}")
                print("--- End Audit Log ---\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\n!!! FATAL ERROR !!!\n{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Shutting down.")
        db.log_audit("SYSTEM_STOP", details={"reason": "normal"})
        if "cap" in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

