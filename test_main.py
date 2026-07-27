import cv2
import time
import os
import threading
import sys
import numpy as np
import datetime
import traceback

try:
    from test_vision import VisionSystem
except ImportError:
    print("[CRITICAL] Could not import 'vision'.")
    raise

try:
    from test_database import EventDatabase
    from test_assistant import AIAssistant
except ImportError as e:
    print(f"[CRITICAL] Could not import AI modules: {e}")
    raise

try:
    from test_notifier import AlertManager
    NOTIFIER_AVAILABLE = True
except ImportError:
    NOTIFIER_AVAILABLE = False
    print("[WARN] notifier.py not found.")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[WARN] pyttsx3 not installed.")

# ---------------------------------------------------------------------------
# VOICE ANNOUNCEMENT HELPER
# ---------------------------------------------------------------------------
_tts_lock = threading.Lock()
_tts_engine = None

def _ensure_tts():
    global _tts_engine
    if not TTS_AVAILABLE:
        return
    if _tts_engine is None:
        try:
            _tts_engine = pyttsx3.init()
            _tts_engine.setProperty("rate", 175)
            voices = _tts_engine.getProperty("voices")
            for v in voices:
                if "english" in v.name.lower() or "en" in v.id.lower():
                    _tts_engine.setProperty("voice", v.id)
                    break
        except Exception as e:
            print(f"[TTS] Init failed: {e}")

def voice_announce(text: str):
    if not text or not text.strip():
        return
    print(f"[VOICE] {text}")
    if not TTS_AVAILABLE:
        return
    def _speak():
        with _tts_lock:
            try:
                _ensure_tts()
                if _tts_engine:
                    _tts_engine.stop()
                    _tts_engine.say(text)
                    _tts_engine.runAndWait()
            except Exception as e:
                print(f"[TTS ERROR] {e}")
    threading.Thread(target=_speak, daemon=True).start()

# ---------------------------------------------------------------------------
# TUNABLES
# ---------------------------------------------------------------------------
SKIP_FRAMES = 1
PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  INTELLIGENT SECURITY MONITORING SYSTEM")
    print("=" * 60)

    # 1. Database
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "security.db")
    db = EventDatabase(db_path=db_path)
    db.setup_database()
    print(f"[INFO] Database ready.")

    # 2. Alert Manager
    alert_mgr = None
    if NOTIFIER_AVAILABLE:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alert_config.json")
        alert_mgr = AlertManager(config_path=cfg_path)
        if hasattr(alert_mgr, 'enabled') and alert_mgr.enabled: 
            print("[INFO] Alerts active.")
        else:
            print("[WARN] Alerts disabled or not configured.")

    # 3. AI Assistant
    assistant = AIAssistant(db=db, alert_manager=alert_mgr, config={"openai_api_key": os.environ.get("OPENAI_API_KEY", "")})
    if not assistant.is_ready(): print("[WARN] AI Assistant not fully ready.")

    # 4. Vision System
    vision = VisionSystem()
    print("[INFO] VisionSystem ready.")

    # 5. Camera
    print("[DEBUG] Opening camera...")
    cap = cv2.VideoCapture(0)
    
    cv2.destroyAllWindows()

    if not cap.isOpened():
        print("[CRITICAL] Cannot open camera (index 0). Trying index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[FATAL] No camera found.")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESS_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESS_HEIGHT)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[DEBUG] Camera active ({actual_w}x{actual_h}).")

    print("\n" + "=" * 60)
    print("  SYSTEM READY")
    print("  'q' = quit | 'a' = toggle Voice | 'e' = enroll | 't' = test alert")
    print("=" * 60 + "\n")

    voice_announce("Security system online.")
    
    snapshot_dir = "snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    _last_voice_alert = {}
    VOICE_ALERT_COOLDOWN = 30.0

    try:
        frame_count = 0
        start_time = time.time()
        last_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Camera read failed. Retrying...")
                time.sleep(0.5)
                continue
            
            last_frame = frame.copy()
            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            # B. Vision Processing
            try:
                (display_frame, faces, hand_dets, obj_dets,
                 events, tracked_count, dt) = vision.process(display_frame)
            except Exception as e:
                print(f"[ERROR] Vision processing failed (skipping frame): {e}")
                traceback.print_exc()
                cv2.imshow("Security Feed", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            # C. Event Handling
            for event in events:
                try:
                    event_type = event[0]
                    name = event[1] if len(event) > 1 else "SYSTEM"
                    confidence = event[2] if len(event) > 2 else 0.0
                    details = event[3] if len(event) > 3 else ""
                    
                    # Screenshot logic
                    snapshot_path = None
                    if event_type in ["DANGEROUS_OBJECT", "SPOOF_DETECTED", "EVACUATION_ALERT", "FIRE_DETECTED"]:
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        fname = f"{snapshot_dir}/{event_type}_{name}_{ts}.jpg"
                        try:
                            cv2.imwrite(fname, last_frame)
                            snapshot_path = fname
                            print(f"[SNAPSHOT] Saved {fname}")
                        except Exception as e:
                            print(f"[ERROR] Snapshot failed: {e}")

                    # Log DB
                    person_id = None
                    if name not in ("SYSTEM", "UNKNOWN") and not name.startswith("STRANGER_"):
                        person_id = db.get_person_id(name)
                    
                    db.log_event(
                        event_type=event_type, person_id=person_id,
                        confidence=confidence, details=details,
                        severity=2 if event_type in ["DANGEROUS_OBJECT", "SPOOF_DETECTED"] else 1,
                        location="Camera 1",
                    )
                except Exception as e:
                    print(f"[ERROR] Event logging failed: {e}")

                # Send Alerts
                try:
                    if alert_mgr and hasattr(alert_mgr, 'enabled') and alert_mgr.enabled:
                        # FIX: Check if method exists to prevent crash
                        if hasattr(alert_mgr, 'check_and_alert'):
                            alert_mgr.check_and_alert(event_type, name, confidence, details, snapshot_path)
                        else:
                            print(f"[ALERT] {event_type} | {name} | {details}")
                except Exception as e:
                    print(f"[ERROR] Alert failed: {e}")

                # Voice Alerts
                try:
                    now = time.time()
                    if now - _last_voice_alert.get(event_type, 0) >= VOICE_ALERT_COOLDOWN:
                        if event_type == "DANGEROUS_OBJECT":
                            voice_announce(f"Warning! Dangerous object detected: {name}")
                            _last_voice_alert[event_type] = now
                        elif event_type == "SPOOF_DETECTED":
                            voice_announce("Warning! Spoof or fake face detected.")
                            _last_voice_alert[event_type] = now
                        elif event_type == "EVACUATION_ALERT":
                            voice_announce("Emergency! Possible evacuation detected.")
                            _last_voice_alert[event_type] = now
                except Exception: pass

            # D. UI Overlay
            fps = frame_count / (time.time() - start_time + 0.0001)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            status = "AI: ON" if assistant.is_listening() else "AI: OFF"
            color = (0, 255, 255) if assistant.is_listening() else (100, 100, 100)
            cv2.putText(display_frame, status, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            cv2.imshow("Security Feed", display_frame)
            frame_count += 1

            # E. Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting.")
                break
            if key == ord("a"):
                if assistant.is_listening():
                    assistant.stop_listening()
                else:
                    assistant.start_listening()
            if key == ord("t"):
                print("[TEST] Testing alert...")
                if alert_mgr: 
                    if hasattr(alert_mgr, 'test_alert'):
                        alert_mgr.test_alert()
                    else:
                        print("[TEST] Alert Manager has no test_alert method")
            
            if key == ord("e"):
                unknowns = vision.get_current_unknowns()
                print(f"\n[ENROLL] Found {len(unknowns)} unknown candidates.")
                
                if not unknowns:
                    print("[ENROLL] No unknown face currently tracked.")
                    print("[ENROLL] Wait until you see 'UNKNOWN' or 'STRANGER' on screen.")
                    continue

                for i, (oid, label, conf) in enumerate(unknowns):
                    print(f"  [{i}] ID: {oid} | Label: {label} | Conf: {conf:.2f}")

                choice = input("  Select ID to enroll [number] or 'c' to cancel: ").strip()
                if choice.lower() == 'c': continue
                
                try:
                    idx = int(choice)
                    if 0 <= idx < len(unknowns):
                        oid, _, _ = unknowns[idx]
                        person_name = input("  Enter Name: ").strip()
                        if person_name:
                            success = vision.enroll_unknown_face(oid, person_name)
                            if success:
                                db.log_audit("ENROLL_PERSON", person_name, {"track_id": oid})
                                print(f"  [OK] Enrolled {person_name} successfully.")
                                voice_announce(f"Enrollment successful. Welcome {person_name}.")
                            else:
                                print("  [FAIL] Enrollment failed.")
                    else:
                        print("  [FAIL] Invalid selection.")
                except ValueError:
                    print("  [FAIL] Invalid input.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\n!!! FATAL ERROR !!!")
        print(f"{e}")
        traceback.print_exc()
    finally:
        print("[INFO] Shutting down...")

        try:
            if assistant:
                try:
                    assistant.stop_listening()
                except Exception as e:
                    print(f"[WARN] assistant.stop_listening failed: {e}")

                try:
                    assistant.cleanup()
                except Exception as e:
                    print(f"[WARN] assistant.cleanup failed: {e}")

        except Exception as e:
            print(f"[WARN] Assistant shutdown failed: {e}")

        try:
            if cap and cap.isOpened():
                cap.release()
        except Exception as e:
            print(f"[WARN] Camera release failed: {e}")

        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[WARN] OpenCV window cleanup failed: {e}")

        print("[INFO] Done.")

if __name__ == "__main__":
    main()