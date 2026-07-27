import cv2
import time
import os
import threading

try:
    from vision import VisionSystem
except ImportError:
    print("[CRITICAL] Could not import 'vision'. Make sure 'vision.py' is in this directory.")
    raise

try:
    from database import EventDatabase
    from ai_assistant import AIAssistant
except ImportError as e:
    print(f"[CRITICAL] Could not import AI modules: {e}")
    raise

SKIP_FRAMES = 3
PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
SHOW_FPS = True

print("[INFO] Optimized Mode Active:")
print(f"  [INFO] FPS Optimization: Processing 1 frame, skipping {SKIP_FRAMES - 1} frames.")
print(f"  [INFO] AI Resolution: {PROCESS_WIDTH}x{PROCESS_HEIGHT}")
print(f"  [INFO] Display Resolution: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")

def main():
    print("=" * 60)
    print("  INTELLIGENT SECURITY MONITORING SYSTEM (OPTIMIZED)")
    print("=" * 60)
    print("[DEBUG] Initializing Database...")
    db_path = os.path.join(os.path.dirname(__file__), "security.db")
    db = EventDatabase(db_path=db_path)
    db.setup_database()
    print(f"[INFO] Database ready at: {db_path}")

    print("[DEBUG] Initializing AI Assistant...")
    assistant = AIAssistant(db=db, config={"openai_api_key": os.environ.get("OPENAI_API_KEY", "")})
    
    if not assistant.is_ready():
        print("[WARN] AI Assistant is not fully ready (Check OpenAI API key or audio drivers).")
    
    print("[DEBUG] Initializing VisionSystem (loading AI models)...")
    vision = VisionSystem()
    print("[INFO] VisionSystem ready.")
    print("[DEBUG] Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[CRITICAL] Cannot open camera (index 0).")
        print(" - Is a webcam connected?")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[FATAL] No camera found. Exiting.")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESS_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESS_HEIGHT)
    print("[DEBUG] Camera opened.")

    print("\n" + "=" * 60)
    print("  SYSTEM READY")
    print("  'q' = quit")
    print("  'a' = toggle Voice Assistant")
    print("=" * 60 + "\n")

    try:
        frame_count = 0
        start_time = time.time()
        fps = 0.0
        
        while True:
            for _ in range(SKIP_FRAMES):
                ret, frame = cap.read()
                if not ret:
                    break

            if not ret:
                print("[WARN] Failed to grab frame.")
                break

            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            display_frame, faces, hand_dets, obj_dets, events, tracked_count, dt = vision.process(display_frame)
            
            for event in events:
                try:
                    event_type = event[0]
                    name = event[1] if len(event) > 1 else "SYSTEM"
                    confidence = event[2] if len(event) > 2 else 0.0
                    details = event[3] if len(event) > 3 else ""
                    person_id = None
                    if name not in ("SYSTEM", "UNKNOWN") and not name.startswith("STRANGER_"):
                        person_id = db.get_person_id(name)

                    db.log_event(
                        event_type=event_type,
                        person_id=person_id,
                        confidence=confidence,
                        details=details,
                        severity=0, 
                        location="Camera 1"
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to log event: {e}")
                    print(f"[DEBUG] Event Data: {event}") 

            if SHOW_FPS:
                frame_count += 1
                fps = frame_count / (time.time() - start_time + 0.0001)
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                status_text = "AI: ON" if assistant.is_listening() else "AI: OFF (Press 'A')"
                color = (0, 255, 255) if assistant.is_listening() else (100, 100, 100)
                cv2.putText(display_frame, status_text, (10, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Security Feed", display_frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                print("Quitting.")
                break
            
            if key == ord("a"):
                if assistant.is_listening():
                    print("[INFO] Stopping Voice Assistant...")
                    assistant.stop_listening()
                else:
                    print("[INFO] Starting Voice Assistant...")
                    assistant.start_listening()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\n!!! FATAL ERROR !!!")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Shutting down...")
        
        if assistant:
            assistant.stop_listening()
            assistant.cleanup()
        
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Done.")

if __name__ == "__main__":
    main()