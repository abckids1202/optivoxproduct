import cv2
import sys
import os
from test_vision import VisionSystem

def main():
    print("=" * 60)
    print("  FACE ENROLLMENT TOOL")
    print("=" * 60)

    print("[INFO] Loading Vision System...")
    vision = VisionSystem()
    print("[INFO] Ready.")

    print("[DEBUG] Opening Camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FATAL] Cannot open camera.")
        return

    print("\nInstructions:")
    print("1. Position your face in front of the camera.")
    print("2. Wait until a GREEN box appears around your face.")
    print("3. The label should say 'UNKNOWN' or 'STRANGER'.")
    print("4. Press 'c' on keyboard to capture and enroll.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Process frame for detection only
        # We only need the display_frame and faces info
        (display_frame, faces, _, _, _, _, _) = vision.process(frame)

        # Draw instruction
        cv2.putText(display_frame, "Press 'c' to Capture & Enroll", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Enrollment Mode", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            # 1. Get current unknowns
            unknowns = vision.get_current_unknowns()
            
            if not unknowns:
                print("\n[FAIL] No face detected or face is already known.")
                print("       Make sure the box says 'UNKNOWN'.")
                continue
            
            print(f"\n[INFO] Detected {len(unknowns)} candidate face(s).")
            for i, (oid, label, conf) in enumerate(unknowns):
                print(f"  [{i}] ID: {oid} ({label}, conf: {conf:.2f})")
            
            # 2. Select
            choice = input("Select ID to enroll [number] or 'c' to cancel: ").strip()
            if choice.lower() == 'c': continue
            
            try:
                idx = int(choice)
                if 0 <= idx < len(unknowns):
                    oid, _, _ = unknowns[idx]
                    name = input("Enter Person's Name: ").strip()
                    
                    if name:
                        # 3. Enroll
                        success = vision.enroll_unknown_face(oid, name)
                        if success:
                            print(f"\n[SUCCESS] {name} enrolled successfully!")
                            print("You can now close this window.")
                            # Pause to see result
                            cv2.waitKey(2000)
                            break
                        else:
                            print("[FAIL] Enrollment failed.")
            except ValueError:
                print("[FAIL] Invalid input.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()