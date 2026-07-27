import os
import cv2
import time
import vision  # This imports your vision.py file

def main():
    print("=" * 60)
    print("  BATCH FACE ENROLLMENT TOOL")
    print("=" * 60)
    
    # Initialize the Vision System (loads AI models)
    # This gives us access to the 'enroll_person' function
    vs = vision.VisionSystem()
    
    # Path to your known faces
    # Assumes structure: data/known_faces/PersonName/image.jpg
    known_dir = "known_faces"
    
    if not os.path.exists(known_dir):
        print(f"[ERROR] Directory not found: {known_dir}")
        print("Please create a 'known_faces' folder with subfolders for each person.")
        return

    people_found = 0
    images_processed = 0

    # Iterate through subfolders (Person Names)
    for person_name in os.listdir(known_dir):
        person_path = os.path.join(known_dir, person_name)
        
        # Skip if it's not a folder
        if not os.path.isdir(person_path):
            continue

        print(f"\n--- Processing {person_name} ---")
        
        # Get all images in the person's folder
        valid_images = [f for f in os.listdir(person_path) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not valid_images:
            print(f"  [WARN] No images found for {person_name}")
            continue

        # Enroll each image
        for img_file in valid_images:
            img_path = os.path.join(person_path, img_file)
            frame = cv2.imread(img_path)
            
            if frame is None:
                print(f"  [WARN] Could not read {img_file}")
                continue

            # Call the system's enrollment function
            # This handles quality checks and embedding generation
            success = vs.enroll_person(frame, person_name)
            
            if success is not None:
                print(f"  [OK] Enrolled {img_file}")
                images_processed += 1
            else:
                print(f"  [FAIL] Rejected {img_file} (Quality too low or no face)")
            
            # Small delay to prevent UI freezing or overload
            time.sleep(0.1)
            
            # Limit to max 5 embeddings per person to prevent bloat
            # (Check current count in DB)
            if person_name in vs.face_db:
                current_count = len(vs.face_db[person_name].get("embeddings", []))
                if current_count >= vs._max_embeddings_per_person:
                    print(f"  [INFO] Limit reached for {person_name} (Max {vs._max_embeddings_per_person}).")
                    break

        people_found += 1

    print("\n" + "=" * 60)
    print(f"  BATCH PROCESSING COMPLETE")
    print(f"  People Found:   {people_found}")
    print(f"  Images Stored:  {images_processed}")
    print(f"  Database File:  {vision.CONFIG['DATABASE_FILE']}")
    print("=" * 60)

if __name__ == "__main__":
    main()