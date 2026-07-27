# build_face_database.py
# build_face_database.py

import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm
from scipy.spatial.distance import cdist
import argparse

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces"
DATABASE_FILE = "data/face_database.pkl"
# --- FIX 1: Increase quality threshold to use only the best images ---
# Using a higher threshold ensures the database is built with clear, high-quality reference photos.
QUALITY_THRESHOLD = 75 # Only use images with quality >= 75

def calculate_per_person_threshold(database):
    """
    Calculates a per-person recognition threshold.
    The threshold is set to be halfway between the average intra-person distance
    and the minimum average inter-person distance.
    """
    print("[INFO] Calculating per-person recognition thresholds...")
    names = list(database.keys())
    if len(names) < 2:
        for name in names:
            # A reasonable default threshold if only one person is in the DB
            database[name]['threshold'] = 0.35
        return database

    for i, name1 in enumerate(names):
        embeddings1 = np.array(database[name1]['embeddings'])
        
        if len(embeddings1) > 1:
            intra_distances = cdist(embeddings1, embeddings1, 'cosine')
            np.fill_diagonal(intra_distances, np.nan)
            avg_intra_distance = np.nanmean(intra_distances)
        else:
            # A small default if only one image. This person will be harder to recognize accurately.
            avg_intra_distance = 0.1 

        min_avg_inter_distance = float('inf')
        for j, name2 in enumerate(names):
            if i == j:
                continue
            embeddings2 = np.array(database[name2]['embeddings'])
            inter_distances = cdist(embeddings1, embeddings2, 'cosine')
            avg_inter_distance = np.mean(inter_distances)
            if avg_inter_distance < min_avg_inter_distance:
                min_avg_inter_distance = avg_inter_distance
        
        if min_avg_inter_distance == float('inf'):
            threshold = 0.35
        else:
            threshold = (avg_intra_distance + min_avg_inter_distance) / 2.0
            threshold = max(0.2, min(0.5, threshold))
        
        database[name1]['threshold'] = threshold
        print(f"[INFO] - Threshold for {name1}: {threshold:.3f}")
        
    return database

def build_database(known_faces_dir, db_file):
    if not os.path.exists(known_faces_dir):
        print(f"[ERROR] Directory '{known_faces_dir}' not found.")
        print("Please create it and add subdirectories for each person (e.g., known_faces/Arsene_Pierre/).")
        return

    print("[INFO] Initializing InsightFace for database creation...")
    face_app = FaceAnalysis(name='buffalo_l')
    face_app.prepare(ctx_id=-1, det_size=(640, 640))

    face_database = {}
    image_quality_scorer = ImageQualityScorer({})

    for person_name in tqdm(os.listdir(known_faces_dir), desc="Processing people"):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"\n[INFO] Processing {person_name}...")
        embeddings = []
        qualities = []

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"[WARN] Could not read image: {image_path}")
                    continue

                quality_result = image_quality_scorer.score(image)
                if quality_result["overall_quality"] < QUALITY_THRESHOLD:
                    print(f"[WARN] Skipping low-quality image: {image_path} (Quality: {quality_result['overall_quality']})")
                    continue
                
                faces = face_app.get(image)
                if len(faces) == 1:
                    embedding = faces[0].embedding
                    if embedding is not None:
                        embeddings.append(embedding)
                        qualities.append(quality_result["overall_quality"])
                elif len(faces) > 1:
                    print(f"[WARN] Multiple faces in {image_path}. Skipping.")
                else:
                    print(f"[WARN] No face detected in {image_path}. Skipping.")

            except Exception as e:
                print(f"[ERROR] Failed to process {image_path}: {e}")

        if embeddings:
            face_database[person_name] = {
                "embeddings": embeddings,
                "qualities": qualities,
            }
            # --- FIX 2: Add a warning for users with only one image ---
            if len(embeddings) < 2:
                print(f"[WARN] Only {len(embeddings)} image found for {person_name}. Recognition accuracy will be lower. Consider adding more images.")
            else:
                print(f"[SUCCESS] Added {len(embeddings)} high-quality embeddings for {person_name}.")
        else:
            print(f"[ERROR] No valid embeddings found for {person_name}.")

    if not face_database:
        print("[ERROR] No faces were added to the database. Exiting.")
        return

    face_database = calculate_per_person_threshold(face_database)

    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    with open(db_file, 'wb') as f:
        pickle.dump(face_database, f)
    
    print(f"\n[SUCCESS] Face database saved to '{db_file}' with {len(face_database)} people.")
    total_embeddings = sum(len(data['embeddings']) for data in face_database.values())
    print(f"[INFO] Total embeddings in database: {total_embeddings}")


class ImageQualityScorer:
    def __init__(self, config):
        self.config = config

    def score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness_score = np.mean(gray)
        contrast_score = np.std(gray)
        
        is_blurry = blur_score < 100.0
        is_too_dark = brightness_score < 50
        is_too_bright = brightness_score > 200
        is_low_contrast = contrast_score < 50
        
        quality_score = 0
        if not is_blurry:
            quality_score += 40
        if not is_too_dark and not is_too_bright:
            quality_score += 30
        if not is_low_contrast:
            quality_score += 30
            
        return {
            "blur_score": blur_score,
            "brightness_score": brightness_score,
            "contrast_score": contrast_score,
            "overall_quality": quality_score,
            "is_acceptable": quality_score > 70
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a face recognition database from a directory of images.")
    parser.add_argument("--source", type=str, default=KNOWN_FACES_DIR, help="Source directory containing subfolders of face images.")
    parser.add_argument("--output", type=str, default=DATABASE_FILE, help="Output file path for the pickle database.")
    args = parser.parse_args()

    build_database(args.source, args.output)