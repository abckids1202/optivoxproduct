# performance.py
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import faiss
import pickle
import os
import time

# --- FAISS Indexer for High-Performance Search ---
class FAISSIndexer:
    """A wrapper around a FAISS index for fast and efficient face embedding search."""
    def __init__(self, embedding_dim=512):
        self.embedding_dim = embedding_dim
        # IndexFlatIP is for Inner Product search, equivalent to Cosine Similarity for L2-normalized vectors.
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.known_names = []

    def add_embeddings(self, names, embeddings):
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {embeddings.shape[1]}")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.known_names.extend(names)
        print(f"[FAISS] Added {len(names)} embeddings. Total: {self.index.ntotal}")

    def search(self, query_embedding, k=1):
        if self.index.ntotal == 0:
            return []
        query_embedding = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for i in range(k):
            idx = indices[0][i]
            dist = distances[0][i]
            if idx != -1:
                name = self.known_names[idx]
                cosine_distance = 1 - dist
                results.append((name, cosine_distance))
        return results

    def save(self, index_path, names_path):
        try:
            faiss.write_index(self.index, index_path)
            with open(names_path, 'wb') as f:
                pickle.dump(self.known_names, f)
            print(f"[FAISS] Saved index to {index_path} and names to {names_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save FAISS data: {e}")
            return False

    def load(self, index_path, names_path):
        try:
            self.index = faiss.read_index(index_path)
            with open(names_path, 'rb') as f:
                self.known_names = pickle.load(f)
            print(f"[FAISS] Loaded index with {len(self.known_names)} names.")
            return True
        except Exception as e:
            print(f"[WARNING] Could not load FAISS data. Starting fresh. Error: {e}")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.known_names = []
            return False

# --- Face Analyzer ---
class FaceAnalyzer:
    def __init__(self, db_path="face_database.pkl", providers=['CPUExecutionProvider']):
        self.db_path = db_path
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx=0, det_size=(640, 640))
        
        self.known_faces = {}
        self.indexer = FAISSIndexer(embedding_dim=512)
        self._load_known_faces()

    def detect(self, frame, bbox):
        """Detects a face within a given bounding box and returns the cropped face."""
        x1, y1, x2, y2 = map(int, bbox)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
        return face_crop

    def get_embedding(self, face_img):
        """Extracts a 512-D embedding from a face image."""
        if face_img is None:
            return None
        faces = self.app.get(face_img)
        if not faces:
            return None
        # Assuming the largest face is the target
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return face.embedding

    def recognize(self, embedding, threshold=0.5):
        """Identifies a face embedding against the known database."""
        if embedding is None or self.indexer.index.ntotal == 0:
            return "Unknown", 0.0
        
        results = self.indexer.search(embedding, k=1)
        if results:
            name, distance = results[0]
            if distance < threshold:
                return name, distance
        return "Unknown", 0.0

    def enroll(self, name, embedding):
        """Adds a new face to the database."""
        if embedding is None:
            print("[ERROR] Cannot enroll, embedding is None.")
            return False
        if name in self.known_faces:
            print(f"[WARN] Face for '{name}' already exists. Overwriting.")
            # To overwrite, we would need to remove the old one, which is complex with FAISS.
            # For simplicity, we'll just add the new one. A better system would handle updates.
        
        self.known_faces[name] = embedding
        self.indexer.add_embeddings([name], np.array([embedding]))
        self._save_known_faces()
        print(f"[INFO] Successfully enrolled '{name}'.")
        return True

    def _save_known_faces(self):
        """Saves the known faces dictionary to a pickle file."""
        # We save the simple dict for compatibility, but the FAISS index is the primary source.
        # A more robust system would save the FAISS index directly.
        # Let's save the names and embeddings list to be loaded by the FAISS indexer.
        data_to_save = {
            'names': self.indexer.known_names,
            'embeddings': [self.known_faces[name] for name in self.indexer.known_names]
        }
        with open(self.db_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"[INFO] Face database saved to {self.db_path}")

    # --- FIX START ---
    # The original method was not robust against empty or corrupted files.
    # This new version includes comprehensive error handling and validation.
    def _load_known_faces(self):
        """Loads known faces from the pickle file into the FAISS index."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    names = data.get('names', [])
                    embeddings_list = data.get('embeddings', [])

                    # Check if the lists are non-empty and of equal length
                    if embeddings_list and names and len(embeddings_list) == len(names):
                        print(f"[INFO] Loading {len(names)} known faces from database...")
                        embeddings = np.array(embeddings_list)
                        
                        # Add to the FAISS indexer
                        self.indexer.add_embeddings(names, embeddings)
                        
                        # Populate the dictionary for quick lookup
                        self.known_faces = {name: emb for name, emb in zip(names, embeddings_list)}
                        print("[INFO] Known faces loaded successfully.")
                    else:
                        print("[INFO] Face database file is empty or malformed. No known faces loaded.")
                        self.known_faces = {}

            except (pickle.UnpicklingError, EOFError, Exception) as e:
                print(f"[ERROR] Could not read face database. It may be corrupted. Starting fresh. Error: {e}")
                self.known_faces = {}
                # Ensure a clean state if loading fails
                self.indexer = FAISSIndexer(512)
        else:
            print("[INFO] Face database not found. Recognition will be disabled until a face is enrolled.")
            self.known_faces = {}
    # --- FIX END ---
