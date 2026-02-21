import json
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Union, Any

try:
    import numpy as np
except ImportError:
    np = None


class EmbeddingStore:
    """Persistent storage for person re-identification embeddings."""

    def __init__(self, db_path: str = "embeddings.json"):
        """Initialize the embedding store.

        Args:
            db_path: Path to the JSON file storing embeddings
        """
        self.db_path = db_path
        self.embeddings: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        """Load embeddings from disk."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                    self.embeddings = data
                print(f"Loaded {len(self.embeddings)} persons from embedding store")
            except Exception as e:
                print(f"Error loading embeddings from {self.db_path}: {e}")
                self.embeddings = {}
        else:
            self.embeddings = {}

    def _save(self):
        """Save embeddings to disk."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable = {}
            for person_id, data in self.embeddings.items():
                embedding = data["embedding"]
                # Handle both numpy arrays and lists
                if np is not None and isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                elif not isinstance(embedding, list):
                    # Convert to list if it's neither numpy array nor list
                    embedding = list(embedding)

                serializable[person_id] = {
                    "embedding": embedding,
                    "camera": data["camera"],
                    "timestamp": data["timestamp"],
                    "confidence": data["confidence"],
                }

            with open(self.db_path, "w") as f:
                json.dump(serializable, f)
        except Exception as e:
            print(f"Error saving embeddings to {self.db_path}: {e}")

    def store_embedding(
        self,
        person_id: str,
        embedding: Union[List, Any],
        camera: str,
        confidence: float = 0.0,
    ) -> None:
        """Store a person's embedding.

        Args:
            person_id: Unique identifier for the person
            embedding: Feature vector (numpy array or list)
            camera: Camera where the person was detected
            confidence: Confidence score for this detection
        """
        self.embeddings[person_id] = {
            "embedding": embedding,
            "camera": camera,
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
        }
        self._save()

    def get_all_embeddings(self) -> Dict[str, Tuple[Union[List, Any], str, float]]:
        """Get all stored embeddings.

        Returns:
            Dict mapping person_id to (embedding, camera, confidence)
        """
        result = {}
        for person_id, data in self.embeddings.items():
            embedding = data["embedding"]
            if isinstance(embedding, list):
                # Convert to numpy array if numpy is available
                if np is not None:
                    embedding = np.array(embedding)
            result[person_id] = (embedding, data["camera"], data["confidence"])
        return result

    def get_embedding(self, person_id: str) -> Optional[Union[List, Any]]:
        """Get embedding for a specific person.

        Args:
            person_id: Unique identifier for the person

        Returns:
            Feature vector or None if not found
        """
        if person_id not in self.embeddings:
            return None

        embedding = self.embeddings[person_id]["embedding"]
        if isinstance(embedding, list):
            if np is not None:
                embedding = np.array(embedding)
        return embedding

    def person_exists(self, person_id: str) -> bool:
        """Check if a person is in the store.

        Args:
            person_id: Unique identifier for the person

        Returns:
            True if person exists, False otherwise
        """
        return person_id in self.embeddings

    def delete_person(self, person_id: str) -> bool:
        """Delete a person from the store.

        Args:
            person_id: Unique identifier for the person

        Returns:
            True if deleted, False if not found
        """
        if person_id in self.embeddings:
            del self.embeddings[person_id]
            self._save()
            return True
        return False

    def get_all_person_ids(self) -> List[str]:
        """Get list of all stored person IDs.

        Returns:
            List of person IDs
        """
        return list(self.embeddings.keys())

    def clear(self):
        """Clear all embeddings from the store."""
        self.embeddings = {}
        self._save()
