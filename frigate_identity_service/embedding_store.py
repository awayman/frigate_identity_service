import json
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Union, Any

try:
    import numpy as np
except ImportError:
    np = None


class EmbeddingStore:
    """Persistent storage for person re-identification embeddings.

    Stores up to MAX_EMBEDDINGS_PER_PERSON embeddings per person, keeping
    the most recent ones. This allows recency-weighted matching and ensures
    embeddings reflect how a person looks today rather than weeks ago.
    """

    MAX_EMBEDDINGS_PER_PERSON = 10

    def __init__(self, db_path: str = "embeddings.json"):
        """Initialize the embedding store.

        Args:
            db_path: Path to the JSON file storing embeddings
        """
        self.db_path = db_path
        self.embeddings: Dict[str, List[Dict]] = {}
        self._load()

    def _load(self):
        """Load embeddings from disk, migrating old single-embedding format if needed."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                self.embeddings = self._migrate_to_new_format(data)
                total = sum(len(v) for v in self.embeddings.values())
                print(f"Loaded {len(self.embeddings)} persons ({total} total embeddings) from embedding store")
            except Exception as e:
                print(f"Error loading embeddings from {self.db_path}: {e}")
                self.embeddings = {}
        else:
            self.embeddings = {}

    def _migrate_to_new_format(self, data: dict) -> Dict[str, List[Dict]]:
        """Migrate old single-embedding format {person: {embedding, ...}} to
        new multi-embedding format {person: [{embedding, ...}, ...]}."""
        migrated = {}
        for person_id, entry in data.items():
            if isinstance(entry, list):
                # Already new format
                migrated[person_id] = entry
            elif isinstance(entry, dict) and "embedding" in entry:
                # Old format: single embedding dict — wrap in a list
                migrated[person_id] = [{
                    "embedding": entry["embedding"],
                    "camera": entry.get("camera", "unknown"),
                    "timestamp": entry.get("timestamp", datetime.now().isoformat()),
                    "confidence": entry.get("confidence", 0.0),
                }]
        return migrated

    def _save(self):
        """Save embeddings to disk."""
        try:
            serializable = {}
            for person_id, embeddings_list in self.embeddings.items():
                serializable[person_id] = []
                for emb_entry in embeddings_list:
                    embedding = emb_entry["embedding"]
                    if np is not None and isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                    elif not isinstance(embedding, list):
                        embedding = list(embedding)
                    serializable[person_id].append({
                        "embedding": embedding,
                        "camera": emb_entry["camera"],
                        "timestamp": emb_entry["timestamp"],
                        "confidence": emb_entry["confidence"],
                    })

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
        """Prepend a new embedding for a person, retaining at most MAX_EMBEDDINGS_PER_PERSON.

        The most recent embedding is always stored first, so index 0 is the
        best candidate for a quick single-embedding lookup.

        Args:
            person_id: Unique identifier for the person
            embedding: Feature vector (numpy array or list)
            camera: Camera where the person was detected
            confidence: Confidence score for this detection
        """
        new_entry = {
            "embedding": embedding,
            "camera": camera,
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
        }

        if person_id not in self.embeddings:
            self.embeddings[person_id] = [new_entry]
        else:
            self.embeddings[person_id].insert(0, new_entry)
            # Trim to max allowed
            self.embeddings[person_id] = self.embeddings[person_id][:self.MAX_EMBEDDINGS_PER_PERSON]

        self._save()

    def get_all_embeddings(self) -> Dict[str, List[Tuple[Union[List, Any], str, float, str]]]:
        """Get all stored embeddings with timestamps for recency weighting.

        Returns:
            Dict mapping person_id to a list of
            (embedding, camera, confidence, timestamp_iso) tuples,
            ordered most-recent first.
        """
        result = {}
        for person_id, embeddings_list in self.embeddings.items():
            result[person_id] = []
            for emb_entry in embeddings_list:
                embedding = emb_entry["embedding"]
                if isinstance(embedding, list) and np is not None:
                    embedding = np.array(embedding)
                result[person_id].append((
                    embedding,
                    emb_entry["camera"],
                    emb_entry["confidence"],
                    emb_entry["timestamp"],
                ))
        return result

    def get_embedding(self, person_id: str) -> Optional[Union[List, Any]]:
        """Get the most recent embedding for a specific person.

        Args:
            person_id: Unique identifier for the person

        Returns:
            Feature vector or None if not found
        """
        if person_id not in self.embeddings or not self.embeddings[person_id]:
            return None

        embedding = self.embeddings[person_id][0]["embedding"]
        if isinstance(embedding, list) and np is not None:
            embedding = np.array(embedding)
        return embedding

    def person_exists(self, person_id: str) -> bool:
        """Check if a person is in the store.

        Args:
            person_id: Unique identifier for the person

        Returns:
            True if person exists, False otherwise
        """
        return person_id in self.embeddings and len(self.embeddings[person_id]) > 0

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
        print("[EMBEDDINGS] Cleared all embeddings from store")
