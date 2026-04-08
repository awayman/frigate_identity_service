import json
import os
import threading
from datetime import datetime, timedelta, timezone
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
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        """Load embeddings from disk, migrating old single-embedding format if needed."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                self.embeddings = self._migrate_to_new_format(data)
                total = sum(len(v) for v in self.embeddings.values())
                print(
                    f"Loaded {len(self.embeddings)} persons ({total} total embeddings) from embedding store"
                )
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
                migrated[person_id] = [
                    {
                        "embedding": entry["embedding"],
                        "camera": entry.get("camera", "unknown"),
                        "timestamp": entry.get("timestamp", datetime.now().isoformat()),
                        "confidence": entry.get("confidence", 0.0),
                        "negative": bool(entry.get("negative", False)),
                    }
                ]
        return migrated

    def _save(self):
        """Save embeddings to disk."""
        try:
            with self._lock:
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

                serializable = {}
                for person_id, embeddings_list in self.embeddings.items():
                    serializable[person_id] = []
                    for emb_entry in embeddings_list:
                        embedding = emb_entry["embedding"]
                        if np is not None and isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()
                        elif not isinstance(embedding, list):
                            embedding = list(embedding)
                        entry_dict = {
                            "embedding": embedding,
                            "camera": emb_entry["camera"],
                            "timestamp": emb_entry["timestamp"],
                            "confidence": emb_entry["confidence"],
                            "negative": bool(emb_entry.get("negative", False)),
                        }
                        if emb_entry.get("event_id"):
                            entry_dict["event_id"] = emb_entry["event_id"]
                        if emb_entry.get("negative_at"):
                            entry_dict["negative_at"] = emb_entry["negative_at"]
                        if emb_entry.get("negative_reason"):
                            entry_dict["negative_reason"] = emb_entry["negative_reason"]
                        serializable[person_id].append(entry_dict)

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
        event_id: Optional[str] = None,
    ) -> None:
        """Prepend a new embedding for a person, retaining at most MAX_EMBEDDINGS_PER_PERSON.

        The most recent embedding is always stored first, so index 0 is the
        best candidate for a quick single-embedding lookup.

        Args:
            person_id: Unique identifier for the person
            embedding: Feature vector (numpy array or list)
            camera: Camera where the person was detected
            confidence: Confidence score for this detection
            event_id: Optional Frigate event ID that produced this embedding
        """
        new_entry: Dict[str, Any] = {
            "embedding": embedding,
            "camera": camera,
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "negative": False,
        }
        if event_id:
            new_entry["event_id"] = event_id

        with self._lock:
            if person_id not in self.embeddings:
                self.embeddings[person_id] = [new_entry]
            else:
                self.embeddings[person_id].insert(0, new_entry)
                # Trim to max allowed
                self.embeddings[person_id] = self.embeddings[person_id][
                    : self.MAX_EMBEDDINGS_PER_PERSON
                ]

        self._save()

    def get_all_embeddings(
        self,
        *,
        include_negative: bool = False,
    ) -> Dict[str, List[Tuple[Union[List, Any], str, float, str]]]:
        """Get all stored embeddings with timestamps for recency weighting.

        Returns:
            Dict mapping person_id to a list of
            (embedding, camera, confidence, timestamp_iso) tuples,
            ordered most-recent first.
        """
        result = {}
        with self._lock:
            for person_id, embeddings_list in self.embeddings.items():
                result[person_id] = []
                for emb_entry in embeddings_list:
                    if not include_negative and emb_entry.get("negative", False):
                        continue
                    embedding = emb_entry["embedding"]
                    if isinstance(embedding, list) and np is not None:
                        embedding = np.array(embedding)
                    result[person_id].append(
                        (
                            embedding,
                            emb_entry["camera"],
                            emb_entry["confidence"],
                            emb_entry["timestamp"],
                        )
                    )
        return result

    def get_embedding(self, person_id: str) -> Optional[Union[List, Any]]:
        """Get the most recent embedding for a specific person.

        Args:
            person_id: Unique identifier for the person

        Returns:
            Feature vector or None if not found
        """
        with self._lock:
            if person_id not in self.embeddings or not self.embeddings[person_id]:
                return None

            embedding = None
            for entry in self.embeddings[person_id]:
                if entry.get("negative", False):
                    continue
                embedding = entry["embedding"]
                break
            if embedding is None:
                return None
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
        with self._lock:
            return person_id in self.embeddings and len(self.embeddings[person_id]) > 0

    def delete_person(self, person_id: str) -> bool:
        """Delete a person from the store.

        Args:
            person_id: Unique identifier for the person

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
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
        with self._lock:
            return list(self.embeddings.keys())

    def clear(self):
        """Clear all embeddings from the store."""
        with self._lock:
            self.embeddings = {}
            self._save()
        print("[EMBEDDINGS] Cleared all embeddings from store")

    @staticmethod
    def _parse_timestamp(timestamp_value: Any) -> Optional[datetime]:
        """Parse an ISO8601 timestamp string into a naive UTC datetime."""
        if not isinstance(timestamp_value, str):
            return None

        normalized = timestamp_value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None

        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    def prune_expired(self, max_age_hours: float) -> Dict[str, int]:
        """Prune embeddings older than max_age_hours.

        Invalid or missing timestamps are preserved to avoid accidental data loss.

        Args:
            max_age_hours: Maximum age to retain, in hours.

        Returns:
            A dictionary with removal and remaining item counts.
        """
        if max_age_hours <= 0:
            raise ValueError("max_age_hours must be > 0")

        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        removed_embeddings = 0
        removed_persons = 0

        with self._lock:
            for person_id, entries in list(self.embeddings.items()):
                retained_entries = []
                for entry in entries:
                    entry_timestamp = self._parse_timestamp(entry.get("timestamp"))
                    if entry_timestamp is None or entry_timestamp >= cutoff:
                        retained_entries.append(entry)
                        continue
                    removed_embeddings += 1

                if retained_entries:
                    self.embeddings[person_id] = retained_entries
                else:
                    removed_persons += 1
                    del self.embeddings[person_id]

            if removed_embeddings > 0:
                self._save()

            remaining_persons = len(self.embeddings)
            remaining_embeddings = sum(
                len(entries) for entries in self.embeddings.values()
            )

        return {
            "removed_embeddings": removed_embeddings,
            "removed_persons": removed_persons,
            "remaining_persons": remaining_persons,
            "remaining_embeddings": remaining_embeddings,
        }

    def remove_embeddings_by_event_id(
        self,
        person_id: str,
        event_id: str,
        *,
        fallback_to_latest: bool = False,
    ) -> int:
        """Remove all embeddings for *person_id* that were created from *event_id*.

        If no entry carries a matching event_id (e.g. embeddings stored before
        this feature was added) the method falls back to removing the most
        recent embedding for the person.

        Returns:
            Number of embedding entries removed (0 if person not found).
        """
        if not person_id:
            return 0

        if not event_id and not fallback_to_latest:
            return 0

        with self._lock:
            entries = self.embeddings.get(person_id)
            if not entries:
                return 0

            matched = [e for e in entries if e.get("event_id") == event_id]
            if matched:
                retained = [e for e in entries if e.get("event_id") != event_id]
                removed_count = len(matched)
            elif fallback_to_latest:
                # Fallback: remove the most recent entry (index 0)
                retained = entries[1:]
                removed_count = 1
            else:
                retained = entries
                removed_count = 0

            if retained:
                self.embeddings[person_id] = retained
            else:
                del self.embeddings[person_id]

            if removed_count:
                self._save()

        return removed_count

    def get_latest_event_id(self, person_id: str) -> Optional[str]:
        """Return the event_id of the most recent embedding for *person_id*, or None."""
        with self._lock:
            entries = self.embeddings.get(person_id)
            if not entries:
                return None
            for entry in entries:
                if entry.get("negative", False):
                    continue
                eid = entry.get("event_id")
                if eid:
                    return eid
        return None

    def mark_embeddings_by_event_id(
        self,
        person_id: str,
        event_id: str,
        *,
        fallback_to_latest: bool = False,
        negative_reason: str = "false_positive_feedback",
    ) -> int:
        """Mark embeddings as negative so matcher excludes them by default.

        Returns the number of newly marked entries.
        """
        if not person_id:
            return 0

        if not event_id and not fallback_to_latest:
            return 0

        with self._lock:
            entries = self.embeddings.get(person_id)
            if not entries:
                return 0

            now_iso = datetime.now().isoformat()
            marked_count = 0

            matched_indices = [
                idx for idx, entry in enumerate(entries) if entry.get("event_id") == event_id
            ]

            if matched_indices:
                for idx in matched_indices:
                    entry = entries[idx]
                    if entry.get("negative", False):
                        continue
                    entry["negative"] = True
                    entry["negative_at"] = now_iso
                    entry["negative_reason"] = negative_reason
                    marked_count += 1
            elif fallback_to_latest:
                for entry in entries:
                    if entry.get("negative", False):
                        continue
                    entry["negative"] = True
                    entry["negative_at"] = now_iso
                    entry["negative_reason"] = negative_reason
                    marked_count = 1
                    break

            if marked_count:
                self._save()

        return marked_count

    def get_stats(self) -> Dict[str, int]:
        """Return summary stats for stored embeddings."""
        with self._lock:
            return {
                "persons": len(self.embeddings),
                "embeddings": sum(len(entries) for entries in self.embeddings.values()),
            }
