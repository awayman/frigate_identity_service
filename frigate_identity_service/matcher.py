from datetime import datetime

try:
    from scipy.spatial.distance import cosine
except ImportError:
    cosine = None

from typing import Dict, Tuple, Optional, List, Any, Union


class EmbeddingMatcher:
    """Matcher for finding best embedding matches using cosine similarity.

    Supports the new multi-embedding format where each person has a list of
    (embedding, camera, confidence, timestamp) tuples ordered most-recent first.
    Recency weighting gives recent embeddings a higher effective similarity score.
    """

    # After this many hours, an embedding's weight levels off at 0.5
    RECENCY_DECAY_HOURS = 24

    @staticmethod
    def _recency_weight(timestamp_str: str) -> float:
        """Return a weight in [0.5, 1.0] based on how recent the embedding is.

        Decays linearly from 1.0 (just captured) to 0.5 at RECENCY_DECAY_HOURS.
        """
        try:
            age_hours = (
                datetime.now() - datetime.fromisoformat(timestamp_str)
            ).total_seconds() / 3600
            return max(
                0.5, 1.0 - age_hours / (EmbeddingMatcher.RECENCY_DECAY_HOURS * 2)
            )
        except Exception:
            return 0.8

    @staticmethod
    def _best_similarity_for_person(
        query_embedding: Union[List, Any],
        embeddings_data: Union[
            List[Tuple],  # new: [(emb, camera, conf, ts), ...]
            Tuple,  # old: (emb, camera, conf)
        ],
    ) -> float:
        """Return the best recency-weighted similarity for a single person."""
        if cosine is None:
            return 0.5

        # Old format: bare tuple with 3 elements
        if isinstance(embeddings_data, tuple) and len(embeddings_data) == 3:
            stored_embedding, _, _ = embeddings_data
            return 1.0 - cosine(query_embedding, stored_embedding)

        # New format: list of tuples
        if not isinstance(embeddings_data, list) or not embeddings_data:
            return 0.0

        best = 0.0
        for item in embeddings_data:
            if len(item) == 4:
                stored_embedding, _, _, timestamp = item
                weight = EmbeddingMatcher._recency_weight(timestamp)
            elif len(item) == 3:
                stored_embedding, _, _ = item
                weight = 0.8
            else:
                continue
            weighted = (1.0 - cosine(query_embedding, stored_embedding)) * weight
            if weighted > best:
                best = weighted
        return best

    @staticmethod
    def find_best_match(
        query_embedding: Union[List, Any],
        stored_embeddings: Dict[str, Any],
        threshold: float = 0.6,
    ) -> Tuple[Optional[str], float]:
        """Find the best matching person from stored embeddings.

        Accepts both old format (person_id → (emb, camera, conf)) and new
        format (person_id → [(emb, camera, conf, timestamp), ...]).
        Recent embeddings receive a higher effective similarity score.

        Args:
            query_embedding: Feature vector from query image (list or numpy array)
            stored_embeddings: Dict mapping person_id to embeddings
            threshold: Minimum cosine similarity to consider a match (0-1)

        Returns:
            Tuple of (matched_person_id, similarity_score) or (None, 0.0) if no match
        """
        if not stored_embeddings:
            return None, 0.0

        if cosine is None:
            return list(stored_embeddings.keys())[0], 0.5

        best_person = None
        best_similarity = 0.0

        for person_id, embeddings_data in stored_embeddings.items():
            similarity = EmbeddingMatcher._best_similarity_for_person(
                query_embedding, embeddings_data
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_person = person_id

        if best_similarity >= threshold:
            return best_person, best_similarity
        return None, best_similarity

    @staticmethod
    def find_top_k_matches(
        query_embedding: Union[List, Any],
        stored_embeddings: Dict[str, Any],
        k: int = 5,
        threshold: float = 0.0,
    ) -> list:
        """Find top-k matching persons with recency weighting.

        Args:
            query_embedding: Feature vector from query image
            stored_embeddings: Dict mapping person_id to embeddings
            k: Number of top matches to return
            threshold: Minimum similarity to include in results

        Returns:
            List of (person_id, similarity_score) tuples, sorted by similarity descending
        """
        if not stored_embeddings or cosine is None:
            return []

        similarities = []
        for person_id, embeddings_data in stored_embeddings.items():
            similarity = EmbeddingMatcher._best_similarity_for_person(
                query_embedding, embeddings_data
            )
            if similarity >= threshold:
                similarities.append((person_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    @staticmethod
    def batch_match(
        query_embeddings: Union[List[List], Any],
        stored_embeddings: Dict[str, Any],
        threshold: float = 0.6,
    ) -> list:
        """Match multiple query embeddings against stored embeddings.

        Args:
            query_embeddings: Array of feature vectors (shape: (n, embedding_dim))
            stored_embeddings: Dict mapping person_id to embeddings
            threshold: Minimum cosine similarity to consider a match

        Returns:
            List of (matched_person_id, similarity_score) tuples, one per query
        """
        return [
            EmbeddingMatcher.find_best_match(qe, stored_embeddings, threshold)
            for qe in query_embeddings
        ]
