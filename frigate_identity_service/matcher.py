from datetime import datetime
import math

try:
    from scipy.spatial.distance import cosine
except ImportError:
    cosine = None

from typing import Dict, Tuple, Optional, List, Any, Union


class EmbeddingMatcher:
    """Matcher for finding best embedding matches using cosine similarity.

    Supports the new multi-embedding format where each person has a list of
    (embedding, camera, confidence, timestamp) tuples ordered most-recent first.
    Recency weighting gives recent embeddings a higher effective similarity score
    that decays as embeddings age, aligned with the retention policy.

    Args:
        max_age_hours: Maximum age of embeddings before pruning (from retention policy)
        decay_mode: How recency weight decays - 'linear', 'exponential', or 'none'
        weight_floor: Minimum weight for oldest embeddings (0.0-0.9)
        use_confidence_weighting: Whether to incorporate detection confidence into weights
    """

    def __init__(
        self,
        max_age_hours: float = 48.0,
        decay_mode: str = "linear",
        weight_floor: float = 0.3,
        use_confidence_weighting: bool = False,
    ):
        self.max_age_hours = max_age_hours
        self.decay_mode = decay_mode.lower()
        self.weight_floor = weight_floor
        self.use_confidence_weighting = use_confidence_weighting

        # Validate decay mode
        if self.decay_mode not in {"linear", "exponential", "none"}:
            raise ValueError(
                f"decay_mode must be 'linear', 'exponential', or 'none', got '{decay_mode}'"
            )

    def _recency_weight(self, timestamp_str: str) -> float:
        """Return a weight based on how recent the embedding is.

        Weight ranges from 1.0 (just captured) to weight_floor (at max_age_hours).
        Decay behavior depends on decay_mode:
        - 'linear': Uniform linear decay from 1.0 to floor
        - 'exponential': Fast initial decay, slower later (half-life at max_age/3)
        - 'none': Constant weight of 1.0 for all embeddings

        Args:
            timestamp_str: ISO format timestamp string

        Returns:
            Weight value between weight_floor and 1.0
        """
        if self.decay_mode == "none":
            return 1.0

        try:
            age_hours = (
                datetime.now() - datetime.fromisoformat(timestamp_str)
            ).total_seconds() / 3600
        except Exception:
            # If timestamp parsing fails, return mid-range weight
            return (1.0 + self.weight_floor) / 2

        # Clamp age to valid range
        if age_hours <= 0:
            return 1.0
        if age_hours >= self.max_age_hours:
            return self.weight_floor

        # Calculate weight based on decay mode
        if self.decay_mode == "linear":
            # Linear decay from 1.0 to weight_floor over max_age_hours
            weight = 1.0 - (1.0 - self.weight_floor) * (age_hours / self.max_age_hours)
        else:  # exponential
            # Exponential decay: half-life at max_age_hours / 3
            half_life = self.max_age_hours / 3
            decay_factor = math.exp(-math.log(2) * age_hours / half_life)
            # Scale to range [weight_floor, 1.0]
            weight = self.weight_floor + (1.0 - self.weight_floor) * decay_factor

        return max(self.weight_floor, min(1.0, weight))

    def _confidence_weight(self, confidence: float) -> float:
        """Convert detection confidence (0.0-1.0) to a quality multiplier.

        Maps confidence to range [0.7, 1.0] so high-confidence embeddings
        have slightly more weight.

        Args:
            confidence: Detection confidence score (0.0-1.0)

        Returns:
            Quality weight between 0.7 and 1.0
        """
        if not self.use_confidence_weighting:
            return 1.0
        # Scale from [0, 1] to [0.7, 1.0]
        return 0.7 + (0.3 * max(0.0, min(1.0, confidence)))

    def _best_similarity_for_person(
        self,
        query_embedding: Union[List, Any],
        embeddings_data: Union[
            List[Tuple],  # new: [(emb, camera, conf, ts), ...]
            Tuple,  # old: (emb, camera, conf)
        ],
    ) -> float:
        """Return the best weighted similarity for a single person.

        Applies recency weighting and optionally confidence weighting to each
        stored embedding, returning the best weighted similarity score.

        Args:
            query_embedding: Feature vector from query image
            embeddings_data: Either list of (emb, camera, conf, ts) tuples or
                           legacy (emb, camera, conf) tuple

        Returns:
            Best weighted similarity score (0.0-1.0)
        """
        if cosine is None:
            return 0.5

        # Old format: bare tuple with 3 elements (no timestamp, no recency weighting)
        if isinstance(embeddings_data, tuple) and len(embeddings_data) == 3:
            stored_embedding, _, confidence = embeddings_data
            base_similarity = 1.0 - cosine(query_embedding, stored_embedding)
            conf_weight = self._confidence_weight(confidence)
            return base_similarity * conf_weight

        # New format: list of tuples
        if not isinstance(embeddings_data, list) or not embeddings_data:
            return 0.0

        best = 0.0
        for item in embeddings_data:
            if len(item) == 4:
                stored_embedding, _, confidence, timestamp = item
                recency_w = self._recency_weight(timestamp)
                conf_w = self._confidence_weight(confidence)
                combined_weight = recency_w * conf_w
            elif len(item) == 3:
                stored_embedding, _, confidence = item
                # No timestamp: use mid-range recency weight
                recency_w = (1.0 + self.weight_floor) / 2
                conf_w = self._confidence_weight(confidence)
                combined_weight = recency_w * conf_w
            else:
                continue

            base_similarity = 1.0 - cosine(query_embedding, stored_embedding)
            weighted = base_similarity * combined_weight

            if weighted > best:
                best = weighted
        return best

    def find_best_match(
        self,
        query_embedding: Union[List, Any],
        stored_embeddings: Dict[str, Any],
        threshold: float = 0.6,
    ) -> Tuple[Optional[str], float]:
        """Find the best matching person from stored embeddings.

        Accepts both old format (person_id → (emb, camera, conf)) and new
        format (person_id → [(emb, camera, conf, timestamp), ...]).
        Recent embeddings receive higher effective similarity scores based on
        configured decay mode. Optionally incorporates detection confidence.

        Args:
            query_embedding: Feature vector from query image (list or numpy array)
            stored_embeddings: Dict mapping person_id to embeddings
            threshold: Minimum weighted similarity to consider a match (0-1)

        Returns:
            Tuple of (matched_person_id, similarity_score) or (None, best_score) if no match
        """
        if not stored_embeddings:
            return None, 0.0

        if cosine is None:
            return list(stored_embeddings.keys())[0], 0.5

        best_person = None
        best_similarity = 0.0

        for person_id, embeddings_data in stored_embeddings.items():
            similarity = self._best_similarity_for_person(
                query_embedding, embeddings_data
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_person = person_id

        if best_similarity >= threshold:
            return best_person, best_similarity
        return None, best_similarity

    def find_top_k_matches(
        self,
        query_embedding: Union[List, Any],
        stored_embeddings: Dict[str, Any],
        k: int = 5,
        threshold: float = 0.0,
    ) -> list:
        """Find top-k matching persons with recency and confidence weighting.

        Args:
            query_embedding: Feature vector from query image
            stored_embeddings: Dict mapping person_id to embeddings
            k: Number of top matches to return
            threshold: Minimum weighted similarity to include in results

        Returns:
            List of (person_id, similarity_score) tuples, sorted by similarity descending
        """
        if not stored_embeddings or cosine is None:
            return []

        similarities = []
        for person_id, embeddings_data in stored_embeddings.items():
            similarity = self._best_similarity_for_person(
                query_embedding, embeddings_data
            )
            if similarity >= threshold:
                similarities.append((person_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def batch_match(
        self,
        query_embeddings: Union[List[List], Any],
        stored_embeddings: Dict[str, Any],
        threshold: float = 0.6,
    ) -> list:
        """Match multiple query embeddings against stored embeddings.

        Args:
            query_embeddings: Array of feature vectors (shape: (n, embedding_dim))
            stored_embeddings: Dict mapping person_id to embeddings
            threshold: Minimum weighted similarity to consider a match

        Returns:
            List of (matched_person_id, similarity_score) tuples, one per query
        """
        return [
            self.find_best_match(qe, stored_embeddings, threshold)
            for qe in query_embeddings
        ]
