try:
    from scipy.spatial.distance import cosine
except ImportError:
    cosine = None

from typing import Dict, Tuple, Optional, List, Any, Union


class EmbeddingMatcher:
    """Matcher for finding best embedding matches using cosine similarity."""

    @staticmethod
    def find_best_match(
        query_embedding: Union[List, Any],
        stored_embeddings: Dict[str, Tuple[Union[List, Any], str, float]],
        threshold: float = 0.6,
    ) -> Tuple[Optional[str], float]:
        """Find the best matching person from stored embeddings.

        Args:
            query_embedding: Feature vector from query image (list or numpy array)
            stored_embeddings: Dict mapping person_id to (embedding, camera, confidence)
            threshold: Minimum cosine similarity to consider a match (0-1)
                      Note: cosine similarity = 1 - cosine distance

        Returns:
            Tuple of (matched_person_id, similarity_score) or (None, 0.0) if no match
        """
        if not stored_embeddings:
            return None, 0.0

        if cosine is None:
            # If scipy not available, just return first person (no actual matching)
            person_id = list(stored_embeddings.keys())[0]
            return person_id, 0.5

        best_person = None
        best_similarity = 0.0

        for person_id, (
            stored_embedding,
            camera,
            confidence,
        ) in stored_embeddings.items():
            # Compute cosine distance (0 = identical, 1 = orthogonal)
            distance = cosine(query_embedding, stored_embedding)
            # Convert to similarity (1 - distance)
            similarity = 1.0 - distance

            if similarity > best_similarity:
                best_similarity = similarity
                best_person = person_id

        # Only return match if above threshold
        if best_similarity >= threshold:
            return best_person, best_similarity
        else:
            return None, best_similarity

    @staticmethod
    def find_top_k_matches(
        query_embedding: Union[List, Any],
        stored_embeddings: Dict[str, Tuple[Union[List, Any], str, float]],
        k: int = 5,
        threshold: float = 0.0,
    ) -> list:
        """Find top-k matching persons.

        Args:
            query_embedding: Feature vector from query image
            stored_embeddings: Dict mapping person_id to (embedding, camera, confidence)
            k: Number of top matches to return
            threshold: Minimum similarity to include in results

        Returns:
            List of (person_id, similarity_score) tuples, sorted by similarity descending
        """
        if not stored_embeddings or cosine is None:
            return []

        similarities = []

        for person_id, (
            stored_embedding,
            camera,
            confidence,
        ) in stored_embeddings.items():
            distance = cosine(query_embedding, stored_embedding)
            similarity = 1.0 - distance

            if similarity >= threshold:
                similarities.append((person_id, similarity))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:k]

    @staticmethod
    def batch_match(
        query_embeddings: Union[List[List], Any],
        stored_embeddings: Dict[str, Tuple[Union[List, Any], str, float]],
        threshold: float = 0.6,
    ) -> list:
        """Match multiple query embeddings against stored embeddings.

        Args:
            query_embeddings: Array of feature vectors (shape: (n, embedding_dim))
            stored_embeddings: Dict mapping person_id to (embedding, camera, confidence)
            threshold: Minimum cosine similarity to consider a match

        Returns:
            List of (matched_person_id, similarity_score) tuples, one per query
        """
        matches = []

        for query_embedding in query_embeddings:
            person_id, similarity = EmbeddingMatcher.find_best_match(
                query_embedding, stored_embeddings, threshold
            )
            matches.append((person_id, similarity))

        return matches
