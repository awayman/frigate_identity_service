"""Unit tests for ReID system."""

import pytest
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_db():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestEmbeddingStore:
    """Test EmbeddingStore module."""

    def test_initialization(self, temp_db):
        """Test embedding store initialization."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        assert store.db_path == temp_db
        assert len(store.embeddings) == 0

    def test_store_and_retrieve(self, temp_db):
        """Test storing and retrieving embeddings."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)

        embedding = np.random.rand(256)
        store.store_embedding("person1", embedding, "camera1", confidence=0.95)

        assert store.person_exists("person1")
        retrieved = store.get_embedding("person1")
        assert retrieved is not None
        assert len(retrieved) == 256


class TestMatcher:
    """Test EmbeddingMatcher module."""

    def test_find_best_match(self):
        """Test finding best matching embedding."""
        from matcher import EmbeddingMatcher

        matcher = EmbeddingMatcher()
        query = np.array([1.0, 0.0, 0.0])
        stored = {
            "person1": (np.array([1.0, 0.0, 0.0]), "camera1", 0.9),
            "person2": (np.array([0.0, 1.0, 0.0]), "camera2", 0.8),
        }

        matched, score = matcher.find_best_match(query, stored, threshold=0.9)
        assert matched == "person1"
        assert score > 0.99


class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self, temp_db):
        """Test complete workflow of storing and matching embeddings."""
        from embedding_store import EmbeddingStore
        from matcher import EmbeddingMatcher

        # Store embedding
        store = EmbeddingStore(temp_db)
        matcher = EmbeddingMatcher()
        emb = np.array([1.0, 0.0, 0.0])
        store.store_embedding("alice", emb, "camera1", confidence=0.95)

        # Retrieve and match
        stored = store.get_all_embeddings()
        matched, score = matcher.find_best_match(emb, stored, threshold=0.5)

        assert matched == "alice"
        assert score > 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
