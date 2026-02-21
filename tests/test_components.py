"""
Unit tests for the ReID system components.

Run with: python -m pytest tests/ -v

Install pytest first:
  pip install pytest pytest-cov
"""

import pytest
import base64
import numpy as np
from PIL import Image
import io
import tempfile
import os


@pytest.fixture
def sample_image():
    """Create a simple test image (100x50 RGB)."""
    img = Image.new("RGB", (100, 50), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return base64.b64encode(img_byte_arr.read()).decode("utf-8")


@pytest.fixture
def sample_image_blue():
    """Create a different test image (100x50 RGB, blue)."""
    img = Image.new("RGB", (100, 50), color="blue")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return base64.b64encode(img_byte_arr.read()).decode("utf-8")


@pytest.fixture
def temp_db():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestEmbeddingStore:
    """Tests for the EmbeddingStore module."""

    def test_embedding_store_initialization(self, temp_db):
        """Test that embedding store initializes correctly."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        assert store.db_path == temp_db
        assert len(store.embeddings) == 0

    def test_store_and_retrieve_embedding(self, temp_db):
        """Test storing and retrieving an embedding."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)

        embedding = np.random.rand(256)
        store.store_embedding("person1", embedding, "camera1", confidence=0.95)

        assert store.person_exists("person1")
        retrieved = store.get_embedding("person1")
        assert retrieved is not None
        assert len(retrieved) == 256

    def test_get_all_embeddings(self, temp_db):
        """Test retrieving all embeddings."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)

        emb1 = np.random.rand(256)
        emb2 = np.random.rand(256)
        store.store_embedding("person1", emb1, "camera1")
        store.store_embedding("person2", emb2, "camera2")

        all_embs = store.get_all_embeddings()
        assert len(all_embs) == 2
        assert "person1" in all_embs
        assert "person2" in all_embs

    def test_delete_person(self, temp_db):
        """Test deleting a person."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)

        embedding = np.random.rand(256)
        store.store_embedding("person1", embedding, "camera1")
        assert store.person_exists("person1")

        store.delete_person("person1")
        assert not store.person_exists("person1")

    def test_persistence(self, temp_db):
        """Test that embeddings persist across instances."""
        from embedding_store import EmbeddingStore

        store1 = EmbeddingStore(temp_db)
        embedding = np.random.rand(256)
        store1.store_embedding("person1", embedding, "camera1")

        store2 = EmbeddingStore(temp_db)
        assert store2.person_exists("person1")
        retrieved = store2.get_embedding("person1")
        assert len(retrieved) == 256


class TestMatcher:
    """Tests for the EmbeddingMatcher module."""

    def test_find_best_match_exact(self):
        """Test finding exact match with identical embeddings."""
        from matcher import EmbeddingMatcher

        query = np.array([1.0, 0.0, 0.0])
        stored = {
            "person1": (np.array([1.0, 0.0, 0.0]), "camera1", 0.9),
            "person2": (np.array([0.0, 1.0, 0.0]), "camera2", 0.8),
        }

        matched, score = EmbeddingMatcher.find_best_match(query, stored, threshold=0.9)
        assert matched == "person1"
        assert score >= 0.99

    def test_find_best_match_below_threshold(self):
        """Test that no match is returned below threshold."""
        from matcher import EmbeddingMatcher

        query = np.array([1.0, 0.0, 0.0])
        stored = {
            "person1": (np.array([0.0, 1.0, 0.0]), "camera1", 0.9),
        }

        matched, score = EmbeddingMatcher.find_best_match(query, stored, threshold=0.9)
        assert matched is None

    def test_find_best_match_empty_store(self):
        """Test matching against empty embedding store."""
        from matcher import EmbeddingMatcher

        query = np.array([1.0, 0.0, 0.0])
        stored = {}

        matched, score = EmbeddingMatcher.find_best_match(query, stored, threshold=0.5)
        assert matched is None
        assert score == 0.0

    def test_find_top_k_matches(self):
        """Test finding top-k matches."""
        from matcher import EmbeddingMatcher

        query = np.array([1.0, 0.0, 0.0])
        stored = {
            "person1": (np.array([1.0, 0.0, 0.0]), "camera1", 0.9),
            "person2": (np.array([0.9, 0.1, 0.0]), "camera2", 0.8),
            "person3": (np.array([0.0, 1.0, 0.0]), "camera3", 0.7),
        }

        top_matches = EmbeddingMatcher.find_top_k_matches(
            query, stored, k=2, threshold=0.0
        )
        assert len(top_matches) <= 2
        assert top_matches[0][1] >= top_matches[1][1]


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_embedding_storage_and_matching(self, temp_db):
        """Test complete workflow: store embeddings and match."""
        from embedding_store import EmbeddingStore
        from matcher import EmbeddingMatcher

        store = EmbeddingStore(temp_db)

        # Create and store embeddings
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])
        store.store_embedding("alice", emb1, "camera1", confidence=0.95)
        store.store_embedding("bob", emb2, "camera2", confidence=0.90)

        # Test matching
        stored = store.get_all_embeddings()
        query = np.array([1.0, 0.0, 0.0])

        matched, score = EmbeddingMatcher.find_best_match(query, stored, threshold=0.5)
        assert matched == "alice"
        assert score > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
