"""
Lightweight unit tests for ReID core modules (no numpy required).
Run with: python -m pytest tests/test_lite.py -v
"""

import json
import pytest
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
    """Test EmbeddingStore without requiring numpy."""
    
    def test_initialization(self, temp_db):
        """Test embedding store initializes correctly."""
        from embedding_store import EmbeddingStore
        store = EmbeddingStore(temp_db)
        assert store.db_path == temp_db
        assert len(store.embeddings) == 0
        assert store.get_all_person_ids() == []
    
    def test_store_and_retrieve(self, temp_db):
        """Test storing and retrieving embeddings."""
        from embedding_store import EmbeddingStore
        
        store = EmbeddingStore(temp_db)
        
        # Use a simple list instead of numpy array
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        store.store_embedding("person1", embedding, "camera1", confidence=0.95)
        
        assert store.person_exists("person1")
        
    def test_persistence(self, temp_db):
        """Test that embeddings persist across instances."""
        from embedding_store import EmbeddingStore
        
        # First instance: store data
        store1 = EmbeddingStore(temp_db)
        embedding = [0.1, 0.2, 0.3]
        store1.store_embedding("person1", embedding, "camera1", confidence=0.95)
        assert store1.person_exists("person1")
        
        # Second instance: load data
        store2 = EmbeddingStore(temp_db)
        assert store2.person_exists("person1")
        
    def test_delete_person(self, temp_db):
        """Test deleting a person."""
        from embedding_store import EmbeddingStore
        
        store = EmbeddingStore(temp_db)
        embedding = [0.1, 0.2]
        store.store_embedding("person1", embedding, "camera1")
        
        assert store.person_exists("person1")
        result = store.delete_person("person1")
        assert result is True
        assert not store.person_exists("person1")
    
    def test_clear(self, temp_db):
        """Test clearing the store."""
        from embedding_store import EmbeddingStore
        
        store = EmbeddingStore(temp_db)
        store.store_embedding("person1", [0.1], "camera1")
        store.store_embedding("person2", [0.2], "camera2")
        
        store.clear()
        assert len(store.get_all_person_ids()) == 0


class TestMatcher:
    """Test Matcher without requiring numpy."""
    
    def test_import(self):
        """Test that matcher module can be imported."""
        from matcher import EmbeddingMatcher
        assert EmbeddingMatcher is not None
    
    def test_empty_store(self):
        """Test matching against empty store."""
        from matcher import EmbeddingMatcher
        
        query = [1.0, 0.0, 0.0]
        stored = {}
        
        matched, score = EmbeddingMatcher.find_best_match(query, stored)
        assert matched is None
        assert score == 0.0


class TestServices:
    """Test that main service modules can be imported."""
    
    def test_import_identity_service(self):
        """Test that identity_service can be parsed."""
        with open("identity_service.py") as f:
            code = f.read()
            assert "on_connect" in code
            assert "on_message" in code
            assert "EmbeddingStore" in code
            assert "EmbeddingMatcher" in code
    
    def test_import_embedding_store(self):
        """Test EmbeddingStore module."""
        from embedding_store import EmbeddingStore
        assert hasattr(EmbeddingStore, "store_embedding")
        assert hasattr(EmbeddingStore, "get_embedding")
        assert hasattr(EmbeddingStore, "person_exists")
    
    def test_import_matcher(self):
        """Test Matcher module."""
        from matcher import EmbeddingMatcher
        assert hasattr(EmbeddingMatcher, "find_best_match")
        assert hasattr(EmbeddingMatcher, "find_top_k_matches")
    
    def test_import_reid_model(self):
        """Test ReIDModel module structure (requires torch - skipped if not available)."""
        try:
            import torch
            from reid_model import ReIDModel
            model_class = ReIDModel
            assert model_class is not None
        except ModuleNotFoundError as e:
            pytest.skip(f"torch not available: {e}")


class TestLoadHaOptions:
    """Tests for the load_ha_options function used by the Home Assistant Add-on."""

    def _get_load_ha_options(self):
        """Import load_ha_options from identity_service without executing the module."""
        import ast
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "identity_service.py",
        )
        with open(src_path) as f:
            source = f.read()

        # Extract and compile just the load_ha_options function
        tree = ast.parse(source)
        func_def = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "load_ha_options"
        )
        module = ast.Module(body=[func_def], type_ignores=[])
        code = compile(module, src_path, "exec")
        ns = {"os": os, "Path": __import__("pathlib").Path, "json": json}
        exec(code, ns)
        return ns["load_ha_options"]

    def test_no_options_file(self, monkeypatch, tmp_path):
        """load_ha_options does nothing when the file is absent."""
        load_ha_options = self._get_load_ha_options()
        monkeypatch.delenv("MQTT_BROKER", raising=False)
        load_ha_options(options_file=str(tmp_path / "options.json"))
        assert "MQTT_BROKER" not in os.environ

    def test_options_loaded_to_env(self, monkeypatch, tmp_path):
        """Options from the JSON file are exported as upper-case env vars."""
        load_ha_options = self._get_load_ha_options()
        monkeypatch.delenv("MQTT_BROKER", raising=False)
        monkeypatch.delenv("MQTT_PORT", raising=False)

        options = {"mqtt_broker": "my-broker", "mqtt_port": 1884}
        options_file = tmp_path / "options.json"
        options_file.write_text(json.dumps(options))

        load_ha_options(options_file=str(options_file))

        assert os.environ["MQTT_BROKER"] == "my-broker"
        assert os.environ["MQTT_PORT"] == "1884"

    def test_existing_env_not_overwritten(self, monkeypatch, tmp_path):
        """Pre-existing environment variables are not overwritten by the options file."""
        load_ha_options = self._get_load_ha_options()
        monkeypatch.setenv("MQTT_BROKER", "original-broker")

        options = {"mqtt_broker": "new-broker"}
        options_file = tmp_path / "options.json"
        options_file.write_text(json.dumps(options))

        load_ha_options(options_file=str(options_file))

        assert os.environ["MQTT_BROKER"] == "original-broker"

    def test_empty_string_values_skipped(self, monkeypatch, tmp_path):
        """Empty string option values are not exported."""
        load_ha_options = self._get_load_ha_options()
        monkeypatch.delenv("MQTT_USERNAME", raising=False)

        options = {"mqtt_username": ""}
        options_file = tmp_path / "options.json"
        options_file.write_text(json.dumps(options))

        load_ha_options(options_file=str(options_file))

        assert "MQTT_USERNAME" not in os.environ

    def test_none_values_skipped(self, monkeypatch, tmp_path):
        """None option values are not exported."""
        load_ha_options = self._get_load_ha_options()
        monkeypatch.delenv("MQTT_PASSWORD", raising=False)

        options = {"mqtt_password": None}
        options_file = tmp_path / "options.json"
        options_file.write_text(json.dumps(options))

        load_ha_options(options_file=str(options_file))

        assert "MQTT_PASSWORD" not in os.environ

    def test_invalid_json_handled_gracefully(self, tmp_path):
        """Malformed JSON in the options file does not raise an exception."""
        load_ha_options = self._get_load_ha_options()
        options_file = tmp_path / "options.json"
        options_file.write_text("not valid json {{{")

        # Should not raise
        load_ha_options(options_file=str(options_file))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
