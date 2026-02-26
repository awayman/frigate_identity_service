"""
Lightweight unit tests for ReID core modules (no numpy required).
Run with: python -m pytest tests/test_lite.py -v
"""

import json
import logging
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
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "frigate_identity_service",
            "identity_service.py",
        )
        with open(src_path) as f:
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
            import torch  # noqa: F401
            from reid_model import ReIDModel

            model_class = ReIDModel
            assert model_class is not None
        except ModuleNotFoundError as e:
            pytest.skip(f"torch not available: {e}")


class TestReIDModelSelection:
    """Tests for configurable model selection in reid_model.py."""

    def test_torchreid_models_defined(self):
        """TORCHREID_MODELS set is defined and contains expected models."""
        from reid_model import TORCHREID_MODELS

        assert isinstance(TORCHREID_MODELS, set)
        assert "osnet_x1_0" in TORCHREID_MODELS
        assert "osnet_x0_25" in TORCHREID_MODELS
        assert "osnet_ibn_x1_0" in TORCHREID_MODELS

    def test_model_name_parameter_in_init(self):
        """ReIDModel.__init__ accepts a model_name parameter."""
        import inspect
        from reid_model import ReIDModel

        sig = inspect.signature(ReIDModel.__init__)
        assert "model_name" in sig.parameters
        assert sig.parameters["model_name"].default == "osnet_x1_0"

    def test_torchreid_availability_flag(self):
        """TORCHREID_AVAILABLE flag is defined as a boolean."""
        from reid_model import TORCHREID_AVAILABLE

        assert isinstance(TORCHREID_AVAILABLE, bool)

    def test_resnet50_fallback_exists(self):
        """The ResNet50 fallback method is present on ReIDModel."""
        from reid_model import ReIDModel

        assert hasattr(ReIDModel, "_load_resnet50_fallback")

    def test_identity_service_reads_reid_model_env(self):
        """identity_service.py reads the REID_MODEL environment variable."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "frigate_identity_service",
            "identity_service.py",
        )
        with open(src_path) as f:
            source = f.read()
        assert "REID_MODEL" in source
        assert 'os.getenv("REID_MODEL"' in source
        assert "model_name=" in source


class TestLoadHaOptions:
    """Tests for the load_ha_options function used by the Home Assistant Add-on."""

    def _get_load_ha_options(self):
        """Import load_ha_options from identity_service without executing the module."""
        import ast

        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "frigate_identity_service",
            "identity_service.py",
        )
        with open(src_path) as f:
            source = f.read()

        # Extract and compile just the load_ha_options function
        tree = ast.parse(source)
        func_def = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "load_ha_options"
        )
        module = ast.Module(body=[func_def], type_ignores=[])
        code = compile(module, src_path, "exec")
        ns = {
            "os": os,
            "Path": __import__("pathlib").Path,
            "json": json,
            "logger": logging.getLogger("test"),
        }
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

    def test_core_mosquitto_applied_without_preset_env(self, monkeypatch, tmp_path):
        """core-mosquitto is applied when MQTT_BROKER is not pre-set by Docker ENV.

        Regression test: the Dockerfile must NOT set ENV MQTT_BROKER=localhost
        because that would prevent load_ha_options from applying the user's
        mqtt_broker = core-mosquitto option.
        """
        load_ha_options = self._get_load_ha_options()
        monkeypatch.delenv("MQTT_BROKER", raising=False)

        options = {"mqtt_broker": "core-mosquitto"}
        options_file = tmp_path / "options.json"
        options_file.write_text(json.dumps(options))

        load_ha_options(options_file=str(options_file))

        assert os.environ["MQTT_BROKER"] == "core-mosquitto"


class TestConnectWithRetry:
    """Tests for the connect_with_retry function."""

    def _get_connect_with_retry(self):
        """Extract connect_with_retry from identity_service without executing the module."""
        import ast

        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "frigate_identity_service",
            "identity_service.py",
        )
        with open(src_path) as f:
            source = f.read()

        tree = ast.parse(source)
        func_def = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "connect_with_retry"
        )
        module = ast.Module(body=[func_def], type_ignores=[])
        code = compile(module, src_path, "exec")
        ns = {"os": os, "time": __import__("time"), "logger": logging.getLogger("test")}
        exec(code, ns)
        return ns["connect_with_retry"]

    def test_succeeds_on_first_attempt(self):
        """connect_with_retry returns True when connect succeeds immediately."""
        connect_with_retry = self._get_connect_with_retry()

        class FakeClient:
            def connect(self, broker, port, keepalive):
                pass

        assert (
            connect_with_retry(
                FakeClient(), "localhost", 1883, max_attempts=3, retry_delay=0
            )
            is True
        )

    def test_retries_then_succeeds(self):
        """connect_with_retry retries after failures and returns True on eventual success."""
        connect_with_retry = self._get_connect_with_retry()

        class FakeClient:
            def __init__(self):
                self.attempts = 0

            def connect(self, broker, port, keepalive):
                self.attempts += 1
                if self.attempts < 3:
                    raise OSError("[Errno 111] Connection refused")

        client = FakeClient()
        assert (
            connect_with_retry(client, "localhost", 1883, max_attempts=5, retry_delay=0)
            is True
        )
        assert client.attempts == 3

    def test_returns_false_after_all_retries_exhausted(self):
        """connect_with_retry returns False when all attempts fail."""
        connect_with_retry = self._get_connect_with_retry()

        class FakeClient:
            def __init__(self):
                self.attempts = 0

            def connect(self, broker, port, keepalive):
                self.attempts += 1
                raise OSError("[Errno 111] Connection refused")

        client = FakeClient()
        assert (
            connect_with_retry(client, "localhost", 1883, max_attempts=3, retry_delay=0)
            is False
        )
        assert client.attempts == 3

    def test_no_retries_when_max_attempts_is_zero(self):
        """connect_with_retry makes no attempts when max_attempts=0."""
        connect_with_retry = self._get_connect_with_retry()

        class FakeClient:
            def connect(self, broker, port, keepalive):
                raise OSError("[Errno 111] Connection refused")

        assert (
            connect_with_retry(
                FakeClient(), "localhost", 1883, max_attempts=0, retry_delay=0
            )
            is False
        )


class TestDebugLoggerInit:
    """Tests for DebugLogger initialization fix (no dir creation when disabled)."""

    def test_disabled_does_not_create_dirs(self, tmp_path):
        """DebugLogger with enabled=False must not create debug directories."""
        from debug_logger import DebugLogger

        non_existent = tmp_path / "nonexistent" / "debug"
        DebugLogger(debug_path=str(non_existent), enabled=False)

        assert not non_existent.exists(), "Directories must not be created when disabled"

    def test_enabled_creates_dirs(self, tmp_path):
        """DebugLogger with enabled=True must create debug directories."""
        from debug_logger import DebugLogger

        debug_path = tmp_path / "debug"
        DebugLogger(debug_path=str(debug_path), enabled=True)

        assert (debug_path / "snapshots").is_dir()
        assert (debug_path / "logs").is_dir()

    def test_set_enabled_creates_dirs(self, tmp_path):
        """Calling set_enabled(True) on a disabled logger must create directories."""
        from debug_logger import DebugLogger

        debug_path = tmp_path / "debug"
        dl = DebugLogger(debug_path=str(debug_path), enabled=False)
        assert not debug_path.exists()

        dl.set_enabled(True)
        assert (debug_path / "snapshots").is_dir()
        assert (debug_path / "logs").is_dir()

    def test_state_file_enabled_creates_dirs(self, tmp_path):
        """When the state file says 'true', directories should be created on init."""
        from debug_logger import DebugLogger

        debug_path = tmp_path / "debug"
        debug_path.mkdir()
        (debug_path / "enabled").write_text("true")

        dl = DebugLogger(debug_path=str(debug_path), enabled=False)
        assert dl.enabled is True
        assert (debug_path / "snapshots").is_dir()
        assert (debug_path / "logs").is_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
