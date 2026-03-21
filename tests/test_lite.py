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
from collections import defaultdict, deque
from datetime import datetime, timedelta
from types import SimpleNamespace

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

    def test_prune_expired_embeddings(self, temp_db):
        """Test age-based pruning removes only expired entries."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        now = datetime.now()
        store.embeddings = {
            "old_person": [
                {
                    "embedding": [0.1, 0.2],
                    "camera": "camera1",
                    "timestamp": (now - timedelta(hours=72)).isoformat(),
                    "confidence": 0.8,
                }
            ],
            "new_person": [
                {
                    "embedding": [0.3, 0.4],
                    "camera": "camera2",
                    "timestamp": (now - timedelta(hours=2)).isoformat(),
                    "confidence": 0.9,
                }
            ],
        }
        store._save()

        stats = store.prune_expired(max_age_hours=24)

        assert stats["removed_embeddings"] == 1
        assert stats["removed_persons"] == 1
        assert store.person_exists("new_person")
        assert not store.person_exists("old_person")

    def test_prune_keeps_invalid_timestamps(self, temp_db):
        """Invalid timestamp entries are preserved to avoid accidental data loss."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        store.embeddings = {
            "person1": [
                {
                    "embedding": [0.1],
                    "camera": "camera1",
                    "timestamp": "not-a-timestamp",
                    "confidence": 0.7,
                }
            ]
        }
        store._save()

        stats = store.prune_expired(max_age_hours=1)

        assert stats["removed_embeddings"] == 0
        assert store.person_exists("person1")


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

        matcher = EmbeddingMatcher()
        matched, score = matcher.find_best_match(query, stored)
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

    def test_identity_service_reads_embedding_retention_env(self):
        """identity_service.py reads embedding retention environment variables."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "frigate_identity_service",
            "identity_service.py",
        )
        with open(src_path) as f:
            source = f.read()

        assert "EMBEDDING_RETENTION_MODE" in source
        assert 'os.getenv("EMBEDDING_MAX_AGE_HOURS"' in source
        assert 'os.getenv("EMBEDDING_PRUNE_INTERVAL_MINUTES"' in source
        assert 'os.getenv("EMBEDDING_FULL_CLEAR_TIME"' in source


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

    def test_mqtt_host_alias_maps_to_mqtt_broker(self, monkeypatch, tmp_path):
        """mqtt_host option is accepted as an alias for MQTT_BROKER."""
        load_ha_options = self._get_load_ha_options()
        monkeypatch.delenv("MQTT_BROKER", raising=False)

        options = {"mqtt_host": "ha-mosquitto"}
        options_file = tmp_path / "options.json"
        options_file.write_text(json.dumps(options))

        load_ha_options(options_file=str(options_file))

        assert os.environ["MQTT_BROKER"] == "ha-mosquitto"

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

        assert not non_existent.exists(), (
            "Directories must not be created when disabled"
        )

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


class TestIdentityLifecycleHandlers:
    """Tests for event lifecycle-specific identity handling."""

    def _load_handler_functions(self):
        """Compile selected identity_service handlers without importing the module."""
        import ast
        import time
        import traceback

        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "frigate_identity_service",
            "identity_service.py",
        )
        with open(src_path) as f:
            source = f.read()

        tree = ast.parse(source)
        wanted = {
            "_cache_recognized_person_event",
            "_store_completed_face_embedding",
            "handle_frigate_event",
            "handle_tracked_object_update",
        }
        func_defs: list[ast.stmt] = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        module = ast.Module(body=func_defs, type_ignores=[])
        code = compile(module, src_path, "exec")

        ns = {
            "json": json,
            "time": time,
            "traceback": traceback,
            "logger": logging.getLogger("test_identity_lifecycle"),
            "recognized_person_events": {},
            "camera_person_queue": defaultdict(lambda: deque(maxlen=3)),
            "REID_AVAILABLE": True,
            "REID_SIMILARITY_THRESHOLD": 0.75,
            "MIN_PERSON_DETECTION_CONFIDENCE": 0.80,
        }
        exec(code, ns)
        return ns

    def test_active_face_recognition_publishes_without_learning(self):
        """Live recognized events should publish immediately but defer embedding storage."""
        ns = self._load_handler_functions()
        fetch_calls = []
        stored_embeddings = []
        published = []

        ns["fetch_snapshot_from_api"] = lambda *args, **kwargs: fetch_calls.append(
            (args, kwargs)
        )
        ns["embedding_store"] = SimpleNamespace(
            store_embedding=lambda *args: stored_embeddings.append(args),
            get_all_embeddings=lambda: {},
        )
        ns["reid_model"] = SimpleNamespace(
            extract_embedding=lambda snapshot: [0.1],
            extract_embedding_from_pil=lambda image: [0.1],
        )
        ns["debug_logger"] = SimpleNamespace(
            log_facial_recognition=lambda **kwargs: None,
            log_reid_match=lambda **kwargs: None,
            log_reid_no_match=lambda **kwargs: None,
        )
        ns["embedding_matcher"] = SimpleNamespace(
            find_best_match=lambda *args, **kwargs: (None, 0.0),
            find_top_k_matches=lambda *args, **kwargs: [],
        )
        ns["publish_identity_event"] = lambda *args: published.append(args)

        payload = {
            "type": "update",
            "after": {
                "id": "evt-1",
                "camera": "front",
                "label": "person",
                "sub_label": ["Alice", 0.96],
                "current_zones": ["porch"],
                "top_score": 0.91,
                "frame_time": 1234.5,
            },
        }
        msg = SimpleNamespace(
            topic="frigate/events",
            payload=json.dumps(payload).encode("utf-8"),
        )

        ns["handle_frigate_event"](SimpleNamespace(), msg)

        assert fetch_calls == []
        assert stored_embeddings == []
        assert published and published[0][4] == "facial_recognition"
        assert ns["recognized_person_events"]["evt-1"]["person_id"] == "Alice"
        assert ns["camera_person_queue"]["front"][0]["person_id"] == "Alice"

    def test_completed_event_stores_final_face_embedding(self):
        """Completed person events should learn from the final event snapshot."""
        ns = self._load_handler_functions()
        fetch_calls = []
        stored_embeddings = []
        debug_logs = []
        published = []

        ns["fetch_snapshot_from_api"] = lambda *args, **kwargs: (
            fetch_calls.append((args, kwargs)) or "snapshot-base64"
        )
        ns["embedding_store"] = SimpleNamespace(
            store_embedding=lambda *args: stored_embeddings.append(args),
            get_all_embeddings=lambda: {},
        )
        ns["reid_model"] = SimpleNamespace(
            extract_embedding=lambda snapshot: [0.1, 0.2],
            extract_embedding_from_pil=lambda image: [0.3, 0.4],
        )
        ns["debug_logger"] = SimpleNamespace(
            log_facial_recognition=lambda **kwargs: debug_logs.append(kwargs),
            log_reid_match=lambda **kwargs: None,
            log_reid_no_match=lambda **kwargs: None,
        )
        ns["embedding_matcher"] = SimpleNamespace(
            find_best_match=lambda *args, **kwargs: (None, 0.0),
            find_top_k_matches=lambda *args, **kwargs: [],
        )
        ns["publish_identity_event"] = lambda *args: published.append(args)

        ns["_cache_recognized_person_event"](
            "evt-2", "Alice", "front", 0.97, 5678.9, ["porch"]
        )
        end_payload = {
            "type": "end",
            "after": {
                "id": "evt-2",
                "camera": "front",
                "label": "person",
                "sub_label": None,
                "current_zones": ["porch"],
                "top_score": 0.80,
                "frame_time": 5680.0,
            },
        }
        msg = SimpleNamespace(
            topic="frigate/events",
            payload=json.dumps(end_payload).encode("utf-8"),
        )

        ns["handle_frigate_event"](SimpleNamespace(), msg)

        assert len(fetch_calls) == 1
        assert stored_embeddings == [("Alice", [0.1, 0.2], "front", 0.97)]
        assert debug_logs and debug_logs[0]["person_id"] == "Alice"
        assert published == []
        assert "evt-2" not in ns["recognized_person_events"]

    def test_active_unknown_person_still_runs_reid(self):
        """Unrecognized active events should still fetch snapshots for live ReID."""
        ns = self._load_handler_functions()
        fetch_calls = []
        published = []
        reid_logs = []

        ns["fetch_snapshot_from_api"] = lambda *args, **kwargs: (
            fetch_calls.append((args, kwargs)) or "snapshot-base64"
        )
        ns["embedding_store"] = SimpleNamespace(
            store_embedding=lambda *args: None,
            get_all_embeddings=lambda: {"Bob": ([0.9], "driveway", 0.8)},
        )
        ns["reid_model"] = SimpleNamespace(
            extract_embedding=lambda snapshot: [0.1],
            extract_embedding_from_pil=lambda image: [0.1],
        )
        ns["debug_logger"] = SimpleNamespace(
            log_facial_recognition=lambda **kwargs: None,
            log_reid_match=lambda **kwargs: reid_logs.append(kwargs),
            log_reid_no_match=lambda **kwargs: None,
        )
        ns["embedding_matcher"] = SimpleNamespace(
            find_best_match=lambda *args, **kwargs: ("Bob", 0.88),
            find_top_k_matches=lambda *args, **kwargs: [("Bob", 0.88)],
        )
        ns["publish_identity_event"] = lambda *args: published.append(args)

        payload = {
            "type": "update",
            "after": {
                "id": "evt-3",
                "camera": "driveway",
                "label": "person",
                "sub_label": None,
                "current_zones": ["driveway"],
                "top_score": 0.84,
                "frame_time": 6789.0,
            },
        }
        msg = SimpleNamespace(
            topic="frigate/events",
            payload=json.dumps(payload).encode("utf-8"),
        )

        ns["handle_frigate_event"](SimpleNamespace(), msg)

        assert len(fetch_calls) == 1
        assert published and published[0][4] == "reid_model"
        assert reid_logs and reid_logs[0]["chosen_person_id"] == "Bob"
        assert ns["camera_person_queue"]["driveway"][0]["person_id"] == "Bob"

    def test_face_updates_cache_identity_without_learning_immediately(self):
        """Face updates should publish and cache identity but wait for end-event learning."""
        ns = self._load_handler_functions()
        fetch_calls = []
        stored_embeddings = []
        published = []

        ns["fetch_snapshot_from_api"] = lambda *args, **kwargs: fetch_calls.append(
            (args, kwargs)
        )
        ns["embedding_store"] = SimpleNamespace(
            store_embedding=lambda *args: stored_embeddings.append(args),
            get_all_embeddings=lambda: {},
        )
        ns["reid_model"] = SimpleNamespace(
            extract_embedding=lambda snapshot: [0.1],
            extract_embedding_from_pil=lambda image: [0.1],
        )
        ns["debug_logger"] = SimpleNamespace(
            log_facial_recognition=lambda **kwargs: None,
            log_reid_match=lambda **kwargs: None,
            log_reid_no_match=lambda **kwargs: None,
        )
        ns["publish_identity_event"] = lambda *args: published.append(args)

        payload = {
            "type": "face",
            "id": "evt-4",
            "name": "Carol",
            "score": 0.93,
            "camera": "backyard",
            "timestamp": 7890.1,
        }
        msg = SimpleNamespace(
            topic="frigate/tracked_object_update",
            payload=json.dumps(payload).encode("utf-8"),
        )

        ns["handle_tracked_object_update"](SimpleNamespace(), msg)

        assert fetch_calls == []
        assert stored_embeddings == []
        assert published and published[0][4] == "face_recognition_update"
        assert ns["recognized_person_events"]["evt-4"]["person_id"] == "Carol"
        assert ns["camera_person_queue"]["backyard"][0]["person_id"] == "Carol"

    def test_low_confidence_person_detection_is_ignored(self):
        """Low-confidence person detections should be ignored before ReID/publish."""
        ns = self._load_handler_functions()
        fetch_calls = []
        published = []

        ns["fetch_snapshot_from_api"] = lambda *args, **kwargs: (
            fetch_calls.append((args, kwargs)) or "snapshot-base64"
        )
        ns["embedding_store"] = SimpleNamespace(
            store_embedding=lambda *args: None,
            get_all_embeddings=lambda: {"Bob": ([0.9], "driveway", 0.8)},
        )
        ns["reid_model"] = SimpleNamespace(
            extract_embedding=lambda snapshot: [0.1],
            extract_embedding_from_pil=lambda image: [0.1],
        )
        ns["debug_logger"] = SimpleNamespace(
            log_facial_recognition=lambda **kwargs: None,
            log_reid_match=lambda **kwargs: None,
            log_reid_no_match=lambda **kwargs: None,
        )
        ns["embedding_matcher"] = SimpleNamespace(
            find_best_match=lambda *args, **kwargs: ("Bob", 0.88),
            find_top_k_matches=lambda *args, **kwargs: [("Bob", 0.88)],
        )
        ns["publish_identity_event"] = lambda *args: published.append(args)

        payload = {
            "type": "update",
            "after": {
                "id": "evt-low-1",
                "camera": "driveway",
                "label": "person",
                "sub_label": None,
                "current_zones": ["driveway"],
                "top_score": 0.78,
                "frame_time": 6789.0,
            },
        }
        msg = SimpleNamespace(
            topic="frigate/events",
            payload=json.dumps(payload).encode("utf-8"),
        )

        ns["handle_frigate_event"](SimpleNamespace(), msg)

        assert fetch_calls == []
        assert published == []
        assert len(ns["camera_person_queue"]["driveway"]) == 0

    def test_low_confidence_face_update_is_ignored(self):
        """Low-confidence face updates should not publish or cache identity."""
        ns = self._load_handler_functions()
        published = []

        ns["publish_identity_event"] = lambda *args: published.append(args)
        ns["fetch_snapshot_from_api"] = lambda *args, **kwargs: None
        ns["embedding_store"] = SimpleNamespace(
            store_embedding=lambda *args: None,
            get_all_embeddings=lambda: {},
        )
        ns["reid_model"] = SimpleNamespace(
            extract_embedding=lambda snapshot: [0.1],
            extract_embedding_from_pil=lambda image: [0.1],
        )
        ns["debug_logger"] = SimpleNamespace(
            log_facial_recognition=lambda **kwargs: None,
            log_reid_match=lambda **kwargs: None,
            log_reid_no_match=lambda **kwargs: None,
        )

        payload = {
            "type": "face",
            "id": "evt-low-face",
            "name": "Carol",
            "score": 0.78,
            "camera": "backyard",
            "timestamp": 7890.1,
        }
        msg = SimpleNamespace(
            topic="frigate/tracked_object_update",
            payload=json.dumps(payload).encode("utf-8"),
        )

        ns["handle_tracked_object_update"](SimpleNamespace(), msg)

        assert published == []
        assert "evt-low-face" not in ns["recognized_person_events"]
        assert len(ns["camera_person_queue"]["backyard"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
