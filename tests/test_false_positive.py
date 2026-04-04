"""Unit tests for the false-positive feedback pipeline.

Tests cover:
    - handle_false_positive_feedback(): message parsing, embedding removal, ACK publishing
    - Snapshot refresh behaviour after removal
    - Graceful handling of edge cases (unknown person, missing fields, bad JSON)
    - EmbeddingStore false-positive helpers (tested separately in test_components.py)

The tests import identity_service with the ReID model mocked out so no weights
are downloaded and no real MQTT broker is required.

Run with:
    cd frigate_identity_service
    ../.venv/Scripts/python -m pytest tests/test_false_positive.py -v
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup so the identity_service package can be imported without install
# ---------------------------------------------------------------------------
SERVICE_DIR = Path(__file__).resolve().parents[1] / "frigate_identity_service"
if str(SERVICE_DIR) not in sys.path:
        sys.path.insert(0, str(SERVICE_DIR))

# ---------------------------------------------------------------------------
# Patch reid_model BEFORE importing identity_service to prevent weight download
# and any MQTT/scheduler startup code that runs at module level.
# identity_service.py catches RuntimeError from ReIDModel() and sets
# REID_AVAILABLE=False, which is the correct no-model test state.
# ---------------------------------------------------------------------------
def _patch_heavy_imports():
        """Inject stubs for modules that have heavy side-effects on import."""
        # Reid model: raise RuntimeError so identity_service sets reid_model=None
        mock_reid_module = MagicMock()
        mock_reid_module.ReIDModel = MagicMock(side_effect=RuntimeError("mocked for tests"))
        sys.modules.setdefault("reid_model", mock_reid_module)

        # apscheduler: prevent real scheduler creation
        mock_sched = MagicMock()
        mock_sched.schedulers = MagicMock()
        mock_sched.schedulers.background = MagicMock()
        mock_sched.schedulers.background.BackgroundScheduler = MagicMock(
                return_value=MagicMock()
        )
        sys.modules.setdefault("apscheduler", mock_sched)
        sys.modules.setdefault("apscheduler.schedulers", mock_sched.schedulers)
        sys.modules.setdefault(
                "apscheduler.schedulers.background", mock_sched.schedulers.background
        )


_patch_heavy_imports()


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db():
    """Temporary EmbeddingStore JSON path."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)


def _make_image_bytes(color: str = "red") -> bytes:
    """Return a minimal JPEG image as bytes."""
    img = Image.new("RGB", (20, 20), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_image_b64(color: str = "red") -> str:
    return base64.b64encode(_make_image_bytes(color)).decode()


class MockMessage:
    """Minimal MQTT message stub."""

    def __init__(self, payload: Any, topic: str = "frigate_identity/feedback/false_positive"):
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload).encode()
        elif isinstance(payload, str):
            payload = payload.encode()
        self.payload = payload
        self.topic = topic


class MockMQTTClient:
    """Minimal MQTT client spy that records publish calls."""

    def __init__(self):
        self.published: list[tuple[str, Any, bool]] = []

    def publish(self, topic: str, payload: Any = None, retain: bool = False, **kwargs):
        self.published.append((topic, payload, retain))

    def get_published(self, topic: str) -> list[Any]:
        return [p for t, p, _ in self.published if t == topic]


# ---------------------------------------------------------------------------
# Tests for handle_false_positive_feedback
# ---------------------------------------------------------------------------


class TestHandleFalsePositiveFeedback:
    """Unit tests for handle_false_positive_feedback()."""

    def _import_handler(self, store, fetch_side_effect=None):
        """Import the handler with embedding_store and fetch patched."""
        import identity_service as svc

        # Swap in test store
        svc.embedding_store = store

        if fetch_side_effect is not None:
            # provide a callable or a return value
            if callable(fetch_side_effect):
                svc.fetch_snapshot_from_api = fetch_side_effect
            else:
                svc.fetch_snapshot_from_api = MagicMock(return_value=fetch_side_effect)
        else:
            svc.fetch_snapshot_from_api = MagicMock(return_value=None)

        return svc.handle_false_positive_feedback

    def test_valid_payload_removes_embedding_by_event_id(self, temp_db):
        """Valid payload with event_id results in embedding removal and ok ACK."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        store.store_embedding("alice", np.random.rand(256), "cam1", event_id="evt-fp")
        store.store_embedding("alice", np.random.rand(256), "cam1", event_id="evt-ok")

        client = MockMQTTClient()
        handler = self._import_handler(store, fetch_side_effect=None)
        msg = MockMessage({"person_id": "alice", "event_id": "evt-fp"})

        handler(client, msg)

        # Bad embedding gone, good one remains
        assert store.person_exists("alice")
        remaining = [e.get("event_id") for e in store.embeddings["alice"]]
        assert "evt-fp" not in remaining
        assert "evt-ok" in remaining

        # ACK published with status=ok
        acks = client.get_published("frigate_identity/feedback/false_positive_ack")
        assert len(acks) == 1
        ack = json.loads(acks[0])
        assert ack["status"] == "ok"
        assert ack["person_id"] == "alice"
        assert ack["event_id"] == "evt-fp"
        assert ack["embeddings_removed"] == 1

    def test_valid_payload_no_event_id_removes_most_recent(self, temp_db):
        """Payload without event_id falls back to removing the most recent embedding."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        store.store_embedding("bob", np.random.rand(256), "cam1", event_id="evt-old")
        store.store_embedding("bob", np.random.rand(256), "cam1", event_id="evt-newer")

        client = MockMQTTClient()
        handler = self._import_handler(store)
        # Omit event_id
        msg = MockMessage({"person_id": "bob"})

        handler(client, msg)

        # One embedding removed (most recent = evt-newer, index 0)
        assert store.person_exists("bob")
        assert len(store.embeddings["bob"]) == 1
        acks = client.get_published("frigate_identity/feedback/false_positive_ack")
        ack = json.loads(acks[0])
        assert ack["status"] == "ok"
        assert ack["embeddings_removed"] == 1

    def test_last_embedding_removed_clears_person(self, temp_db):
        """When the last embedding is removed the snapshot is cleared (empty payload)."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        store.store_embedding("carol", np.random.rand(256), "cam1", event_id="evt-only")

        client = MockMQTTClient()
        handler = self._import_handler(store)
        msg = MockMessage({"person_id": "carol", "event_id": "evt-only"})

        handler(client, msg)

        assert not store.person_exists("carol")
        # Empty retained payload published to clear dashboard snapshot
        cleared = client.get_published("identity/snapshots/carol")
        assert len(cleared) == 1
        assert cleared[0] == b""

    def test_snapshot_refreshed_after_removal(self, temp_db):
        """When a second embedding remains, the service re-publishes a fresh snapshot."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        store.store_embedding("dave", np.random.rand(256), "cam1", event_id="evt-bad")
        store.store_embedding("dave", np.random.rand(256), "cam1", event_id="evt-good")

        fresh_img_b64 = _make_image_b64("green")
        client = MockMQTTClient()
        handler = self._import_handler(store, fetch_side_effect=fresh_img_b64)
        msg = MockMessage({"person_id": "dave", "event_id": "evt-bad"})

        handler(client, msg)

        snapshots = client.get_published("identity/snapshots/dave")
        assert len(snapshots) == 1
        assert snapshots[0] == base64.b64decode(fresh_img_b64)

        acks = client.get_published("frigate_identity/feedback/false_positive_ack")
        ack = json.loads(acks[0])
        assert ack["snapshot_refreshed"] is True

    def test_unknown_person_returns_ok_with_zero_removed(self, temp_db):
        """An unknown person_id should succeed gracefully (0 removed)."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        client = MockMQTTClient()
        handler = self._import_handler(store)
        msg = MockMessage({"person_id": "nobody", "event_id": "evt-123"})

        handler(client, msg)

        acks = client.get_published("frigate_identity/feedback/false_positive_ack")
        ack = json.loads(acks[0])
        assert ack["status"] == "ok"
        assert ack["embeddings_removed"] == 0

    def test_missing_person_id_is_ignored(self, temp_db):
        """Payload without person_id should be silently dropped (no ACK)."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        client = MockMQTTClient()
        handler = self._import_handler(store)
        msg = MockMessage({"event_id": "evt-123"})

        handler(client, msg)

        assert len(client.published) == 0

    def test_invalid_json_is_ignored(self, temp_db):
        """Malformed JSON payload should be silently dropped (no ACK, no crash)."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        client = MockMQTTClient()
        handler = self._import_handler(store)

        class BadMsg:
            payload = b"not-json"
            topic = "frigate_identity/feedback/false_positive"

        handler(client, BadMsg())
        assert len(client.published) == 0

    def test_ack_contains_required_fields(self, temp_db):
        """ACK payload must include all documented fields."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        store.store_embedding("eve", np.random.rand(256), "cam1", event_id="evt-x")

        client = MockMQTTClient()
        handler = self._import_handler(store)
        msg = MockMessage({"person_id": "eve", "event_id": "evt-x"})
        handler(client, msg)

        acks = client.get_published("frigate_identity/feedback/false_positive_ack")
        ack = json.loads(acks[0])
        for field in ("person_id", "event_id", "status", "embeddings_removed",
                       "snapshot_refreshed", "message", "timestamp"):
            assert field in ack, f"ACK missing field: {field}"

    def test_invalid_event_id_type_returns_error_ack(self, temp_db):
        """event_id must be string/null; wrong types return an error ACK."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        client = MockMQTTClient()
        handler = self._import_handler(store)
        msg = MockMessage({"person_id": "eve", "event_id": 12345})

        handler(client, msg)

        acks = client.get_published("frigate_identity/feedback/false_positive_ack")
        assert len(acks) == 1
        ack = json.loads(acks[0])
        assert ack["status"] == "error"
        assert "Invalid event_id" in ack["message"]

    def test_duplicate_event_report_is_idempotent(self, temp_db):
        """A second report for the same event must not remove unrelated embeddings."""
        from embedding_store import EmbeddingStore
        import identity_service as svc

        store = EmbeddingStore(temp_db)
        store.store_embedding("alice", np.random.rand(256), "cam1", event_id="evt-bad")
        store.store_embedding("alice", np.random.rand(256), "cam1", event_id="evt-keep")

        svc.recognized_person_events["evt-bad"] = {
            "person_id": "alice",
            "camera": "cam1",
            "confidence": 0.9,
            "timestamp": 1,
            "zones": [],
        }

        client = MockMQTTClient()
        handler = self._import_handler(store)
        msg = MockMessage({"person_id": "alice", "event_id": "evt-bad"})

        handler(client, msg)
        assert len(store.embeddings["alice"]) == 1

        handler(client, msg)
        # Still only one embedding; duplicate event report should be ignored.
        assert len(store.embeddings["alice"]) == 1
        remaining = [e.get("event_id") for e in store.embeddings["alice"]]
        assert remaining == ["evt-keep"]

        acks = client.get_published("frigate_identity/feedback/false_positive_ack")
        duplicate_ack = json.loads(acks[-1])
        assert duplicate_ack["status"] == "ok"
        assert duplicate_ack["embeddings_removed"] == 0

        assert svc.recognized_person_events["evt-bad"].get("false_positive") is True
        svc.recognized_person_events.pop("evt-bad", None)

    def test_person_update_published_for_false_positive(self, temp_db):
        """Service publishes an identity/person update with false_positive=true."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        store.store_embedding("zoe", np.random.rand(256), "cam1", event_id="evt-z")

        client = MockMQTTClient()
        handler = self._import_handler(store)
        msg = MockMessage({"person_id": "zoe", "event_id": "evt-z"})
        handler(client, msg)

        updates = client.get_published("identity/person/zoe")
        assert updates
        payload = json.loads(updates[0])
        assert payload["false_positive"] is True
        assert payload["source"] == "false_positive_feedback"


# ---------------------------------------------------------------------------
# Tests for _publish_fp_ack helper
# ---------------------------------------------------------------------------


class TestPublishFpAck:
    """Tests for the _publish_fp_ack helper directly."""

    def test_publishes_to_correct_topic(self):
        """_publish_fp_ack should publish to the provided topic."""
        import identity_service as svc

        client = MockMQTTClient()
        svc._publish_fp_ack(
            client,
            "test/ack",
            person_id="foo",
            event_id="e1",
            status="ok",
            embeddings_removed=1,
            snapshot_refreshed=False,
            message="done",
        )
        assert any(t == "test/ack" for t, _, _ in client.published)

    def test_ack_payload_is_valid_json(self):
        """_publish_fp_ack payload must be valid JSON."""
        import identity_service as svc

        client = MockMQTTClient()
        svc._publish_fp_ack(
            client,
            "test/ack",
            person_id="bar",
            event_id=None,
            status="error",
            embeddings_removed=0,
            snapshot_refreshed=False,
            message="something went wrong",
        )
        topic, payload, _ = client.published[0]
        parsed = json.loads(payload)
        assert parsed["status"] == "error"
        assert parsed["person_id"] == "bar"
        assert parsed["event_id"] is None


# ---------------------------------------------------------------------------
# Dashboard card structure tests (no HA runtime needed)
# ---------------------------------------------------------------------------
