"""Integration tests for the false-positive feedback end-to-end pipeline.

The tests spin up an in-process mock MQTT broker (using a simple queue-based
stub) and exercise the full round-trip:

  HA publishes false-positive command
      → identity service handler processes it
      → embedding marked negative in store
      → snapshot refreshed (or cleared)
      → ACK published back

No real MQTT broker or Frigate instance is required.

Run with:
  cd frigate_identity_service
  ../.venv/Scripts/python -m pytest tests/test_false_positive_integration.py -v
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup


# Patch reid_model and apscheduler before importing identity_service
@pytest.fixture(autouse=True)
def _patch_heavy(monkeypatch):
    mock_reid = MagicMock()
    mock_reid.ReIDModel = MagicMock(side_effect=RuntimeError("mocked"))
    monkeypatch.setitem(sys.modules, "reid_model", mock_reid)
    mock_sched = MagicMock()
    mock_sched.schedulers = MagicMock()
    mock_sched.schedulers.background = MagicMock()
    mock_sched.schedulers.background.BackgroundScheduler = MagicMock(
        return_value=MagicMock()
    )
    monkeypatch.setitem(sys.modules, "apscheduler", mock_sched)
    monkeypatch.setitem(sys.modules, "apscheduler.schedulers", mock_sched.schedulers)
    monkeypatch.setitem(
        sys.modules,
        "apscheduler.schedulers.background",
        mock_sched.schedulers.background,
    )


# ---------------------------------------------------------------------------

SERVICE_DIR = Path(__file__).resolve().parents[1] / "frigate_identity_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jpeg_bytes(color: str = "blue") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)


class InProcessMQTTClient:
    """Synchronous in-process MQTT client that delivers messages immediately.

    This mimics paho's publish() / subscribe() interface just enough for the
    integration tests to pass without a real broker.
    """

    def __init__(self):
        self._published: list[tuple[str, Any, bool]] = []
        self._subscriptions: dict[str, list] = {}
        self._lock = threading.Lock()

    def publish(self, topic: str, payload: Any = None, retain: bool = False, **kwargs):
        with self._lock:
            self._published.append((topic, payload, retain))
        # Deliver immediately to in-process subscribers
        self._deliver(topic, payload)

    def subscribe(self, topic: str, callback=None):
        with self._lock:
            self._subscriptions.setdefault(topic, [])
            if callback:
                self._subscriptions[topic].append(callback)

    def _deliver(self, topic: str, payload: Any):
        with self._lock:
            cbs = list(self._subscriptions.get(topic, []))

        class _Msg:
            pass

        msg = _Msg()
        msg.topic = topic
        if isinstance(payload, bytes):
            msg.payload = payload
        elif payload is None:
            msg.payload = b""
        else:
            msg.payload = (
                payload if isinstance(payload, bytes) else str(payload).encode()
            )
        for cb in cbs:
            try:
                cb(self, None, msg)
            except Exception:
                pass

    def get_published(self, topic: str) -> list[Any]:
        with self._lock:
            return [p for t, p, _ in self._published if t == topic]

    def clear(self):
        with self._lock:
            self._published.clear()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestFalsePositiveEndToEnd:
    """End-to-end pipeline: HA submits false positive → service processes → ACK received."""

    def _setup(self, temp_db, snapshot_bytes: bytes | None = None):
        """Return (client, store, handler) with test fixtures wired up."""
        from embedding_store import EmbeddingStore
        import identity_service as svc

        store = EmbeddingStore(temp_db)
        svc.embedding_store = store

        if snapshot_bytes is not None:
            svc.fetch_snapshot_from_api = MagicMock(return_value=_b64(snapshot_bytes))
        else:
            svc.fetch_snapshot_from_api = MagicMock(return_value=None)

        client = InProcessMQTTClient()
        return client, store, svc.handle_false_positive_feedback

    # ------------------------------------------------------------------

    def test_submission_marks_embedding_and_sends_ack(self, temp_db):
        """Core flow: report false positive → embedding marked negative → ok ACK."""
        client, store, handler = self._setup(temp_db)
        import identity_service as svc

        store.store_embedding("alice", np.random.rand(256), "cam1", event_id="evt-bad")
        store.store_embedding("alice", np.random.rand(256), "cam1", event_id="evt-good")
        svc.recognized_person_events["evt-bad"] = {
            "person_id": "alice",
            "camera": "cam1",
            "confidence": 0.9,
            "timestamp": 1,
            "zones": [],
        }

        payload = json.dumps({"person_id": "alice", "event_id": "evt-bad"}).encode()

        class Msg:
            topic = "frigate_identity/feedback/false_positive"

        Msg.payload = payload
        handler(client, Msg())

        # Embedding marked negative
        assert store.person_exists("alice")
        by_event = {e.get("event_id"): e for e in store.embeddings["alice"]}
        assert by_event["evt-bad"].get("negative") is True
        assert by_event["evt-good"].get("negative") is False

        # ACK published
        ack_payloads = client.get_published(
            "frigate_identity/feedback/false_positive_ack"
        )
        assert ack_payloads
        ack = json.loads(ack_payloads[0])
        assert ack["status"] == "ok"
        assert ack["embeddings_removed"] >= 1
        assert svc.recognized_person_events["evt-bad"].get("false_positive") is True

        updates = client.get_published("identity/person/alice")
        assert updates
        person_update = json.loads(updates[0])
        assert person_update["false_positive"] is True
        svc.recognized_person_events.pop("evt-bad", None)

    def test_duplicate_report_is_idempotent(self, temp_db):
        """Second report for same event does not mark additional embeddings."""
        client, store, handler = self._setup(temp_db)
        store.store_embedding("iris", np.random.rand(256), "cam1", event_id="evt-a")
        store.store_embedding("iris", np.random.rand(256), "cam1", event_id="evt-b")

        class Msg:
            topic = "frigate_identity/feedback/false_positive"
            payload = json.dumps({"person_id": "iris", "event_id": "evt-a"}).encode()

        handler(client, Msg())
        by_event = {e.get("event_id"): e for e in store.embeddings["iris"]}
        assert by_event["evt-a"].get("negative") is True
        assert by_event["evt-b"].get("negative") is False

        handler(client, Msg())
        by_event = {e.get("event_id"): e for e in store.embeddings["iris"]}
        assert by_event["evt-a"].get("negative") is True
        assert by_event["evt-b"].get("negative") is False

    def test_snapshot_updated_after_removal(self, temp_db):
        """When a next embedding exists and fetch succeeds the snapshot is updated."""
        fresh = _jpeg_bytes("green")
        client, store, handler = self._setup(temp_db, snapshot_bytes=fresh)

        store.store_embedding("bob", np.random.rand(256), "cam1", event_id="evt-bad")
        store.store_embedding("bob", np.random.rand(256), "cam1", event_id="evt-next")

        class Msg:
            topic = "frigate_identity/feedback/false_positive"
            payload = json.dumps({"person_id": "bob", "event_id": "evt-bad"}).encode()

        handler(client, Msg())

        snaps = client.get_published("identity/snapshots/bob")
        assert snaps
        assert snaps[0] == fresh  # refreshed

        ack = json.loads(
            client.get_published("frigate_identity/feedback/false_positive_ack")[0]
        )
        assert ack["snapshot_refreshed"] is True

    def test_all_positive_embeddings_marked_clears_snapshot(self, temp_db):
        """When the last positive embedding is marked the retained snapshot is cleared."""
        client, store, handler = self._setup(temp_db, snapshot_bytes=None)

        store.store_embedding("carol", np.random.rand(256), "cam1", event_id="evt-only")

        class Msg:
            topic = "frigate_identity/feedback/false_positive"
            payload = json.dumps(
                {"person_id": "carol", "event_id": "evt-only"}
            ).encode()

        handler(client, Msg())

        assert store.person_exists("carol")
        assert store.embeddings["carol"][0].get("negative") is True
        snaps = client.get_published("identity/snapshots/carol")
        assert snaps
        assert snaps[0] == b""  # cleared

    def test_ack_published_even_when_fetch_fails(self, temp_db):
        """ACK should still be published if snapshot fetch fails."""
        client, store, handler = self._setup(temp_db, snapshot_bytes=None)
        store.store_embedding("dave", np.random.rand(256), "cam1", event_id="evt-1")
        store.store_embedding("dave", np.random.rand(256), "cam1", event_id="evt-2")

        class Msg:
            topic = "frigate_identity/feedback/false_positive"
            payload = json.dumps({"person_id": "dave", "event_id": "evt-1"}).encode()

        handler(client, Msg())

        acks = client.get_published("frigate_identity/feedback/false_positive_ack")
        assert len(acks) == 1
        ack = json.loads(acks[0])
        assert ack["status"] == "ok"
        assert ack["snapshot_refreshed"] is False

    def test_concurrent_false_positives_are_independent(self, temp_db):
        """Two concurrent false-positive submissions for different persons are independent."""
        from embedding_store import EmbeddingStore
        import identity_service as svc

        store = EmbeddingStore(temp_db)
        svc.embedding_store = store
        svc.fetch_snapshot_from_api = MagicMock(return_value=None)

        store.store_embedding("eve", np.random.rand(256), "cam1", event_id="evt-e1")
        store.store_embedding("frank", np.random.rand(256), "cam2", event_id="evt-f1")

        client = InProcessMQTTClient()
        handler = svc.handle_false_positive_feedback

        results = []
        errors = []

        def _submit(person, event_id):
            class Msg:
                topic = "frigate_identity/feedback/false_positive"
                payload = json.dumps(
                    {"person_id": person, "event_id": event_id}
                ).encode()

            try:
                handler(client, Msg())
                results.append(person)
            except Exception as exc:
                errors.append((person, exc))

        threads = [
            threading.Thread(target=_submit, args=("eve", "evt-e1")),
            threading.Thread(target=_submit, args=("frank", "evt-f1")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Errors in concurrent test: {errors}"
        assert store.person_exists("eve")
        assert store.person_exists("frank")
        assert store.embeddings["eve"][0].get("negative") is True
        assert store.embeddings["frank"][0].get("negative") is True
        assert len(results) == 2

    def test_message_schema_submitted_at_optional(self, temp_db):
        """Payload without submitted_at should process without error."""
        client, store, handler = self._setup(temp_db)
        store.store_embedding("grace", np.random.rand(256), "cam1", event_id="evt-g")

        class Msg:
            topic = "frigate_identity/feedback/false_positive"
            payload = json.dumps(
                {"person_id": "grace"}
            ).encode()  # no event_id or submitted_at

        handler(client, Msg())
        acks = client.get_published("frigate_identity/feedback/false_positive_ack")
        assert acks
        assert json.loads(acks[0])["status"] == "ok"
