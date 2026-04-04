"""Deterministic Docker Mosquitto integration test for false-positive flow."""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from queue import Empty, Queue
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
import paho.mqtt.client as mqtt


SERVICE_DIR = Path(__file__).resolve().parents[1] / "frigate_identity_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))


def _patch_heavy_imports() -> None:
    """Stub heavy imports so importing identity_service is side-effect safe."""
    mock_reid_module = MagicMock()
    mock_reid_module.ReIDModel = MagicMock(side_effect=RuntimeError("mocked for tests"))
    sys.modules.setdefault("reid_model", mock_reid_module)

    mock_sched = MagicMock()
    mock_sched.schedulers = MagicMock()
    mock_sched.schedulers.background = MagicMock()
    mock_sched.schedulers.background.BackgroundScheduler = MagicMock(return_value=MagicMock())
    sys.modules.setdefault("apscheduler", mock_sched)
    sys.modules.setdefault("apscheduler.schedulers", mock_sched.schedulers)
    sys.modules.setdefault("apscheduler.schedulers.background", mock_sched.schedulers.background)


_patch_heavy_imports()


def _new_client() -> mqtt.Client:
    """Create a paho client across both v1 and v2 APIs."""
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except Exception:
        return mqtt.Client()


def _jpeg_b64() -> str:
    img = Image.new("RGB", (16, 16), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for(predicate, timeout: float = 10.0, interval: float = 0.1) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def docker_mosquitto():
    if shutil.which("docker") is None:
        pytest.skip("docker not available")

    port = _free_port()
    name = f"frigate-fp-test-{uuid.uuid4().hex[:8]}"

    run_cmd = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        name,
        "-p",
        f"{port}:1883",
        "eclipse-mosquitto:2",
    ]
    start = subprocess.run(run_cmd, capture_output=True, text=True)
    if start.returncode != 0:
        pytest.skip(f"unable to start docker mosquitto: {start.stderr.strip()}")

    probe_client = _new_client()

    def _probe_connected() -> bool:
        try:
            probe_client.connect("127.0.0.1", port, 5)
            probe_client.disconnect()
            return True
        except Exception:
            return False

    if not _wait_for(_probe_connected, timeout=12.0):
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        pytest.skip("mosquitto container did not become ready")

    try:
        yield ("127.0.0.1", port, name)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def test_false_positive_flow_with_docker_mosquitto(docker_mosquitto, temp_db_path):
    """Dashboard-like publish triggers service processing and observable updates."""
    from embedding_store import EmbeddingStore
    import identity_service as svc

    host, port, _name = docker_mosquitto

    store = EmbeddingStore(temp_db_path)
    svc.embedding_store = store
    svc.fetch_snapshot_from_api = MagicMock(return_value=_jpeg_b64())

    # Seed two embeddings so removal + refresh path can be observed.
    store.store_embedding("alice", np.random.rand(256), "cam1", event_id="evt-bad")
    store.store_embedding("alice", np.random.rand(256), "cam1", event_id="evt-good")

    svc.recognized_person_events["evt-bad"] = {
        "person_id": "alice",
        "camera": "cam1",
        "confidence": 0.9,
        "timestamp": 1,
        "zones": [],
    }

    events: Queue[tuple[str, bytes]] = Queue()

    observer = _new_client()

    def _observer_on_connect(client, _userdata, _flags, _rc, _props=None):
        client.subscribe("frigate_identity/feedback/false_positive_ack")
        client.subscribe("identity/snapshots/alice")
        client.subscribe("identity/person/alice")

    def _observer_on_message(_client, _userdata, msg):
        payload = msg.payload if isinstance(msg.payload, bytes) else bytes(msg.payload)
        events.put((msg.topic, payload))

    observer.on_connect = _observer_on_connect
    observer.on_message = _observer_on_message
    observer.connect(host, port, 30)
    observer.loop_start()

    service_client = _new_client()

    def _service_on_connect(client, userdata, flags, rc, properties=None):
        svc.on_connect(client, userdata, flags, rc, properties)

    service_client.on_connect = _service_on_connect
    service_client.on_message = svc.on_message
    service_client.connect(host, port, 30)
    service_client.loop_start()

    publisher = _new_client()
    publisher.connect(host, port, 30)
    publisher.loop_start()

    try:
        ready = _wait_for(lambda: observer.is_connected() and service_client.is_connected(), timeout=10.0)
        assert ready, "MQTT clients did not connect in time"

        cmd = {
            "person_id": "alice",
            "event_id": "evt-bad",
            "camera": "cam1",
            "submitted_at": int(time.time() * 1000),
        }
        publisher.publish("frigate_identity/feedback/false_positive", json.dumps(cmd), qos=1)

        seen_topics: dict[str, bytes] = {}
        deadline = time.time() + 12.0
        while time.time() < deadline and len(seen_topics) < 3:
            try:
                topic, payload = events.get(timeout=0.5)
                seen_topics[topic] = payload
            except Empty:
                pass

        assert "frigate_identity/feedback/false_positive_ack" in seen_topics
        assert "identity/snapshots/alice" in seen_topics
        assert "identity/person/alice" in seen_topics

        ack = json.loads(seen_topics["frigate_identity/feedback/false_positive_ack"].decode("utf-8"))
        assert ack["status"] == "ok"
        assert ack["embeddings_removed"] == 1

        person_update = json.loads(seen_topics["identity/person/alice"].decode("utf-8"))
        assert person_update["false_positive"] is True
        assert person_update["source"] == "false_positive_feedback"

        assert svc.recognized_person_events["evt-bad"].get("false_positive") is True

        # Duplicate report should be idempotent for explicit event_id.
        publisher.publish("frigate_identity/feedback/false_positive", json.dumps(cmd), qos=1)

        duplicate_ack = None
        deadline2 = time.time() + 8.0
        while time.time() < deadline2:
            try:
                topic, payload = events.get(timeout=0.5)
            except Empty:
                continue
            if topic == "frigate_identity/feedback/false_positive_ack":
                candidate = json.loads(payload.decode("utf-8"))
                if candidate.get("embeddings_removed") == 0:
                    duplicate_ack = candidate
                    break

        assert duplicate_ack is not None
        assert duplicate_ack["status"] == "ok"
        assert duplicate_ack["embeddings_removed"] == 0

        remaining = store.embeddings.get("alice", [])
        assert len(remaining) == 1
        assert remaining[0].get("event_id") == "evt-good"
    finally:
        publisher.loop_stop()
        publisher.disconnect()
        service_client.loop_stop()
        service_client.disconnect()
        observer.loop_stop()
        observer.disconnect()
        svc.recognized_person_events.pop("evt-bad", None)
