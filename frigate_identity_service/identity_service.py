import json
import logging
import time
from collections import defaultdict, deque
import os
import traceback
import requests
import base64
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

from embedding_store import EmbeddingStore
from reid_model import ReIDModel
from matcher import EmbeddingMatcher
from mqtt_utils import get_mqtt_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# TODO: After code is stable, review logging levels and reduce INFO verbosity
# Currently many detection/event logs are at INFO for debugging the MQTT integration.
# Consider demoting non-critical events to DEBUG level for production use.


def load_env_file(env_file_path):
    """Load environment variables from a .env file."""
    if not Path(env_file_path).exists():
        return

    try:
        with open(env_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        # Only set if not already in environment
                        if key not in os.environ:
                            os.environ[key] = value
    except Exception as e:
        logger.warning("Failed to load %s: %s", env_file_path, e)


def load_ha_options(options_file="/data/options.json"):
    """Load configuration from Home Assistant Add-on options file.

    When deployed as a Home Assistant Add-on, the Supervisor writes the
    user's add-on configuration to /data/options.json.  This function reads
    that file and maps each option (e.g. ``mqtt_broker``) to the
    corresponding upper-case environment variable (e.g. ``MQTT_BROKER``),
    but only when the variable has not already been set in the environment.
    """
    if not Path(options_file).exists():
        return

    try:
        with open(options_file, "r") as f:
            options = json.load(f)

        for key, value in options.items():
            env_key = key.upper()
            if env_key not in os.environ and value not in (None, ""):
                os.environ[env_key] = str(value)

        logger.info("Loaded Home Assistant Add-on configuration from %s", options_file)
    except Exception as e:
        logger.warning("Failed to load %s: %s", options_file, e)


# Load configuration from Home Assistant Add-on options (if running as HA add-on)
load_ha_options()

# Load configuration from .env.integration-test if it exists
test_env_path = Path(__file__).parent / ".env.integration-test"
if test_env_path.exists():
    logger.info("Loading configuration from %s", test_env_path)
    load_env_file(str(test_env_path))

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
MQTT_CONNECT_RETRIES = int(os.getenv("MQTT_CONNECT_RETRIES", "30"))
MQTT_CONNECT_RETRY_DELAY = int(os.getenv("MQTT_CONNECT_RETRY_DELAY", "5"))
FRIGATE_HOST = os.getenv("FRIGATE_HOST", "http://localhost:5000")
REID_MODEL = os.getenv("REID_MODEL", "osnet_x1_0")
REID_DEVICE = os.getenv("REID_DEVICE", "auto")
REID_SIMILARITY_THRESHOLD = float(os.getenv("REID_SIMILARITY_THRESHOLD", "0.6"))


def get_default_embeddings_path():
    """Return container-appropriate default path for embeddings.
    
    When running in a container, defaults to /data/embeddings.json for persistence.
    Otherwise uses embeddings.json in the current directory.
    """
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return "/data/embeddings.json"
    return "embeddings.json"


EMBEDDINGS_DB_PATH = os.getenv("EMBEDDINGS_DB_PATH", get_default_embeddings_path())
SNAPSHOT_CORRELATION_WINDOW = float(os.getenv("SNAPSHOT_CORRELATION_WINDOW", "2.0"))
MAX_TRACKED_PERSONS_PER_CAMERA = int(os.getenv("MAX_TRACKED_PERSONS_PER_CAMERA", "3"))

# Initialize modules
logger.info("Initializing embedding store...")
embedding_store = EmbeddingStore(EMBEDDINGS_DB_PATH)

logger.info("Initializing ReID model (%s)...", REID_MODEL)
device = None if REID_DEVICE == "auto" else REID_DEVICE
try:
    reid_model = ReIDModel(device=device, model_name=REID_MODEL)
    logger.info("ReID system ready!")
    REID_AVAILABLE = True
except RuntimeError as e:
    logger.warning("%s", e)
    logger.warning(
        "ReID matching will be disabled. Only basic temporal tracking will be available."
    )
    reid_model = None
    REID_AVAILABLE = False

# Track recent person detections per camera for snapshot correlation
camera_person_queue = defaultdict(lambda: deque(maxlen=MAX_TRACKED_PERSONS_PER_CAMERA))

# Cache snapshots to avoid redundant API calls
snapshot_cache = {}  # event_id -> (base64_image, timestamp)
CACHE_TTL = 60  # seconds


def on_connect(client, userdata, flags, rc, properties=None):
    logger.info("Connected to MQTT Broker at %s:%s", MQTT_BROKER, MQTT_PORT)
    logger.info("Frigate API endpoint: %s", FRIGATE_HOST)

    # Subscribe to tracked object events (new/update/end for all objects)
    # See: https://docs.frigate.video/integrations/mqtt#frigateevents
    client.subscribe("frigate/events")
    logger.info("Subscribed to: frigate/events")

    # Subscribe to tracked object metadata updates (face recognition, LPR, etc.)
    # See: https://docs.frigate.video/integrations/mqtt#frigatetracked_object_update
    client.subscribe("frigate/tracked_object_update")
    logger.info("Subscribed to: frigate/tracked_object_update")

    # Subscribe to person snapshots for fast display
    client.subscribe("frigate/+/person/snapshot")
    logger.info("Subscribed to: frigate/+/person/snapshot")

    # Optional: Subscribe to car/truck for vehicle detection
    client.subscribe("frigate/+/car/snapshot")
    client.subscribe("frigate/+/truck/snapshot")
    logger.info("Subscribed to vehicle snapshots")

    logger.info(
        "Frigate Identity Service started successfully (model=%s, device=%s, threshold=%.2f)",
        REID_MODEL,
        REID_DEVICE,
        REID_SIMILARITY_THRESHOLD,
    )


def on_message(client, userdata, msg):
    """Route messages to appropriate handlers based on topic"""
    try:
        if msg.topic == "frigate/events":
            handle_frigate_event(client, msg)
        elif msg.topic == "frigate/tracked_object_update":
            handle_tracked_object_update(client, msg)
        elif "/snapshot" in msg.topic:
            handle_snapshot_for_display(client, msg)
    except Exception as e:
        logger.error("Error processing MQTT message on %s: %s", msg.topic, e)
        traceback.print_exc()


def fetch_snapshot_from_api(event_id, crop=True, quality=85, height=400):
    """
    Fetch snapshot from Frigate API for a specific event.
    Uses caching to avoid redundant API calls.

    Returns: base64-encoded JPEG string, or None if failed
    """
    now = time.time()

    # Check cache
    if event_id in snapshot_cache:
        cached_img, cached_time = snapshot_cache[event_id]
        if now - cached_time < CACHE_TTL:
            return cached_img

    try:
        # Use thumbnail endpoint with crop parameter
        url = f"{FRIGATE_HOST}/api/events/{event_id}/thumbnail.jpg"
        params = {"crop": "1" if crop else "0", "quality": quality, "h": height}

        response = requests.get(url, params=params, timeout=5)

        if response.status_code == 200:
            # Convert to base64 for ReID model
            image_base64 = base64.b64encode(response.content).decode("utf-8")

            # Cache the result
            snapshot_cache[event_id] = (image_base64, now)

            return image_base64
        else:
            logger.warning(
                "[API] Failed to fetch snapshot for %s: HTTP %s",
                event_id,
                response.status_code,
            )
            return None

    except requests.exceptions.RequestException as e:
        logger.error("[API] Error fetching snapshot for %s: %s", event_id, e)
        return None


def publish_identity_event(
    client, person_id, camera, confidence, source, zones, event_id, timestamp
):
    """Publish identity event to Home Assistant"""
    identity_event = {
        "person_id": person_id,
        "camera": camera,
        "confidence": float(confidence),
        "source": source,
        "frigate_zones": zones,
        "event_id": event_id,
        "timestamp": int(timestamp * 1000)
        if isinstance(timestamp, float)
        else timestamp,
        "snapshot_url": f"{FRIGATE_HOST}/api/events/{event_id}/thumbnail.jpg?crop=1",
    }

    # Publish to person-specific topic
    client.publish(f"identity/person/{person_id}", json.dumps(identity_event))
    logger.info(
        "[%s] %s at %s (zones: %s, confidence: %.3f)",
        source.upper(),
        person_id,
        camera,
        zones,
        confidence,
    )


def handle_frigate_event(client, msg):
    """
    Process Frigate event messages from the 'frigate/events' topic.

    Payload structure (per Frigate docs):
        {"type": "new"|"update"|"end", "before": {...}, "after": {...}}

    The 'after' dict contains the current state of the tracked object with
    fields like id, camera, label, sub_label, current_zones, top_score, etc.
    sub_label is either null or ["Name", score].
    """
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except json.JSONDecodeError:
        logger.warning("[EVENT] Could not decode JSON from %s", msg.topic)
        return

    event_type = payload.get("type")  # "new", "update", or "end"
    after = payload.get("after", {})

    camera = after.get("camera")
    event_id = after.get("id")
    label = after.get("label")
    raw_sub_label = after.get("sub_label")  # null or ["Name", score]
    current_zones = after.get("current_zones", [])
    confidence = after.get("top_score", after.get("score", 0))
    timestamp = after.get("frame_time", time.time())

    if label != "person":
        return  # Only process person objects

    # Parse sub_label: Frigate sends ["Name", score] or null
    sub_label = None
    if isinstance(raw_sub_label, list) and len(raw_sub_label) >= 2:
        sub_label = raw_sub_label[0]
    elif isinstance(raw_sub_label, str) and raw_sub_label:
        sub_label = raw_sub_label

    logger.info(
        "[EVENT] %s person on %s (event=%s, zones=%s, score=%.2f, sub_label=%s)",
        event_type,
        camera,
        event_id,
        current_zones,
        confidence,
        sub_label,
    )

    # Skip end events — person has left the frame
    if event_type == "end":
        return

    # Add to camera tracking queue for snapshot correlation
    detection_record = {
        "event_id": event_id,
        "timestamp": timestamp,
        "zones": current_zones,
        "confidence": confidence,
    }

    # SCENARIO A: Frigate identified face via facial recognition (sub_label set)
    if sub_label:
        person_id = sub_label
        detection_record["person_id"] = person_id

        # Add to correlation queue
        camera_person_queue[camera].append(detection_record)

        # ACCURATE PATH: Fetch via API for embedding storage
        snapshot_base64 = fetch_snapshot_from_api(event_id, crop=True)

        if snapshot_base64 and REID_AVAILABLE:
            try:
                embedding = reid_model.extract_embedding(snapshot_base64)
                embedding_store.store_embedding(
                    person_id, embedding, camera, confidence
                )
                logger.info("[EMBEDDING] Stored accurate embedding for %s", person_id)
            except Exception as e:
                logger.error(
                    "[EMBEDDING] Error storing embedding for %s: %s", person_id, e
                )

        # Publish identity event (HA doesn't wait for embedding storage)
        publish_identity_event(
            client,
            person_id,
            camera,
            confidence,
            "facial_recognition",
            current_zones,
            event_id,
            timestamp,
        )

    # SCENARIO B: Person detected but no face visible - try ReID
    else:
        if not REID_AVAILABLE:
            return

        # Try ReID matching via API (accurate)
        snapshot_base64 = fetch_snapshot_from_api(event_id, crop=True)

        if not snapshot_base64:
            logger.warning("[REID] Could not fetch snapshot for event %s", event_id)
            return

        try:
            # Extract embedding and match to stored persons
            query_embedding = reid_model.extract_embedding(snapshot_base64)
            stored_embeddings = embedding_store.get_all_embeddings()
            person_id, similarity_score = EmbeddingMatcher.find_best_match(
                query_embedding, stored_embeddings, threshold=REID_SIMILARITY_THRESHOLD
            )

            if person_id:
                detection_record["person_id"] = person_id
                camera_person_queue[camera].append(detection_record)

                publish_identity_event(
                    client,
                    person_id,
                    camera,
                    similarity_score,
                    "reid_model",
                    current_zones,
                    event_id,
                    timestamp,
                )
            else:
                logger.info(
                    "[REID] No match found for event %s (best score: %.3f)",
                    event_id,
                    similarity_score,
                )

        except Exception as e:
            logger.error("[REID] Error processing event %s: %s", event_id, e)
            traceback.print_exc()


def handle_tracked_object_update(client, msg):
    """
    Process Frigate tracked_object_update messages (face recognition, LPR, etc.).

    Face recognition payload:
        {"type": "face", "id": "...", "name": "John", "score": 0.95,
         "camera": "front_door_cam", "timestamp": ...}
    """
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except json.JSONDecodeError:
        logger.warning("[TRACKED_UPDATE] Could not decode JSON from %s", msg.topic)
        return

    update_type = payload.get("type")

    if update_type == "face":
        person_id = payload.get("name")
        event_id = payload.get("id")
        score = payload.get("score", 0.0)
        camera = payload.get("camera")
        timestamp = payload.get("timestamp", time.time())

        if not person_id or not event_id:
            return

        logger.info(
            "[FACE_UPDATE] Face recognized: %s on %s (event=%s, score=%.2f)",
            person_id,
            camera,
            event_id,
            score,
        )

        # Add to correlation queue
        detection_record = {
            "event_id": event_id,
            "timestamp": timestamp,
            "zones": [],
            "confidence": score,
            "person_id": person_id,
        }
        camera_person_queue[camera].append(detection_record)

        # Fetch snapshot and store embedding for ReID learning
        snapshot_base64 = fetch_snapshot_from_api(event_id, crop=True)
        if snapshot_base64 and REID_AVAILABLE:
            try:
                embedding = reid_model.extract_embedding(snapshot_base64)
                embedding_store.store_embedding(
                    person_id, embedding, camera, score
                )
                logger.info("[EMBEDDING] Stored embedding from face update for %s", person_id)
            except Exception as e:
                logger.error(
                    "[EMBEDDING] Error storing embedding for %s: %s", person_id, e
                )

        # Publish identity event
        publish_identity_event(
            client,
            person_id,
            camera,
            score,
            "face_recognition_update",
            [],
            event_id,
            timestamp,
        )
    else:
        logger.debug("[TRACKED_UPDATE] Ignoring update type: %s", update_type)


def handle_snapshot_for_display(client, msg):
    """
    FAST PATH: Match MQTT snapshot to recent person detection for live dashboard.
    Uses temporal correlation - occasional mismatches acceptable for display.
    """
    # Extract camera and object type from topic: frigate/{camera}/{object_type}/snapshot
    topic_parts = msg.topic.split("/")
    if len(topic_parts) < 4:
        return

    camera = topic_parts[1]
    object_type = topic_parts[2]

    image_bytes = msg.payload
    now = time.time()

    if object_type == "person":
        logger.info("[SNAPSHOT] Person snapshot received from %s", camera)

    # Handle vehicle snapshots
    if object_type in ["car", "truck"]:
        # Publish vehicle detection event
        vehicle_event = {
            "vehicle_type": object_type,
            "camera": camera,
            "timestamp": int(now * 1000),
        }
        client.publish("identity/vehicle/detected", json.dumps(vehicle_event))

        # Store snapshot for HA display
        client.publish(f"identity/snapshots/vehicle_{camera}", image_bytes, retain=True)
        logger.info("[VEHICLE] %s detected at %s", object_type, camera)
        return

    # Handle person snapshots
    if object_type != "person":
        return

    # Get recent person detections on this camera
    recent_detections = camera_person_queue.get(camera, deque())

    if not recent_detections:
        logger.info(
            "[SNAPSHOT] No correlated person update for %s — update messages may not be arriving (check MQTT topic format)",
            camera,
        )
        return

    # Match to most recent person within correlation window
    matched_person = None
    active_persons = []

    for detection in reversed(recent_detections):  # Most recent first
        if now - detection["timestamp"] <= SNAPSHOT_CORRELATION_WINDOW:
            if "person_id" in detection:
                active_persons.append(detection)
                if matched_person is None:
                    matched_person = detection

    if not matched_person or "person_id" not in matched_person:
        logger.debug("[SNAPSHOT] No correlation match found for %s snapshot", camera)
        return

    person_id = matched_person["person_id"]

    # Determine correlation confidence
    if len(active_persons) == 1:
        confidence_note = "high_confidence"
    elif len(active_persons) > 1:
        confidence_note = "low_confidence_multi_person"
        logger.warning(
            "[SNAPSHOT] %d persons active on %s, snapshot may be mismatched",
            len(active_persons),
            camera,
        )
    else:
        confidence_note = "unknown"

    # FAST PUBLISH: Send snapshot directly to HA for dashboard
    client.publish(f"identity/snapshots/{person_id}", image_bytes, retain=True)

    # Update snapshot metadata
    snapshot_metadata = {
        "person_id": person_id,
        "camera": camera,
        "timestamp": int(now * 1000),
        "source": "mqtt_snapshot",
        "correlation_confidence": confidence_note,
        "active_persons_count": len(active_persons),
        "zones": matched_person.get("zones", []),
    }
    client.publish(
        f"identity/snapshots/{person_id}/metadata", json.dumps(snapshot_metadata)
    )

    logger.info(
        "[SNAPSHOT-FAST] Published snapshot for %s at %s (%s)",
        person_id,
        camera,
        confidence_note,
    )


def connect_with_retry(client, broker, port, max_attempts, retry_delay):
    """Attempt to connect to the MQTT broker, retrying on failure.

    Args:
        client: paho MQTT client instance.
        broker: MQTT broker hostname or IP.
        port: MQTT broker port.
        max_attempts: Total number of connection attempts to make.
        retry_delay: Seconds to wait between attempts.

    Returns:
        True if connected successfully, False if all attempts were exhausted.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            client.connect(broker, port, 60)
            return True
        except Exception as e:
            logger.error(
                "Failed to connect to MQTT broker at %s:%s (attempt %d/%d): %s",
                broker,
                port,
                attempt,
                max_attempts,
                e,
            )
            if attempt < max_attempts:
                logger.info("Retrying in %d seconds...", retry_delay)
                time.sleep(retry_delay)
    return False


def schedule_nightly_embedding_cleanup():
    """Start a background scheduler that clears all embeddings at midnight.

    This ensures the ReID database only contains embeddings from the current
    day, preventing stale appearance data from causing false matches.
    Frigate facial recognition will repopulate the store throughout each day.
    """
    scheduler = BackgroundScheduler()

    def _clear_embeddings():
        logger.info("[CLEANUP] Nightly cleanup: clearing all embeddings for daily refresh")
        embedding_store.clear()
        logger.info("[CLEANUP] Embedding store cleared. Store will rebuild as people are seen today.")

    scheduler.add_job(_clear_embeddings, "cron", hour=0, minute=0)
    scheduler.start()
    logger.info("[SCHEDULER] Nightly embedding cleanup scheduled (runs at midnight)")
    return scheduler


client = get_mqtt_client()
client.on_connect = on_connect
client.on_message = on_message

if MQTT_USERNAME:
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

# Start the nightly cleanup scheduler before entering the MQTT loop
scheduler = schedule_nightly_embedding_cleanup()

if connect_with_retry(
    client, MQTT_BROKER, MQTT_PORT, MQTT_CONNECT_RETRIES, MQTT_CONNECT_RETRY_DELAY
):
    client.loop_forever()
else:
    logger.error(
        "Could not connect to MQTT broker after %d attempts. Exiting.",
        MQTT_CONNECT_RETRIES,
    )
    raise SystemExit(1)
