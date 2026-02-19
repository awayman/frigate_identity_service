import paho.mqtt.client as mqtt
import json
import time
from collections import defaultdict, deque
import os
import traceback
import requests
import base64

from embedding_store import EmbeddingStore
from reid_model import ReIDModel
from matcher import EmbeddingMatcher
from mqtt_utils import get_mqtt_client

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
FRIGATE_HOST = os.getenv("FRIGATE_HOST", "http://localhost:5000")
REID_MODEL = os.getenv("REID_MODEL", "osnet_x1_0")
REID_DEVICE = os.getenv("REID_DEVICE", "auto")
REID_SIMILARITY_THRESHOLD = float(os.getenv("REID_SIMILARITY_THRESHOLD", "0.6"))
EMBEDDINGS_DB_PATH = os.getenv("EMBEDDINGS_DB_PATH", "embeddings.json")
SNAPSHOT_CORRELATION_WINDOW = float(os.getenv("SNAPSHOT_CORRELATION_WINDOW", "2.0"))
MAX_TRACKED_PERSONS_PER_CAMERA = int(os.getenv("MAX_TRACKED_PERSONS_PER_CAMERA", "3"))
CONFIDENCE_DECAY_START_MINUTES = float(os.getenv("CONFIDENCE_DECAY_START_MINUTES", "5.0"))
CONFIDENCE_DECAY_FULL_MINUTES = float(os.getenv("CONFIDENCE_DECAY_FULL_MINUTES", "15.0"))
CONFIDENCE_DECAY_RATE_PER_MINUTE = float(os.getenv("CONFIDENCE_DECAY_RATE_PER_MINUTE", "0.10"))

# Initialize modules
print("Initializing embedding store...")
embedding_store = EmbeddingStore(EMBEDDINGS_DB_PATH)

print(f"Initializing ReID model: {REID_MODEL}")
device = None if REID_DEVICE == "auto" else REID_DEVICE
try:
    reid_model = ReIDModel(model_name=REID_MODEL, device=device)
    print("ReID system ready!")
    REID_AVAILABLE = True
except RuntimeError as e:
    print(f"WARNING: {e}")
    print("ReID matching will be disabled. Only basic temporal tracking will be available.")
    reid_model = None
    REID_AVAILABLE = False

# Track recent person detections per camera for snapshot correlation
camera_person_queue = defaultdict(lambda: deque(maxlen=MAX_TRACKED_PERSONS_PER_CAMERA))

# Track last-seen timestamps for confidence decay
person_last_seen: dict[str, float] = {}  # person_id -> unix timestamp

# Cache snapshots to avoid redundant API calls
snapshot_cache = {}  # event_id -> (base64_image, timestamp)
CACHE_TTL = 60  # seconds

def calculate_effective_confidence(
    base_confidence: float,
    last_seen_timestamp: float | None,
    decay_start_minutes: float = 5.0,
    decay_rate_per_minute: float = 0.10,
    full_decay_minutes: float = 15.0,
) -> tuple[float, float]:
    """
    Calculate effective confidence with time-based decay.
    
    Returns (effective_confidence, minutes_since_seen).
    - No decay until decay_start_minutes
    - Decay by decay_rate_per_minute for each minute after that
    - Returns 0.0 after full_decay_minutes
    - Returns base_confidence unchanged if last_seen_timestamp is None (fresh detection)
    """
    # Fresh detection - no previous timestamp
    if last_seen_timestamp is None:
        return base_confidence, 0.0
    
    # Calculate time elapsed
    current_time = time.time()
    minutes_since_seen = (current_time - last_seen_timestamp) / 60.0
    
    # Full decay reached
    if minutes_since_seen >= full_decay_minutes:
        return 0.0, minutes_since_seen
    
    # No decay yet
    if minutes_since_seen <= decay_start_minutes:
        return base_confidence, minutes_since_seen
    
    # Linear decay phase
    minutes_in_decay = minutes_since_seen - decay_start_minutes
    decay_amount = minutes_in_decay * decay_rate_per_minute
    effective_confidence = max(0.0, base_confidence - decay_amount)
    
    return effective_confidence, minutes_since_seen

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"Connected to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}")
    print(f"Frigate API endpoint: {FRIGATE_HOST}")
    
    # Subscribe to tracked object updates (includes face recognition via sub_label)
    client.subscribe("frigate/+/+/update")
    print("Subscribed to: frigate/+/+/update")
    
    # Subscribe to person snapshots for fast display
    client.subscribe("frigate/+/person/snapshot")
    print("Subscribed to: frigate/+/person/snapshot")
    
    # Optional: Subscribe to car/truck for vehicle detection
    client.subscribe("frigate/+/car/snapshot")
    client.subscribe("frigate/+/truck/snapshot")
    print("Subscribed to vehicle snapshots")

def on_message(client, userdata, msg):
    """Route messages to appropriate handlers based on topic"""
    try:
        if "/update" in msg.topic:
            handle_tracked_object_update(client, msg)
        elif "/snapshot" in msg.topic:
            handle_snapshot_for_display(client, msg)
    except Exception as e:
        print(f"Error processing MQTT message on {msg.topic}: {e}")
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
        params = {
            "crop": "1" if crop else "0",
            "quality": quality,
            "h": height
        }
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            # Convert to base64 for ReID model
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            
            # Cache the result
            snapshot_cache[event_id] = (image_base64, now)
            
            return image_base64
        else:
            print(f"[API] Failed to fetch snapshot for {event_id}: HTTP {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"[API] Error fetching snapshot for {event_id}: {e}")
        return None

def publish_identity_event(client, person_id, camera, confidence, source, zones, event_id, timestamp):
    """Publish identity event to Home Assistant"""
    # Calculate effective confidence with decay
    last_seen = person_last_seen.get(person_id)
    effective_confidence, minutes_since_seen = calculate_effective_confidence(
        confidence,
        last_seen,
        decay_start_minutes=CONFIDENCE_DECAY_START_MINUTES,
        decay_rate_per_minute=CONFIDENCE_DECAY_RATE_PER_MINUTE,
        full_decay_minutes=CONFIDENCE_DECAY_FULL_MINUTES,
    )
    confidence_decayed = effective_confidence < confidence
    
    identity_event = {
        "person_id": person_id,
        "camera": camera,
        "confidence": float(confidence),
        "effective_confidence": float(effective_confidence),
        "last_seen": last_seen,
        "minutes_since_seen": float(minutes_since_seen),
        "confidence_decayed": confidence_decayed,
        "source": source,
        "frigate_zones": zones,
        "event_id": event_id,
        "timestamp": int(timestamp * 1000) if isinstance(timestamp, float) else timestamp,
        "snapshot_url": f"{FRIGATE_HOST}/api/events/{event_id}/thumbnail.jpg?crop=1"
    }
    
    # Publish to person-specific topic
    client.publish(f"identity/person/{person_id}", json.dumps(identity_event))
    print(f"[{source.upper()}] {person_id} at {camera} (zones: {zones}, confidence: {confidence:.3f}, effective: {effective_confidence:.3f})")

def handle_tracked_object_update(client, msg):
    """
    Process Frigate tracked object update.
    Builds correlation data for snapshot matching and fetches accurate embeddings via API.
    """
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except json.JSONDecodeError:
        print(f"[UPDATE] Warning: Could not decode JSON from {msg.topic}")
        return
    
    camera = payload.get("camera")
    event_id = payload.get("id")
    label = payload.get("label")
    sub_label = payload.get("sub_label")  # Face recognition result from Frigate
    current_zones = payload.get("current_zones", [])
    confidence = payload.get("top_score", payload.get("score", 0))
    timestamp = payload.get("frame_time", time.time())
    
    if label != "person":
        return  # Only process person objects
    
    # Add to camera tracking queue for snapshot correlation
    detection_record = {
        "event_id": event_id,
        "timestamp": timestamp,
        "zones": current_zones,
        "confidence": confidence
    }
    
    # SCENARIO A: Frigate identified face via facial recognition
    if sub_label and sub_label != "":
        person_id = sub_label
        detection_record["person_id"] = person_id
        
        # Add to correlation queue
        camera_person_queue[camera].append(detection_record)
        
        # ACCURATE PATH: Fetch via API for embedding storage
        snapshot_base64 = fetch_snapshot_from_api(event_id, crop=True)
        
        if snapshot_base64 and REID_AVAILABLE:
            try:
                embedding = reid_model.extract_embedding(snapshot_base64)
                embedding_store.store_embedding(person_id, embedding, camera, confidence)
                print(f"[EMBEDDING] Stored accurate embedding for {person_id}")
            except Exception as e:
                print(f"[EMBEDDING] Error storing embedding for {person_id}: {e}")
        
        # Update last-seen timestamp
        person_last_seen[person_id] = timestamp
        
        # Publish identity event (HA doesn't wait for embedding storage)
        publish_identity_event(client, person_id, camera, confidence,
                              "facial_recognition", current_zones, event_id, timestamp)
    
    # SCENARIO B: Person detected but no face visible - try ReID
    else:
        if not REID_AVAILABLE:
            return
        
        # Try ReID matching via API (accurate)
        snapshot_base64 = fetch_snapshot_from_api(event_id, crop=True)
        
        if not snapshot_base64:
            print(f"[REID] Could not fetch snapshot for event {event_id}")
            return
        
        try:
            # Extract embedding and match to stored persons
            query_embedding = reid_model.extract_embedding(snapshot_base64)
            stored_embeddings = embedding_store.get_all_embeddings()
            person_id, similarity_score = EmbeddingMatcher.find_best_match(
                query_embedding,
                stored_embeddings,
                threshold=REID_SIMILARITY_THRESHOLD
            )
            
            if person_id:
                detection_record["person_id"] = person_id
                camera_person_queue[camera].append(detection_record)
                
                # Update last-seen timestamp
                person_last_seen[person_id] = timestamp
                
                publish_identity_event(client, person_id, camera, similarity_score,
                                     "reid_model", current_zones, event_id, timestamp)
            else:
                print(f"[REID] No match found for event {event_id} (best score: {similarity_score:.3f})")
                
        except Exception as e:
            print(f"[REID] Error processing event {event_id}: {e}")
            traceback.print_exc()

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
    
    # Handle vehicle snapshots
    if object_type in ["car", "truck"]:
        # Publish vehicle detection event
        vehicle_event = {
            "vehicle_type": object_type,
            "camera": camera,
            "timestamp": int(now * 1000)
        }
        client.publish("identity/vehicle/detected", json.dumps(vehicle_event))
        
        # Store snapshot for HA display
        client.publish(f"identity/snapshots/vehicle_{camera}", image_bytes, retain=True)
        print(f"[VEHICLE] {object_type} detected at {camera}")
        return
    
    # Handle person snapshots
    if object_type != "person":
        return
    
    # Get recent person detections on this camera
    recent_detections = camera_person_queue.get(camera, deque())
    
    if not recent_detections:
        print(f"[SNAPSHOT] Received snapshot from {camera}, but no recent person detection")
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
        print(f"[SNAPSHOT] No correlation match found for {camera} snapshot")
        return
    
    person_id = matched_person["person_id"]
    
    # Determine correlation confidence
    if len(active_persons) == 1:
        confidence_note = "high_confidence"
    elif len(active_persons) > 1:
        confidence_note = "low_confidence_multi_person"
        print(f"[SNAPSHOT] Warning: {len(active_persons)} persons active on {camera}, snapshot may be mismatched")
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
        "zones": matched_person.get("zones", [])
    }
    client.publish(f"identity/snapshots/{person_id}/metadata", json.dumps(snapshot_metadata))
    
    print(f"[SNAPSHOT-FAST] Published snapshot for {person_id} at {camera} ({confidence_note})")

client = get_mqtt_client()
client.on_connect = on_connect
client.on_message = on_message

try:
    # If username is provided, configure credentials
    if MQTT_USERNAME:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()
except Exception as e:
    print(f"Failed to connect to MQTT broker: {e}")
