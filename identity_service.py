import paho.mqtt.client as mqtt
import json
import time
from collections import defaultdict, deque
import os
import traceback
import requests
import base64
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

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
PERSONS_CONFIG_PATH = os.getenv("PERSONS_CONFIG_PATH", "persons.yaml")

# Global persons configuration dictionary
persons_config = {}
persons_config_lock = threading.Lock()

def validate_persons_config(config: dict) -> bool:
    """
    Validate the persons configuration dictionary.
    
    Args:
        config: Dictionary of person configurations keyed by person_id
        
    Returns:
        True if valid (with warnings for individual entries), False if completely invalid
    """
    if not isinstance(config, dict):
        print("[CONFIG] ERROR: Persons config must be a dictionary")
        return False
    
    if len(config) == 0:
        print("[CONFIG] WARNING: Persons config is empty")
        return True
    
    valid_roles = ["child", "adult", "unknown"]
    
    for person_id, person_data in config.items():
        if not isinstance(person_data, dict):
            print(f"[CONFIG] WARNING: Invalid entry for {person_id}: must be a dictionary")
            continue
        
        # Check for required name or display_name
        if "name" not in person_data and "display_name" not in person_data:
            print(f"[CONFIG] WARNING: Person {person_id} missing required 'name' or 'display_name' field")
        
        # Validate role if present
        if "role" in person_data:
            if person_data["role"] not in valid_roles:
                print(f"[CONFIG] WARNING: Person {person_id} has invalid role '{person_data['role']}'. Must be one of: {valid_roles}")
        
        # Validate age if present
        if "age" in person_data:
            age = person_data["age"]
            if not isinstance(age, int) or age <= 0:
                print(f"[CONFIG] WARNING: Person {person_id} has invalid age '{age}'. Must be a positive integer")
        
        # Validate supervision_required if present
        if "supervision_required" in person_data:
            if not isinstance(person_data["supervision_required"], bool):
                print(f"[CONFIG] WARNING: Person {person_id} has invalid supervision_required '{person_data['supervision_required']}'. Must be a boolean")
    
    return True

def load_persons_config(config_path: str) -> dict:
    """
    Load and parse persons configuration from YAML file.
    
    Args:
        config_path: Path to the persons.yaml file
        
    Returns:
        Dictionary of person configurations, or empty dict on error
    """
    if not os.path.exists(config_path):
        print(f"[CONFIG] WARNING: Persons config file not found at {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            print(f"[CONFIG] WARNING: Persons config file {config_path} is empty")
            return {}
        
        if not validate_persons_config(config):
            print(f"[CONFIG] ERROR: Invalid persons config in {config_path}")
            return {}
        
        print(f"[CONFIG] Successfully loaded persons config with {len(config)} entries")
        return config
    
    except yaml.YAMLError as e:
        print(f"[CONFIG] ERROR: Failed to parse YAML in {config_path}: {e}")
        return {}
    except Exception as e:
        print(f"[CONFIG] ERROR: Failed to load persons config from {config_path}: {e}")
        return {}

def reload_persons_config():
    """Reload the persons configuration from disk."""
    global persons_config
    new_config = load_persons_config(PERSONS_CONFIG_PATH)
    with persons_config_lock:
        persons_config = new_config
    print(f"[CONFIG] INFO: Persons config reloaded successfully")

class PersonsConfigFileHandler(FileSystemEventHandler):
    """File system event handler for watching persons config file changes."""
    
    def __init__(self, config_path):
        self.config_path = os.path.abspath(config_path)
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        if os.path.abspath(event.src_path) == self.config_path:
            print(f"[CONFIG] Detected change in {self.config_path}, reloading...")
            try:
                reload_persons_config()
            except Exception as e:
                print(f"[CONFIG] ERROR: Failed to reload config after file change: {e}")

def start_config_file_watcher():
    """Start a file watcher for the persons config file."""
    config_path = os.path.abspath(PERSONS_CONFIG_PATH)
    
    if not os.path.exists(config_path):
        print(f"[CONFIG] Config file {config_path} does not exist, skipping file watcher")
        return None
    
    watch_directory = os.path.dirname(config_path)
    
    event_handler = PersonsConfigFileHandler(config_path)
    observer = Observer()
    observer.schedule(event_handler, watch_directory, recursive=False)
    observer.daemon = True
    observer.start()
    
    print(f"[CONFIG] Started file watcher for {config_path}")
    return observer

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

# Load persons configuration
print(f"Loading persons configuration from {PERSONS_CONFIG_PATH}...")
persons_config = load_persons_config(PERSONS_CONFIG_PATH)

# Start file watcher for hot-reload
config_watcher = start_config_file_watcher()

# Track recent person detections per camera for snapshot correlation
camera_person_queue = defaultdict(lambda: deque(maxlen=MAX_TRACKED_PERSONS_PER_CAMERA))

# Cache snapshots to avoid redundant API calls
snapshot_cache = {}  # event_id -> (base64_image, timestamp)
CACHE_TTL = 60  # seconds

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
    identity_event = {
        "person_id": person_id,
        "camera": camera,
        "confidence": float(confidence),
        "source": source,
        "frigate_zones": zones,
        "event_id": event_id,
        "timestamp": int(timestamp * 1000) if isinstance(timestamp, float) else timestamp,
        "snapshot_url": f"{FRIGATE_HOST}/api/events/{event_id}/thumbnail.jpg?crop=1"
    }
    
    # Add person config data if available
    with persons_config_lock:
        if person_id in persons_config:
            person_data = persons_config[person_id]
            person_config_data = {}
            
            # Include relevant fields from person config
            if "display_name" in person_data:
                person_config_data["display_name"] = person_data["display_name"]
            elif "name" in person_data:
                person_config_data["display_name"] = person_data["name"]
            
            if "role" in person_data:
                person_config_data["role"] = person_data["role"]
            
            if "age" in person_data:
                person_config_data["age"] = person_data["age"]
            
            if "supervision_required" in person_data:
                person_config_data["supervision_required"] = person_data["supervision_required"]
            
            if person_config_data:
                identity_event["person_config"] = person_config_data
    
    # Publish to person-specific topic
    client.publish(f"identity/person/{person_id}", json.dumps(identity_event))
    print(f"[{source.upper()}] {person_id} at {camera} (zones: {zones}, confidence: {confidence:.3f})")

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
