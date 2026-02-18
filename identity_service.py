import paho.mqtt.client as mqtt
import json
import time
from collections import defaultdict
import os
import traceback

from embedding_store import EmbeddingStore
from reid_model import ReIDModel
from matcher import EmbeddingMatcher

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
REID_MODEL = os.getenv("REID_MODEL", "osnet_x1_0")
REID_DEVICE = os.getenv("REID_DEVICE", "auto")
REID_SIMILARITY_THRESHOLD = float(os.getenv("REID_SIMILARITY_THRESHOLD", "0.6"))
EMBEDDINGS_DB_PATH = os.getenv("EMBEDDINGS_DB_PATH", "embeddings.json")

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

# Track known persons and their last seen location/time
person_tracking = defaultdict(lambda: {
    "last_camera": None,
    "last_timestamp": None,
    "confidence": 0
})

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"Connected to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}")
    client.subscribe("frigate/events")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        
        # Handle face recognition events
        if payload.get("type") == "face":
            person_name = payload.get("name")
            confidence = payload.get("score")
            camera = payload.get("camera")
            timestamp = payload.get("timestamp")
            event_id = payload.get("id")
            image_base64 = payload.get("image")  # Base64-encoded person crop
            
            # Store tracking information
            person_tracking[person_name]["last_camera"] = camera
            person_tracking[person_name]["last_timestamp"] = timestamp
            person_tracking[person_name]["confidence"] = confidence
            
            # Extract and store embedding if image is provided
            if image_base64 and REID_AVAILABLE:
                try:
                    embedding = reid_model.extract_embedding(image_base64)
                    embedding_store.store_embedding(person_name, embedding, camera, confidence)
                    print(f"[FACE] Stored embedding for {person_name} at {camera}")
                except Exception as e:
                    print(f"[FACE] Warning: Could not extract embedding for {person_name}: {e}")
            
            # Publish identity event
            identity_event = {
                "person_id": person_name,
                "confidence": confidence,
                "checkpoint": camera,
                "timestamp": timestamp,
                "frigate_event_id": event_id,
                "camera": camera,
                "source": "facial_recognition"
            }
            
            client.publish("identity/person/recognized", json.dumps(identity_event))
            print(f"[FACE] {person_name} recognized at {camera} (confidence: {confidence:.2f})")
        
        # Handle person detection events without facial recognition
        elif payload.get("type") == "person":
            camera = payload.get("camera")
            timestamp = payload.get("timestamp")
            event_id = payload.get("id")
            image_base64 = payload.get("image")  # Base64-encoded person crop
            
            if not REID_AVAILABLE:
                print(f"[REID] ReID model not available, skipping re-identification at {camera}")
                return
            
            if not image_base64:
                print(f"[REID] Warning: Person detection event has no image, skipping re-id")
                return
            
            try:
                # Extract embedding from the person crop
                query_embedding = reid_model.extract_embedding(image_base64)
                
                # Get all stored embeddings
                stored_embeddings = embedding_store.get_all_embeddings()
                
                # Find best match
                matched_person, similarity_score = EmbeddingMatcher.find_best_match(
                    query_embedding,
                    stored_embeddings,
                    threshold=REID_SIMILARITY_THRESHOLD
                )
                
                # If we found a match, propagate the identity
                if matched_person:
                    # Ensure numeric types are JSON-serializable (convert numpy.float32 etc. to native Python float)
                    score = float(similarity_score)
                    identity_event = {
                        "person_id": matched_person,
                        "confidence": score,
                        "camera": camera,
                        "timestamp": int(timestamp) if timestamp is not None else None,
                        "frigate_event_id": event_id,
                        "source": "reid_model",
                        "similarity_score": score
                    }

                    client.publish("identity/person/tracked", json.dumps(identity_event))
                    print(f"[REID] {matched_person} tracked at {camera} (similarity: {score:.4f})")
                else:
                    # No match found - this might be a new person or a low-confidence match
                    print(f"[REID] No person match found at {camera} (best similarity was {similarity_score:.4f})")

            except Exception as e:
                print(f"[REID] Error processing person detection at {camera}: {e}")
                traceback.print_exc()

    except Exception as e:
        print(f"Error processing MQTT message: {e}")
        traceback.print_exc()

client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
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
