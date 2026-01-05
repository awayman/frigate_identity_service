import paho.mqtt.client as mqtt
import json
import time
from collections import defaultdict
import os

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))

# Track known persons and their last seen location/time
person_tracking = defaultdict(lambda: {
    "last_camera": None,
    "last_timestamp": None,
    "confidence": 0
})

def on_connect(client, userdata, flags, rc):
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
            
            # Store tracking information
            person_tracking[person_name]["last_camera"] = camera
            person_tracking[person_name]["last_timestamp"] = timestamp
            person_tracking[person_name]["confidence"] = confidence
            
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
        
        # Handle person detection events on weaker cameras
        elif payload.get("type") == "person":
            camera = payload.get("camera")
            timestamp = payload.get("timestamp")
            event_id = payload.get("id")
            
            # Try to match with recently seen persons using temporal/spatial context
            matched_person = None
            best_score = 0
            
            for person_name, tracking_data in person_tracking.items():
                # Simple heuristic: if person was seen recently (within 30 seconds)
                # on a nearby camera, assume it's the same person
                time_diff = abs(timestamp - tracking_data["last_timestamp"])
                
                if time_diff < 30:  # Within 30 seconds
                    # In a real system, you'd use spatial proximity here
                    score = tracking_data["confidence"] * (1 - time_diff / 30)
                    if score > best_score:
                        best_score = score
                        matched_person = person_name
            
            # If we found a match, propagate the identity
            if matched_person and best_score > 0.5:
                identity_event = {
                    "person_id": matched_person,
                    "confidence": best_score,
                    "camera": camera,
                    "timestamp": timestamp,
                    "frigate_event_id": event_id,
                    "source": "reid_propagation"
                }
                
                client.publish("identity/person/tracked", json.dumps(identity_event))
                print(f"[REID] {matched_person} tracked at {camera} (confidence: {best_score:.2f})")

    except Exception as e:
        print(f"Error processing MQTT message: {e}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()
except Exception as e:
    print(f"Failed to connect to MQTT broker: {e}")