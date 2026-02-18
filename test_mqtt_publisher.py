"""
MQTT Test Publisher - Simulate Frigate events for testing.

Usage:
  python test_mqtt_publisher.py --broker localhost --port 1883

This publishes mock face and person detection events to your MQTT broker.
Monitor the identity_service.py output to see if matching is working.
"""

import os
import paho.mqtt.client as mqtt
import json
import time
import base64
import argparse
from PIL import Image
import io


def create_test_image(color_name="red", width=256, height=128):
    """Create a test person crop image."""
    colors = {
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'yellow': (255, 255, 0),
    }
    color = colors.get(color_name, (128, 128, 128))
    img = Image.new('RGB', (width, height), color=color)
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return base64.b64encode(img_byte_arr.read()).decode('utf-8')


def publish_face_event(client, person_name, confidence, camera, color='red'):
    """Publish a face recognition event."""
    event = {
        "type": "face",
        "name": person_name,
        "score": confidence,
        "camera": camera,
        "timestamp": int(time.time()),
        "id": f"face_{person_name}_{int(time.time()*1000)}",
        "image": create_test_image(color)
    }
    
    client.publish("frigate/events", json.dumps(event))
    print(f"Published FACE event: {person_name} at {camera} (confidence: {confidence:.2f})")


def publish_person_event(client, camera, color='red'):
    """Publish a person detection event (no face recognition)."""
    event = {
        "type": "person",
        "camera": camera,
        "timestamp": int(time.time()),
        "id": f"person_{camera}_{int(time.time()*1000)}",
        "image": create_test_image(color)
    }
    
    client.publish("frigate/events", json.dumps(event))
    print(f"Published PERSON event: {camera}")


def on_connect(client, userdata, flags, rc, properties=None):
    print(f"Connected to MQTT broker (code: {rc})")


def on_disconnect(client, userdata, *args):
    rc = None
    if len(args) >= 1:
        rc = args[0]
    if rc is not None and rc != 0:
        print(f"Unexpected disconnection (code: {rc})")


def main():
    parser = argparse.ArgumentParser(description='Publish mock Frigate MQTT events')
    parser.add_argument('--broker', default='localhost', help='MQTT broker address')
    parser.add_argument('--port', type=int, default=1883, help='MQTT broker port')
    parser.add_argument('--username', help='MQTT username (or set MQTT_USERNAME env)')
    parser.add_argument('--password', help='MQTT password (or set MQTT_PASSWORD env)')
    parser.add_argument('--test', choices=['basic', 'multiface', 'reid'], default='basic',
                       help='Test scenario to run')
    
    args = parser.parse_args()
    
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    # Determine credentials: CLI args take precedence, then environment variables
    mqtt_user = args.username if args.username else os.getenv("MQTT_USERNAME")
    mqtt_pass = args.password if args.password else os.getenv("MQTT_PASSWORD")
    if mqtt_user:
        client.username_pw_set(mqtt_user, mqtt_pass)
    
    try:
        print(f"Connecting to {args.broker}:{args.port}...")
        client.connect(args.broker, args.port, 60)
        client.loop_start()
        time.sleep(1)
        
        if args.test == 'basic':
            print("\n=== Test 1: Face Detection ===")
            publish_face_event(client, "alice", 0.98, "front_door", color='red')
            time.sleep(2)
            
            print("\n=== Test 2: Person Detection (should match alice) ===")
            publish_person_event(client, "porch", color='red')
            time.sleep(2)
        
        elif args.test == 'multiface':
            print("\n=== Test: Multiple Face Detections ===")
            publish_face_event(client, "alice", 0.95, "front_door", color='red')
            time.sleep(1)
            publish_face_event(client, "bob", 0.93, "back_door", color='blue')
            time.sleep(1)
            
            print("\n=== Person detections matching different people ===")
            publish_person_event(client, "hallway", color='red')
            time.sleep(1)
            publish_person_event(client, "garage", color='blue')
            time.sleep(2)
        
        elif args.test == 'reid':
            print("\n=== Test: Re-ID Matching ===")
            print("Publishing face of 'alice'...")
            publish_face_event(client, "alice", 0.96, "entry", color='red')
            time.sleep(2)
            
            print("Publishing person (same person, same color)...")
            publish_person_event(client, "hallway", color='red')
            time.sleep(2)
            
            print("Publishing person (different color - should NOT match)...")
            publish_person_event(client, "garage", color='yellow')
            time.sleep(2)
        
        print("\n=== Tests complete ===")
        print("Check identity_service.py output for matching results")
        print("Check Home Assistant MQTT sensors: identity/person/recognized and identity/person/tracked")
        
        client.loop_stop()
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
