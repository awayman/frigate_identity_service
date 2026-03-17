#!/usr/bin/env python3
"""
Mock Frigate NVR service for testing frigate_identity_service.

Simulates realistic Frigate MQTT events including:
- Person detection events with base64-encoded face images
- Tracked object updates
- Snapshot frames

This service generates synthetic colored rectangle images to simulate
face detections and publishes them to the same MQTT topics that
a real Frigate instance would use.
"""

import base64
import json
import logging
import os
import time
from io import BytesIO
from typing import Any, Dict

import numpy as np
import paho.mqtt.client as mqtt
from PIL import Image, ImageDraw

# Configure logging
logging.basicConfig(
    level=os.getenv("LOGGING_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# MQTT Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC_PREFIX = os.getenv("MQTT_TOPIC_PREFIX", "frigate")
EVENT_INTERVAL = int(os.getenv("EVENT_INTERVAL", 20))  # seconds between event cycles

# Test camera name
CAMERA_NAME = "front_door"

# Test person configurations - each represents a detected person
TEST_PERSONS = [
    {
        "id": "person_1",
        "label": "person",
        "color": (255, 0, 0),  # Red
        "confidence": 0.95,
    },
    {
        "id": "person_2",
        "label": "person",
        "color": (0, 255, 0),  # Green
        "confidence": 0.92,
    },
    {
        "id": "person_3",
        "label": "person",
        "color": (0, 0, 255),  # Blue
        "confidence": 0.88,
    },
]


def generate_synthetic_image(
    width: int = 320,
    height: int = 240,
    color: tuple = (255, 0, 0),
    with_face_box: bool = True,
) -> bytes:
    """
    Generate a synthetic image with a colored rectangle (simulating a face).

    Args:
        width: Image width in pixels
        height: Image height in pixels
        color: RGB color tuple for the rectangle
        with_face_box: Whether to draw a face detection box

    Returns:
        JPEG image data as bytes
    """
    # Create a blank image with a random background
    bg_color = (
        np.random.randint(50, 100),
        np.random.randint(50, 100),
        np.random.randint(50, 100),
    )
    image = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)

    if with_face_box:
        # Draw a colored rectangle to simulate face detection
        box_width = width // 3
        box_height = height // 2
        x_offset = (width - box_width) // 2
        y_offset = (height - box_height) // 3

        # Draw filled rectangle
        draw.rectangle(
            [
                (x_offset, y_offset),
                (x_offset + box_width, y_offset + box_height),
            ],
            fill=color,
            outline=(255, 255, 255),
            width=2,
        )

        # Add a label text
        text_x = x_offset + 5
        text_y = y_offset + 5
        draw.text(
            (text_x, text_y),
            "Face",
            fill=(255, 255, 255),
            font=None,
        )

    # Convert to JPEG bytes
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def encode_image_to_base64(image_data: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_data).decode("utf-8")


def create_event_payload(person: Dict[str, Any], event_index: int) -> Dict[str, Any]:
    """Create a realistic Frigate event payload."""
    event_id = f"{person['id']}_event_{event_index}"

    return {
        "type": "new",
        "before": {
            "id": event_id,
            "label": person["label"],
            "stationary": False,
            "motionless_count": 0,
            "frame": 1000 + event_index,
            "region": [50, 50, 100, 150],
            "confidence": person["confidence"],
            "rel_mass": 0.15,
        },
        "after": {
            "id": event_id,
            "label": person["label"],
            "stationary": False,
            "motionless_count": 0,
            "frame": 1000 + event_index,
            "region": [50, 50, 100, 150],
            "confidence": person["confidence"],
            "rel_mass": 0.15,
        },
        "camera": CAMERA_NAME,
    }


def create_tracked_object_update(
    person: Dict[str, Any], frame_number: int
) -> Dict[str, Any]:
    """Create a tracked object update payload (face metadata)."""
    return {
        "type": "update",
        "camera": CAMERA_NAME,
        "frame_time": time.time(),
        "object_type": person["label"],
        "object_id": person["id"],
        "region": [50, 50, 100, 150],
        "confidence": person["confidence"],
        "label": person["label"],
        "motion": 0.5,
        "frame_count": frame_number,
        "attributes": [
            {
                "name": "face_detected",
                "value": True,
                "confidence": person["confidence"],
            }
        ],
    }


def create_snapshot_payload(
    person: Dict[str, Any], image_base64: str
) -> Dict[str, Any]:
    """Create a person snapshot payload with base64-encoded image."""
    return {
        "type": "snapshot",
        "camera": CAMERA_NAME,
        "object_id": person["id"],
        "label": person["label"],
        "confidence": person["confidence"],
        "frame_time": time.time(),
        "image": image_base64,
    }


def on_connect(
    client: mqtt.Client,
    userdata: Any,
    connect_flags: Any,
    rc: int,
    properties: Any = None,
) -> None:
    """MQTT connection callback."""
    if rc == 0:
        logger.info(f"Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
    else:
        logger.error(f"Failed to connect to MQTT broker, return code {rc}")


def on_disconnect(
    client: mqtt.Client,
    userdata: Any,
    disconnect_flags: Any,
    rc: int,
    properties: Any = None,
) -> None:
    """MQTT disconnection callback."""
    if rc != 0:
        logger.warning(f"Unexpected MQTT disconnection, return code {rc}")
    else:
        logger.info("Disconnected from MQTT broker")


def on_publish(
    client: mqtt.Client, userdata: Any, mid: int, properties: Any = None, rc: int = None
) -> None:
    """MQTT publish callback (for paho-mqtt 2.x compatibility)."""
    pass  # Silent logging for publish events


def publish_event_cycle(client: mqtt.Client, event_index: int) -> None:
    """Publish one complete cycle of events for all test persons."""
    try:
        for person in TEST_PERSONS:
            # Generate a synthetic image for this person
            image_data = generate_synthetic_image(color=person["color"])
            image_base64 = encode_image_to_base64(image_data)

            # Publish event
            event_payload = create_event_payload(person, event_index)
            event_topic = f"{MQTT_TOPIC_PREFIX}/events"
            client.publish(event_topic, json.dumps(event_payload), qos=1)
            logger.debug(f"Published event to {event_topic}: {person['id']}")

            # Publish tracked object update
            tracked_update = create_tracked_object_update(person, 1000 + event_index)
            tracked_topic = f"{MQTT_TOPIC_PREFIX}/tracked_object_update"
            client.publish(tracked_topic, json.dumps(tracked_update), qos=1)
            logger.debug(f"Published update to {tracked_topic}: {person['id']}")

            # Publish snapshot with image
            snapshot_payload = create_snapshot_payload(person, image_base64)
            snapshot_topic = f"{MQTT_TOPIC_PREFIX}/{CAMERA_NAME}/person/snapshot"
            client.publish(snapshot_topic, json.dumps(snapshot_payload), qos=1)
            logger.info(
                f"Published snapshot to {snapshot_topic}: {person['id']} "
                f"(confidence: {person['confidence']})"
            )

            # Small delay between person publications
            time.sleep(0.5)

    except Exception as e:
        logger.error(f"Error publishing event cycle: {e}", exc_info=True)


def main() -> None:
    """Main loop for mock Frigate service."""
    logger.info("Mock Frigate service starting...")
    logger.info(f"MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    logger.info(f"Event interval: {EVENT_INTERVAL} seconds")
    logger.info(f"Camera: {CAMERA_NAME}")

    # Create MQTT client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish = on_publish

    # Connect to MQTT broker
    max_retries = 30
    retry_count = 0
    while retry_count < max_retries:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            client.loop_start()
            logger.info("MQTT client loop started")
            break
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(
                    f"Failed to connect to MQTT broker (attempt {retry_count}/{max_retries}): {e}"
                )
                time.sleep(5)
            else:
                logger.error(
                    f"Failed to connect after {max_retries} attempts. Exiting."
                )
                exit(1)

    # Main event publishing loop
    event_index = 0
    try:
        logger.info("Starting event publishing loop...")
        while True:
            publish_event_cycle(client, event_index)
            event_index += 1
            time.sleep(EVENT_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        client.loop_stop()
        client.disconnect()
        logger.info("Mock Frigate service stopped")


if __name__ == "__main__":
    main()
