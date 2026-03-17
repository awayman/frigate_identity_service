"""Frigate API helper functions for integration testing.

Provides reusable functions for:
- Fetching recent events from Frigate API
- Getting snapshot URLs and fetching snapshot data
- Validating JPEG images
"""

import logging
from typing import Optional, List, Tuple
import requests
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


def fetch_recent_events(
    session: requests.Session,
    frigate_url: str,
    limit: int = 10,
    object_type: str = "person",
    after: Optional[int] = None,
    before: Optional[int] = None,
) -> List[dict]:
    """Fetch recent events from Frigate API.

    Uses /api/events with query params because this endpoint consistently
    honors limit and time filters (`after`/`before`) across Frigate versions.

    Args:
        session: requests.Session with Frigate auth and config
        frigate_url: Base URL for Frigate (e.g., http://localhost:5000)
        limit: Maximum number of events to fetch
        object_type: Label type to filter by (e.g., 'person')
        after: Optional Unix timestamp (seconds) lower bound (inclusive)
        before: Optional Unix timestamp (seconds) upper bound (exclusive)

    Returns:
        List of event dictionaries from Frigate API
    """
    try:
        url = f"{frigate_url}/api/events"
        params = {
            "limit": limit,
            "has_snapshot": 1,
            "label": object_type,
        }
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        events = response.json()
        logger.info(f"Fetched {len(events)} recent {object_type} events from Frigate")
        return events
    except requests.RequestException as e:
        logger.error(f"Failed to fetch recent events from Frigate: {e}")
        return []


def fetch_known_faces(
    session: requests.Session,
    frigate_url: str,
) -> dict:
    """Fetch known faces from Frigate's facial recognition system.

    Returns a dictionary mapping person names to their face image filenames.
    Also includes 'train' data with event IDs that have facial recognition results.

    Args:
        session: requests.Session with Frigate auth and config
        frigate_url: Base URL for Frigate

    Returns:
        Dictionary with person names as keys and list of face images/events as values
    """
    try:
        url = f"{frigate_url}/api/faces"
        response = session.get(url, timeout=10)
        response.raise_for_status()
        faces = response.json()

        # Count known persons (exclude 'train' key)
        known_persons = [k for k in faces.keys() if k != "train"]
        logger.info(
            f"Fetched {len(known_persons)} known faces from Frigate: {known_persons}"
        )

        return faces
    except requests.RequestException as e:
        logger.error(f"Failed to fetch known faces from Frigate: {e}")
        return {}


def parse_face_training_events(faces_data: dict) -> List[Tuple[str, str, float]]:
    """Parse face training data to extract event IDs with person names.

    The 'train' key in faces data contains filenames like:
    '1772147544.726833-7azxn1-1772147548.138356-Hannah-0.89.webp'
    which decode to: event_id-camera-timestamp-person_name-confidence

    Args:
        faces_data: Dictionary from fetch_known_faces()

    Returns:
        List of (event_id, person_name, confidence) tuples
    """
    parsed_events: List[Tuple[float, str, str, float]] = []

    train_data = faces_data.get("train", [])
    for filename in train_data:
        try:
            # Parse filename: event_id-camera-timestamp-person_name-confidence.webp
            parts = filename.replace(".webp", "").split("-")
            if len(parts) >= 5:
                event_id = f"{parts[0]}-{parts[1]}"  # First two parts = event_id
                event_timestamp = float(parts[2])  # Third part = event timestamp
                person_name = parts[3]  # Fourth part = person name
                confidence = float(parts[4])  # Fifth part = confidence

                # Only include recognized faces (not "unknown" and high confidence)
                if person_name != "unknown" and confidence >= 0.80:
                    parsed_events.append(
                        (event_timestamp, event_id, person_name, confidence)
                    )
        except (ValueError, IndexError):
            # Skip malformed filenames
            continue

    # Prefer newest facial-recognition training events first
    parsed_events.sort(key=lambda event: event[0], reverse=True)
    training_events = [
        (event_id, person_name, confidence)
        for _, event_id, person_name, confidence in parsed_events
    ]

    logger.info(f"Parsed {len(training_events)} facial recognition training events")
    return training_events


def fetch_event_by_id(
    session: requests.Session,
    frigate_url: str,
    event_id: str,
) -> Optional[dict]:
    """Fetch a specific event by ID from Frigate API.

    Args:
        session: requests.Session with Frigate auth and config
        frigate_url: Base URL for Frigate
        event_id: Event ID to fetch

    Returns:
        Event dictionary or None if not found
    """
    try:
        url = f"{frigate_url}/api/events/{event_id}"
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.debug(f"Failed to fetch event {event_id}: {e}")
        return None


def get_event_snapshot_url(frigate_url: str, event_id: str) -> str:
    """Get the snapshot URL for a specific Frigate event.

    Args:
        frigate_url: Base URL for Frigate
        event_id: Event ID from Frigate API

    Returns:
        Full URL to the event snapshot
    """
    frigate_url = frigate_url.rstrip("/")
    return f"{frigate_url}/api/events/{event_id}/thumbnail.jpg"


def fetch_snapshot_bytes(
    session: requests.Session,
    url: str,
    crop: bool = True,
    quality: int = 85,
    height: int = 400,
    timeout: int = 10,
) -> Optional[bytes]:
    """Fetch snapshot image bytes from a URL.

    Args:
        session: requests.Session with timeout and auth configured
        url: Full URL to snapshot
        crop: Whether to crop to the detection box
        quality: JPEG quality (1-100)
        height: Maximum height in pixels
        timeout: Request timeout in seconds

    Returns:
        Image bytes if successful, None if failed
    """
    try:
        params = {
            "crop": "1" if crop else "0",
            "quality": quality,
            "h": height,
        }
        response = session.get(url, params=params, timeout=timeout)
        response.raise_for_status()

        if response.status_code == 200:
            return response.content
        else:
            logger.warning(f"Failed to fetch snapshot: HTTP {response.status_code}")
            return None
    except requests.RequestException as e:
        logger.error(f"Error fetching snapshot: {e}")
        return None


def validate_image(image_bytes: bytes) -> bool:
    """Validate that image bytes are a valid image (JPEG, WebP, PNG, etc.).

    Args:
        image_bytes: Raw image bytes to validate

    Returns:
        True if valid image format, False otherwise
    """
    try:
        if len(image_bytes) < 8:
            return False

        # Check for common image format magic bytes
        # JPEG: FF D8 FF
        # PNG: 89 50 4E 47
        # WebP: RIFF...WEBP
        is_jpeg = image_bytes[0:2] == b"\xff\xd8"
        is_png = image_bytes[0:4] == b"\x89PNG"
        is_webp = image_bytes[0:4] == b"RIFF" and image_bytes[8:12] == b"WEBP"

        if not (is_jpeg or is_png or is_webp):
            logger.debug(f"Unknown image format, first 12 bytes: {image_bytes[0:12]}")
            # Still try PIL as a fallback

        # Try to open with PIL to ensure it's a valid image
        img = Image.open(BytesIO(image_bytes))
        img.verify()

        return True
    except Exception as e:
        logger.warning(f"Image validation failed: {e}")
        return False


# Backwards compatibility alias
validate_jpeg = validate_image
