"""
Meshtastic GPS tracker integration for Frigate Identity Service.

Receives Meshtastic node position updates via MQTT and correlates them with
Frigate camera zones and person tracks to produce a weighted identity signal.

This module is opt-in and disabled by default.  Face recognition and ReID
remain the authoritative identity mechanisms.
"""

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Haversine distance helper
# ---------------------------------------------------------------------------

_EARTH_RADIUS_M = 6_371_000.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in metres between two GPS co-ordinates."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return _EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Data classes (plain dicts wrapped in thin helpers)
# ---------------------------------------------------------------------------


class TrackerPosition:
    """Normalised Meshtastic position update."""

    __slots__ = (
        "tracker_id",
        "person_id",
        "asset_id",
        "display_name",
        "latitude",
        "longitude",
        "accuracy_m",
        "altitude_m",
        "speed_mps",
        "heading_degrees",
        "battery_percent",
        "received_at",
        "device_timestamp",
    )

    def __init__(
        self,
        tracker_id: str,
        person_id: Optional[str],
        asset_id: Optional[str],
        display_name: Optional[str],
        latitude: float,
        longitude: float,
        accuracy_m: Optional[float],
        altitude_m: Optional[float],
        speed_mps: Optional[float],
        heading_degrees: Optional[float],
        battery_percent: Optional[int],
        received_at: str,
        device_timestamp: Optional[int],
    ):
        self.tracker_id = tracker_id
        self.person_id = person_id
        self.asset_id = asset_id
        self.display_name = display_name
        self.latitude = latitude
        self.longitude = longitude
        self.accuracy_m = accuracy_m
        self.altitude_m = altitude_m
        self.speed_mps = speed_mps
        self.heading_degrees = heading_degrees
        self.battery_percent = battery_percent
        self.received_at = received_at
        self.device_timestamp = device_timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}

    @property
    def age_seconds(self) -> float:
        """Seconds since this position was received."""
        try:
            received = datetime.fromisoformat(self.received_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return (now - received).total_seconds()
        except Exception:
            return float("inf")


# ---------------------------------------------------------------------------
# Tracker config loader
# ---------------------------------------------------------------------------


def load_tracker_config(path: str) -> Dict[str, Dict[str, Any]]:
    """Load tracker→person/asset mappings from a JSON file.

    Returns an empty dict when the file does not exist or cannot be parsed.
    Only entries with ``"enabled": true`` (the default when the key is absent)
    are included.
    """
    p = Path(path)
    if not p.exists():
        logger.debug("[Meshtastic] Tracker config not found at %s", path)
        return {}
    try:
        with open(p, "r") as fh:
            raw: Dict[str, Any] = json.load(fh)
    except Exception as exc:
        logger.warning("[Meshtastic] Failed to load tracker config %s: %s", path, exc)
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for node_id, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("enabled", True):
            result[node_id] = cfg
    return result


# ---------------------------------------------------------------------------
# Camera zone config loader
# ---------------------------------------------------------------------------


def load_camera_zones(path: str) -> Dict[str, Dict[str, Any]]:
    """Load camera GPS zone definitions from a JSON file.

    Expected format::

        {
            "cameras": {
                "driveway": {"latitude": 40.1231, "longitude": -73.1239, "radius_meters": 35}
            }
        }

    Returns an empty dict when the file does not exist or lacks the key.
    """
    p = Path(path)
    if not p.exists():
        logger.debug("[Meshtastic] Camera zone config not found at %s", path)
        return {}
    try:
        with open(p, "r") as fh:
            raw: Dict[str, Any] = json.load(fh)
    except Exception as exc:
        logger.warning("[Meshtastic] Failed to load camera zone config %s: %s", path, exc)
        return {}
    return raw.get("cameras", {})


# ---------------------------------------------------------------------------
# Core integration class
# ---------------------------------------------------------------------------


class MeshtasticTracker:
    """Meshtastic integration component.

    Args:
        mqtt_topic:             MQTT topic pattern, e.g. ``meshtastic/+/json/receive``.
        max_position_age_s:     Discard positions older than this (seconds).
        max_gps_accuracy_m:     Discard positions with accuracy worse than this (metres).
        association_radius_m:   Radius within which a tracker is considered near a camera.
        tracker_weight:         Weight applied to tracker score in combined identity scoring.
        tracker_config:         Dict mapping Meshtastic node IDs to person/asset config.
        camera_zones:           Dict mapping camera names to GPS zone definitions.
    """

    def __init__(
        self,
        mqtt_topic: str = "meshtastic/+/json/receive",
        max_position_age_s: float = 90.0,
        max_gps_accuracy_m: float = 50.0,
        association_radius_m: float = 75.0,
        tracker_weight: float = 0.30,
        tracker_config: Optional[Dict[str, Dict[str, Any]]] = None,
        camera_zones: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.mqtt_topic = mqtt_topic
        self.max_position_age_s = max_position_age_s
        self.max_gps_accuracy_m = max_gps_accuracy_m
        self.association_radius_m = association_radius_m
        self.tracker_weight = tracker_weight
        self.tracker_config: Dict[str, Dict[str, Any]] = tracker_config or {}
        self.camera_zones: Dict[str, Dict[str, Any]] = camera_zones or {}

        # Latest valid position per tracker_id
        self._last_positions: Dict[str, TrackerPosition] = {}

    # ------------------------------------------------------------------
    # MQTT message parsing
    # ------------------------------------------------------------------

    def parse_position_message(
        self, topic: str, payload_bytes: bytes
    ) -> Optional[TrackerPosition]:
        """Parse a raw Meshtastic MQTT message into a TrackerPosition.

        Returns ``None`` when the payload is malformed, the node ID is unknown,
        the position is invalid, or the tracker is disabled.  Missing optional
        fields are tolerated.
        """
        # Extract node ID from topic or payload
        tracker_id = self._extract_node_id(topic, payload_bytes)
        if tracker_id is None:
            logger.debug("[Meshtastic] Could not extract node ID from topic=%s", topic)
            return None

        # Parse JSON
        try:
            data: Dict[str, Any] = json.loads(payload_bytes.decode("utf-8"))
        except Exception as exc:
            logger.warning("[Meshtastic] Malformed payload on %s: %s", topic, exc)
            return None

        # Look up tracker config (node ID must be configured)
        cfg = self.tracker_config.get(tracker_id)
        if cfg is None:
            logger.debug("[Meshtastic] Unknown tracker ID: %s", tracker_id)
            return None

        # Extract position sub-object (Meshtastic JSON gateway wraps position)
        position = self._extract_position(data)
        if position is None:
            logger.debug("[Meshtastic] No position data in payload for %s", tracker_id)
            return None

        lat = position.get("latitude_i") if position.get("latitude_i") is not None else position.get("latitude")
        lon = position.get("longitude_i") if position.get("longitude_i") is not None else position.get("longitude")

        # Meshtastic encodes lat/lon as integer * 1e-7
        if isinstance(lat, int) and abs(lat) > 180:
            lat = lat * 1e-7
        if isinstance(lon, int) and abs(lon) > 180:
            lon = lon * 1e-7

        # Validate coordinates
        if lat is None or lon is None:
            logger.debug("[Meshtastic] Missing lat/lon for tracker %s", tracker_id)
            return None
        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            logger.debug("[Meshtastic] Non-numeric lat/lon for tracker %s", tracker_id)
            return None
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            logger.debug("[Meshtastic] Invalid lat/lon (%.6f, %.6f) for tracker %s", lat, lon, tracker_id)
            return None

        accuracy_m: Optional[float] = None
        # Use PDOP or HDOP as a rough accuracy proxy; do NOT use precision_bits
        # (that field counts obfuscation bits, not metres).
        raw_acc = position.get("PDOP") or position.get("hdop")
        if raw_acc is not None:
            try:
                accuracy_m = float(raw_acc)
            except (TypeError, ValueError):
                pass

        # Also accept a direct "accuracy_m" field that some gateways provide
        if accuracy_m is None and "accuracy_m" in position:
            try:
                accuracy_m = float(position["accuracy_m"])
            except (TypeError, ValueError):
                pass

        altitude_m: Optional[float] = None
        if "altitude" in position:
            try:
                altitude_m = float(position["altitude"])
            except (TypeError, ValueError):
                pass

        speed_mps: Optional[float] = None
        if "ground_speed" in position:
            try:
                speed_mps = float(position["ground_speed"])
            except (TypeError, ValueError):
                pass

        heading: Optional[float] = None
        if "ground_track" in position:
            try:
                heading = float(position["ground_track"])
            except (TypeError, ValueError):
                pass

        battery: Optional[int] = None
        device_metrics = data.get("payload", {}).get("device_metrics") or data.get("device_metrics")
        if device_metrics and "battery_level" in device_metrics:
            try:
                battery = int(device_metrics["battery_level"])
            except (TypeError, ValueError):
                pass

        device_ts: Optional[int] = None
        raw_ts = position.get("time") or data.get("timestamp")
        if raw_ts is not None:
            try:
                device_ts = int(raw_ts)
            except (TypeError, ValueError):
                pass

        received_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        pos = TrackerPosition(
            tracker_id=tracker_id,
            person_id=cfg.get("person_id"),
            asset_id=cfg.get("asset_id"),
            display_name=cfg.get("display_name"),
            latitude=lat,
            longitude=lon,
            accuracy_m=accuracy_m,
            altitude_m=altitude_m,
            speed_mps=speed_mps,
            heading_degrees=heading,
            battery_percent=battery,
            received_at=received_at,
            device_timestamp=device_ts,
        )
        return pos

    def _extract_node_id(self, topic: str, payload_bytes: bytes) -> Optional[str]:
        """Try to extract the Meshtastic node ID from the topic or payload."""
        # Common topic patterns:
        #   meshtastic/<gateway>/json/receive/<node_id>
        #   meshtastic/<gateway>/json/receive  (node ID in payload)
        parts = topic.split("/")
        # Check if last segment looks like a node ID (!xxxxxxxx)
        for part in reversed(parts):
            if part.startswith("!") and len(part) > 4:
                return part

        # Fall back to payload "from" or "fromId" field
        try:
            data = json.loads(payload_bytes.decode("utf-8"))
            for key in ("from", "fromId", "from_id", "sender"):
                val = data.get(key)
                if val:
                    # Meshtastic uses unsigned int node IDs; convert to !hex
                    if isinstance(val, int):
                        return f"!{val:08x}"
                    if isinstance(val, str):
                        return val if val.startswith("!") else f"!{val}"
        except Exception:
            pass
        return None

    def _extract_position(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Navigate Meshtastic JSON envelope to find position object."""
        # Gateway format: {"payload": {"position": {...}}}
        payload = data.get("payload", {})
        if isinstance(payload, dict):
            pos = payload.get("position")
            if isinstance(pos, dict):
                return pos

        # Alternative flat format or nested differently
        pos = data.get("position")
        if isinstance(pos, dict):
            return pos

        # Some gateways flatten all fields at top level
        if "latitude_i" in data or "latitude" in data:
            return data

        return None

    # ------------------------------------------------------------------
    # Position store
    # ------------------------------------------------------------------

    def update_position(self, position: TrackerPosition) -> bool:
        """Store a parsed position after validating age and accuracy.

        Returns ``True`` when the position was accepted, ``False`` when discarded.
        """
        if position.age_seconds > self.max_position_age_s:
            logger.debug(
                "[Meshtastic] Stale position for %s (age=%.1fs > %.1fs)",
                position.tracker_id,
                position.age_seconds,
                self.max_position_age_s,
            )
            return False

        if (
            position.accuracy_m is not None
            and position.accuracy_m > self.max_gps_accuracy_m
        ):
            logger.debug(
                "[Meshtastic] Poor accuracy for %s (%.1fm > %.1fm)",
                position.tracker_id,
                position.accuracy_m,
                self.max_gps_accuracy_m,
            )
            return False

        self._last_positions[position.tracker_id] = position
        logger.info(
                "[Meshtastic] Position updated for %s (person=%s)",
                position.tracker_id,
                position.person_id,
            )
        return True

    def get_position(self, tracker_id: str) -> Optional[TrackerPosition]:
        """Return the most recent valid position for *tracker_id*, or None."""
        pos = self._last_positions.get(tracker_id)
        if pos is None:
            return None
        # Re-check age on retrieval
        if pos.age_seconds > self.max_position_age_s:
            logger.debug(
                "[Meshtastic] Evicting stale position for %s (age=%.1fs)",
                tracker_id,
                pos.age_seconds,
            )
            del self._last_positions[tracker_id]
            return None
        return pos

    def get_all_valid_positions(self) -> List[TrackerPosition]:
        """Return all non-stale positions, evicting stale ones."""
        stale = [
            tid
            for tid, pos in self._last_positions.items()
            if pos.age_seconds > self.max_position_age_s
        ]
        for tid in stale:
            del self._last_positions[tid]
        return list(self._last_positions.values())

    # ------------------------------------------------------------------
    # Camera zone correlation
    # ------------------------------------------------------------------

    def cameras_near_position(
        self, lat: float, lon: float, extra_radius_m: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Return cameras whose configured zone overlaps the given GPS position.

        Args:
            lat, lon:       GPS co-ordinates to test.
            extra_radius_m: Additional radius (e.g. GPS accuracy) to add to each
                            camera's configured radius before testing overlap.

        Returns:
            List of (camera_name, distance_m) sorted by ascending distance.
        """
        results: List[Tuple[str, float]] = []
        for cam_name, zone in self.camera_zones.items():
            cam_lat = zone.get("latitude")
            cam_lon = zone.get("longitude")
            radius = zone.get("radius_meters", self.association_radius_m)
            if cam_lat is None or cam_lon is None:
                continue
            dist = haversine_distance(lat, lon, cam_lat, cam_lon)
            if dist <= (radius + extra_radius_m):
                results.append((cam_name, dist))
        results.sort(key=lambda x: x[1])
        return results

    def trackers_near_camera(self, camera_name: str) -> List[Tuple[TrackerPosition, float]]:
        """Return all current valid tracker positions near *camera_name*.

        Returns list of (position, distance_m) sorted by ascending distance.
        """
        zone = self.camera_zones.get(camera_name)
        if zone is None:
            return []
        cam_lat = zone.get("latitude")
        cam_lon = zone.get("longitude")
        if cam_lat is None or cam_lon is None:
            return []
        radius = zone.get("radius_meters", self.association_radius_m)

        results: List[Tuple[TrackerPosition, float]] = []
        for pos in self.get_all_valid_positions():
            extra = pos.accuracy_m or 0.0
            dist = haversine_distance(pos.latitude, pos.longitude, cam_lat, cam_lon)
            if dist <= (radius + extra):
                results.append((pos, dist))
        results.sort(key=lambda x: x[1])
        return results

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute_tracker_score(
        self,
        position: TrackerPosition,
        camera_name: str,
    ) -> float:
        """Return a 0–1 score indicating how strongly the tracker supports
        associating *camera_name* with the tracker's person/asset.

        Score components:
        - Proximity: how close the tracker is to the camera zone.
        - Recency: how fresh the position is.
        - Accuracy: penalise poor GPS accuracy.
        """
        zone = self.camera_zones.get(camera_name)
        if zone is None:
            return 0.0
        cam_lat = zone.get("latitude")
        cam_lon = zone.get("longitude")
        if cam_lat is None or cam_lon is None:
            return 0.0

        radius = zone.get("radius_meters", self.association_radius_m)
        dist = haversine_distance(position.latitude, position.longitude, cam_lat, cam_lon)

        # Proximity component: 1.0 at distance=0, 0.0 at distance=radius
        max_dist = radius + (position.accuracy_m or 0.0)
        if max_dist <= 0:
            proximity = 0.0
        elif dist > max_dist:
            proximity = 0.0
        else:
            proximity = max(0.0, 1.0 - dist / max_dist)

        # Recency component: 1.0 when fresh, 0.0 at max_position_age_s
        age = position.age_seconds
        recency = max(0.0, 1.0 - age / self.max_position_age_s) if self.max_position_age_s > 0 else 0.0

        # Accuracy penalty: 1.0 when perfect GPS, 0.5 at max_gps_accuracy_m
        if position.accuracy_m is None or self.max_gps_accuracy_m <= 0:
            accuracy_factor = 1.0
        else:
            accuracy_factor = max(0.5, 1.0 - 0.5 * position.accuracy_m / self.max_gps_accuracy_m)

        score = proximity * recency * accuracy_factor
        return round(min(1.0, score), 4)

    def tracker_signal_for_person(
        self, person_id: str, camera_name: str
    ) -> Optional[Dict[str, Any]]:
        """Return tracker correlation metadata for *person_id* at *camera_name*.

        Returns ``None`` when no recent tracker position is available for that person.
        """
        for pos in self.get_all_valid_positions():
            if pos.person_id != person_id:
                continue
            score = self.compute_tracker_score(pos, camera_name)
            if score <= 0.0:
                continue

            zone = self.camera_zones.get(camera_name)
            cam_lat = zone.get("latitude") if zone else None
            cam_lon = zone.get("longitude") if zone else None
            dist = (
                haversine_distance(pos.latitude, pos.longitude, cam_lat, cam_lon)
                if cam_lat is not None and cam_lon is not None
                else None
            )

            return {
                "tracker_id": pos.tracker_id,
                "person_id": person_id,
                "tracker_age_seconds": round(pos.age_seconds, 1),
                "gps_accuracy_m": pos.accuracy_m,
                "nearest_camera": camera_name,
                "distance_to_camera_m": round(dist, 1) if dist is not None else None,
                "association_confidence": score,
                "used_for_identity": True,
            }
        return None

    # ------------------------------------------------------------------
    # MQTT topic matching
    # ------------------------------------------------------------------

    def matches_topic(self, topic: str) -> bool:
        """Return True when *topic* matches the configured Meshtastic MQTT pattern."""
        pattern_parts = self.mqtt_topic.split("/")
        topic_parts = topic.split("/")
        if len(pattern_parts) != len(topic_parts):
            # Allow trailing wildcard '#'
            if "#" in self.mqtt_topic:
                return topic.startswith(self.mqtt_topic.split("#")[0].rstrip("/"))
            return False
        for pp, tp in zip(pattern_parts, topic_parts):
            if pp == "#":
                return True
            if pp != "+" and pp != tp:
                return False
        return True


# ---------------------------------------------------------------------------
# Factory / convenience builder
# ---------------------------------------------------------------------------


def build_meshtastic_tracker_from_env() -> Optional["MeshtasticTracker"]:
    """Construct a MeshtasticTracker from environment variables.

    Returns ``None`` when ``MESHTASTIC_ENABLED`` is not ``"true"``.
    """
    enabled = os.getenv("MESHTASTIC_ENABLED", "false").lower() == "true"
    if not enabled:
        return None

    topic = os.getenv("MESHTASTIC_MQTT_TOPIC", "meshtastic/+/json/receive")
    max_age = float(os.getenv("MESHTASTIC_MAX_POSITION_AGE_SECONDS", "90"))
    max_accuracy = float(os.getenv("MESHTASTIC_MAX_GPS_ACCURACY_METERS", "50"))
    assoc_radius = float(os.getenv("MESHTASTIC_ASSOCIATION_RADIUS_METERS", "75"))
    weight = float(os.getenv("MESHTASTIC_WEIGHT", "0.30"))
    tracker_config_path = os.getenv("MESHTASTIC_TRACKER_CONFIG_PATH", "trackers.json")
    camera_zones_path = os.getenv("MESHTASTIC_CAMERA_ZONES_PATH", "camera_zones.json")

    tracker_config = load_tracker_config(tracker_config_path)
    camera_zones = load_camera_zones(camera_zones_path)

    logger.info(
        "[Meshtastic] Enabled — topic=%s, max_age=%.0fs, max_accuracy=%.0fm, "
        "assoc_radius=%.0fm, weight=%.2f, trackers=%d, cameras=%d",
        topic,
        max_age,
        max_accuracy,
        assoc_radius,
        weight,
        len(tracker_config),
        len(camera_zones),
    )

    return MeshtasticTracker(
        mqtt_topic=topic,
        max_position_age_s=max_age,
        max_gps_accuracy_m=max_accuracy,
        association_radius_m=assoc_radius,
        tracker_weight=weight,
        tracker_config=tracker_config,
        camera_zones=camera_zones,
    )
