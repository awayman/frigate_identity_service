"""
Tests for the Meshtastic GPS tracker integration.

Run with: python -m pytest tests/test_meshtastic.py -v
"""

import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make sure the frigate_identity_service package is importable
sys.path.insert(
    0,
    str(Path(__file__).parent.parent / "frigate_identity_service"),
)

from meshtastic_tracker import (
    MeshtasticTracker,
    TrackerPosition,
    haversine_distance,
    load_camera_zones,
    load_tracker_config,
    build_meshtastic_tracker_from_env,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TRACKER_CONFIG = {
    "!a1b2c3d4": {
        "person_id": "alice",
        "display_name": "Alice",
        "enabled": True,
    },
    "!e5f6g7h8": {
        "asset_id": "tractor",
        "display_name": "Tractor",
        "enabled": True,
    },
    "!disabled01": {
        "person_id": "bob",
        "display_name": "Bob",
        "enabled": False,
    },
}

SAMPLE_CAMERA_ZONES = {
    "driveway": {
        "latitude": 40.1231,
        "longitude": -73.1239,
        "radius_meters": 35,
    },
    "garden": {
        "latitude": 40.1237,
        "longitude": -73.1232,
        "radius_meters": 30,
    },
}


@pytest.fixture
def tracker():
    return MeshtasticTracker(
        mqtt_topic="meshtastic/+/json/receive",
        max_position_age_s=90.0,
        max_gps_accuracy_m=50.0,
        association_radius_m=75.0,
        tracker_weight=0.30,
        tracker_config=SAMPLE_TRACKER_CONFIG,
        camera_zones=SAMPLE_CAMERA_ZONES,
    )


def _make_payload(
    node_id: str = "!a1b2c3d4",
    lat: float = 40.1231,
    lon: float = -73.1239,
    accuracy_m: float = 10.0,
    altitude: float = 42.0,
    device_ts: int = 1787500000,
) -> bytes:
    payload = {
        "from": int(node_id.lstrip("!"), 16),
        "payload": {
            "position": {
                "latitude_i": int(lat * 1e7),
                "longitude_i": int(lon * 1e7),
                "accuracy_m": accuracy_m,
                "altitude": altitude,
                "time": device_ts,
            }
        },
    }
    return json.dumps(payload).encode()


def _make_position(
    tracker_id: str = "!a1b2c3d4",
    person_id: str = "alice",
    lat: float = 40.1231,
    lon: float = -73.1239,
    accuracy_m: float = 10.0,
    age_offset_seconds: float = 0.0,
) -> TrackerPosition:
    """Build a TrackerPosition with a controllable received_at age."""
    received_at = (
        datetime.now(timezone.utc) - timedelta(seconds=age_offset_seconds)
    ).isoformat().replace("+00:00", "Z")
    return TrackerPosition(
        tracker_id=tracker_id,
        person_id=person_id,
        asset_id=None,
        display_name="Alice",
        latitude=lat,
        longitude=lon,
        accuracy_m=accuracy_m,
        altitude_m=42.0,
        speed_mps=1.2,
        heading_degrees=90.0,
        battery_percent=82,
        received_at=received_at,
        device_timestamp=1787500000,
    )


# ---------------------------------------------------------------------------
# haversine_distance
# ---------------------------------------------------------------------------


class TestHaversineDistance:
    def test_same_point_is_zero(self):
        assert haversine_distance(40.0, -73.0, 40.0, -73.0) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        # ~111 km per degree of latitude
        dist = haversine_distance(0.0, 0.0, 1.0, 0.0)
        assert 110_000 < dist < 112_000

    def test_returns_metres(self):
        # 10 m movement northward (≈0.00009 degrees)
        dist = haversine_distance(40.0, -73.0, 40.00009, -73.0)
        assert dist == pytest.approx(10.0, rel=0.05)


# ---------------------------------------------------------------------------
# load_tracker_config
# ---------------------------------------------------------------------------


class TestLoadTrackerConfig:
    def test_returns_empty_for_missing_file(self, tmp_path):
        result = load_tracker_config(str(tmp_path / "missing.json"))
        assert result == {}

    def test_returns_empty_for_malformed_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json")
        result = load_tracker_config(str(p))
        assert result == {}

    def test_loads_enabled_trackers(self, tmp_path):
        p = tmp_path / "trackers.json"
        p.write_text(json.dumps(SAMPLE_TRACKER_CONFIG))
        result = load_tracker_config(str(p))
        assert "!a1b2c3d4" in result
        assert "!e5f6g7h8" in result

    def test_excludes_disabled_trackers(self, tmp_path):
        p = tmp_path / "trackers.json"
        p.write_text(json.dumps(SAMPLE_TRACKER_CONFIG))
        result = load_tracker_config(str(p))
        assert "!disabled01" not in result

    def test_includes_tracker_with_no_enabled_key(self, tmp_path):
        cfg = {"!abc": {"person_id": "charlie"}}
        p = tmp_path / "trackers.json"
        p.write_text(json.dumps(cfg))
        result = load_tracker_config(str(p))
        assert "!abc" in result


# ---------------------------------------------------------------------------
# load_camera_zones
# ---------------------------------------------------------------------------


class TestLoadCameraZones:
    def test_returns_empty_for_missing_file(self, tmp_path):
        result = load_camera_zones(str(tmp_path / "missing.json"))
        assert result == {}

    def test_returns_empty_for_malformed_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{invalid")
        result = load_camera_zones(str(p))
        assert result == {}

    def test_loads_cameras(self, tmp_path):
        p = tmp_path / "zones.json"
        p.write_text(json.dumps({"cameras": SAMPLE_CAMERA_ZONES}))
        result = load_camera_zones(str(p))
        assert "driveway" in result
        assert result["driveway"]["radius_meters"] == 35

    def test_missing_cameras_key(self, tmp_path):
        p = tmp_path / "zones.json"
        p.write_text(json.dumps({"other": {}}))
        result = load_camera_zones(str(p))
        assert result == {}


# ---------------------------------------------------------------------------
# TrackerPosition
# ---------------------------------------------------------------------------


class TestTrackerPosition:
    def test_age_seconds_recent(self):
        pos = _make_position(age_offset_seconds=5.0)
        assert pos.age_seconds == pytest.approx(5.0, abs=1.0)

    def test_age_seconds_stale(self):
        pos = _make_position(age_offset_seconds=200.0)
        assert pos.age_seconds > 190.0

    def test_to_dict_contains_all_slots(self):
        pos = _make_position()
        d = pos.to_dict()
        for slot in TrackerPosition.__slots__:
            assert slot in d

    def test_age_seconds_invalid_timestamp(self):
        pos = _make_position()
        pos.received_at = "not-a-date"
        assert pos.age_seconds == float("inf")


# ---------------------------------------------------------------------------
# parse_position_message
# ---------------------------------------------------------------------------


class TestParsePositionMessage:
    def test_valid_payload(self, tracker):
        payload = _make_payload()
        pos = tracker.parse_position_message("meshtastic/gw/json/receive", payload)
        assert pos is not None
        assert pos.tracker_id == "!a1b2c3d4"
        assert pos.person_id == "alice"
        assert pos.latitude == pytest.approx(40.1231, rel=1e-4)
        assert pos.longitude == pytest.approx(-73.1239, rel=1e-4)

    def test_unknown_tracker_returns_none(self, tracker):
        payload = json.dumps({
            "from": int("deadbeef", 16),
            "payload": {"position": {"latitude_i": 400000000, "longitude_i": -730000000}},
        }).encode()
        pos = tracker.parse_position_message("meshtastic/gw/json/receive", payload)
        assert pos is None

    def test_malformed_json_returns_none(self, tracker):
        pos = tracker.parse_position_message("meshtastic/gw/json/receive", b"not json{{{")
        assert pos is None

    def test_missing_gps_fields_returns_none(self, tracker):
        payload = json.dumps({
            "from": int("a1b2c3d4", 16),
            "payload": {"position": {}},
        }).encode()
        pos = tracker.parse_position_message("meshtastic/gw/json/receive", payload)
        assert pos is None

    def test_node_id_from_topic(self, tracker):
        payload = json.dumps({
            "payload": {
                "position": {
                    "latitude_i": int(40.1231 * 1e7),
                    "longitude_i": int(-73.1239 * 1e7),
                }
            }
        }).encode()
        pos = tracker.parse_position_message(
            "meshtastic/gw/json/receive/!a1b2c3d4", payload
        )
        assert pos is not None
        assert pos.tracker_id == "!a1b2c3d4"

    def test_missing_optional_fields_tolerated(self, tracker):
        payload = json.dumps({
            "from": int("a1b2c3d4", 16),
            "payload": {
                "position": {
                    "latitude_i": int(40.1231 * 1e7),
                    "longitude_i": int(-73.1239 * 1e7),
                    # No accuracy, altitude, speed, etc.
                }
            },
        }).encode()
        pos = tracker.parse_position_message("meshtastic/gw/json/receive", payload)
        assert pos is not None
        assert pos.accuracy_m is None
        assert pos.altitude_m is None
        assert pos.speed_mps is None

    def test_integer_encoded_lat_lon(self, tracker):
        """latitude_i / longitude_i encoded as int * 1e7."""
        lat_i = int(40.1231 * 1e7)
        lon_i = int(-73.1239 * 1e7)
        payload = json.dumps({
            "from": int("a1b2c3d4", 16),
            "payload": {"position": {"latitude_i": lat_i, "longitude_i": lon_i}},
        }).encode()
        pos = tracker.parse_position_message("meshtastic/gw/json/receive", payload)
        assert pos is not None
        assert abs(pos.latitude - 40.1231) < 0.001
        assert abs(pos.longitude - -73.1239) < 0.001

    def test_node_name_change_doesnt_affect_node_id(self, tracker):
        """Node ID from payload 'from' field is stable even if display_name changes."""
        payload = json.dumps({
            "from": int("a1b2c3d4", 16),
            "longName": "Alice's Renamed Node",
            "payload": {
                "position": {
                    "latitude_i": int(40.1231 * 1e7),
                    "longitude_i": int(-73.1239 * 1e7),
                }
            },
        }).encode()
        pos = tracker.parse_position_message("meshtastic/gw/json/receive", payload)
        assert pos is not None
        assert pos.tracker_id == "!a1b2c3d4"

    def test_empty_payload(self, tracker):
        pos = tracker.parse_position_message("meshtastic/gw/json/receive", b"")
        assert pos is None

    def test_invalid_coordinates_rejected(self, tracker):
        payload = json.dumps({
            "from": int("a1b2c3d4", 16),
            "payload": {"position": {"latitude_i": 9999999999, "longitude_i": 9999999999}},
        }).encode()
        pos = tracker.parse_position_message("meshtastic/gw/json/receive", payload)
        assert pos is None


# ---------------------------------------------------------------------------
# update_position / get_position
# ---------------------------------------------------------------------------


class TestUpdateAndGetPosition:
    def test_fresh_position_accepted(self, tracker):
        pos = _make_position(age_offset_seconds=5.0)
        accepted = tracker.update_position(pos)
        assert accepted is True
        assert tracker.get_position(pos.tracker_id) is not None

    def test_stale_position_rejected(self, tracker):
        pos = _make_position(age_offset_seconds=200.0)
        accepted = tracker.update_position(pos)
        assert accepted is False
        assert tracker.get_position(pos.tracker_id) is None

    def test_poor_accuracy_rejected(self, tracker):
        pos = _make_position(accuracy_m=100.0)  # > max_gps_accuracy_m=50
        accepted = tracker.update_position(pos)
        assert accepted is False

    def test_good_accuracy_accepted(self, tracker):
        pos = _make_position(accuracy_m=30.0)
        accepted = tracker.update_position(pos)
        assert accepted is True

    def test_get_position_evicts_stale(self, tracker):
        pos = _make_position(age_offset_seconds=5.0)
        tracker.update_position(pos)
        # Manually age the position beyond the max
        pos.received_at = (
            datetime.now(timezone.utc) - timedelta(seconds=200)
        ).isoformat().replace("+00:00", "Z")
        # Re-store the modified position directly
        tracker._last_positions[pos.tracker_id] = pos
        assert tracker.get_position(pos.tracker_id) is None

    def test_duplicate_message_overwrites_previous(self, tracker):
        pos1 = _make_position(lat=40.0, age_offset_seconds=10.0)
        pos2 = _make_position(lat=40.1231, age_offset_seconds=2.0)
        tracker.update_position(pos1)
        tracker.update_position(pos2)
        stored = tracker.get_position("!a1b2c3d4")
        assert stored.latitude == pytest.approx(40.1231, abs=0.001)


# ---------------------------------------------------------------------------
# cameras_near_position / trackers_near_camera
# ---------------------------------------------------------------------------


class TestCameraCorrelation:
    def test_camera_near_tracker(self, tracker):
        # Position exactly at driveway camera
        cameras = tracker.cameras_near_position(40.1231, -73.1239)
        names = [c[0] for c in cameras]
        assert "driveway" in names

    def test_camera_far_from_tracker(self, tracker):
        # Position 1 km away
        cameras = tracker.cameras_near_position(40.1331, -73.1339)
        assert len(cameras) == 0

    def test_tracker_near_camera(self, tracker):
        pos = _make_position(lat=40.1231, lon=-73.1239, age_offset_seconds=5.0)
        tracker.update_position(pos)
        results = tracker.trackers_near_camera("driveway")
        assert len(results) > 0
        assert results[0][0].tracker_id == "!a1b2c3d4"

    def test_tracker_not_near_camera(self, tracker):
        pos = _make_position(lat=40.2000, lon=-73.2000, age_offset_seconds=5.0)
        tracker.update_position(pos)
        results = tracker.trackers_near_camera("driveway")
        assert len(results) == 0

    def test_multiple_trackers_near_same_camera(self, tracker):
        pos_alice = _make_position(
            tracker_id="!a1b2c3d4", person_id="alice",
            lat=40.1231, lon=-73.1239, age_offset_seconds=5.0,
        )
        pos_tractor = TrackerPosition(
            tracker_id="!e5f6g7h8",
            person_id=None,
            asset_id="tractor",
            display_name="Tractor",
            latitude=40.1231,
            longitude=-73.1239,
            accuracy_m=10.0,
            altitude_m=None,
            speed_mps=None,
            heading_degrees=None,
            battery_percent=None,
            received_at=(datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat().replace("+00:00", "Z"),
            device_timestamp=None,
        )
        tracker.update_position(pos_alice)
        tracker.update_position(pos_tractor)
        results = tracker.trackers_near_camera("driveway")
        assert len(results) == 2

    def test_one_tracker_near_multiple_cameras(self, tracker):
        # Position between driveway and garden
        mid_lat = (40.1231 + 40.1237) / 2
        mid_lon = (-73.1239 + -73.1232) / 2
        cameras = tracker.cameras_near_position(mid_lat, mid_lon, extra_radius_m=100)
        assert len(cameras) >= 1

    def test_camera_without_coordinates_skipped(self, tracker):
        tracker.camera_zones["broken"] = {"radius_meters": 50}
        cameras = tracker.cameras_near_position(40.1231, -73.1239)
        names = [c[0] for c in cameras]
        assert "broken" not in names


# ---------------------------------------------------------------------------
# compute_tracker_score
# ---------------------------------------------------------------------------


class TestComputeTrackerScore:
    def test_score_at_camera_location(self, tracker):
        pos = _make_position(lat=40.1231, lon=-73.1239, accuracy_m=5.0, age_offset_seconds=1.0)
        score = tracker.compute_tracker_score(pos, "driveway")
        assert 0.5 <= score <= 1.0

    def test_score_is_zero_outside_radius(self, tracker):
        pos = _make_position(lat=40.2000, lon=-73.2000, accuracy_m=5.0, age_offset_seconds=1.0)
        score = tracker.compute_tracker_score(pos, "driveway")
        assert score == 0.0

    def test_score_decreases_with_age(self, tracker):
        pos_fresh = _make_position(lat=40.1231, lon=-73.1239, accuracy_m=5.0, age_offset_seconds=1.0)
        pos_old = _make_position(lat=40.1231, lon=-73.1239, accuracy_m=5.0, age_offset_seconds=80.0)
        score_fresh = tracker.compute_tracker_score(pos_fresh, "driveway")
        score_old = tracker.compute_tracker_score(pos_old, "driveway")
        assert score_fresh > score_old

    def test_stale_position_score_is_zero(self, tracker):
        pos = _make_position(lat=40.1231, lon=-73.1239, accuracy_m=5.0, age_offset_seconds=200.0)
        score = tracker.compute_tracker_score(pos, "driveway")
        assert score == pytest.approx(0.0, abs=0.01)

    def test_poor_accuracy_reduces_score(self, tracker):
        pos_good = _make_position(lat=40.1231, lon=-73.1239, accuracy_m=5.0, age_offset_seconds=1.0)
        pos_poor = _make_position(lat=40.1231, lon=-73.1239, accuracy_m=45.0, age_offset_seconds=1.0)
        score_good = tracker.compute_tracker_score(pos_good, "driveway")
        score_poor = tracker.compute_tracker_score(pos_poor, "driveway")
        assert score_good > score_poor

    def test_score_for_unknown_camera_is_zero(self, tracker):
        pos = _make_position(lat=40.1231, lon=-73.1239)
        score = tracker.compute_tracker_score(pos, "nonexistent_camera")
        assert score == 0.0


# ---------------------------------------------------------------------------
# tracker_signal_for_person
# ---------------------------------------------------------------------------


class TestTrackerSignalForPerson:
    def test_returns_signal_when_tracker_near_camera(self, tracker):
        pos = _make_position(lat=40.1231, lon=-73.1239, accuracy_m=10.0, age_offset_seconds=5.0)
        tracker.update_position(pos)
        signal = tracker.tracker_signal_for_person("alice", "driveway")
        assert signal is not None
        assert signal["person_id"] == "alice"
        assert signal["nearest_camera"] == "driveway"
        assert signal["association_confidence"] > 0.0

    def test_returns_none_when_no_tracker_for_person(self, tracker):
        signal = tracker.tracker_signal_for_person("bob", "driveway")
        assert signal is None

    def test_returns_none_when_tracker_out_of_range(self, tracker):
        pos = _make_position(lat=40.2000, lon=-73.2000, accuracy_m=10.0, age_offset_seconds=5.0)
        tracker.update_position(pos)
        signal = tracker.tracker_signal_for_person("alice", "driveway")
        assert signal is None

    def test_tracker_person_mismatch_strong_face_match(self, tracker):
        """Tracker for alice should not return a signal for bob."""
        pos = _make_position(lat=40.1231, lon=-73.1239, person_id="alice", age_offset_seconds=5.0)
        tracker.update_position(pos)
        signal = tracker.tracker_signal_for_person("bob", "driveway")
        assert signal is None


# ---------------------------------------------------------------------------
# matches_topic
# ---------------------------------------------------------------------------


class TestMatchesTopic:
    def test_matches_single_wildcard(self, tracker):
        assert tracker.matches_topic("meshtastic/gw1/json/receive") is True
        assert tracker.matches_topic("meshtastic/gw2/json/receive") is True

    def test_no_match_wrong_prefix(self, tracker):
        assert tracker.matches_topic("frigate/events") is False

    def test_no_match_extra_segment(self, tracker):
        # Default pattern has 4 segments; 5 segments should not match single-level wildcard
        assert tracker.matches_topic("meshtastic/gw/json/receive/extra") is False

    def test_hash_wildcard(self):
        t = MeshtasticTracker(mqtt_topic="meshtastic/#", tracker_config={})
        assert t.matches_topic("meshtastic/gw/json/receive") is True
        assert t.matches_topic("meshtastic/anything/else") is True


# ---------------------------------------------------------------------------
# build_meshtastic_tracker_from_env
# ---------------------------------------------------------------------------


class TestBuildFromEnv:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("MESHTASTIC_ENABLED", raising=False)
        result = build_meshtastic_tracker_from_env()
        assert result is None

    def test_enabled(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MESHTASTIC_ENABLED", "true")
        monkeypatch.setenv("MESHTASTIC_TRACKER_CONFIG_PATH", str(tmp_path / "t.json"))
        monkeypatch.setenv("MESHTASTIC_CAMERA_ZONES_PATH", str(tmp_path / "c.json"))
        result = build_meshtastic_tracker_from_env()
        assert result is not None
        assert isinstance(result, MeshtasticTracker)

    def test_custom_parameters_applied(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MESHTASTIC_ENABLED", "true")
        monkeypatch.setenv("MESHTASTIC_MAX_POSITION_AGE_SECONDS", "120")
        monkeypatch.setenv("MESHTASTIC_MAX_GPS_ACCURACY_METERS", "30")
        monkeypatch.setenv("MESHTASTIC_WEIGHT", "0.45")
        monkeypatch.setenv("MESHTASTIC_TRACKER_CONFIG_PATH", str(tmp_path / "t.json"))
        monkeypatch.setenv("MESHTASTIC_CAMERA_ZONES_PATH", str(tmp_path / "c.json"))
        result = build_meshtastic_tracker_from_env()
        assert result.max_position_age_s == 120.0
        assert result.max_gps_accuracy_m == 30.0
        assert result.tracker_weight == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# MQTT reconnect / duplicate tolerance
# ---------------------------------------------------------------------------


class TestMqttResilience:
    def test_duplicate_position_updates_are_accepted(self, tracker):
        """Receiving the same tracker update twice should not raise an error."""
        payload = _make_payload()
        pos1 = tracker.parse_position_message("meshtastic/gw/json/receive", payload)
        pos2 = tracker.parse_position_message("meshtastic/gw/json/receive", payload)
        assert pos1 is not None
        assert pos2 is not None
        # Both accepted without error; second overwrites first
        tracker.update_position(pos1)
        tracker.update_position(pos2)
        stored = tracker.get_position("!a1b2c3d4")
        assert stored is not None

    def test_position_survives_after_reconnect(self, tracker):
        """Re-subscribing (simulated by calling matches_topic again) works cleanly."""
        pos = _make_position(age_offset_seconds=1.0)
        tracker.update_position(pos)
        # Simulate re-subscription without losing stored positions
        assert tracker.matches_topic("meshtastic/gw/json/receive") is True
        assert tracker.get_position("!a1b2c3d4") is not None


# ---------------------------------------------------------------------------
# get_all_valid_positions
# ---------------------------------------------------------------------------


class TestGetAllValidPositions:
    def test_returns_fresh_positions(self, tracker):
        pos = _make_position(age_offset_seconds=5.0)
        tracker.update_position(pos)
        all_pos = tracker.get_all_valid_positions()
        assert len(all_pos) == 1

    def test_evicts_stale_on_retrieval(self, tracker):
        pos = _make_position(age_offset_seconds=5.0)
        tracker.update_position(pos)
        # Age the position beyond max
        tracker._last_positions[pos.tracker_id].received_at = (
            datetime.now(timezone.utc) - timedelta(seconds=200)
        ).isoformat().replace("+00:00", "Z")
        all_pos = tracker.get_all_valid_positions()
        assert len(all_pos) == 0
