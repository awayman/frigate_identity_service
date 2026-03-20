import json
import logging
import sys
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
from debug_logger import DebugLogger
from snapshot_crop import (
    build_local_crop_rect,
    crop_snapshot_bytes,
    crop_snapshot_pil,
    pil_to_jpeg_bytes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress APScheduler's verbose job execution logs
logging.getLogger("apscheduler").setLevel(logging.WARNING)

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
    user's add-on configuration to /data/options.json. This function reads
    that file and maps each option (e.g. ``mqtt_broker``/``mqtt_host``) to the
    corresponding environment variable (e.g. ``MQTT_BROKER``),
    but only when the variable has not already been set in the environment.

    If the file cannot be read (permissions), environment variables must be
    set by the Home Assistant Supervisor instead.
    """
    path_obj = Path(options_file)
    logger.info("Loading Home Assistant add-on configuration from %s", options_file)

    # Log any MQTT-related env vars that were already set (these will block options.json)
    pre_set_mqtt_vars = {}
    for var in ["MQTT_BROKER", "MQTT_PORT", "MQTT_USERNAME", "MQTT_PASSWORD"]:
        if var in os.environ:
            pre_set_mqtt_vars[var] = os.environ[var]
    if pre_set_mqtt_vars:
        logger.debug(
            "Pre-existing MQTT env vars will block options.json overrides: %s",
            list(pre_set_mqtt_vars.keys()),
        )

    if not path_obj.exists():
        logger.debug(
            "Options file not found at %s (normal for non-HA deployments)", options_file
        )
        return

    logger.debug("Options file found")

    try:
        logger.debug("Reading options file...")
        with open(options_file, "r") as f:
            raw = f.read()
            options = json.loads(raw)

        logger.debug("Successfully parsed options.json (%d keys)", len(options))

        loaded_vars = {}
        option_to_env = {
            "mqtt_host": "MQTT_BROKER",
            "mqtt_broker": "MQTT_BROKER",
            "mqtt_server": "MQTT_BROKER",
        }

        # Sensitive keys that should not be logged in plaintext
        sensitive_keys = {"mqtt_password", "mqtt_username", "frigate_password"}

        for key, value in options.items():
            env_key = option_to_env.get(key.lower(), key.upper())

            # Log with redaction for sensitive values
            if key.lower() in sensitive_keys:
                logger.debug(
                    "Processing option '%s' -> env var '%s' (***redacted***)",
                    key,
                    env_key,
                )
            else:
                logger.debug(
                    "Processing option '%s' -> env var '%s' = %s", key, env_key, value
                )

            if value in (None, ""):
                continue

            if env_key in os.environ:
                logger.debug(
                    "Env var '%s' already set, skipping options.json value", env_key
                )
                continue

            os.environ[env_key] = str(value)
            loaded_vars[env_key] = str(value)

        logger.info(
            "Loaded %d configuration variables from options.json", len(loaded_vars)
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Variables set: %s", list(loaded_vars.keys()))

    except PermissionError:
        logger.error(
            "Cannot read %s: Permission denied. "
            "Container may be running as non-root user. "
            "For HA add-on mode, ensure Dockerfile runs as root (UID 0).",
            options_file,
        )
    except json.JSONDecodeError as e:
        logger.error("Failed to parse %s (invalid JSON): %s", options_file, e)
    except Exception as e:
        logger.error("Unexpected error loading %s: %s", options_file, e)


# Load configuration from Home Assistant Add-on options (if running as HA add-on)
load_ha_options()

# Load configuration from .env.integration-test if it exists
test_env_path = Path(__file__).parent / ".env.integration-test"
if test_env_path.exists():
    logger.info("Loading configuration from %s", test_env_path)
    load_env_file(str(test_env_path))

MQTT_BROKER = os.getenv("MQTT_BROKER") or os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
MQTT_CONNECT_RETRIES = int(os.getenv("MQTT_CONNECT_RETRIES", "30"))
MQTT_CONNECT_RETRY_DELAY = int(os.getenv("MQTT_CONNECT_RETRY_DELAY", "5"))

# Log the actual MQTT configuration being used
logger.info(
    "MQTT Configuration: broker=%s:%d, auth=%s",
    MQTT_BROKER,
    MQTT_PORT,
    "yes" if MQTT_USERNAME and MQTT_PASSWORD else "no",
)
FRIGATE_HOST = os.getenv("FRIGATE_HOST", "http://localhost:5000")
REID_MODEL = os.getenv("REID_MODEL", "osnet_x1_0")
REID_DEVICE = os.getenv("REID_DEVICE", "auto")
REID_SIMILARITY_THRESHOLD = float(os.getenv("REID_SIMILARITY_THRESHOLD", "0.75"))
SNAPSHOT_FETCH_MODE = os.getenv("SNAPSHOT_FETCH_MODE", "clean_if_available").lower()
SNAPSHOT_LOCAL_CROP = os.getenv("SNAPSHOT_LOCAL_CROP", "true").lower() == "true"
# Asymmetric padding: more vertical context (head/feet) than horizontal.
SNAPSHOT_CROP_PADDING_X = float(os.getenv("SNAPSHOT_CROP_PADDING_X", "0.05"))
SNAPSHOT_CROP_PADDING_Y = float(os.getenv("SNAPSHOT_CROP_PADDING_Y", "0.20"))


def validate_config():
    """Validate configuration values and exit with clear error if invalid.

    This function validates ranges and types for critical configuration values,
    logging the sanitized config (with secrets redacted) for debugging purposes.
    """
    errors = []

    # Validate MQTT port
    if not (1 <= MQTT_PORT <= 65535):
        errors.append(f"MQTT_PORT must be between 1 and 65535, got {MQTT_PORT}")

    # Validate Frigate host URL format
    if not FRIGATE_HOST.startswith(("http://", "https://")):
        errors.append(
            f"FRIGATE_HOST must start with http:// or https://, got '{FRIGATE_HOST}'"
        )

    # Validate ReID similarity threshold
    if not (0.0 <= REID_SIMILARITY_THRESHOLD <= 1.0):
        errors.append(
            f"REID_SIMILARITY_THRESHOLD must be between 0.0 and 1.0, got {REID_SIMILARITY_THRESHOLD}"
        )

    # Validate snapshot correlation window
    snapshot_window = float(os.getenv("SNAPSHOT_CORRELATION_WINDOW", "2.0"))
    if not (0.1 <= snapshot_window <= 10.0):
        errors.append(
            f"SNAPSHOT_CORRELATION_WINDOW must be between 0.1 and 10.0, got {snapshot_window}"
        )

    # Validate snapshot fetch mode
    snapshot_fetch_mode = os.getenv(
        "SNAPSHOT_FETCH_MODE", "clean_if_available"
    ).lower()
    valid_snapshot_fetch_modes = {"thumbnail", "snapshot", "clean_if_available"}
    if snapshot_fetch_mode not in valid_snapshot_fetch_modes:
        errors.append(
            "SNAPSHOT_FETCH_MODE must be one of "
            f"{sorted(valid_snapshot_fetch_modes)}, got '{snapshot_fetch_mode}'"
        )

    # Validate snapshot crop padding (legacy symmetric and per-axis)
    # Validate snapshot crop padding (per-axis)
    snapshot_crop_padding_x = float(os.getenv("SNAPSHOT_CROP_PADDING_X", "0.05"))
    if not (0.0 <= snapshot_crop_padding_x <= 1.0):
        errors.append(
            "SNAPSHOT_CROP_PADDING_X must be between 0.0 and 1.0, "
            f"got {snapshot_crop_padding_x}"
        )
    snapshot_crop_padding_y = float(os.getenv("SNAPSHOT_CROP_PADDING_Y", "0.20"))
    if not (0.0 <= snapshot_crop_padding_y <= 1.0):
        errors.append(
            "SNAPSHOT_CROP_PADDING_Y must be between 0.0 and 1.0, "
            f"got {snapshot_crop_padding_y}"
        )

    # Validate snapshot local crop toggle
    snapshot_local_crop = os.getenv("SNAPSHOT_LOCAL_CROP", "true").lower()
    if snapshot_local_crop not in {"true", "false"}:
        errors.append(
            "SNAPSHOT_LOCAL_CROP must be 'true' or 'false', "
            f"got '{snapshot_local_crop}'"
        )

    # Validate max tracked persons
    max_persons = int(os.getenv("MAX_TRACKED_PERSONS_PER_CAMERA", "3"))
    if not (1 <= max_persons <= 20):
        errors.append(
            f"MAX_TRACKED_PERSONS_PER_CAMERA must be between 1 and 20, got {max_persons}"
        )

    # Validate debug retention days
    retention = int(os.getenv("DEBUG_RETENTION_DAYS", "7"))
    if not (1 <= retention <= 90):
        errors.append(f"DEBUG_RETENTION_DAYS must be between 1 and 90, got {retention}")

    # Validate embedding retention mode
    retention_mode = os.getenv("EMBEDDING_RETENTION_MODE", "age_prune").lower()
    valid_retention_modes = {"age_prune", "full_clear_daily", "manual"}
    if retention_mode not in valid_retention_modes:
        errors.append(
            "EMBEDDING_RETENTION_MODE must be one of "
            f"{sorted(valid_retention_modes)}, got '{retention_mode}'"
        )

    # Validate embedding max age
    max_age_hours = float(os.getenv("EMBEDDING_MAX_AGE_HOURS", "48"))
    if not (1 <= max_age_hours <= 720):
        errors.append(
            f"EMBEDDING_MAX_AGE_HOURS must be between 1 and 720, got {max_age_hours}"
        )

    # Validate prune interval
    prune_interval = int(os.getenv("EMBEDDING_PRUNE_INTERVAL_MINUTES", "30"))
    if not (1 <= prune_interval <= 1440):
        errors.append(
            "EMBEDDING_PRUNE_INTERVAL_MINUTES must be between 1 and 1440, "
            f"got {prune_interval}"
        )

    # Validate full clear time format HH:MM
    full_clear_time = os.getenv("EMBEDDING_FULL_CLEAR_TIME", "00:00")
    try:
        hour_str, minute_str = full_clear_time.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
    except (ValueError, AttributeError):
        errors.append(
            "EMBEDDING_FULL_CLEAR_TIME must use 24h HH:MM format, "
            f"got '{full_clear_time}'"
        )

    # Validate recency decay mode
    recency_decay_mode = os.getenv("RECENCY_DECAY_MODE", "linear").lower()
    valid_decay_modes = {"linear", "exponential", "none"}
    if recency_decay_mode not in valid_decay_modes:
        errors.append(
            "RECENCY_DECAY_MODE must be one of "
            f"{sorted(valid_decay_modes)}, got '{recency_decay_mode}'"
        )

    # Validate recency weight floor
    recency_weight_floor = float(os.getenv("RECENCY_WEIGHT_FLOOR", "0.3"))
    if not (0.0 <= recency_weight_floor <= 0.9):
        errors.append(
            "RECENCY_WEIGHT_FLOOR must be between 0.0 and 0.9, "
            f"got {recency_weight_floor}"
        )

    # Validate use_confidence_weighting
    use_confidence_weighting_str = os.getenv(
        "USE_CONFIDENCE_WEIGHTING", "false"
    ).lower()
    if use_confidence_weighting_str not in {"true", "false"}:
        errors.append(
            "USE_CONFIDENCE_WEIGHTING must be 'true' or 'false', "
            f"got '{use_confidence_weighting_str}'"
        )

    # If any validation errors, log and exit
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error("  - %s", error)
        logger.error("Please fix the configuration and restart the service.")
        sys.exit(1)

    # Log sanitized config (with secrets redacted)
    mqtt_password_display = "***" if MQTT_PASSWORD else "(not set)"
    logger.info("Configuration validated successfully:")
    logger.info(
        "  MQTT: %s:%d (user: %s, password: %s)",
        MQTT_BROKER,
        MQTT_PORT,
        MQTT_USERNAME or "(not set)",
        mqtt_password_display,
    )
    logger.info("  Frigate: %s", FRIGATE_HOST)
    logger.info(
        "  ReID: model=%s, device=%s, threshold=%.2f",
        REID_MODEL,
        REID_DEVICE,
        REID_SIMILARITY_THRESHOLD,
    )
    logger.info(
        "  Snapshots: mode=%s, local_crop=%s, crop_padding_x=%.2f, crop_padding_y=%.2f",
        snapshot_fetch_mode,
        snapshot_local_crop,
        snapshot_crop_padding_x,
        snapshot_crop_padding_y,
    )

    # Determine default paths for display
    embeddings_path = os.getenv("EMBEDDINGS_DB_PATH") or (
        "/data/embeddings.json"
        if (os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"))
        else "embeddings.json"
    )
    debug_path = os.getenv("DEBUG_LOG_PATH") or (
        "/data/debug"
        if (os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"))
        else "debug"
    )

    logger.info("  Embeddings: %s", embeddings_path)
    logger.info(
        "  Embedding retention: mode=%s, max_age_hours=%.1f, prune_interval_minutes=%d, full_clear_time=%s",
        retention_mode,
        max_age_hours,
        prune_interval,
        full_clear_time,
    )

    logger.info(
        "  Recency weighting: decay_mode=%s, weight_floor=%.2f, use_confidence=%s",
        recency_decay_mode,
        recency_weight_floor,
        use_confidence_weighting_str,
    )

    # Log retention mode explanation
    if retention_mode == "age_prune":
        logger.info(
            "  → Embeddings older than %.1f hours will be pruned every %d minutes (continuous identity)",
            max_age_hours,
            prune_interval,
        )
    elif retention_mode == "full_clear_daily":
        logger.info(
            "  → All embeddings will be cleared daily at %s (legacy mode, brief recognition gap after clear)",
            full_clear_time,
        )
    else:
        logger.info(
            "  → No automatic embedding cleanup (manual mode, embeddings persist until explicitly cleared)"
        )

    logger.info(
        "  Debug: enabled=%s, path=%s",
        os.getenv("DEBUG_LOGGING_ENABLED", "false"),
        debug_path,
    )


# Validate configuration
validate_config()

# Warn if running as HA addon but using localhost fallback (indicates config issue)
ha_options_path = Path("/data/options.json")
if ha_options_path.exists() and MQTT_BROKER == "localhost":
    logger.error(
        "HA add-on mode detected but MQTT_BROKER=localhost. "
        "This means /data/options.json could not be read. "
        "Check Dockerfile runs as root (UID 0) and /data mount is accessible."
    )
elif not ha_options_path.exists():
    logger.debug("Not running as HA add-on (no /data/options.json)")
else:
    logger.debug("Successfully loaded configuration from HA options")


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


def get_default_debug_path():
    """Return container-appropriate default path for debug logging.

    When running in a container, defaults to /data/debug for persistence.
    Otherwise uses ./debug in the current directory.
    """
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return "/data/debug"
    return "debug"


DEBUG_LOGGING_ENABLED = os.getenv("DEBUG_LOGGING_ENABLED", "false").lower() == "true"
DEBUG_LOG_PATH = os.getenv("DEBUG_LOG_PATH", get_default_debug_path())
DEBUG_SAVE_EMBEDDINGS = os.getenv("DEBUG_SAVE_EMBEDDINGS", "false").lower() == "true"
DEBUG_RETENTION_DAYS = int(os.getenv("DEBUG_RETENTION_DAYS", "7"))
EMBEDDING_RETENTION_MODE = os.getenv("EMBEDDING_RETENTION_MODE", "age_prune").lower()
EMBEDDING_MAX_AGE_HOURS = float(os.getenv("EMBEDDING_MAX_AGE_HOURS", "48"))
EMBEDDING_PRUNE_INTERVAL_MINUTES = int(
    os.getenv("EMBEDDING_PRUNE_INTERVAL_MINUTES", "30")
)
EMBEDDING_FULL_CLEAR_TIME = os.getenv("EMBEDDING_FULL_CLEAR_TIME", "00:00")
RECENCY_DECAY_MODE = os.getenv("RECENCY_DECAY_MODE", "linear").lower()
RECENCY_WEIGHT_FLOOR = float(os.getenv("RECENCY_WEIGHT_FLOOR", "0.3"))
USE_CONFIDENCE_WEIGHTING = (
    os.getenv("USE_CONFIDENCE_WEIGHTING", "true").lower() == "true"
)

# Initialize modules
logger.info("Initializing embedding store...")
embedding_store = EmbeddingStore(EMBEDDINGS_DB_PATH)

logger.info(
    "Initializing embedding matcher (decay_mode=%s, floor=%.2f, use_confidence=%s)...",
    RECENCY_DECAY_MODE,
    RECENCY_WEIGHT_FLOOR,
    USE_CONFIDENCE_WEIGHTING,
)
embedding_matcher = EmbeddingMatcher(
    max_age_hours=EMBEDDING_MAX_AGE_HOURS,
    decay_mode=RECENCY_DECAY_MODE,
    weight_floor=RECENCY_WEIGHT_FLOOR,
    use_confidence_weighting=USE_CONFIDENCE_WEIGHTING,
)

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

# Initialize debug logger
logger.info("Initializing debug logger at %s...", DEBUG_LOG_PATH)
debug_logger = DebugLogger(
    debug_path=DEBUG_LOG_PATH,
    enabled=DEBUG_LOGGING_ENABLED,
    save_embeddings=DEBUG_SAVE_EMBEDDINGS,
    retention_days=DEBUG_RETENTION_DAYS,
)

# Track recent person detections per camera for snapshot correlation
camera_person_queue = defaultdict(lambda: deque(maxlen=MAX_TRACKED_PERSONS_PER_CAMERA))

# Track recognized person identities until the tracked object finishes so we can
# learn embeddings from Frigate's final best-frame snapshot.
recognized_person_events = {}

# Cache snapshots to avoid redundant API calls
snapshot_cache = {}  # cache_key -> (base64_image, timestamp)
event_details_cache = {}  # event_id -> (event_payload, timestamp)
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

    # Subscribe to debug control commands
    client.subscribe("frigate_identity/debug/set")
    logger.info("Subscribed to: frigate_identity/debug/set")

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
        elif msg.topic == "frigate_identity/debug/set":
            handle_debug_control(client, msg)
        elif "/snapshot" in msg.topic:
            handle_snapshot_for_display(client, msg)
    except Exception as e:
        logger.error("Error processing MQTT message on %s: %s", msg.topic, e)
        traceback.print_exc()


def handle_debug_control(client, msg):
    """
    Handle debug mode control messages from Home Assistant.

    Payload: {"enabled": true/false}
    Publishes state to: frigate_identity/debug/state
    """
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        enabled = payload.get("enabled", False)

        debug_logger.set_enabled(enabled)

        # Publish current state
        state_message = {"enabled": debug_logger.enabled, "path": DEBUG_LOG_PATH}
        client.publish(
            "frigate_identity/debug/state",
            json.dumps(state_message),
            retain=True,
        )

        logger.info(
            "[DEBUG] Debug logging toggled: %s",
            "ENABLED" if enabled else "DISABLED",
        )

    except json.JSONDecodeError:
        logger.warning("[DEBUG] Invalid JSON in debug control message")
    except Exception as e:
        logger.error("[DEBUG] Error handling debug control: %s", e)


def _normalize_relative_rect(rect):
    """Normalize a relative [x, y, width, height] rectangle from Frigate metadata."""
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return None

    try:
        x_pos, y_pos, width, height = (float(value) for value in rect)
    except (TypeError, ValueError):
        return None

    if min(x_pos, y_pos, width, height) < 0:
        return None

    if width <= 0 or height <= 0:
        return None

    # Frigate event payloads expose relative coordinates. Ignore anything that
    # looks like absolute pixels because we cannot map those safely here.
    if max(x_pos, y_pos, width, height) > 1.5:
        return None

    return (x_pos, y_pos, width, height)


def _extract_snapshot_crop_geometry(event_payload):
    """Extract box/region geometry from MQTT or REST event payloads."""
    if not isinstance(event_payload, dict):
        return None

    payload_candidates = []
    data_payload = event_payload.get("data")
    if isinstance(data_payload, dict):
        payload_candidates.append(data_payload)
    payload_candidates.append(event_payload)

    for payload in payload_candidates:
        box = _normalize_relative_rect(payload.get("box"))
        region = _normalize_relative_rect(payload.get("region"))
        if box or region:
            return {"box": box, "region": region}

    return None


def _build_local_crop_rect(crop_geometry):
    """Delegate to :func:`snapshot_crop.build_local_crop_rect` with service-level padding."""
    return build_local_crop_rect(
        crop_geometry,
        padding_x=SNAPSHOT_CROP_PADDING_X,
        padding_y=SNAPSHOT_CROP_PADDING_Y,
    )


def _crop_snapshot_bytes(image_bytes, crop_geometry, quality=85):
    """Delegate to :func:`snapshot_crop.crop_snapshot_bytes` with service-level padding."""
    return crop_snapshot_bytes(
        image_bytes,
        crop_geometry,
        quality=quality,
        padding_x=SNAPSHOT_CROP_PADDING_X,
        padding_y=SNAPSHOT_CROP_PADDING_Y,
    )


def _crop_snapshot_pil(image_bytes, crop_geometry):
    """Delegate to :func:`snapshot_crop.crop_snapshot_pil` with service-level padding."""
    return crop_snapshot_pil(
        image_bytes,
        crop_geometry,
        padding_x=SNAPSHOT_CROP_PADDING_X,
        padding_y=SNAPSHOT_CROP_PADDING_Y,
    )


def _fetch_event_details(event_id):
    """Fetch full event metadata from Frigate and cache it briefly."""
    now = time.time()

    if event_id in event_details_cache:
        cached_payload, cached_time = event_details_cache[event_id]
        if now - cached_time < CACHE_TTL:
            return cached_payload

    try:
        url = f"{FRIGATE_HOST}/api/events/{event_id}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        payload = response.json()
        event_details_cache[event_id] = (payload, now)
        return payload
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.warning("[API] Failed to fetch event details for %s: %s", event_id, e)
        return None


def _build_snapshot_cache_key(event_id, crop, quality, height, crop_geometry):
    """Build a cache key that distinguishes snapshot strategy and crop geometry."""
    crop_key = None
    if crop_geometry:
        crop_key = tuple(crop_geometry.get("box") or crop_geometry.get("region") or ())

    return (
        event_id,
        SNAPSHOT_FETCH_MODE,
        bool(crop),
        int(quality),
        int(height or 0),
        crop_key,
    )


def _build_snapshot_candidates(event_id, crop, quality, height):
    """Build preferred Frigate snapshot endpoints in fallback order."""
    candidates = []

    if SNAPSHOT_FETCH_MODE == "clean_if_available":
        candidates.append(
            {
                "kind": "clean",
                "url": f"{FRIGATE_HOST}/api/events/{event_id}/snapshot-clean.webp",
                "params": None,
            }
        )

    if SNAPSHOT_FETCH_MODE in {"snapshot", "clean_if_available"}:
        snapshot_params = {
            "bbox": 0,
            "timestamp": 0,
            "quality": quality,
        }
        if height:
            snapshot_params["height"] = height
        snapshot_params["crop"] = 1 if crop else 0
        candidates.append(
            {
                "kind": "snapshot",
                "url": f"{FRIGATE_HOST}/api/events/{event_id}/snapshot.jpg",
                "params": snapshot_params,
            }
        )

    candidates.append(
        {
            "kind": "thumbnail",
            "url": f"{FRIGATE_HOST}/api/events/{event_id}/thumbnail.jpg",
            "params": None,
        }
    )

    return candidates


def build_identity_snapshot_urls(event_id):
    """Build stable Frigate URLs for downstream consumers."""
    return {
        "snapshot_url": (
            f"{FRIGATE_HOST}/api/events/{event_id}/snapshot.jpg?crop=1&bbox=0&timestamp=0"
        ),
        "clean_snapshot_url": (
            f"{FRIGATE_HOST}/api/events/{event_id}/snapshot-clean.webp"
        ),
    }


def fetch_snapshot_from_api(
    event_id,
    crop=True,
    quality=85,
    height=400,
    event_payload=None,
    _pil_out=None,
):
    """
    Fetch snapshot from Frigate API for a specific event.
    Uses caching to avoid redundant API calls.

    Args:
        event_id: Frigate event ID.
        crop: Whether to request/apply a person crop.
        quality: JPEG quality for encoding.
        height: Optional height hint passed to Frigate's snapshot endpoint.
        event_payload: MQTT event payload used to extract crop geometry.
        _pil_out: Optional list.  When provided and the clean-crop path is
            taken, the pre-JPEG PIL Image is appended so callers can pass it
            directly to :meth:`ReIDModel.extract_embedding_from_pil`, avoiding
            the lossy JPEG encode/decode round-trip.  On cache hits this list
            remains empty and callers should fall back to the base64 path.

    Returns:
        base64-encoded JPEG string, or None if failed.
    """
    now = time.time()
    crop_geometry = _extract_snapshot_crop_geometry(event_payload)

    if crop and SNAPSHOT_FETCH_MODE == "clean_if_available" and SNAPSHOT_LOCAL_CROP:
        if crop_geometry is None:
            crop_geometry = _extract_snapshot_crop_geometry(_fetch_event_details(event_id))

    cache_key = _build_snapshot_cache_key(event_id, crop, quality, height, crop_geometry)

    # Check cache
    if cache_key in snapshot_cache:
        cached_img, cached_time = snapshot_cache[cache_key]
        if now - cached_time < CACHE_TTL:
            return cached_img

    for candidate in _build_snapshot_candidates(event_id, crop, quality, height):
        try:
            response = requests.get(
                candidate["url"], params=candidate["params"], timeout=5
            )

            if response.status_code != 200:
                logger.debug(
                    "[API] Snapshot candidate %s failed for %s: HTTP %s",
                    candidate["kind"],
                    event_id,
                    response.status_code,
                )
                continue

            image_bytes = response.content
            if (
                crop
                and candidate["kind"] == "clean"
                and SNAPSHOT_LOCAL_CROP
                and crop_geometry is not None
            ):
                # Crop to PIL first — lossless intermediate for the ReID path.
                pil_crop = _crop_snapshot_pil(image_bytes, crop_geometry)
                if pil_crop is not None:
                    if _pil_out is not None:
                        _pil_out.append(pil_crop)
                    # JPEG-encode for cache and debug logger (only one encode).
                    image_bytes = pil_to_jpeg_bytes(pil_crop, quality=quality)
                else:
                    # Degenerate bbox — fall back to bytes-only crop.
                    cropped_bytes = _crop_snapshot_bytes(
                        image_bytes, crop_geometry, quality=quality
                    )
                    if cropped_bytes:
                        image_bytes = cropped_bytes

            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            snapshot_cache[cache_key] = (image_base64, now)
            return image_base64
        except requests.exceptions.RequestException as e:
            logger.debug(
                "[API] Snapshot candidate %s errored for %s: %s",
                candidate["kind"],
                event_id,
                e,
            )

    logger.warning("[API] Failed to fetch snapshot for %s using all candidates", event_id)
    return None


def _cache_recognized_person_event(
    event_id,
    person_id,
    camera,
    confidence,
    timestamp,
    zones,
):
    """Cache recognized identity metadata until the tracked event completes."""
    if not event_id or not person_id:
        return

    existing = recognized_person_events.get(event_id, {})
    existing_confidence = float(existing.get("confidence", 0.0) or 0.0)
    current_confidence = float(confidence or 0.0)

    recognized_person_events[event_id] = {
        "person_id": person_id,
        "camera": camera or existing.get("camera"),
        "confidence": max(existing_confidence, current_confidence),
        "timestamp": timestamp
        if timestamp is not None
        else existing.get("timestamp", time.time()),
        "zones": list(zones or existing.get("zones", [])),
    }


def _store_completed_face_embedding(
    event_id,
    person_id,
    camera,
    confidence,
    zones,
    timestamp,
    event_payload,
):
    """Store a facial-recognition embedding from the completed event snapshot."""
    if not REID_AVAILABLE or reid_model is None:
        return

    pil_out = []
    snapshot_base64 = fetch_snapshot_from_api(
        event_id,
        crop=True,
        event_payload=event_payload,
        _pil_out=pil_out,
    )

    if not snapshot_base64:
        logger.warning(
            "[EMBEDDING] Could not fetch completed snapshot for recognized event %s",
            event_id,
        )
        return

    try:
        if pil_out:
            embedding = reid_model.extract_embedding_from_pil(pil_out[0])
        else:
            embedding = reid_model.extract_embedding(snapshot_base64)

        embedding_store.store_embedding(person_id, embedding, camera, confidence)
        logger.info("[EMBEDDING] Stored final facial embedding for %s", person_id)

        debug_logger.log_facial_recognition(
            event_id=event_id,
            snapshot_base64=snapshot_base64,
            person_id=person_id,
            camera=camera,
            confidence=confidence,
            zones=zones,
            timestamp=timestamp,
        )
    except Exception as e:
        logger.error(
            "[EMBEDDING] Error storing final embedding for %s: %s", person_id, e
        )


def publish_identity_event(
    client, person_id, camera, confidence, source, zones, event_id, timestamp
):
    """Publish identity event to Home Assistant"""
    snapshot_urls = build_identity_snapshot_urls(event_id)
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
        "snapshot_url": snapshot_urls["snapshot_url"],
        "clean_snapshot_url": snapshot_urls["clean_snapshot_url"],
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

    if sub_label:
        _cache_recognized_person_event(
            event_id,
            sub_label,
            camera,
            confidence,
            timestamp,
            current_zones,
        )

    # Learn face embeddings only after Frigate has finalized the best frame.
    if event_type == "end":
        recognized_event = recognized_person_events.pop(event_id, None)
        if recognized_event:
            _store_completed_face_embedding(
                event_id,
                recognized_event["person_id"],
                recognized_event.get("camera") or camera,
                recognized_event.get("confidence", confidence),
                recognized_event.get("zones", current_zones),
                recognized_event.get("timestamp", timestamp),
                after,
            )
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
        if not REID_AVAILABLE or reid_model is None:
            return

        # Try ReID matching via API (accurate)
        pil_out_b = []
        snapshot_base64 = fetch_snapshot_from_api(
            event_id,
            crop=True,
            event_payload=after,
            _pil_out=pil_out_b,
        )

        if not snapshot_base64:
            logger.warning("[REID] Could not fetch snapshot for event %s", event_id)
            return

        try:
            # Extract embedding and match to stored persons
            if pil_out_b:
                query_embedding = reid_model.extract_embedding_from_pil(pil_out_b[0])
            else:
                query_embedding = reid_model.extract_embedding(snapshot_base64)
            stored_embeddings = embedding_store.get_all_embeddings()
            person_id, similarity_score = embedding_matcher.find_best_match(
                query_embedding, stored_embeddings, threshold=REID_SIMILARITY_THRESHOLD
            )

            if person_id:
                detection_record["person_id"] = person_id
                camera_person_queue[camera].append(detection_record)

                # Log to debug logger for analysis
                top_matches = embedding_matcher.find_top_k_matches(
                    query_embedding, stored_embeddings, k=5
                )
                debug_logger.log_reid_match(
                    event_id=event_id,
                    snapshot_base64=snapshot_base64,
                    matches=top_matches,
                    chosen_person_id=person_id,
                    chosen_similarity=similarity_score,
                    camera=camera,
                    zones=current_zones,
                    timestamp=timestamp,
                )

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
                # Log no-match for debugging
                top_matches = embedding_matcher.find_top_k_matches(
                    query_embedding, stored_embeddings, k=5
                )
                debug_logger.log_reid_no_match(
                    event_id=event_id,
                    snapshot_base64=snapshot_base64,
                    matches=top_matches,
                    best_similarity=similarity_score,
                    threshold=REID_SIMILARITY_THRESHOLD,
                    camera=camera,
                    zones=current_zones,
                    timestamp=timestamp,
                )

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
        _cache_recognized_person_event(
            event_id,
            person_id,
            camera,
            score,
            timestamp,
            [],
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

        # Log multi-person correlation issue for debugging
        queue_state = [
            {
                "person_id": d.get("person_id", "unknown"),
                "timestamp": d.get("timestamp", 0),
                "zones": d.get("zones", []),
            }
            for d in list(recent_detections)
        ]
        debug_logger.log_correlation_issue(
            camera=camera,
            active_persons_count=len(active_persons),
            queue_state=queue_state,
            timestamp=now,
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


def _parse_clock_time(time_value):
    """Parse HH:MM (24h) into hour, minute."""
    hour_str, minute_str = time_value.split(":", 1)
    return int(hour_str), int(minute_str)


def schedule_embedding_maintenance():
    """Start background jobs for embedding retention, debug cleanup, and heartbeat."""
    scheduler = BackgroundScheduler()

    def _clear_embeddings():
        logger.info("[CLEANUP] Full embedding clear triggered by retention policy")
        embedding_store.clear()
        logger.info(
            "[CLEANUP] Embedding store cleared. Store will rebuild as people are recognized."
        )

    def _prune_embeddings():
        stats = embedding_store.prune_expired(EMBEDDING_MAX_AGE_HOURS)
        if stats["removed_embeddings"] > 0:
            logger.info(
                "[CLEANUP] Pruned %d expired embeddings (%d persons removed, %d persons / %d embeddings remain)",
                stats["removed_embeddings"],
                stats["removed_persons"],
                stats["remaining_persons"],
                stats["remaining_embeddings"],
            )

    def _cleanup_debug_logs():
        logger.info("[CLEANUP] Running debug log retention cleanup...")
        deleted_count = debug_logger.cleanup_old_logs()
        if deleted_count > 0:
            logger.info(
                "[CLEANUP] Debug log cleanup completed: %d days of logs removed",
                deleted_count,
            )

    def _heartbeat():
        """Log service health status every 5 minutes"""
        uptime_seconds = int(time.time() - service_start_time)
        hours = uptime_seconds // 3600
        minutes = (uptime_seconds % 3600) // 60
        seconds = uptime_seconds % 60
        uptime_str = f"{hours}h {minutes}m {seconds}s"

        store_stats = embedding_store.get_stats()
        mqtt_status = "connected" if client.is_connected() else "disconnected"
        logger.info(
            "[HEARTBEAT] Service running | Uptime: %s | Persons tracked: %d | Embeddings: %d | MQTT: %s | ReID: %s",
            uptime_str,
            store_stats["persons"],
            store_stats["embeddings"],
            mqtt_status,
            "enabled" if REID_AVAILABLE else "disabled",
        )

    if EMBEDDING_RETENTION_MODE == "age_prune":
        scheduler.add_job(
            _prune_embeddings,
            "interval",
            minutes=EMBEDDING_PRUNE_INTERVAL_MINUTES,
        )
        logger.info(
            "[SCHEDULER] Embedding retention mode=age_prune (max_age_hours=%.1f, interval=%d minutes)",
            EMBEDDING_MAX_AGE_HOURS,
            EMBEDDING_PRUNE_INTERVAL_MINUTES,
        )
    elif EMBEDDING_RETENTION_MODE == "full_clear_daily":
        clear_hour, clear_minute = _parse_clock_time(EMBEDDING_FULL_CLEAR_TIME)
        scheduler.add_job(
            _clear_embeddings, "cron", hour=clear_hour, minute=clear_minute
        )
        logger.info(
            "[SCHEDULER] Embedding retention mode=full_clear_daily (runs at %02d:%02d)",
            clear_hour,
            clear_minute,
        )
    else:
        logger.info(
            "[SCHEDULER] Embedding retention mode=manual (no automatic embedding cleanup)"
        )

    scheduler.add_job(_cleanup_debug_logs, "cron", hour=1, minute=0)
    scheduler.add_job(_heartbeat, "interval", minutes=5)
    scheduler.start()
    logger.info("[SCHEDULER] Debug log cleanup scheduled (runs at 01:00)")
    logger.info("[SCHEDULER] Health heartbeat scheduled (every 5 minutes)")
    return scheduler


# Track service start time for uptime calculation
service_start_time = time.time()

client = get_mqtt_client()
client.on_connect = on_connect
client.on_message = on_message

if MQTT_USERNAME:
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

# Start maintenance scheduler before entering the MQTT loop
scheduler = schedule_embedding_maintenance()

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
