# Implementation Summary: Two-Tier ReID Architecture

## What Was Implemented

### 1. Identity Service Updates (frigate_identity_service/)

**New Features:**
- ✅ Two-tier snapshot system (MQTT for speed, API for accuracy)
- ✅ Subscribes to Frigate's native MQTT topics:
  - `frigate/+/+/update` (tracked objects with face recognition)
  - `frigate/+/person/snapshot` (fast person snapshots)
  - `frigate/+/car/snapshot` and `frigate/+/truck/snapshot` (vehicle detection)
- ✅ API snapshot fetching with caching for accurate embeddings
- ✅ Camera-based person correlation queue (tracks recent detections per camera)
- ✅ Publishes to person-specific topics: `identity/person/{person_id}`
- ✅ Republishes snapshots to: `identity/snapshots/{person_id}`
- ✅ Vehicle detection support

**New Environment Variables:**
- `FRIGATE_HOST` - Frigate API endpoint (default: http://localhost:5000)
- `SNAPSHOT_CORRELATION_WINDOW` - Seconds to correlate snapshots (default: 2.0)
- `MAX_TRACKED_PERSONS_PER_CAMERA` - Queue size for correlation (default: 3)

**Updated Dependencies:**
- Added `requests>=2.31.0` for API calls

### 2. Home Assistant Integration Updates (frigate_identity_ha/)

**New Sensors:**
- ✅ `sensor.frigate_identity_last_person` - Most recently detected person (updated)
- ✅ `sensor.frigate_identity_all_persons` - Tracks all currently detected persons with full attributes

**Enhanced Attributes:**
- `frigate_zones` - Zones the person is detected in
- `event_id` - Frigate event ID for API lookups
- `snapshot_url` - Direct URL to cropped snapshot
- `source` - Identification source (facial_recognition, reid_model)
- `last_seen` - Timestamp of last detection

### 3. Documentation

**Created Files:**
- ✅ `CONFIGURATION_EXAMPLES.md` - Complete Home Assistant configuration guide including:
  - MQTT camera setup for live person snapshots
  - Template sensors for per-person tracking
  - Supervision detection sensors
  - Safety automation examples
  - Dashboard configuration
  - Frigate MQTT snapshot configuration
  
- ✅ `.env.example` - Example environment configuration for identity service

**Updated Files:**
- ✅ `README.md` (identity service) - Documents new architecture and environment variables
- ✅ `README.md` (HA integration) - Points to configuration examples

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│ FRIGATE                                                 │
│ - Facial recognition (sub_label in /update messages)   │
│ - MQTT snapshots (cropped, fast)                       │
│ - API snapshots (accurate, on-demand)                  │
└────────────┬────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────┐
│ IDENTITY SERVICE (Two-Tier Processing)                 │
│                                                          │
│ Fast Path (MQTT Snapshots):                            │
│ - Correlate snapshot to recent person detection       │
│ - Publish to identity/snapshots/{person_id}           │
│ - ~50-100ms latency for live dashboard                │
│                                                          │
│ Accurate Path (API Snapshots):                         │
│ - Fetch via event_id for guaranteed correct crop      │
│ - Extract and store embeddings for ReID               │
│ - ~300-500ms latency for accurate storage             │
└────────────┬────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────┐
│ HOME ASSISTANT                                          │
│ - MQTT camera entities (live person snapshots)        │
│ - Template sensors (per-person location tracking)     │
│ - Binary sensors (supervision detection)              │
│ - Automations (safety alerts)                         │
│ - Dashboard (live monitoring)                          │
└─────────────────────────────────────────────────────────┘
```

## Data Flow Example

1. **Frigate detects Alice** → Publishes to `frigate/backyard/1234-abc/update` with `sub_label: "Alice"`
2. **Identity Service receives update**:
   - Adds to camera_person_queue for correlation
   - Fetches accurate snapshot via API: `/api/events/1234-abc/thumbnail.jpg?crop=1`
   - Extracts and stores embedding
   - Publishes: `identity/person/Alice` with zones, confidence, snapshot_url
3. **Frigate publishes snapshot** → `frigate/backyard/person/snapshot` (JPEG bytes)
4. **Identity Service correlates**:
   - Matches to Alice (most recent person on backyard camera)
   - Republishes: `identity/snapshots/Alice` (fast display)
5. **Home Assistant receives**:
   - Updates `sensor.frigate_identity_all_persons` attributes
   - Updates `camera.alice_snapshot` entity
   - Template sensors extract Alice-specific data
   - Automations check zones for safety alerts

## Testing the Implementation

### 1. Start the Identity Service

```bash
cd frigate_identity_service
cp .env.example .env
# Edit .env with your MQTT broker and Frigate host
pip install -r requirements.txt
python identity_service.py
```

Expected output:
```
Initializing embedding store...
Initializing ReID model (osnet_x1_0)...
ReID system ready!
Connected to MQTT Broker at localhost:1883
Frigate API endpoint: http://localhost:5000
Subscribed to: frigate/+/+/update
Subscribed to: frigate/+/person/snapshot
```

### 2. Install Home Assistant Integration

1. Copy the integration to Home Assistant:
   ```bash
   cp -r frigate_identity_ha/frigate-identity-ha/custom_components/frigate_identity \
     ~/.homeassistant/custom_components/
   ```

2. Restart Home Assistant

3. Add the integration via UI (Configuration → Integrations → Add Integration → Frigate Identity)

### 3. Configure MQTT Cameras (Optional)

Add to `configuration.yaml`:
```yaml
mqtt:
  camera:
    - name: "Alice Snapshot"
      topic: "identity/snapshots/Alice"
```

### 4. Test Detection

Walk in front of a Frigate camera and check:
- Identity service logs show face recognition or ReID matching
- Home Assistant sensor updates: `sensor.frigate_identity_last_person`
- MQTT camera entity shows person snapshot (if configured)

## Next Steps

### Phase 1: Basic Usage (Implemented ✅)
- Identity service running with two-tier snapshots
- Home Assistant sensors tracking persons
- MQTT camera entities for live snapshots

### Phase 2: Advanced Features (Configuration Examples Provided)
- Per-person template sensors
- Supervision detection
- Safety automations (child near street, vehicle alerts)
- Dashboard with live snapshots

### Phase 3: Future Enhancements (Design Phase)
- Config flow for defining persons in HA UI
- Dynamic entity creation per person
- Zone-based rules engine
- Time-based confidence decay
- Historical tracking and analytics
- Custom Lovelace cards for visual zone map

## Files Modified/Created

### Identity Service
- ✏️ `identity_service.py` - Complete rewrite for two-tier architecture
- ✏️ `requirements.txt` - Added requests
- ✏️ `README.md` - Updated documentation
- ➕ `.env.example` - Example configuration

### Home Assistant Integration
- ✏️ `sensor.py` - Added FrigateIdentityAllPersonsSensor, enhanced attributes
- ✏️ `README.md` - Updated documentation
- ➕ `CONFIGURATION_EXAMPLES.md` - Comprehensive configuration guide

## Known Issues / Type Checker Warnings

The following Pylance warnings are **expected and safe to ignore**:
- `Import "requests" could not be resolved` - Package not installed in dev environment
- `"extract_embedding" is not a known attribute of "None"` - Code checks REID_AVAILABLE before calling
- These will not cause runtime issues when dependencies are installed

## Questions?

Refer to:
- [CONFIGURATION_EXAMPLES.md](https://github.com/awayman/frigate-identity-ha/blob/main/CONFIGURATION_EXAMPLES.md) for Home Assistant setup
- [README.md](README.md) for Identity Service deployment
- GitHub issues for support
