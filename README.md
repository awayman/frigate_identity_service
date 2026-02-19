**Frigate Identity Service**

Lightweight ReID service that provides person identification continuity for Frigate. Uses facial recognition as the primary identity source and ReID (re-identification) to maintain identity when faces are not visible.

**Architecture:**

- **Two-Tier Snapshot System:**
  - MQTT snapshots for fast dashboard display (~50-100ms latency)
  - API snapshots for accurate embedding storage (~300-500ms latency)
  
- **Identity Sources:**
  - Frigate facial recognition (primary, highest confidence)
  - ReID model matching (continuity when face not visible)

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MQTT_BROKER` | `localhost` | MQTT broker hostname |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_USERNAME` | (optional) | MQTT authentication username |
| `MQTT_PASSWORD` | (optional) | MQTT authentication password |
| `FRIGATE_HOST` | `http://localhost:5000` | Frigate HTTP API endpoint |
| `REID_MODEL` | `osnet_x1_0` | ReID model name (timm compatible) |
| `REID_DEVICE` | `auto` | Device for ReID (`auto`, `cuda`, `cpu`) |
| `REID_SIMILARITY_THRESHOLD` | `0.6` | Minimum similarity score for ReID match |
| `EMBEDDINGS_DB_PATH` | `embeddings.json` | Path to store person embeddings |
| `SNAPSHOT_CORRELATION_WINDOW` | `2.0` | Seconds to correlate MQTT snapshots to persons |
| `MAX_TRACKED_PERSONS_PER_CAMERA` | `3` | Max persons tracked per camera for correlation |

**MQTT Topics:**

**Subscriptions:**
- `frigate/+/+/update` - Tracked object updates (includes face recognition)
- `frigate/+/person/snapshot` - Person snapshots (fast display)
- `frigate/+/car/snapshot` - Vehicle detection
- `frigate/+/truck/snapshot` - Vehicle detection

**Publications:**
- `identity/person/{person_id}` - Person identity events
- `identity/snapshots/{person_id}` - Person-specific snapshot images
- `identity/snapshots/{person_id}/metadata` - Snapshot correlation metadata
- `identity/vehicle/detected` - Vehicle detection events

**Files:**
- **`identity_service.py`**: Main service consuming Frigate events and publishing identity messages. See [identity_service.py](identity_service.py).
- **`requirements.txt`**: Python dependencies. See [requirements.txt](requirements.txt).
- **`Dockerfile`**: Container image build. See [Dockerfile](Dockerfile).

**Local setup (Windows PowerShell)**

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
& .\.venv\bin\Activate.ps1
```

2. Install dependencies and run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python identity_service.py
```

If activation fails, install into the venv directly:

```powershell
.\.venv\bin\python.exe -m pip install -r requirements.txt
.\.venv\bin\python.exe identity_service.py
```

**Docker**

Build and run the container:

```bash
docker build -t frigate-identity .
docker run --env MQTT_BROKER=host.docker.internal --env MQTT_PORT=1883 frigate-identity
```

**Development / VS Code**

- Select the workspace interpreter (Command Palette → "Python: Select Interpreter") and choose the `.venv` interpreter.
- If the editor still flags imports, reload the window or restart the Python language server.

**Configuration Files**

- `.env` - Environment configuration (copy from `.env.example`)
- `persons.yaml` - Person roles, ages, and supervision requirements
- `embeddings.json` - Stored person embeddings (auto-generated)

**Testing**

Run the test script to validate your setup:

```powershell
python test_system.py
```

This will check:
- MQTT broker connectivity
- Frigate API accessibility
- MQTT topic subscriptions
- Publish test events

**Troubleshooting**
- Import error for `paho.mqtt.client`: ensure `paho-mqtt` is installed in the active interpreter (`python -m pip install paho-mqtt`).
- Unable to connect to MQTT broker: check `MQTT_BROKER`/`MQTT_PORT` env vars and network reachability.
- No snapshots appearing: verify Frigate MQTT config has `crop: true` enabled

**Home Assistant Integration**

The Home Assistant integration has been moved to a separate repository for HACS compatibility:

📦 **[Frigate Identity HA](https://github.com/awayman/frigate-identity-ha)**

This repository contains:
- The core Frigate Identity Service (this repo)
- Standalone deployment via Docker or Python
- Home Assistant Add-on manifest for deployment as a Home Assistant add-on

For Home Assistant integration, see the separate repository above for:
- HACS installation instructions
- Home Assistant custom component setup
- Integration configuration and sensors
