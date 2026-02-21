**Frigate Identity Service**

Lightweight ReID service that provides person identification continuity for Frigate. Uses facial recognition as the primary identity source and ReID (re-identification) to maintain identity when faces are not visible.

**Architecture:**

- **Two-Tier Snapshot System:**
  - MQTT snapshots for fast dashboard display (~50-100ms latency)
  - API snapshots for accurate embedding storage (~300-500ms latency)
  
- **Identity Sources:**
  - Frigate facial recognition (primary, highest confidence)
  - ReID model matching (continuity when face not visible)

**ReID Model Selection:**

The service uses [torchreid](https://github.com/KaiyangZhou/deep-person-reid) to load
dedicated person re-identification models such as OSNet.  Set the `REID_MODEL`
environment variable (or the `reid_model` option in `config.yaml`) to choose
a model:

| Model | Embedding dim | Notes |
|-------|--------------|-------|
| `osnet_x1_0` (default) | 512 | Best accuracy, recommended |
| `osnet_x0_75` | 512 | Lighter variant |
| `osnet_x0_5` | 512 | Lighter variant |
| `osnet_x0_25` | 512 | Lightest OSNet |
| `osnet_ibn_x1_0` | 512 | OSNet + Instance Batch Norm |
| `osnet_ain_x1_0` | 512 | OSNet + Attention Instance Norm |
| `resnet50` | 2048 | Generic ImageNet fallback (no torchreid required) |

If torchreid is not installed and a torchreid model is requested, the service
automatically falls back to ResNet50.  Both GPU (`cuda`) and CPU-only modes are
fully supported.

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MQTT_BROKER` | `localhost` | MQTT broker hostname |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_USERNAME` | (optional) | MQTT authentication username |
| `MQTT_PASSWORD` | (optional) | MQTT authentication password |
| `FRIGATE_HOST` | `http://localhost:5000` | Frigate HTTP API endpoint |
| `REID_MODEL` | `osnet_x1_0` | ReID model name (`osnet_x1_0`, `osnet_x0_75`, `osnet_x0_5`, `osnet_x0_25`, `osnet_ibn_x1_0`, `osnet_ain_x1_0`, or `resnet50`) |
| `REID_DEVICE` | `auto` | Device for ReID (`auto`, `cuda`, `cpu`) |
| `REID_SIMILARITY_THRESHOLD` | `0.6` | Minimum similarity score for ReID match |
| `EMBEDDINGS_DB_PATH` | `embeddings.json` | Path to store person embeddings |
| `SNAPSHOT_CORRELATION_WINDOW` | `2.0` | Seconds to correlate MQTT snapshots to persons |
| `MAX_TRACKED_PERSONS_PER_CAMERA` | `3` | Max persons tracked per camera for correlation |

**MQTT Topics:**

**Subscriptions:**
- `frigate/events` - Tracked object updates (new/update/end); contains face recognition via `sub_label` field ([Frigate docs](https://docs.frigate.video/integrations/mqtt#frigateevents))
- `frigate/tracked_object_update` - Face recognition and LPR metadata updates ([Frigate docs](https://docs.frigate.video/integrations/mqtt#frigatetracked_object_update))
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
- **`requirements.txt`**: Python dependencies (GPU-capable). See [requirements.txt](requirements.txt).
- **`requirements-cpu.txt`**: CPU-only Python dependencies for Home Assistant Add-on. See [requirements-cpu.txt](requirements-cpu.txt).
- **`Dockerfile`**: Container image build (CPU-only by default; pass `--build-arg USE_GPU=true` for GPU). See [Dockerfile](Dockerfile).
- **`config.yaml`**: Home Assistant Add-on manifest. See [config.yaml](config.yaml).
- **`run.sh`**: Container entry point used by the Home Assistant Add-on. See [run.sh](run.sh).

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

Build and run the container (CPU-only, suitable for most deployments):

```bash
docker build -t frigate-identity .
docker run --env MQTT_BROKER=host.docker.internal --env MQTT_PORT=1883 frigate-identity
```

To build with GPU (CUDA) support:

```bash
docker build --build-arg USE_GPU=true -t frigate-identity-gpu .
docker run --gpus all --env MQTT_BROKER=host.docker.internal --env MQTT_PORT=1883 frigate-identity-gpu
```

**Home Assistant Add-on**

This repository can be used directly as a Home Assistant Add-on repository.  GPU acceleration is not required when deployed as an Add-on; the service falls back to CPU-based ReID automatically.

1. In Home Assistant, navigate to **Settings → Add-ons → Add-on Store**.
2. Click the three-dot menu (⋮) and select **Repositories**.
3. Add this repository URL: `https://github.com/awayman/frigate_identity_service`
4. Find **Frigate Identity Service** in the store and click **Install**.
5. Configure the add-on options (MQTT broker, Frigate host, etc.) and click **Start**.

Configuration is written by the Supervisor to `/data/options.json` and is read automatically on startup.  All options from `config.yaml` map to the environment variables listed above (e.g. `mqtt_broker` → `MQTT_BROKER`).

> **Note:** GPU acceleration is not available in Home Assistant Add-on deployments.  The ReID model runs on CPU, which is sufficient for most home use cases.  For GPU-accelerated deployments, use the standalone Docker image built with `--build-arg USE_GPU=true`.

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
