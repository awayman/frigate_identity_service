**Frigate Identity Service**

Lightweight service to consume Frigate MQTT events, publish identity events for faces, and propagate identities to weaker cameras using ReID heuristics.

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

**Troubleshooting**
- Import error for `paho.mqtt.client`: ensure `paho-mqtt` is installed in the active interpreter (`python -m pip install paho-mqtt`).
- Unable to connect to MQTT broker: check `MQTT_BROKER`/`MQTT_PORT` env vars and network reachability.

**Next steps**
- Add `camera_topology.yaml` for spatial matching and persist tracked identities between restarts.

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
