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

**Home Assistant Add-on**

This repository can be used as a Home Assistant add-on. A minimal add-on manifest is included in `config.json`.

Install / build instructions:

- Place this repository in your Home Assistant add-ons folder (e.g., `/addons/local/frigate_identity_service`) or use the Supervisor "Add-on store -> Repositories" method.
- Configure add-on options in the Supervisor UI, or set `mqtt_broker` / `mqtt_port` in the add-on configuration.
- The add-on uses host networking by default (`host_network: true`) so it can reach your MQTT broker on the host.

Example add-on options (Supervisor UI):

```json
{
	"mqtt_broker": "core-mosquitto",
	"mqtt_port": 1883
}
```

When running as an add-on, Home Assistant Supervisor will set environment variables from the add-on options. The service reads `MQTT_BROKER` and `MQTT_PORT` from the environment; the add-on manifest maps the `mqtt_broker` option into the container environment at runtime.

If you want me to convert this into a more complete add-on (web UI, health checks, validated options schema, multi-arch build), tell me which features to prioritize.

**HACS (Custom Integration) Install**

This repository also includes a minimal HACS-compatible custom integration scaffold under `custom_components/frigate_identity`.

To install via HACS (Community Store):

1. In HACS, go to "Settings -> Custom repositories" and add this repository URL as a "Integration".
2. Install "Frigate Identity" from HACS and restart Home Assistant.
3. The integration is minimal and depends on the `mqtt` integration; enable/configure the add-on or ensure your MQTT broker is reachable.

After installation you can configure the add-on (if used) and enable the integration from Settings → Devices & Services.

If you'd like, I can expand the custom integration to expose entities (sensors/switches), provide services for manual re-id propagation, or integrate with the add-on to control lifecycle.
