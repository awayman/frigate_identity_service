# Docker Compose Test Environment Setup

This document describes the integrated test environment using Docker Compose. This setup provides a lightweight, self-contained environment for development and integration testing without requiring Home Assistant, a real Frigate instance, or manual prerequisite configuration.

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Python 3.10+ and requirements installed (for running integration tests)

### Start the Environment

```bash
# Start all services in the background
docker-compose up -d

# View service logs
docker-compose logs -f

# Run integration tests
python run_integration_tests.py

# Stop services
docker-compose down
```

That's it! The environment will:
1. Start an MQTT broker (Mosquitto)
2. Start a mock Frigate service that publishes synthetic test events
3. Start the identity-service with pre-downloaded models

Expected startup time: **2-3 minutes** (mostly model download and initialization)

## Services

### 1. Mosquitto (MQTT Broker)
- **Container:** `frigate_identity_mosquitto`
- **Port:** `1883` (MQTT), `9001` (WebSocket)
- **Volume:** `mosquitto_data`, `mosquitto_logs`
- **Status:** Healthy when broker is accepting connections
- **Purpose:** Message broker for all inter-service communication

#### Verify Mosquitto is running:
```bash
# Monitor MQTT traffic (requires mosquitto-clients installed locally)
mosquitto_sub -h localhost -t "frigate/#"

# Or via Docker
docker exec frigate_identity_mosquitto mosquitto_sub -h localhost -t "frigate/#" -n 1
```

### 2. Frigate Mock Service
- **Container:** `frigate_identity_frigate_mock`
- **Status:** Healthy when MQTT broker is healthy
- **Purpose:** Simulates Frigate NVR by publishing synthetic person detection events
- **Topics Published:**
  - `frigate/events` - Person detection events
  - `frigate/tracked_object_update` - Object tracking updates
  - `frigate/{camera}/person/snapshot` - Person snapshots with base64-encoded images

#### Configuration:
- `EVENT_INTERVAL=20` - Publishes a complete event cycle every 20 seconds
- Tests 3 different colored "persons" to simulate multiple people detection

#### Verify events are being published:
```bash
docker-compose logs frigate-mock
# Should show: "Published snapshot to frigate/front_door/person/snapshot"
```

### 3. Identity Service
- **Container:** `frigate_identity_identity_service`
- **Port:** `5000` (internal, not exposed)
- **Volumes:**
  - `/data` - Persistent storage for embeddings and debug logs
  - `embeddings.json` - Learned person embeddings database
- **Status:** Healthy after models load (1-2 minutes)
- **Purpose:** Main identity service that processes Frigate events

#### Verify service is ready:
```bash
docker-compose logs identity-service
# Look for messages like: "Successfully connected to MQTT broker"
# And: "Models loaded successfully"

# Or check service health
docker ps | grep frigate_identity_identity_service
```

## Integration Testing

### Run Full Integration Test Suite

```bash
# Make sure services are running
docker-compose up -d

# Run integration tests
python run_integration_tests.py

# Expected output:
# ✅ MQTT Broker: Connected successfully
# ✅ Identity Service: Running and processing events
# ✅ Event Processing: X events processed, Y published
```

### Run Specific Tests

```bash
# Run only MQTT connectivity tests
python -c "from frigate_identity_service.mqtt_utils import create_mqtt_client; \
  client = create_mqtt_client('localhost', 1883); print('MQTT OK')"

# Manual event monitoring
docker exec frigate_identity_mosquitto mosquitto_sub -h localhost -t "frigate/#" -v
```

### Testing Different Scenarios

The `frigate-mock` service publishes three test persons with different colors. You can verify that the identity service correctly:
1. Receives person detection events from the mock Frigate
2. Processes face images from snapshots
3. Publishes recognized person identities back to MQTT
4. Stores embeddings for future matching

Monitor identity-service logs to see processing:
```bash
docker-compose logs -f identity-service
```

Look for messages indicating successful MQTT connections, model loading, and event processing.

## Persistent Data

The compose setup creates Docker volumes that persist between restarts:

- `mosquitto_data`: MQTT broker persistent storage
- `mosquitto_logs`: MQTT broker logs
- `identity_data`: Identity service embeddings and debug logs

To preserve embeddings across restarts:
```bash
# Embeddings are automatically saved to ./embeddings.json (volume mount)
# They persist as long as you don't run: docker-compose down -v
```

To reset all data:
```bash
docker-compose down -v
```

## Environment Configuration

Edit `.env.docker-compose` to modify behavior:

```bash
# Change how frequently mock Frigate publishes events
EVENT_INTERVAL=20          # seconds between event cycles (default: 20)

# Adjust identity service configuration
SIMILARITY_THRESHOLD=0.6   # Face match confidence threshold
LOGGING_LEVEL=INFO         # DEBUG, INFO, WARNING, ERROR

# Model configuration
MODEL_REID=osnet_x1_0      # ReID model type
ENCODING_MODEL_SIZE=medium # Face encoding model size
```

## Troubleshooting

### "Connection refused" errors
**Problem:** Service fails to connect to MQTT.

**Solution:**
```bash
# Check Mosquitto health
docker-compose ps mosquitto

# Wait for broker to become healthy
docker-compose logs mosquitto

# If stuck, rebuild services
docker-compose down
docker-compose up -d
```

### Identity service won't start / very slow startup
**Problem:** Service takes a long time to initialize (1-2 minutes).

**Expected Behavior:** First startup downloads and initializes ML models (1-2 GB).

**Solution:**
```bash
# Check initialization progress
docker-compose logs -f identity-service

# Look for: "Downloading model..." / "Models loaded successfully"

# If stuck for > 5 minutes, check logs for errors:
docker-compose logs identity-service | tail -50
```

### No events being published
**Problem:** Mock Frigate service not publishing events.

**Solution:**
```bash
# Check mock Frigate service status
docker-compose logs frigate-mock

# If it exited, check for MQTT connection errors:
docker-compose logs frigate-mock | grep -i error

# Restart the service
docker-compose restart frigate-mock
```

### MQTT topic data not appearing
**Problem:** Events published but identity service not processing them.

**Solution:**
```bash
# Verify service is subscribed to topics
docker-compose logs identity-service | grep -i "subscribed\|subscribe"

# Monitor raw MQTT traffic
docker exec frigate_identity_mosquitto mosquitto_sub -h localhost -t "#" -v

# Check if service is receiving events
docker-compose logs -f identity-service | grep -i "event\|received"
```

## Differences from Real Frigate

The mock Frigate service is **not** a full Frigate implementation. Key differences:

| Feature | Real Frigate | Mock Frigate |
|---------|--------------|--------------|
| Live camera streams | ✅ RTSP/HTTP | ❌ Not available |
| Full detection processing | ✅ YOLOv8, tracking | ⚠️ Synthetic events only |
| API endpoints | ✅ REST API on port 5000 | ❌ Not implemented |
| Configuration UI | ✅ Web interface | ❌ Not available |
| Event filtering/retention | ✅ Database backend | ❌ Not available |

**Use real Frigate for:** Testing integration with actual camera streams, testing detection accuracy, performance benchmarking.

**Use mock Frigate for:** Development, integration testing, client-side event processing verification, debugging communication flows.

## Monitoring and Debugging

### View All Service Logs
```bash
docker-compose logs -f
```

### View Specific Service Logs
```bash
docker-compose logs -f mosquitto
docker-compose logs -f frigate-mock
docker-compose logs -f identity-service
```

### Monitor MQTT in Real-time
```bash
docker exec frigate_identity_mosquitto mosquitto_sub -h localhost -t "#" -v
```

### Check Service Health
```bash
docker-compose ps
```

### Access Service Containers
```bash
# Open shell in identity-service
docker-compose exec identity-service /bin/bash

# Check embeddings database
docker-compose exec identity-service cat /data/embeddings.json | jq .
```

### View Service Resource Usage
```bash
docker stats
```

## Advanced Usage

### Run Services Individually
```bash
# Start only MQTT broker
docker-compose up -d mosquitto

# Start only identity service (MQTT must be running)
docker-compose up -d identity-service

# Start only mock Frigate (MQTT must be running)
docker-compose up -d frigate-mock
```

### Custom Configuration
Create a `.env` file (not committed) to override defaults:
```bash
# .env
EVENT_INTERVAL=10
LOGGING_LEVEL=DEBUG
SIMILARITY_THRESHOLD=0.5
```

Then run:
```bash
docker-compose up -d
```

### Build for GPU Support
```bash
docker-compose build --build-arg USE_GPU=true identity-service
docker-compose up -d
```

Note: Requires NVIDIA Docker runtime and compatible GPU.

### Rebuild Services
```bash
# Rebuild all services
docker-compose build --no-cache

# Rebuild specific service
docker-compose build --no-cache identity-service

# Then restart
docker-compose up -d
```

## Integration with Real Frigate

To test with a **real Frigate instance** instead of the mock:

1. Ensure real Frigate is running and accessible
2. Stop the mock service:
   ```bash
   docker-compose stop frigate-mock
   ```
3. Update `FRIGATE_HOST` in your `.env`:
   ```bash
   FRIGATE_HOST=<your-frigate-ip>
   ```
4. Restart identity service:
   ```bash
   docker-compose up -d
   ```

The identity service will now connect to real Frigate and process actual camera events.

## Cleanup

```bash
# Stop services (volumes preserved)
docker-compose down

# Stop and remove volumes (resets embeddings)
docker-compose down -v

# Remove images (forces rebuild on next `up`)
docker-compose down -v --rmi all
```

## Performance Notes

- **Startup time:** 2-3 minutes (model download and initialization)
- **Model download size:** ~1.2 GB (first run only)
- **CPU usage:** ~30-40% per service (CPU-only mode)
- **Memory usage:** ~800MB-1.2GB per service
- **Event processing latency:** < 100ms per event (depends on model size)

Subsequent startups (~30 seconds) after cached models are available.

## Next Steps

See [TESTING.md](TESTING.md) for:
- Writing custom test scenarios
- Validating model accuracy
- Performance benchmarking
- Integration with Home Assistant

See [README.md](README.md) for:
- Configuration options
- Deployment to Home Assistant
- Production usage
