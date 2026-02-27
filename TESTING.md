# Testing Guide

## Quick Start with Docker Compose (Recommended)

The easiest way to set up a complete test environment is using Docker Compose. This provides Frigate, MQTT broker, and all dependencies in isolated containers:

```powershell
# Start all services
docker-compose up -d

# Run integration tests
python run_integration_tests.py

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

See [DOCKER_COMPOSE_SETUP.md](DOCKER_COMPOSE_SETUP.md) for detailed documentation, troubleshooting, and advanced usage.

---

## Prerequisites (Manual Setup)

If not using Docker Compose, you'll need:

1. **Frigate Instance** - Running and configured to send person crops via MQTT
2. **Home Assistant** - With MQTT integration enabled
3. **MQTT Broker** - Accessible from both Frigate and your test machine
4. **Python Environment** - With dependencies installed from requirements.txt

## Setup Steps (Manual)

### 1. Install Test Dependencies

```powershell
pip install -r requirements.txt
pip install pytest pytest-cov  # For unit tests
```

### 2. Configure Frigate to Send Images via MQTT

In your Frigate `config.yaml`, verify MQTT is enabled:

```yaml
mqtt:
  host: 127.0.0.1
  port: 1883
  client_id: frigate

objects:
  track:
    - person
```

Verify Frigate is publishing to `frigate/events` topic:
```powershell
# Using MQTT Explorer or mosquitto_sub (if installed)
mosquitto_sub -h localhost -p 1883 -t "frigate/events"
```

### 3. Start the Identity Service

```powershell
# Terminal 1: Start identity_service
$env:MQTT_BROKER = "localhost"
$env:MQTT_PORT = "1883"
$env:EMBEDDINGS_DB_PATH = "./embeddings.json"
python identity_service.py
```

You should see:
```
Initializing embedding store...
Initializing ReID model (osnet_x1_0)...
ReID system ready!
Connected to MQTT Broker at localhost:1883
```

### 4. Run Unit Tests

```powershell
# Terminal 2: Run unit tests
pytest tests/test_components.py -v

# With coverage report
pytest tests/test_components.py -v --cov=. --cov-report=html
```

Expected output:
```
test_components.py::TestEmbeddingStore::test_embedding_store_initialization PASSED
test_components.py::TestEmbeddingStore::test_store_and_retrieve_embedding PASSED
test_components.py::TestMatcher::test_find_best_match_exact PASSED
...
========================= 10 passed in 0.32s =========================
```

### 5. Run Live Tests

```powershell
# Terminal 2: Publish test events
python test_mqtt_publisher.py --broker localhost --port 1883 --test basic
```

## Test Scenarios

### Scenario 1: Basic Face + Re-ID Matching

```powershell
python test_mqtt_publisher.py --broker localhost --port 1883 --test basic
```

**Expected behavior:**
1. Service receives face event (alice at front_door with red image)
2. Service extracts embedding and stores it
3. Service receives person event (porch with red image)
4. Service matches person to alice using cosine similarity
5. Service publishes: `identity/person/tracked` with person_id="alice"

**Check Home Assistant:**
- Go to Developer Tools  MQTT  Listen to: `identity/person/#`
- You should see both `identity/person/recognized` and `identity/person/tracked` messages

**Service Output:**
```
[FACE] alice recognized at front_door (confidence: 0.98)
[FACE] Stored embedding for alice at front_door
[REID] alice tracked at porch (similarity: 0.8234)
```

### Scenario 2: Multiple People

```powershell
python test_mqtt_publisher.py --broker localhost --port 1883 --test multiface
```

**Expected behavior:**
- Alice (red image) and Bob (blue image) are stored as known persons
- Person detections in hallway (red) match alice
- Person detections in garage (blue) match bob

**Service Output:**
```
[FACE] alice recognized at front_door (confidence: 0.95)
[FACE] Stored embedding for alice at front_door
[FACE] bob recognized at back_door (confidence: 0.93)
[FACE] Stored embedding for bob at back_door
[REID] alice tracked at hallway (similarity: 0.82)
[REID] bob tracked at garage (similarity: 0.79)
```

### Scenario 3: Re-ID Accuracy

```powershell
python test_mqtt_publisher.py --broker localhost --port 1883 --test reid
```

**Expected behavior:**
- Same color (red)  should match alice above 0.6 threshold
- Different color (yellow)  should NOT match, similarity < 0.6

**Service Output:**
```
[FACE] alice recognized at entry (confidence: 0.96)
[FACE] Stored embedding for alice at entry
[REID] alice tracked at hallway (similarity: 0.85)
[REID] No person match found at garage (best similarity was 0.42)
```

## Troubleshooting

### Service won't connect to MQTT broker

```powershell
# Check MQTT broker is running and accessible
mosquitto_sub -h localhost -p 1883 -t "frigate/events"

# If it blocks forever, broker is not accessible
# Check firewall, port, and broker address
```

**Solution:**
- Verify MQTT broker is running: `mosquitto -v` or check Docker containers
- Check firewall allows port 1883
- Verify correct IP address for MQTT_BROKER env var

### No embeddings being stored

Check service logs for error messages like:
```
[FACE] Warning: Could not extract embedding for alice: ...
```

If image data is missing, Frigate may not be sending it. Check that:
1. MQTT events include `"image"` field with base64-encoded data
2. Image is valid PNG/JPEG (not corrupted)

### Re-ID matches are incorrect (too many false positives)

Adjust the similarity threshold to be more strict:
```powershell
$env:REID_SIMILARITY_THRESHOLD = "0.7"  # More strict (default 0.6)
python identity_service.py
```

Test with different thresholds:
- **0.5** - Very loose (many false positives)
- **0.6** - Default (balanced)
- **0.7** - Stricter (fewer false positives, more misses)
- **0.8** - Very strict (only exact matches)

### Model loading fails with GPU errors

Force CPU-only mode:
```powershell
$env:REID_DEVICE = "cpu"
python identity_service.py
```

### Database file permission errors

Ensure the embeddings database path is writable:
```powershell
$env:EMBEDDINGS_DB_PATH = "$PWD/embeddings.json"
python identity_service.py
```

## Monitoring Identity Events

### Home Assistant MQTT Sensors

Go to **Developer Tools  MQTT  Subscribe to: `identity/person/#`**

You'll see events like:
```json
{
  "person_id": "alice",
  "confidence": 0.8234,
  "camera": "porch",
  "timestamp": 1702000000,
  "frigate_event_id": "abc123",
  "source": "reid_model",
  "similarity_score": 0.8234
}
```

### Service Logs

Watch the service output for patterns:
```
[FACE] alice recognized at front_door (confidence: 0.98)
[FACE] Stored embedding for alice at front_door
[REID] alice tracked at porch (similarity: 0.8234)
[REID] No person match found at garage (best similarity was 0.32)
```

## Next Steps After Testing

1. **Tune Similarity Threshold** - Adjust `REID_SIMILARITY_THRESHOLD` based on false positive/negative rate
2. **Monitor Performance** - Track CPU/memory usage with `Process Explorer` or Task Manager
3. **Collect Real Data** - Test with actual Frigate person crops from your cameras
4. **Integrate with Automations** - Use the `identity/person/tracked` events in Home Assistant automations

## Performance Notes

- **First Run** - Downloading ReID model takes 1-2 minutes
- **Model Inference** - ~50-100ms per person crop (CPU), ~10-20ms (GPU)
- **Embedding Comparison** - ~1ms per stored person
- **Memory** - ~2GB for ResNet-based model loaded in memory

## Custom Test Images

To test with real screenshots from your Frigate instance:

```powershell
# Capture a person crop from Frigate Web UI
# Save as test_alice.png

# Encode to base64 and use in test events:
$imageData = [System.IO.File]::ReadAllBytes("test_alice.png")
$base64 = [System.Convert]::ToBase64String($imageData)

# Modify test_mqtt_publisher.py to use your image
```

## CI/CD Integration

Run tests automatically on code changes:

```powershell
# Run all tests
pytest tests/ -v

# Run only unit tests (skip slow model tests)
pytest tests/test_components.py -v -m "not slow"

# Generate coverage report
pytest tests/test_components.py -v --cov=. --cov-report=html
```

View coverage report: `htmlcov/index.html`
