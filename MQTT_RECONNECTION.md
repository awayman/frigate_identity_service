# MQTT Reconnection and Error Recovery

This document describes the MQTT reconnection and error recovery features implemented in the Frigate Identity Service.

## Overview

The Frigate Identity Service now includes robust MQTT reconnection and error recovery logic to handle network disruptions and broker outages. This is critical for a child safety system where silent failures are unacceptable.

## Features

### 1. Automatic Reconnection

The service automatically attempts to reconnect to the MQTT broker when the connection is lost. This uses paho-mqtt's built-in reconnection mechanism configured via `client.reconnect_delay_set()`.

**Key behaviors:**
- Initial reconnection delay: 1 second (configurable via `RECONNECT_INITIAL_DELAY`)
- Maximum reconnection delay: 60 seconds (configurable via `MAX_RECONNECT_DELAY`)
- Unlimited retry attempts by default
- Exponential backoff between attempts

### 2. Exponential Backoff

Reconnection delays follow an exponential backoff pattern with jitter:
- Attempt 1: 1s
- Attempt 2: 2s
- Attempt 3: 4s
- Attempt 4: 8s
- Attempt 5: 16s
- Attempt 6: 32s
- Attempt 7+: 60s (capped)

Jitter (±25%) is added to prevent synchronized reconnection storms when multiple clients reconnect simultaneously.

### 3. Message Queuing

When the MQTT broker is unavailable, outbound messages are queued in memory rather than dropped:
- **Queue capacity:** 100 messages (configurable)
- **Thread-safe:** Uses `collections.deque` with locks
- **FIFO ordering:** Oldest messages are dropped if queue fills
- **Automatic republish:** Queued messages are published upon reconnection

### 4. Status Callbacks

The connection manager provides callbacks for monitoring connection state:
- `on_disconnect_callback`: Called when connection is lost
- `on_reconnect_callback`: Called when reconnection succeeds

These callbacks are used to log connection state changes at appropriate levels:
- **WARNING** for disconnection
- **INFO** for successful reconnection

### 5. Frigate API Retry Logic

The `fetch_snapshot_from_api()` function now includes retry logic for resilient API access:
- **Max retries:** 3 attempts
- **Retry delay:** 0.5 seconds between attempts
- **Error handling:** Logs warnings instead of raising exceptions
- **Graceful degradation:** Returns `None` on failure instead of crashing

## Configuration

Add the following environment variables to your `.env` file:

```bash
# MQTT Reconnection Configuration
# Initial delay for reconnection attempts (seconds)
RECONNECT_INITIAL_DELAY=1

# Maximum delay for reconnection attempts (seconds)
MAX_RECONNECT_DELAY=60
```

## Architecture

### MQTTConnectionManager Class

The `MQTTConnectionManager` class in `mqtt_utils.py` provides the core reconnection functionality:

```python
from mqtt_utils import MQTTConnectionManager, get_mqtt_client

client = get_mqtt_client()
connection_manager = MQTTConnectionManager(
    client,
    initial_delay=1,
    max_delay=60,
    max_queue_size=100,
    max_retries=-1  # Unlimited
)

# Set up callbacks
connection_manager.set_disconnect_callback(on_disconnect_status)
connection_manager.set_reconnect_callback(on_reconnect_status)

# Attach to client for access in handlers
client._connection_manager = connection_manager
```

### Integration with identity_service.py

The `identity_service.py` integrates the connection manager:

1. **on_connect handler:** Notifies connection manager and drains queued messages
2. **on_disconnect handler:** Logs disconnection and updates connection state
3. **publish_or_queue wrapper:** All publish calls check connection state and queue if disconnected

## Usage Example

### Normal Operation

```python
# Publishing when connected
connection_manager.publish_or_queue("test/topic", "payload")
# → Message published immediately
```

### Disconnected Operation

```python
# Connection lost
# Messages are automatically queued
connection_manager.publish_or_queue("test/topic", "payload")
# → Message queued for later

# Multiple messages can be queued
for i in range(10):
    connection_manager.publish_or_queue(f"test/topic/{i}", f"payload {i}")

# Connection restored
# → All queued messages are republished automatically
```

### Monitoring Connection State

```python
def on_disconnect_status(reason_code, reason_string):
    print(f"[MQTT] Connection lost: {reason_string}")
    # Optional: Send alert to monitoring system

def on_reconnect_status(reconnect_count):
    print(f"[MQTT] Reconnected after {reconnect_count} attempt(s)")
    # Optional: Log reconnection success

connection_manager.set_disconnect_callback(on_disconnect_status)
connection_manager.set_reconnect_callback(on_reconnect_status)
```

## Testing

Run the comprehensive unit tests:

```bash
python -m pytest tests/test_mqtt_reconnection.py -v
```

**Test coverage:**
- ✓ Connection manager initialization
- ✓ Exponential backoff calculation
- ✓ Message queuing when disconnected
- ✓ Message publishing when connected
- ✓ Queue size limits
- ✓ Republish on reconnect
- ✓ Disconnect/reconnect callbacks
- ✓ API retry logic
- ✓ paho-mqtt 1.x/2.x compatibility

## Logging

The service provides detailed logging for connection events:

```
[MQTT] WARNING: Disconnected from MQTT Broker (reason_code=7)
[MQTT] INFO: Automatic reconnection will be attempted...
[MQTT] Republishing 5 queued messages...
[MQTT] Finished republishing queued messages
[MQTT] INFO: Successfully reconnected after 1 attempt(s)
```

## Performance Considerations

### Memory Usage
- Queue capacity: 100 messages × ~1KB average = ~100KB maximum
- Connection manager overhead: Negligible (<1KB)
- No memory leaks: Queue uses `deque` with `maxlen` to prevent unbounded growth

### CPU Impact
- Minimal: Exponential backoff reduces reconnection attempts over time
- Thread-safe: Uses locks for queue access, but contention is rare
- No busy-waiting: paho-mqtt handles reconnection in background

### Network Impact
- Exponential backoff prevents connection storms
- Jitter reduces synchronized reconnection attempts
- Frigate API retry logic uses short delays (0.5s) for quick recovery

## Backward Compatibility

The implementation maintains full backward compatibility:
- ✓ Works with paho-mqtt 1.x and 2.x
- ✓ Existing code continues to work without modification
- ✓ Connection manager is optional (degrades gracefully)
- ✓ Environment variables have sensible defaults

## Limitations

1. **Queue capacity:** Limited to 100 messages to prevent memory issues during extended outages
2. **No persistence:** Queued messages are lost if the service crashes
3. **FIFO only:** No message prioritization (all messages treated equally)
4. **No QoS handling:** Queued messages use original QoS level without verification

## Future Enhancements

Potential improvements for future releases:
- [ ] Persistent message queue (Redis/SQLite)
- [ ] Message prioritization (critical alerts first)
- [ ] Circuit breaker pattern for repeated failures
- [ ] Health check endpoint for monitoring
- [ ] Metrics export (reconnection count, queue depth, etc.)

## Troubleshooting

### Queue filling up
If you see warnings about the queue being full:
```
[MQTT] Warning: Message queue is 90/100
```

**Solutions:**
- Check MQTT broker connectivity
- Verify network stability
- Consider increasing `max_queue_size` (with caution)

### Repeated reconnection failures
If reconnection attempts never succeed:
```
[MQTT] WARNING: Disconnected from MQTT Broker (reason_code=5)
```

**Common causes:**
- MQTT broker is down
- Incorrect broker address/port
- Authentication failure
- Network firewall blocking connection

**Debug steps:**
1. Check MQTT broker logs
2. Verify `MQTT_BROKER` and `MQTT_PORT` settings
3. Test broker connectivity: `mosquitto_sub -h localhost -p 1883 -t "#"`
4. Review username/password if authentication is enabled

## Related Files

- `mqtt_utils.py` - Connection manager implementation
- `identity_service.py` - Integration with service
- `tests/test_mqtt_reconnection.py` - Unit tests
- `.env.example` - Configuration documentation
- `TODO.md` - Implementation status

## References

- [paho-mqtt documentation](https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html)
- [MQTT specification](https://mqtt.org/mqtt-specification/)
- [Exponential backoff](https://en.wikipedia.org/wiki/Exponential_backoff)
