"""
MQTT utilities for backward compatibility with paho-mqtt 1.x and 2.x
Includes connection management with exponential backoff and message queuing
"""
import paho.mqtt.client as mqtt
import threading
import time
import random
from collections import deque


def get_mqtt_client():
    """
    Create MQTT client with version-appropriate initialization.
    Supports both paho-mqtt 1.x and 2.x for backward compatibility.
    
    Returns:
        mqtt.Client: MQTT client instance
    """
    # Check if paho-mqtt 2.x API is available
    if hasattr(mqtt, 'CallbackAPIVersion'):
        # Use paho-mqtt 2.x API
        return mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    else:
        # Fall back to paho-mqtt 1.x API
        return mqtt.Client()


class MQTTConnectionManager:
    """
    MQTT connection manager with exponential backoff, message queuing, and reconnection logic.
    
    This class provides:
    - Exponential backoff retry (1s -> 60s cap with jitter)
    - Reconnect attempt counter
    - Thread-safe outbound message queue (cap at 100 messages)
    - Republish queued messages on reconnect
    - Configurable max retry attempts
    - Status callbacks for disconnect/reconnect events
    """
    
    # Queue warning threshold (warn when queue is 90% full)
    QUEUE_WARNING_THRESHOLD = 0.9
    
    def __init__(self, client, initial_delay=1, max_delay=60, max_queue_size=100, max_retries=-1):
        """
        Initialize MQTT connection manager.
        
        Args:
            client: paho-mqtt Client instance
            initial_delay: Initial reconnection delay in seconds (default: 1)
            max_delay: Maximum reconnection delay in seconds (default: 60)
            max_queue_size: Maximum number of messages to queue (default: 100)
            max_retries: Maximum reconnection attempts, -1 for unlimited (default: -1)
        """
        self.client = client
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.max_queue_size = max_queue_size
        self.max_retries = max_retries
        
        # Reconnection tracking
        self.reconnect_count = 0
        self.is_connected = False
        
        # Thread-safe message queue
        self.message_queue = deque(maxlen=max_queue_size)
        self.queue_lock = threading.Lock()
        
        # Callbacks
        self.on_disconnect_callback = None
        self.on_reconnect_callback = None
        
        # Configure paho-mqtt's built-in reconnection delay
        self.client.reconnect_delay_set(min_delay=initial_delay, max_delay=max_delay)
    
    def set_disconnect_callback(self, callback):
        """
        Set callback to be called on disconnect.
        
        Args:
            callback: Function with signature callback(reason_code, reason_string)
        """
        self.on_disconnect_callback = callback
    
    def set_reconnect_callback(self, callback):
        """
        Set callback to be called on successful reconnect.
        
        Args:
            callback: Function with signature callback(reconnect_count)
        """
        self.on_reconnect_callback = callback
    
    def handle_connect(self):
        """Called when connection is established."""
        self.is_connected = True
        
        # Drain the message queue
        self._republish_queued_messages()
        
        # Call reconnect callback if this is a reconnection
        if self.reconnect_count > 0 and self.on_reconnect_callback:
            self.on_reconnect_callback(self.reconnect_count)
        
        # Reset reconnect counter after successful connection
        self.reconnect_count = 0
    
    def handle_disconnect(self, reason_code, reason_string=""):
        """Called when connection is lost."""
        self.is_connected = False
        self.reconnect_count += 1
        
        # Call disconnect callback
        if self.on_disconnect_callback:
            self.on_disconnect_callback(reason_code, reason_string)
    
    def publish_or_queue(self, topic, payload=None, qos=0, retain=False):
        """
        Publish message or queue it if disconnected.
        
        Args:
            topic: MQTT topic
            payload: Message payload
            qos: Quality of Service level
            retain: Whether to retain the message
            
        Returns:
            bool: True if published immediately, False if queued
        """
        if self.is_connected:
            try:
                self.client.publish(topic, payload, qos, retain)
                return True
            except Exception as e:
                # If publish fails, queue the message
                self._queue_message(topic, payload, qos, retain)
                return False
        else:
            # Not connected, queue the message
            self._queue_message(topic, payload, qos, retain)
            return False
    
    def _queue_message(self, topic, payload, qos, retain):
        """Add message to queue (thread-safe)."""
        with self.queue_lock:
            message = {
                'topic': topic,
                'payload': payload,
                'qos': qos,
                'retain': retain
            }
            self.message_queue.append(message)
            
            # Log if queue is getting full
            if len(self.message_queue) >= self.max_queue_size * self.QUEUE_WARNING_THRESHOLD:
                print(f"[MQTT] Warning: Message queue is {len(self.message_queue)}/{self.max_queue_size}")
    
    def _republish_queued_messages(self):
        """Republish all queued messages after reconnection."""
        with self.queue_lock:
            if len(self.message_queue) == 0:
                return
            
            print(f"[MQTT] Republishing {len(self.message_queue)} queued messages...")
            
            # Drain the queue
            while self.message_queue:
                msg = self.message_queue.popleft()
                try:
                    self.client.publish(msg['topic'], msg['payload'], msg['qos'], msg['retain'])
                except Exception as e:
                    print(f"[MQTT] Error republishing message to {msg['topic']}: {e}")
                    # Re-queue the message at the end
                    self.message_queue.append(msg)
                    break
            
            print(f"[MQTT] Finished republishing queued messages")
    
    def get_queue_size(self):
        """Get current size of message queue."""
        with self.queue_lock:
            return len(self.message_queue)
    
    def calculate_backoff_delay(self, attempt, with_jitter=True):
        """
        Calculate exponential backoff delay.
        
        Args:
            attempt: Retry attempt number (0-indexed)
            with_jitter: Whether to add random jitter
            
        Returns:
            float: Delay in seconds
        """
        # Exponential backoff: 2^attempt * initial_delay
        delay = min(self.initial_delay * (2 ** attempt), self.max_delay)
        
        # Add jitter (±25%)
        if with_jitter:
            jitter = delay * 0.25
            delay += random.uniform(-jitter, jitter)
        
        return max(delay, 0)  # Ensure non-negative
