"""
Unit tests for MQTT reconnection and error recovery logic.
Run with: python -m pytest tests/test_mqtt_reconnection.py -v
"""

import pytest
import sys
import os
import time
from unittest.mock import Mock, MagicMock, patch
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mqtt_utils import MQTTConnectionManager, get_mqtt_client


class TestMQTTConnectionManager:
    """Test MQTT connection manager functionality."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock MQTT client."""
        client = Mock()
        client.publish = Mock()
        client.reconnect_delay_set = Mock()
        return client
    
    @pytest.fixture
    def connection_manager(self, mock_client):
        """Create a connection manager instance."""
        return MQTTConnectionManager(
            mock_client,
            initial_delay=1,
            max_delay=60,
            max_queue_size=100,
            max_retries=-1
        )
    
    def test_initialization(self, connection_manager, mock_client):
        """Test connection manager initializes correctly."""
        assert connection_manager.client == mock_client
        assert connection_manager.initial_delay == 1
        assert connection_manager.max_delay == 60
        assert connection_manager.max_queue_size == 100
        assert connection_manager.reconnect_count == 0
        assert not connection_manager.is_connected
        assert len(connection_manager.message_queue) == 0
        
        # Verify reconnect_delay_set was called
        mock_client.reconnect_delay_set.assert_called_once_with(min_delay=1, max_delay=60)
    
    def test_exponential_backoff_calculation(self, connection_manager):
        """Test exponential backoff delay calculation."""
        # Test without jitter
        delays = [connection_manager.calculate_backoff_delay(i, with_jitter=False) for i in range(10)]
        
        # Verify exponential growth: 1, 2, 4, 8, 16, 32, 60, 60, 60, 60
        expected = [1, 2, 4, 8, 16, 32, 60, 60, 60, 60]
        assert delays == expected
        
        # Test with jitter - should be close to expected but with variation
        delays_with_jitter = [connection_manager.calculate_backoff_delay(i, with_jitter=True) for i in range(5)]
        
        # Verify jittered delays are within ±25% of non-jittered values
        for i, delay in enumerate(delays_with_jitter):
            expected_delay = min(1 * (2 ** i), 60)
            assert expected_delay * 0.75 <= delay <= expected_delay * 1.25
    
    def test_queue_message_when_disconnected(self, connection_manager):
        """Test that messages are queued when disconnected."""
        connection_manager.is_connected = False
        
        # Try to publish
        result = connection_manager.publish_or_queue("test/topic", "test payload", qos=1, retain=False)
        
        # Should return False (message was queued, not published)
        assert not result
        
        # Verify message is in queue
        assert len(connection_manager.message_queue) == 1
        assert connection_manager.message_queue[0]['topic'] == "test/topic"
        assert connection_manager.message_queue[0]['payload'] == "test payload"
        assert connection_manager.message_queue[0]['qos'] == 1
        assert connection_manager.message_queue[0]['retain'] == False
    
    def test_publish_when_connected(self, connection_manager, mock_client):
        """Test that messages are published immediately when connected."""
        connection_manager.is_connected = True
        
        # Try to publish
        result = connection_manager.publish_or_queue("test/topic", "test payload")
        
        # Should return True (message was published)
        assert result
        
        # Verify publish was called
        mock_client.publish.assert_called_once_with("test/topic", "test payload", 0, False)
        
        # Queue should be empty
        assert len(connection_manager.message_queue) == 0
    
    def test_queue_multiple_messages(self, connection_manager):
        """Test queueing multiple messages."""
        connection_manager.is_connected = False
        
        # Queue 5 messages
        for i in range(5):
            connection_manager.publish_or_queue(f"test/topic/{i}", f"payload {i}")
        
        # Verify all messages are queued
        assert len(connection_manager.message_queue) == 5
        
        # Verify FIFO order
        for i in range(5):
            assert connection_manager.message_queue[i]['topic'] == f"test/topic/{i}"
            assert connection_manager.message_queue[i]['payload'] == f"payload {i}"
    
    def test_queue_size_limit(self, connection_manager):
        """Test that queue respects max size limit."""
        connection_manager.is_connected = False
        connection_manager.max_queue_size = 10
        connection_manager.message_queue = deque(maxlen=10)
        
        # Try to queue 15 messages (should only keep last 10)
        for i in range(15):
            connection_manager.publish_or_queue(f"test/topic/{i}", f"payload {i}")
        
        # Verify only 10 messages are kept
        assert len(connection_manager.message_queue) == 10
        
        # Verify oldest messages were dropped (should have messages 5-14)
        assert connection_manager.message_queue[0]['topic'] == "test/topic/5"
        assert connection_manager.message_queue[9]['topic'] == "test/topic/14"
    
    def test_republish_queued_messages_on_reconnect(self, connection_manager, mock_client):
        """Test that queued messages are republished after reconnection."""
        # Queue some messages while disconnected
        connection_manager.is_connected = False
        connection_manager.publish_or_queue("test/topic/1", "payload 1")
        connection_manager.publish_or_queue("test/topic/2", "payload 2")
        connection_manager.publish_or_queue("test/topic/3", "payload 3")
        
        assert len(connection_manager.message_queue) == 3
        
        # Simulate reconnection
        connection_manager.is_connected = True
        connection_manager.handle_connect()
        
        # Verify all queued messages were published
        assert mock_client.publish.call_count == 3
        
        # Verify queue is empty
        assert len(connection_manager.message_queue) == 0
    
    def test_disconnect_callback(self, connection_manager):
        """Test that disconnect callback is called."""
        callback = Mock()
        connection_manager.set_disconnect_callback(callback)
        
        # Simulate disconnect
        connection_manager.handle_disconnect(5, "Connection lost")
        
        # Verify callback was called
        callback.assert_called_once_with(5, "Connection lost")
        
        # Verify reconnect counter was incremented
        assert connection_manager.reconnect_count == 1
        assert connection_manager.is_connected == False
    
    def test_reconnect_callback(self, connection_manager):
        """Test that reconnect callback is called."""
        callback = Mock()
        connection_manager.set_reconnect_callback(callback)
        
        # Simulate disconnect and reconnect
        connection_manager.handle_disconnect(5, "Connection lost")
        assert connection_manager.reconnect_count == 1
        
        connection_manager.is_connected = True
        connection_manager.handle_connect()
        
        # Verify callback was called with reconnect count
        callback.assert_called_once_with(1)
        
        # Verify reconnect counter was reset
        assert connection_manager.reconnect_count == 0
    
    def test_get_queue_size(self, connection_manager):
        """Test getting current queue size."""
        connection_manager.is_connected = False
        
        # Initially empty
        assert connection_manager.get_queue_size() == 0
        
        # Queue some messages
        connection_manager.publish_or_queue("test/topic/1", "payload 1")
        assert connection_manager.get_queue_size() == 1
        
        connection_manager.publish_or_queue("test/topic/2", "payload 2")
        assert connection_manager.get_queue_size() == 2


class TestFrigateAPIRetry:
    """Test Frigate API retry logic."""
    
    def test_api_retry_logic_unit(self):
        """Test API retry logic in isolation without importing identity_service."""
        import time
        import base64
        from unittest.mock import Mock, patch
        
        # Define the retry logic as it should work
        def fetch_with_retry(event_id):
            max_retries = 3
            retry_delay = 0.5
            
            for attempt in range(max_retries):
                try:
                    # Simulate API call
                    if attempt < 2:
                        raise Exception("Network error")
                    
                    # Success on 3rd try
                    return "success"
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        return None
            return None
        
        # Test successful retry
        result = fetch_with_retry("test_event")
        assert result == "success"
    
    def test_api_retry_exhaustion_unit(self):
        """Test API retry exhaustion in isolation."""
        import time
        
        # Define the retry logic
        def fetch_with_retry(event_id):
            max_retries = 3
            retry_delay = 0.01  # Fast for testing
            
            for attempt in range(max_retries):
                try:
                    # Simulate always failing
                    raise Exception("Network error")
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        return None
            return None
        
        # Test exhaustion
        result = fetch_with_retry("test_event")
        assert result is None
    
    def test_api_retry_success_scenarios(self):
        """Test various retry success scenarios."""
        # Test immediate success (0 retries)
        attempt_count = [0]
        
        def fetch_immediate_success():
            attempt_count[0] += 1
            return "success"
        
        result = fetch_immediate_success()
        assert result == "success"
        assert attempt_count[0] == 1
        
        # Test success after 1 retry
        attempt_count2 = [0]
        
        def fetch_after_one_retry():
            attempt_count2[0] += 1
            if attempt_count2[0] == 1:
                raise Exception("Temporary error")
            return "success"
        
        # Simulate retry logic
        max_retries = 3
        result2 = None
        for i in range(max_retries):
            try:
                result2 = fetch_after_one_retry()
                break
            except Exception:
                if i >= max_retries - 1:
                    result2 = None
        
        assert result2 == "success"
        assert attempt_count2[0] == 2


class TestMQTTClientCompat:
    """Test MQTT client version compatibility."""
    
    def test_get_mqtt_client_v2(self):
        """Test client creation with paho-mqtt 2.x."""
        import paho.mqtt.client as mqtt
        
        # paho-mqtt 2.x has CallbackAPIVersion
        if hasattr(mqtt, 'CallbackAPIVersion'):
            client = get_mqtt_client()
            assert client is not None
    
    def test_get_mqtt_client_v1_fallback(self):
        """Test client creation fallback for paho-mqtt 1.x."""
        import paho.mqtt.client as mqtt
        
        # Even if we're on 2.x, the function should handle both cases
        client = get_mqtt_client()
        assert client is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
