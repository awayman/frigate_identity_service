"""
MQTT utilities for backward compatibility with paho-mqtt 1.x and 2.x
"""
import paho.mqtt.client as mqtt


def get_mqtt_client():
    """
    Create MQTT client with version-appropriate initialization.
    Supports both paho-mqtt 1.x and 2.x for backward compatibility.
    
    Returns:
        mqtt.Client: MQTT client instance
    """
    try:
        # Try paho-mqtt 2.x API first
        if hasattr(mqtt, 'CallbackAPIVersion'):
            return mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        else:
            # Fall back to paho-mqtt 1.x API
            return mqtt.Client()
    except Exception as e:
        print(f"Warning: Error creating MQTT client: {e}")
        # Final fallback
        return mqtt.Client()
