#!/usr/bin/env python3
"""
Test script for Frigate Identity Service
Validates MQTT connectivity, Frigate API access, and publishes test events
"""
import os
import sys
import json
import time
import paho.mqtt.client as mqtt
import requests
from datetime import datetime

from mqtt_utils import get_mqtt_client

# Load environment variables
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
FRIGATE_HOST = os.getenv("FRIGATE_HOST", "http://localhost:5000")

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_fail(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def test_mqtt_connection():
    """Test MQTT broker connectivity"""
    print_header("Testing MQTT Connection")
    
    try:
        client = get_mqtt_client()
        
        connected = False
        def on_connect(client, userdata, flags, rc, properties=None):
            nonlocal connected
            connected = True
        
        client.on_connect = on_connect
        client.connect(MQTT_BROKER, MQTT_PORT, 10)
        client.loop_start()
        
        # Wait for connection
        timeout = 5
        start = time.time()
        while not connected and (time.time() - start) < timeout:
            time.sleep(0.1)
        
        client.loop_stop()
        client.disconnect()
        
        if connected:
            print_success(f"Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
            return True
        else:
            print_fail(f"Failed to connect to MQTT broker (timeout)")
            return False
            
    except Exception as e:
        print_fail(f"MQTT connection error: {e}")
        return False

def test_frigate_api():
    """Test Frigate API accessibility"""
    print_header("Testing Frigate API")
    
    try:
        response = requests.get(f"{FRIGATE_HOST}/api/config", timeout=5)
        if response.status_code == 200:
            config = response.json()
            cameras = list(config.get('cameras', {}).keys())
            print_success(f"Frigate API accessible at {FRIGATE_HOST}")
            print_info(f"Cameras configured: {', '.join(cameras)}")
            return True
        else:
            print_fail(f"Frigate API returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_fail(f"Cannot reach Frigate API: {e}")
        return False

def test_mqtt_subscriptions():
    """Test MQTT topic subscriptions"""
    print_header("Testing MQTT Subscriptions")
    
    received_topics = []
    
    def on_message(client, userdata, msg):
        received_topics.append(msg.topic)
    
    def on_connect(client, userdata, flags, rc, properties=None):
        client.subscribe("identity/#")
        client.subscribe("frigate/#")
    
    try:
        client = get_mqtt_client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        
        print_info("Subscribed to identity/# and frigate/#")
        print_info("Listening for 5 seconds...")
        
        time.sleep(5)
        
        client.loop_stop()
        client.disconnect()
        
        if received_topics:
            print_success(f"Received {len(received_topics)} MQTT messages")
            for topic in list(set(received_topics))[:10]:  # Show unique topics
                print_info(f"  - {topic}")
            return True
        else:
            print_fail("No MQTT messages received (are cameras active?)")
            return False
            
    except Exception as e:
        print_fail(f"MQTT subscription error: {e}")
        return False

def publish_test_event():
    """Publish a test person detection event"""
    print_header("Publishing Test Event")
    
    try:
        client = get_mqtt_client()
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        
        test_event = {
            "person_id": "TestPerson",
            "camera": "test_camera",
            "confidence": 0.95,
            "source": "test_script",
            "frigate_zones": ["test_zone"],
            "event_id": f"test-{int(time.time())}",
            "timestamp": int(time.time() * 1000),
            "snapshot_url": f"{FRIGATE_HOST}/api/test/snapshot.jpg"
        }
        
        client.publish("identity/person/TestPerson", json.dumps(test_event))
        print_success("Published test event to identity/person/TestPerson")
        print_info(f"Event: {json.dumps(test_event, indent=2)}")
        
        client.disconnect()
        return True
        
    except Exception as e:
        print_fail(f"Failed to publish test event: {e}")
        return False

def check_environment():
    """Check environment configuration"""
    print_header("Checking Environment Configuration")
    
    required_vars = {
        "MQTT_BROKER": MQTT_BROKER,
        "MQTT_PORT": MQTT_PORT,
        "FRIGATE_HOST": FRIGATE_HOST
    }
    
    all_set = True
    for var, value in required_vars.items():
        if value:
            print_success(f"{var} = {value}")
        else:
            print_fail(f"{var} is not set")
            all_set = False
    
    return all_set

def main():
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     Frigate Identity Service - Test & Validation          ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    
    results = {}
    
    # Run tests
    results['Environment'] = check_environment()
    results['MQTT Connection'] = test_mqtt_connection()
    results['Frigate API'] = test_frigate_api()
    results['MQTT Subscriptions'] = test_mqtt_subscriptions()
    results['Test Event Publish'] = publish_test_event()
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        color = Colors.OKGREEN if result else Colors.FAIL
        print(f"{color}{status:6} {Colors.ENDC} {test_name}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.ENDC}\n")
    
    if passed == total:
        print_success("All tests passed! System is ready.")
        return 0
    else:
        print_fail(f"{total - passed} test(s) failed. Check configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
