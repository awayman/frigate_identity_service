#!/usr/bin/env python3
"""
Setup wizard for Frigate Identity Service integration tests.

This script guides you through configuring:
- MQTT broker connection
- Home Assistant connection
- Frigate API connection

Validates all connections and saves config to .env.integration-test
"""

import os
import sys
import json
import time
import requests
import urllib3
import paho.mqtt.client as mqtt
from pathlib import Path

# Suppress SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_fail(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def input_with_default(prompt, default):
    """Get user input with a default value."""
    display_prompt = f"{prompt} [{default}]: "
    user_input = input(display_prompt).strip()
    return user_input if user_input else default


def test_mqtt_connection(broker, port, username=None, password=None):
    """Test MQTT broker connectivity."""
    print_info(f"Testing MQTT connection to {broker}:{port}...")
    
    try:
        from mqtt_utils import get_mqtt_client
        client = get_mqtt_client()
        
        if username and password:
            client.username_pw_set(username, password)
        
        connected = [False]
        
        def on_connect(client, userdata, flags, rc, properties=None):
            connected[0] = True
        
        def on_connect_old(client, userdata, flags, rc):
            connected[0] = True
        
        # Try to set the appropriate callback
        try:
            client.on_connect = on_connect
        except:
            client.on_connect = on_connect_old
        
        client.connect(broker, port, 10)
        client.loop_start()
        
        # Wait for connection
        timeout = 5
        start = time.time()
        while not connected[0] and (time.time() - start) < timeout:
            time.sleep(0.1)
        
        client.loop_stop()
        client.disconnect()
        
        if connected[0]:
            print_success(f"MQTT broker is reachable ({broker}:{port})")
            return True
        else:
            print_fail("MQTT connection timeout")
            return False
            
    except Exception as e:
        print_fail(f"MQTT connection failed: {e}")
        return False


def test_frigate_api(frigate_url, api_key=None):
    """Test Frigate API accessibility."""
    print_info(f"Testing Frigate API at {frigate_url}...")
    
    try:
        # Handle trailing slash
        api_url = frigate_url.rstrip('/') + '/api/config'
        headers = {}
        if api_key:
            headers['X-API-Key'] = api_key
        
        response = requests.get(api_url, timeout=5, verify=False, headers=headers)
        
        if response.status_code == 200:
            config = response.json()
            cameras = list(config.get('cameras', {}).keys())
            print_success(f"Frigate API is accessible")
            if cameras:
                print_info(f"Available cameras: {', '.join(cameras)}")
            return True, cameras
        elif response.status_code == 401:
            print_fail(f"Frigate API returned 401 (Unauthorized - check API key)")
            return False, []
        else:
            print_fail(f"Frigate API returned status {response.status_code}")
            return False, []
            
    except requests.exceptions.RequestException as e:
        print_fail(f"Cannot reach Frigate API: {e}")
        return False, []


def test_homeassistant(ha_url, token):
    """Test Home Assistant connectivity."""
    print_info(f"Testing Home Assistant at {ha_url}...")
    
    try:
        # Normalize URL
        if not ha_url.startswith('http'):
            ha_url = f"http://{ha_url}"
        
        api_url = ha_url.rstrip('/') + '/api/'
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(api_url, headers=headers, timeout=5, verify=False)
        
        if response.status_code == 200:
            print_success("Home Assistant is accessible and token is valid")
            return True
        elif response.status_code == 401:
            print_fail("Home Assistant token is invalid")
            return False
        else:
            print_fail(f"Home Assistant API returned status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print_fail(f"Cannot reach Home Assistant: {e}")
        return False


def get_ha_token_instructions():
    """Display instructions for getting a Home Assistant token."""
    print_header("Getting Home Assistant Token")
    print("""
To get a Home Assistant long-lived access token:

1. Open Home Assistant in your browser
2. Click your profile icon (bottom left)
3. Scroll to the bottom
4. Under "Long-Lived Access Tokens", click "Create Token"
5. Give it a name (e.g., "Frigate Identity Tests")
6. Click "Create"
7. Copy the token and paste it here

The token will start with "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9." or similar.
    """)


def setup_mqtt_config():
    """Interactive MQTT configuration."""
    print_header("MQTT Broker Configuration")
    
    print("Enter your MQTT broker details.\n")
    
    broker = input_with_default("MQTT Host", "homeassistant.local")
    port = input_with_default("MQTT Port", "1883")
    
    try:
        port = int(port)
    except ValueError:
        print_fail("Port must be a number")
        return None
    
    use_auth = input("Does your MQTT broker require authentication? (y/n) [n]: ").strip().lower() == 'y'
    
    username = None
    password = None
    if use_auth:
        username = input("MQTT Username: ").strip()
        password = input("MQTT Password: ").strip()
    
    # Test connection
    if test_mqtt_connection(broker, port, username, password):
        return {
            'broker': broker,
            'port': port,
            'username': username,
            'password': password
        }
    else:
        print_warning("Could not verify MQTT connection. Continue anyway? (y/n) [n]: ")
        if input().strip().lower() == 'y':
            return {
                'broker': broker,
                'port': port,
                'username': username,
                'password': password
            }
        else:
            return None


def setup_frigate_config():
    """Interactive Frigate configuration."""
    print_header("Frigate Configuration")
    
    print("Enter your Frigate instance URL.\n")
    print_info("(This is optional for integration testing if you don't have API access)")
    
    frigate_url = input_with_default("Frigate URL", "https://nvr.kd0rty.net")
    
    # Test connection without API key first
    success, cameras = test_frigate_api(frigate_url)
    
    # If it failed with 401, ask for API key
    api_key = None
    if not success:
        print_warning("Frigate API returned 401 (Unauthorized)")
        use_api_key = input("Do you have a Frigate API key? (y/n) [n]: ").strip().lower() == 'y'
        if use_api_key:
            api_key = input("Frigate API Key: ").strip()
            success, cameras = test_frigate_api(frigate_url, api_key)
    
    # Always allow continuing (Frigate is optional for basic testing)
    return {'url': frigate_url, 'api_key': api_key, 'cameras': cameras}


def setup_homeassistant_config():
    """Interactive Home Assistant configuration."""
    print_header("Home Assistant Configuration")
    
    print("Enter your Home Assistant instance details.\n")
    
    ha_url = input_with_default("Home Assistant URL", "homeassistant.local")
    
    print("\nDo you have a Home Assistant long-lived access token?")
    has_token = input("(y/n) [n]: ").strip().lower() == 'y'
    
    if not has_token:
        get_ha_token_instructions()
        print("\nPress Enter after you've created a token...")
        input()
    
    token = input("Home Assistant Token: ").strip()
    
    if not token:
        print_fail("Token cannot be empty")
        return None
    
    # Test connection
    if test_homeassistant(ha_url, token):
        return {'url': ha_url, 'token': token}
    else:
        print_warning("Could not verify Home Assistant connection. Continue anyway? (y/n) [n]: ")
        if input().strip().lower() == 'y':
            return {'url': ha_url, 'token': token}
        else:
            return None


def save_config(mqtt_config, frigate_config, ha_config):
    """Save configuration to .env.integration-test file."""
    env_file = Path(__file__).parent / '.env.integration-test'
    
    config_content = f"""# Frigate Identity Service - Integration Test Configuration
# Generated by setup_integration_tests.py

# MQTT Broker
MQTT_BROKER={mqtt_config['broker']}
MQTT_PORT={mqtt_config['port']}
"""
    
    if mqtt_config['username']:
        config_content += f"MQTT_USERNAME={mqtt_config['username']}\n"
        config_content += f"MQTT_PASSWORD={mqtt_config['password']}\n"
    
    config_content += f"""
# Home Assistant
HOMEASSISTANT_URL=http://{ha_config['url']}:8123 if ':' not in ha_config['url'] else http://{ha_config['url']}\n"""
    config_content += f"HOMEASSISTANT_TOKEN={ha_config['token']}\n"
    
    config_content += f"""
# Frigate
FRIGATE_HOST={frigate_config['url']}
"""
    
    if frigate_config.get('api_key'):
        config_content += f"FRIGATE_API_KEY={frigate_config['api_key']}\n"
    
    config_content += f"""
# Identity Service
EMBEDDINGS_DB_PATH=./embeddings.json
"""
    
    try:
        env_file.write_text(config_content)
        print_success(f"Configuration saved to {env_file}")
        return True
    except Exception as e:
        print_fail(f"Failed to save configuration: {e}")
        return False


def verify_all_connections(mqtt_config, frigate_config, ha_config):
    """Verify all connections one final time."""
    print_header("Verifying All Connections")
    
    all_ok = True
    
    # Test MQTT
    if not test_mqtt_connection(mqtt_config['broker'], mqtt_config['port'], 
                                 mqtt_config['username'], mqtt_config['password']):
        all_ok = False
    
    # Test Frigate
    success, _ = test_frigate_api(frigate_config['url'], frigate_config.get('api_key'))
    if not success:
        all_ok = False
    
    # Test Home Assistant
    if not test_homeassistant(ha_config['url'], ha_config['token']):
        all_ok = False
    
    return all_ok


def main():
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Frigate Identity Service - Integration Test Setup      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")
    
    # Gather configuration
    print("This wizard will configure your integration test environment.\n")
    
    mqtt_config = setup_mqtt_config()
    if not mqtt_config:
        print_fail("MQTT configuration failed. Exiting.")
        return 1
    
    frigate_config = setup_frigate_config()
    if not frigate_config:
        print_fail("Frigate configuration failed. Exiting.")
        return 1
    
    ha_config = setup_homeassistant_config()
    if not ha_config:
        print_fail("Home Assistant configuration failed. Exiting.")
        return 1
    
    # Verify all connections
    if not verify_all_connections(mqtt_config, frigate_config, ha_config):
        print_warning("\nSome connections could not be verified.")
        print("You can still save the configuration and debug later.\n")
        proceed = input("Continue and save config? (y/n) [y]: ").strip().lower() != 'n'
        if not proceed:
            return 1
    
    # Save configuration
    if not save_config(mqtt_config, frigate_config, ha_config):
        print_fail("Failed to save configuration. Exiting.")
        return 1
    
    # Success
    print_header("Setup Complete!")
    print_success("Configuration saved successfully!")
    print_info("Next steps:")
    print_info("  1. python run_integration_tests.py")
    print_info("     (From the frigate_identity_service directory)\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
