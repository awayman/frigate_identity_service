#!/usr/bin/env python3
"""
Integration Test Runner for Frigate Identity Service.

Runs full end-to-end tests with:
- MQTT message publishing
- Identity service processing
- Home Assistant sensor state verification
"""

import os
import sys
import json
import time
import subprocess
import requests
import paho.mqtt.client as mqtt
import signal
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class IntegrationTestRunner:
    def __init__(self):
        self.config = {}
        self.service_process = None
        self.mqtt_messages = defaultdict(list)
        self.test_results = {}
        self.mqtt_client = None
        
    def print_header(self, text):
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")
    
    def print_success(self, text):
        print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")
    
    def print_fail(self, text):
        print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")
    
    def print_info(self, text):
        print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")
    
    def print_warning(self, text):
        print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")
    
    def load_config(self):
        """Load configuration from .env.integration-test."""
        env_file = Path(__file__).parent / '.env.integration-test'
        
        if not env_file.exists():
            self.print_fail(".env.integration-test not found")
            self.print_info("Run: python setup_integration_tests.py")
            return False
        
        self.print_info(f"Loading configuration from {env_file.name}")
        
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            self.config[key] = value
            
            # Set environment variables
            for key, value in self.config.items():
                os.environ[key] = value
            
            self.print_success("Configuration loaded")
            return True
            
        except Exception as e:
            self.print_fail(f"Failed to load configuration: {e}")
            return False
    
    def verify_prerequisites(self):
        """Verify all prerequisites are met."""
        self.print_header("Verifying Prerequisites")
        
        all_ok = True
        
        # Check MQTT
        mqtt_broker = self.config.get('MQTT_BROKER', 'localhost')
        mqtt_port = int(self.config.get('MQTT_PORT', 1883))
        
        self.print_info(f"Testing MQTT connection to {mqtt_broker}:{mqtt_port}...")
        if not self._test_mqtt_connection(mqtt_broker, mqtt_port):
            all_ok = False
        else:
            self.print_success("MQTT broker is accessible")
        
        # Check Frigate API
        frigate_url = self.config.get('FRIGATE_HOST', 'http://localhost:5000')
        self.print_info(f"Testing Frigate API at {frigate_url}...")
        if not self._test_frigate_api(frigate_url):
            all_ok = False
        else:
            self.print_success("Frigate API is accessible")
        
        # Check Home Assistant
        ha_url = self.config.get('HOMEASSISTANT_URL', 'http://homeassistant.local:8123')
        ha_token = self.config.get('HOMEASSISTANT_TOKEN')
        self.print_info(f"Testing Home Assistant at {ha_url}...")
        if not self._test_homeassistant(ha_url, ha_token):
            all_ok = False
        else:
            self.print_success("Home Assistant is accessible")
        
        return all_ok
    
    def _test_mqtt_connection(self, broker, port):
        """Test MQTT connection."""
        try:
            # Try to create client with version 2.0 API first
            try:
                client = mqtt.Client(mqtt.CallbackAPIVersion.V1)
            except (AttributeError, TypeError):
                # Fall back to older API if version 2.0 is not available
                client = mqtt.Client()
            
            username = self.config.get('MQTT_USERNAME')
            password = self.config.get('MQTT_PASSWORD')
            
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
            
            client.connect(broker, port, 5)
            client.loop_start()
            
            # Wait for connection
            timeout = 5
            start = time.time()
            while not connected[0] and (time.time() - start) < timeout:
                time.sleep(0.1)
            
            client.loop_stop()
            client.disconnect()
            
            return connected[0]
        except Exception as e:
            self.print_warning(f"MQTT test failed: {e}")
            return False
    
    def _test_frigate_api(self, frigate_url):
        """Test Frigate API."""
        try:
            api_url = frigate_url.rstrip('/') + '/api/config'
            response = requests.get(api_url, timeout=5, verify=False)
            return response.status_code == 200
        except Exception as e:
            self.print_warning(f"Frigate test failed: {e}")
            return False
    
    def _test_homeassistant(self, ha_url, token):
        """Test Home Assistant."""
        try:
            api_url = ha_url.rstrip('/') + '/api/'
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
            response = requests.get(api_url, headers=headers, timeout=5, verify=False)
            return response.status_code == 200
        except Exception as e:
            self.print_warning(f"Home Assistant test failed: {e}")
            return False
    
    def start_identity_service(self):
        """Start the identity service as a subprocess."""
        self.print_header("Starting Identity Service")
        
        try:
            env = os.environ.copy()
            
            # Start the service
            self.service_process = subprocess.Popen(
                [sys.executable, 'identity_service.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1
            )
            
            self.print_info("Waiting for service to initialize...")
            
            # Wait for service to be ready
            ready = False
            timeout = 30
            start = time.time()
            
            while (time.time() - start) < timeout:
                if self.service_process.poll() is not None:
                    # Process ended unexpectedly
                    self.print_fail("Service process ended unexpectedly")
                    return False
                
                # Read available output
                try:
                    output = self.service_process.stdout.readline()
                    if output:
                        print(f"  {output.rstrip()}")
                        if 'Connected to MQTT' in output or 'ReID system ready' in output:
                            ready = True
                            time.sleep(1)  # Give it a moment to fully initialize
                            break
                except:
                    pass
                
                time.sleep(0.1)
            
            if not ready:
                self.print_warning("Service initialization timeout, but continuing...")
            
            self.print_success("Identity service started")
            return True
            
        except Exception as e:
            self.print_fail(f"Failed to start identity service: {e}")
            return False
    
    def run_test_scenario(self, scenario_name, scenario_type):
        """Run a test scenario using test_mqtt_publisher.py."""
        self.print_header(f"Running Test Scenario: {scenario_name}")
        
        # Clear previous messages
        self.mqtt_messages.clear()
        
        # Start MQTT listener
        listener_ready = [False]
        
        def on_message(client, userdata, msg):
            """Handle MQTT messages."""
            try:
                payload = json.loads(msg.payload.decode('utf-8'))
                self.mqtt_messages[msg.topic].append({
                    'payload': payload,
                    'timestamp': datetime.now().isoformat()
                })
                self.print_info(f"Received on {msg.topic}: person_id={payload.get('person_id', 'N/A')}")
            except:
                pass
        
        def on_connect(client, userdata, flags, rc, properties=None):
            client.subscribe("identity/person/#")
            listener_ready[0] = True
        
        def on_connect_old(client, userdata, flags, rc):
            client.subscribe("identity/person/#")
            listener_ready[0] = True
        
        try:
            # Create MQTT client for listening
            try:
                listen_client = mqtt.Client(mqtt.CallbackAPIVersion.V1)
            except (AttributeError, TypeError):
                listen_client = mqtt.Client()
            
            username = self.config.get('MQTT_USERNAME')
            password = self.config.get('MQTT_PASSWORD')
            
            if username and password:
                listen_client.username_pw_set(username, password)
            
            # Try to set the appropriate callback
            try:
                listen_client.on_connect = on_connect
            except:
                listen_client.on_connect = on_connect_old
            
            listen_client.on_message = on_message
            
            mqtt_broker = self.config.get('MQTT_BROKER', 'localhost')
            mqtt_port = int(self.config.get('MQTT_PORT', 1883))
            
            listen_client.connect(mqtt_broker, mqtt_port, 60)
            listen_client.loop_start()
            
            # Wait for listener to be ready
            timeout = 5
            start = time.time()
            while not listener_ready[0] and (time.time() - start) < timeout:
                time.sleep(0.1)
            
            # Run the test scenario
            self.print_info(f"Running test scenario: {scenario_type}")
            
            result = subprocess.run(
                [sys.executable, 'test_mqtt_publisher.py',
                 '--broker', mqtt_broker,
                 '--port', str(mqtt_port),
                 '--test', scenario_type],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Print publisher output
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        print(f"  {line}")
            
            if result.returncode != 0:
                self.print_warning(f"Publisher returned code {result.returncode}")
                if result.stderr:
                    print(f"  Error: {result.stderr}")
            
            # Wait for messages to arrive
            self.print_info("Waiting for identity service to process events...")
            time.sleep(5)
            
            listen_client.loop_stop()
            listen_client.disconnect()
            
            # Check results
            return self._verify_test_results(scenario_name, scenario_type)
            
        except subprocess.TimeoutExpired:
            self.print_fail("Test scenario timeout")
            return False
        except Exception as e:
            self.print_fail(f"Test scenario failed: {e}")
            return False
    
    def _verify_test_results(self, scenario_name, scenario_type):
        """Verify test results."""
        if not self.mqtt_messages:
            self.print_fail(f"No MQTT messages received for {scenario_name}")
            return False
        
        # Check for expected topics
        received_topics = list(self.mqtt_messages.keys())
        self.print_info(f"Received messages on {len(received_topics)} topic(s)")
        
        for topic in received_topics:
            messages = self.mqtt_messages[topic]
            self.print_info(f"  {topic}: {len(messages)} message(s)")
            for msg in messages:
                person_id = msg['payload'].get('person_id')
                camera = msg['payload'].get('camera')
                confidence = msg['payload'].get('confidence')
                self.print_info(f"    - person_id={person_id}, camera={camera}, confidence={confidence}")
        
        # Verify Home Assistant sensors
        ha_last_person = self._get_ha_sensor_state('sensor.frigate_identity_last_person')
        ha_all_persons = self._get_ha_sensor_state('sensor.frigate_identity_all_persons')
        
        self.print_info("Home Assistant sensor states:")
        if ha_last_person:
            self.print_info(f"  Last Person: {ha_last_person.get('state')}")
        if ha_all_persons:
            self.print_info(f"  All Persons Count: {ha_all_persons.get('state')}")
        
        success = True
        if scenario_type == 'basic':
            success = self._verify_basic_scenario()
        elif scenario_type == 'multiface':
            success = self._verify_multiface_scenario()
        elif scenario_type == 'reid':
            success = self._verify_reid_scenario()
        
        if success:
            self.print_success(f"{scenario_name} PASSED")
        else:
            self.print_warning(f"{scenario_name} may have issues (check output above)")
        
        return success
    
    def _verify_basic_scenario(self):
        """Verify basic scenario results."""
        # Basic scenario should have messages about alice
        for topic, messages in self.mqtt_messages.items():
            for msg in messages:
                if msg['payload'].get('person_id') == 'alice':
                    return True
        return False
    
    def _verify_multiface_scenario(self):
        """Verify multiface scenario results."""
        # Should have messages about both alice and bob
        people = set()
        for topic, messages in self.mqtt_messages.items():
            for msg in messages:
                person_id = msg['payload'].get('person_id')
                if person_id:
                    people.add(person_id)
        
        return len(people) >= 2
    
    def _verify_reid_scenario(self):
        """Verify re-id scenario results."""
        # Should have messages about alice
        for topic, messages in self.mqtt_messages.items():
            for msg in messages:
                if msg['payload'].get('person_id') == 'alice':
                    return True
        return False
    
    def _get_ha_sensor_state(self, entity_id):
        """Get Home Assistant sensor state."""
        try:
            ha_url = self.config.get('HOMEASSISTANT_URL', 'http://homeassistant.local:8123')
            ha_token = self.config.get('HOMEASSISTANT_TOKEN')
            
            api_url = f"{ha_url.rstrip('/')}/api/states/{entity_id}"
            headers = {
                'Authorization': f'Bearer {ha_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(api_url, headers=headers, timeout=5, verify=False)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def cleanup(self):
        """Clean up resources."""
        if self.service_process and self.service_process.poll() is None:
            self.print_info("Stopping identity service...")
            self.service_process.terminate()
            try:
                self.service_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.service_process.kill()
    
    def print_summary(self):
        """Print test summary."""
        self.print_header("Test Summary")
        
        total = len(self.test_results)
        passed = sum(1 for v in self.test_results.values() if v)
        
        for test_name, result in self.test_results.items():
            status = "PASS" if result else "FAIL"
            color = Colors.OKGREEN if result else Colors.FAIL
            print(f"{color}{status:6}{Colors.ENDC} {test_name}")
        
        print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.ENDC}\n")
        
        if passed == total:
            self.print_success("All tests passed!")
            return 0
        else:
            self.print_warning(f"{total - passed} test(s) failed or warning(s)")
            return 1
    
    def run_all_tests(self):
        """Run all integration tests."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}")
        print("╔════════════════════════════════════════════════════════════════════╗")
        print("║   Frigate Identity Service - Integration Test Suite                ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        print(f"{Colors.ENDC}\n")
        
        try:
            # Load configuration
            if not self.load_config():
                return 1
            
            # Verify prerequisites
            if not self.verify_prerequisites():
                self.print_warning("Some prerequisites failed, but continuing...")
            
            # Start service
            if not self.start_identity_service():
                return 1
            
            # Run test scenarios
            test_scenarios = [
                ("Basic Matching", "basic"),
                ("Multiple People", "multiface"),
                ("Re-ID Accuracy", "reid")
            ]
            
            for scenario_name, scenario_type in test_scenarios:
                result = self.run_test_scenario(scenario_name, scenario_type)
                self.test_results[scenario_name] = result
                time.sleep(2)  # Pause between tests
            
            # Print summary
            return self.print_summary()
            
        finally:
            self.cleanup()


def main():
    runner = IntegrationTestRunner()
    return runner.run_all_tests()


if __name__ == "__main__":
    # Ignore SIGINT for clean shutdown
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sys.exit(main())
