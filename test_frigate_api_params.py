"""Test different Frigate API parameters to find facial recognition events."""
import os
import requests
from urllib3.exceptions import InsecureRequestWarning

# Suppress SSL warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

FRIGATE_HOST = os.getenv("FRIGATE_HOST", "http://10.0.0.100:5000")

def test_api_params():
    """Test various API parameters to find facial recognition events."""
    
    print(f"Testing Frigate API at {FRIGATE_HOST}")
    print("=" * 80)
    
    # Test 1: Standard explore endpoint with more results
    print("\n1. Testing /api/events/explore with limit=100")
    print("-" * 80)
    try:
        url = f"{FRIGATE_HOST}/api/events/explore"
        params = {
            "limit": 100,
            "has_snapshot": "true",
            "type": "person",
            "include_thumbnails": "false",
        }
        response = requests.get(url, params=params, timeout=10, verify=False)
        response.raise_for_status()
        events = response.json()
        
        with_sub_label = [e for e in events if e.get("sub_label")]
        print(f"Total events: {len(events)}")
        print(f"Events with sub_label: {len(with_sub_label)}")
        
        if with_sub_label:
            print("\nSample events with sub_label:")
            for event in with_sub_label[:3]:
                print(f"  - Event: {event.get('id')}")
                print(f"    Camera: {event.get('camera')}")
                print(f"    sub_label: {event.get('sub_label')}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Check if there's a sub_label filter
    print("\n2. Testing with sub_label filter (if supported)")
    print("-" * 80)
    try:
        url = f"{FRIGATE_HOST}/api/events/explore"
        params = {
            "limit": 50,
            "has_snapshot": "true",
            "type": "person",
            "has_sublabel": "true",  # Try this parameter
            "include_thumbnails": "false",
        }
        response = requests.get(url, params=params, timeout=10, verify=False)
        response.raise_for_status()
        events = response.json()
        
        with_sub_label = [e for e in events if e.get("sub_label")]
        print(f"Total events: {len(events)}")
        print(f"Events with sub_label: {len(with_sub_label)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Try the standard /api/events endpoint
    print("\n3. Testing standard /api/events endpoint")
    print("-" * 80)
    try:
        url = f"{FRIGATE_HOST}/api/events"
        params = {
            "limit": 100,
            "has_snapshot": 1,
            "label": "person",
        }
        response = requests.get(url, params=params, timeout=10, verify=False)
        response.raise_for_status()
        events = response.json()
        
        with_sub_label = [e for e in events if e.get("sub_label")]
        print(f"Total events: {len(events)}")
        print(f"Events with sub_label: {len(with_sub_label)}")
        
        if with_sub_label:
            print("\nSample events with sub_label:")
            for event in with_sub_label[:3]:
                print(f"  - Event: {event.get('id')}")
                print(f"    Camera: {event.get('camera')}")
                print(f"    sub_label: {event.get('sub_label')}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Get all events without limit and check oldest
    print("\n4. Testing with very high limit to check older events")
    print("-" * 80)
    try:
        url = f"{FRIGATE_HOST}/api/events/explore"
        params = {
            "limit": 500,  # Much larger limit
            "has_snapshot": "true",
            "type": "person",
            "include_thumbnails": "false",
        }
        response = requests.get(url, params=params, timeout=15, verify=False)
        response.raise_for_status()
        events = response.json()
        
        with_sub_label = [e for e in events if e.get("sub_label")]
        print(f"Total events: {len(events)}")
        print(f"Events with sub_label: {len(with_sub_label)}")
        
        if with_sub_label:
            print(f"\nFound {len(with_sub_label)} events with facial recognition!")
            print("\nFirst 5 facial recognition events:")
            for i, event in enumerate(with_sub_label[:5], 1):
                sub_label = event.get("sub_label")
                if isinstance(sub_label, list):
                    name = sub_label[0]
                    conf = sub_label[1] if len(sub_label) > 1 else "N/A"
                else:
                    name = sub_label
                    conf = "N/A"
                print(f"\n{i}. Event: {event.get('id')}")
                print(f"   Camera: {event.get('camera')}")
                print(f"   Person: {name}")
                print(f"   Confidence: {conf}")
                print(f"   Start time: {event.get('start_time')}")
        else:
            print("\nNo facial recognition events found in dataset.")
            print("Checking sample of all events for structure...")
            if events:
                print(f"\nSample event structure:")
                sample = events[0]
                for key in ['id', 'camera', 'label', 'sub_label', 'start_time', 'has_snapshot']:
                    print(f"  {key}: {sample.get(key)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Check API config for facial recognition settings
    print("\n5. Checking Frigate config for facial recognition setup")
    print("-" * 80)
    try:
        url = f"{FRIGATE_HOST}/api/config"
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        config = response.json()
        
        # Check for face recognition configuration
        cameras = config.get("cameras", {})
        print(f"Cameras configured: {list(cameras.keys())}")
        
        # Check if any camera has face recognition enabled
        for camera_name, camera_config in cameras.items():
            objects = camera_config.get("objects", {})
            filters = objects.get("filters", {})
            person_filter = filters.get("person", {})
            
            if person_filter:
                print(f"\n{camera_name}:")
                print(f"  Person filters: {person_filter}")
        
        # Look for face recognition in detect config
        if "face_recognition" in config:
            print(f"\nFace recognition config found: {config['face_recognition']}")
        elif "objects" in config:
            if "face" in config["objects"].get("track", []):
                print("\nFace tracking enabled in global config")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api_params()
