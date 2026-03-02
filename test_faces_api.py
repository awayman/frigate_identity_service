"""Check for Frigate faces API endpoint."""
import os
import requests
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

FRIGATE_HOST = os.getenv("FRIGATE_HOST", "http://10.0.0.100:5000")

def check_faces_api():
    """Test various faces-related API endpoints."""
    
    print(f"Checking faces API at {FRIGATE_HOST}")
    print("=" * 80)
    
    # Test 1: /api/faces
    print("\n1. Testing /api/faces endpoint")
    print("-" * 80)
    try:
        url = f"{FRIGATE_HOST}/api/faces"
        response = requests.get(url, timeout=10, verify=False)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: /api/events with sub_label parameter
    print("\n2. Testing /api/events with sub_label parameter")
    print("-" * 80)
    try:
        url = f"{FRIGATE_HOST}/api/events"
        params = {
            "limit": 50,
            "has_snapshot": 1,
            "label": "person",
            "sub_label": "true",  # Try filtering for sub_label
        }
        response = requests.get(url, params=params, timeout=10, verify=False)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            events = response.json()
            print(f"Total events: {len(events)}")
            with_sub = [e for e in events if e.get("sub_label")]
            print(f"With sub_label: {len(with_sub)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Check review items (might contain face data)
    print("\n3. Testing /api/review endpoint")
    print("-" * 80)
    try:
        url = f"{FRIGATE_HOST}/api/review"
        response = requests.get(url, timeout=10, verify=False)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Review items: {len(data) if isinstance(data, list) else 'Not a list'}")
            if isinstance(data, list) and data:
                print(f"Sample item keys: {list(data[0].keys())}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Check for recognized faces
    print("\n4. Testing /api/events with after parameter (last 24h)")
    print("-" * 80)
    try:
        import time
        after_timestamp = time.time() - (24 * 3600)  # Last 24 hours
        
        url = f"{FRIGATE_HOST}/api/events"
        params = {
            "limit": 100,
            "has_snapshot": 1,
            "label": "person",
            "after": int(after_timestamp),
        }
        response = requests.get(url, params=params, timeout=10, verify=False)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            events = response.json()
            print(f"Events in last 24h: {len(events)}")
            with_sub = [e for e in events if e.get("sub_label")]
            print(f"With sub_label: {len(with_sub)}")
            
            # Check for any interesting fields
            if events:
                sample = events[0]
                print(f"\nSample event fields:")
                for key in sorted(sample.keys()):
                    print(f"  {key}: {sample[key]}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Check recordings API
    print("\n5. Checking all API endpoints from /api/")
    print("-" * 80)
    try:
        # Try to see what endpoints are available
        common_endpoints = [
            "config",
            "events",
            "events/explore", 
            "faces",
            "labels",
            "sub_labels",
            "stats",
            "review",
            "timeline",
        ]
        
        for endpoint in common_endpoints:
            url = f"{FRIGATE_HOST}/api/{endpoint}"
            try:
                response = requests.get(url, timeout=5, verify=False)
                if response.status_code == 200:
                    print(f"  ✓ {endpoint} (200 OK)")
                else:
                    print(f"  ✗ {endpoint} ({response.status_code})")
            except:
                print(f"  ✗ {endpoint} (failed)")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_faces_api()
