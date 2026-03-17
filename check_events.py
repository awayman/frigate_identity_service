"""Quick diagnostic to check what events are available from Frigate."""

import os
import requests
from urllib3.exceptions import InsecureRequestWarning

# Suppress SSL warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

FRIGATE_HOST = os.getenv("FRIGATE_HOST", "http://10.0.0.100:5000")


def fetch_recent_events(limit=50):
    """Fetch recent events from Frigate API."""
    try:
        url = f"{FRIGATE_HOST}/api/events/explore"
        params = {
            "limit": limit,
            "has_snapshot": "true",
            "type": "person",
            "include_thumbnails": "false",
        }
        response = requests.get(url, params=params, timeout=10, verify=False)
        response.raise_for_status()
        events = response.json()
        return events
    except Exception as e:
        print(f"Error fetching events: {e}")
        return []


def main():
    print(f"Checking events from {FRIGATE_HOST}...")
    print("=" * 80)

    events = fetch_recent_events(limit=50)

    if not events:
        print("No events found!")
        return

    print(f"Found {len(events)} total events\n")

    # Filter by camera
    non_zoom = [e for e in events if e.get("camera", "").lower() != "zoom"]
    print(f"Non-zoom camera events: {len(non_zoom)}\n")

    # Categorize by sub_label
    with_facial = []
    without_facial = []

    for event in non_zoom:
        sub_label = event.get("sub_label")
        if sub_label:
            with_facial.append(event)
        else:
            without_facial.append(event)

    print(f"Events WITH facial recognition (sub_label): {len(with_facial)}")
    print(f"Events WITHOUT facial recognition: {len(without_facial)}")
    print("=" * 80)

    if with_facial:
        print("\nSample events WITH facial recognition:")
        print("-" * 80)
        for i, event in enumerate(with_facial[:5]):
            event_id = event.get("id", "unknown")
            camera = event.get("camera", "unknown")
            sub_label = event.get("sub_label")

            # Parse sub_label
            if isinstance(sub_label, list) and len(sub_label) >= 2:
                name = sub_label[0]
                confidence = sub_label[1]
                print(f"{i + 1}. Event: {event_id}")
                print(f"   Camera: {camera}")
                print(f"   Person: {name} (confidence: {confidence})")
            else:
                print(f"{i + 1}. Event: {event_id}")
                print(f"   Camera: {camera}")
                print(f"   sub_label: {sub_label}")
            print()

    if without_facial:
        print("\nSample events WITHOUT facial recognition:")
        print("-" * 80)
        for i, event in enumerate(without_facial[:5]):
            event_id = event.get("id", "unknown")
            camera = event.get("camera", "unknown")
            print(f"{i + 1}. Event: {event_id}")
            print(f"   Camera: {camera}")
            print("   sub_label: None")
            print()

    print("=" * 80)
    print("\nSummary:")
    print(f"  Total events: {len(events)}")
    print(f"  Non-zoom events: {len(non_zoom)}")
    print(f"  With facial recognition: {len(with_facial)}")
    print(f"  Without facial recognition: {len(without_facial)}")


if __name__ == "__main__":
    main()
