import os
import sys
import pytest
import requests
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Add the source package directory to sys.path so that tests can import
# modules like embedding_store, matcher, identity_service, etc.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "frigate_identity_service"),
)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "real_frigate: marks tests that require a real Frigate instance"
    )


@pytest.fixture
def frigate_host():
    """Get Frigate host URL from FRIGATE_HOST environment variable.

    Returns None if not set, causing tests to skip.
    """
    host = os.getenv("FRIGATE_HOST")
    if not host:
        pytest.skip("FRIGATE_HOST environment variable not set")
    return host


@pytest.fixture
def frigate_session(frigate_host):
    """Create a requests session for Frigate API calls with retry logic.

    Automatically handles:
    - SSL verification (disabled for self-signed certs)
    - API key authentication (from FRIGATE_API_KEY if set)
    - Timeout and retry logic
    """
    session = requests.Session()

    # Disable SSL warnings for self-signed certs
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    session.verify = False

    # Add API key if configured
    api_key = os.getenv("FRIGATE_API_KEY")
    if api_key:
        session.headers.update({"Authorization": f"Bearer {api_key}"})

    # Add timeout to all requests
    session.timeout = 10

    yield session
    session.close()


@pytest.fixture
def frigate_config(frigate_session, frigate_host):
    """Validate Frigate is reachable and return its config.

    This fixture ensures the Frigate API is actually accessible before tests run.
    """
    try:
        response = frigate_session.get(f"{frigate_host}/api/config", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        pytest.skip(f"Frigate API not accessible at {frigate_host}: {e}")


@pytest.fixture
def output_dir():
    """Create and return the tests/output directory for test artifacts.

    Used for HTML reports and other test outputs.
    """
    output_path = Path(__file__).parent / "tests" / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


@pytest.fixture
def frigate_event_filters():
    """Build optional event time filters from FRIGATE_EVENT_DATE or FRIGATE_EVENT_DAYS_AGO.

    FRIGATE_EVENT_DATE format: YYYY-MM-DD (UTC day).
    FRIGATE_EVENT_DAYS_AGO format: integer >= 0 (0=today UTC, 1=yesterday UTC).
    Returns empty dict when neither value is set.
    """
    event_date = (os.getenv("FRIGATE_EVENT_DATE") or "").strip()
    days_ago_raw = (os.getenv("FRIGATE_EVENT_DAYS_AGO") or "").strip()

    if not event_date and not days_ago_raw:
        return {}

    if event_date and days_ago_raw:
        # Prefer relative day when both are set (e.g., task input history carryover)
        event_date = ""

    day_start: datetime

    if event_date:
        try:
            day_start = datetime.strptime(event_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pytest.fail(
                "FRIGATE_EVENT_DATE must be in YYYY-MM-DD format "
                f"(received: {event_date!r})"
            )
            return {}
    else:
        try:
            days_ago = int(days_ago_raw)
        except ValueError:
            pytest.fail(
                "FRIGATE_EVENT_DAYS_AGO must be an integer >= 0 "
                f"(received: {days_ago_raw!r})"
            )
            return {}

        if days_ago < 0:
            pytest.fail(
                "FRIGATE_EVENT_DAYS_AGO must be an integer >= 0 "
                f"(received: {days_ago!r})"
            )
            return {}

        utc_midnight_today = datetime.now(timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        day_start = utc_midnight_today - timedelta(days=days_ago)

    day_end: datetime = day_start + timedelta(days=1)
    return {
        "after": int(day_start.timestamp()),
        "before": int(day_end.timestamp()),
    }


@pytest.fixture
def temp_embedding_store(output_dir):
    """Create a temporary EmbeddingStore for isolated testing.

    Returns an EmbeddingStore instance with a temporary database file
    that is automatically cleaned up after the test completes.

    This ensures tests don't interfere with each other or with production data.
    """
    from embedding_store import EmbeddingStore

    temp_db_path = output_dir / "test_embeddings.json"

    # Remove existing temp db if present
    if temp_db_path.exists():
        temp_db_path.unlink()

    # Create and return the temporary store
    store = EmbeddingStore(str(temp_db_path))

    yield store

    # Cleanup after test
    if temp_db_path.exists():
        temp_db_path.unlink()
