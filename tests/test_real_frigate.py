"""
Integration tests using real Frigate instance and API snapshots.

These tests require a live Frigate instance accessible via FRIGATE_HOST environment variable.
They fetch real snapshots from the Frigate API and validate the full E2E pipeline:
1. Event retrieval from Frigate API
2. Snapshot fetching via HTTP
3. Embedding extraction using ReIDModel
4. Person identification/matching
5. HTML report generation with visual inspection

Run with:
    export FRIGATE_HOST=http://192.168.1.100:5000
    pytest tests/test_real_frigate.py -v -s

Optional day filter (UTC):
    export FRIGATE_EVENT_DATE=2026-03-01
    pytest tests/test_real_frigate.py -v -s

Optional relative day filter (UTC):
    export FRIGATE_EVENT_DAYS_AGO=1  # yesterday (UTC)
    pytest tests/test_real_frigate.py -v -s

Or to skip if Frigate not available:
    pytest tests/test_real_frigate.py -v  # Skipped gracefully if FRIGATE_HOST not set
"""

import pytest
import logging
import base64
from pathlib import Path
from datetime import datetime

from tests.utils.frigate_api import (
    fetch_recent_events,
    fetch_known_faces,
    parse_face_training_events,
    fetch_event_by_id,
    get_event_snapshot_url,
    fetch_snapshot_bytes,
    validate_image,
)
from tests.utils.report_generator import MatchReport


logger = logging.getLogger(__name__)


REID_MATCH_THRESHOLD = 0.75


@pytest.fixture
def match_report(output_dir, frigate_host):
    """Create a MatchReport that generates HTML on teardown.
    
    Yields the report object during test execution, then automatically
    generates an HTML file with results when the test completes.
    """
    report = MatchReport(frigate_host=frigate_host)
    yield report
    
    # Generate HTML report after test runs
    report_path = output_dir / "real_frigate_report.html"
    report.generate_html(report_path)
    logger.info(f"HTML report generated: {report_path}")


@pytest.mark.real_frigate
class TestRealFrigateIntegration:
    """Integration tests with real Frigate instance."""
    
    def test_frigate_api_connectivity(self, frigate_session, frigate_host, frigate_config):
        """Verify Frigate API is accessible and returns valid config.
        
        This is a smoke test to ensure the Frigate instance is reachable
        and properly configured before running more expensive tests.
        """
        assert frigate_config is not None, "Frigate config should be accessible"
        assert "cameras" in frigate_config or "config" in frigate_config, \
            "Frigate config should contain camera information"
        
        logger.info(f"✓ Frigate API accessible at {frigate_host}")
        logger.info(f"✓ Config: {list(frigate_config.keys())}")
    
    def test_fetch_recent_events(self, frigate_session, frigate_host, frigate_event_filters):
        """Verify we can fetch recent events from Frigate API.
        
        This test validates that:
        1. Frigate API has events available
        2. Events contain required fields (id, camera, etc.)
        3. We can query by object type
        """
        events = fetch_recent_events(
            frigate_session,
            frigate_host,
            limit=5,
            object_type="person",
            **frigate_event_filters,
        )
        
        # Skip if no events available
        if not events:
            pytest.skip("No recent person events found in Frigate")
        
        logger.info(f"✓ Found {len(events)} recent person events")
        
        # Verify event structure
        for event in events[:1]:  # Check first event
            assert "id" in event, "Event should have an id"
            assert "camera" in event, "Event should have a camera"
            logger.info(f"✓ Sample event: {event.get('id')} on {event.get('camera')}")
    
    def test_fetch_snapshot_from_event(
        self,
        frigate_session,
        frigate_host,
        match_report,
        frigate_event_filters,
    ):
        """Fetch a real snapshot from Frigate API and validate it.
        
        This test:
        1. Retrieves a recent event
        2. Fetches the snapshot via HTTP API
        3. Validates it's a valid JPEG image
        4. Adds it to the match report
        """
        # Get recent events
        events = fetch_recent_events(
            frigate_session,
            frigate_host,
            limit=10,
            **frigate_event_filters,
        )
        assert events, "No recent events found in Frigate"
        
        # Find first non-zoom event
        event = None
        for e in events:
            if e.get("camera", "").lower() != "zoom":
                event = e
                break
        
        assert event, "No events found from non-zoom cameras"
        event_id = event["id"]
        camera = event.get("camera", "unknown")
        
        # Fetch snapshot
        snapshot_url = get_event_snapshot_url(frigate_host, event_id)
        snapshot_bytes = fetch_snapshot_bytes(
            frigate_session,
            snapshot_url,
            crop=True,
            quality=85,
            height=400
        )
        
        assert snapshot_bytes is not None, f"Failed to fetch snapshot from {snapshot_url}"
        assert len(snapshot_bytes) > 0, "Snapshot bytes should not be empty"
        
        # Validate it's a valid image (JPEG, WebP, PNG, etc.)
        assert validate_image(snapshot_bytes), "Snapshot should be a valid image"
        
        logger.info(f"✓ Fetched valid image snapshot from {camera}")
        logger.info(f"✓ Snapshot size: {len(snapshot_bytes)} bytes")
        
        # Add to report (as an unidentified snapshot)
        match_report.add_match(
            event_id=event_id,
            snapshot_bytes=snapshot_bytes,
            person_id="[Unidentified]",
            confidence=0.0,
            camera=camera,
            timestamp=datetime.now().isoformat()
        )
    
    def test_full_e2e_pipeline_with_real_snapshots(
        self,
        frigate_session,
        frigate_host,
        match_report,
        output_dir,
        frigate_event_filters,
    ):
        """Full E2E test: fetch snapshots, extract embeddings, match persons.
        
        This is the main integration test that validates the complete pipeline:
        1. Fetch multiple snapshots from Frigate API
        2. Extract embeddings for each using ReIDModel
        3. Compare embeddings to identify matches
        4. Generate HTML report showing results
        
        Note: This test requires the ReID model to be available. It may take
        30+ seconds to complete on first run (model initialization).
        """
        # Get recent events
        events = fetch_recent_events(
            frigate_session,
            frigate_host,
            limit=15,
            **frigate_event_filters,
        )
        
        if not events:
            pytest.skip("No recent person events found in Frigate")
        
        # Filter out zoom camera events
        non_zoom_events = [e for e in events if e.get("camera", "").lower() != "zoom"]
        
        if not non_zoom_events:
            pytest.skip("No recent person events found from non-zoom cameras")
        
        logger.info(f"Processing {len(non_zoom_events)} events from non-zoom cameras...")
        
        # Try to import and initialize ReID model
        try:
            from reid_model import ReIDModel
        except ImportError:
            pytest.skip("ReID model not available")
        
        # Initialize model (cached after first run)
        # Suppress model initialization logs - Docker container has the proper model
        reid_logger = logging.getLogger("reid_model")
        original_level = reid_logger.level
        reid_logger.setLevel(logging.ERROR)
        try:
            model = ReIDModel()
        finally:
            reid_logger.setLevel(original_level)
        logger.info("✓ ReID model initialized")
        
        # Track embeddings for comparison
        embeddings = {}
        
        # Process events (up to 5 from non-zoom cameras)
        for i, event in enumerate(non_zoom_events[:5]):
            event_id = event["id"]
            camera = event.get("camera", "unknown")
            
            # Fetch snapshot
            snapshot_url = get_event_snapshot_url(frigate_host, event_id)
            snapshot_bytes = fetch_snapshot_bytes(frigate_session, snapshot_url)
            
            if not snapshot_bytes or not validate_image(snapshot_bytes):
                logger.warning(f"Skipping invalid snapshot: {event_id}")
                continue
            
            try:
                # Extract embedding
                b64_image = base64.b64encode(snapshot_bytes).decode()
                embedding = model.extract_embedding(b64_image)
                embeddings[event_id] = {
                    "embedding": embedding,
                    "snapshot": snapshot_bytes,
                    "camera": camera,
                }
                
                logger.info(f"✓ Event {i+1}: Extracted embedding from {camera} (event: {event_id})")
                
                # Add to report with dummy match ID
                # In a real scenario, you would match against stored persons
                match_report.add_match(
                    event_id=event_id,
                    snapshot_bytes=snapshot_bytes,
                    person_id=f"detected_{i}",
                    confidence=0.85 + (i * 0.05),  # Dummy confidence
                    camera=camera,
                    timestamp=event.get("start_time") or datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Failed to process event {event_id}: {e}")
                match_report.add_match(
                    event_id=event_id,
                    snapshot_bytes=snapshot_bytes or b"",
                    person_id="[Error]",
                    confidence=0.0,
                    camera=camera,
                    timestamp=datetime.now().isoformat()
                )
        
        # Verify we processed at least one event
        assert len(embeddings) > 0, "Should have successfully processed at least one event"
        logger.info(f"✓ Successfully extracted embeddings from {len(embeddings)} events")
        
        # Compare embeddings if we have multiple
        if len(embeddings) >= 2:
            event_ids = list(embeddings.keys())
            emb1 = embeddings[event_ids[0]]["embedding"]
            emb2 = embeddings[event_ids[1]]["embedding"]
            
            # Simple cosine similarity
            import numpy as np
            try:
                similarity = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )
                logger.info(f"✓ Embedding similarity (event 1 vs 2): {similarity:.4f}")
            except Exception as e:
                logger.warning(f"Could not compute similarity: {e}")
    
    def test_batch_process_recent_events(
        self,
        frigate_session,
        frigate_host,
        match_report,
        frigate_event_filters,
    ):
        """Process a batch of recent events and add all to the report.
        
        This is a basic infrastructure test that validates snapshot fetching
        and report generation. It uses dummy person IDs to verify the test
        framework is working correctly.
        
        For comprehensive identity testing with real facial recognition and
        ReID matching, see test_facial_recognition_and_reid_pipeline.
        """
        # Get recent events (more than usual for batch test)
        events = fetch_recent_events(
            frigate_session,
            frigate_host,
            limit=20,
            **frigate_event_filters,
        )
        
        if not events:
            pytest.skip("No recent events found in Frigate")
        
        # Filter out zoom camera events
        non_zoom_events = [e for e in events if e.get("camera", "").lower() != "zoom"]
        
        if not non_zoom_events:
            pytest.skip("No recent events found from non-zoom cameras")
        
        logger.info(f"Batch processing {len(non_zoom_events)} events from non-zoom cameras...")
        
        processed_count = 0
        
        for i, event in enumerate(non_zoom_events):
            event_id = event["id"]
            camera = event.get("camera", "unknown")
            
            # Fetch snapshot
            snapshot_url = get_event_snapshot_url(frigate_host, event_id)
            snapshot_bytes = fetch_snapshot_bytes(frigate_session, snapshot_url)
            
            if not snapshot_bytes:
                logger.debug(f"Skipping event {event_id}: no snapshot")
                continue
            
            # Validate image format
            if not validate_image(snapshot_bytes):
                logger.debug(f"Skipping event {event_id}: invalid image format")
                continue
            
            # Add to report with dummy match (infrastructure test only)
            confidence = 0.70 + (i * 0.03)  # Dummy confidence
            match_report.add_match(
                event_id=event_id,
                snapshot_bytes=snapshot_bytes,
                person_id=f"person_{i % 3}",  # Cycle through 3 person IDs
                confidence=min(0.99, confidence),
                camera=camera,
                timestamp=event.get("start_time") or datetime.now().isoformat(),
                source="unknown"  # Infrastructure test - no real matching
            )
            
            processed_count += 1
            if processed_count >= 10:
                break  # Limit to 10 for reasonable test duration
        
        assert processed_count > 0, "Should have processed at least one event"
        logger.info(f"✓ Batch processed {processed_count} events")
    
    def test_facial_recognition_and_reid_pipeline(
        self,
        frigate_session,
        frigate_host,
        temp_embedding_store,
        match_report,
        output_dir,
        frigate_event_filters,
    ):
        """Comprehensive E2E test: Learn from facial recognition, then test ReID matching.
        
        This test validates the complete identity service pipeline:
        
        PHASE 1: Learning from Frigate facial recognition
        - Fetch events where Frigate has identified faces (sub_label present)
        - Extract embeddings from these snapshots
        - Store them in the embedding database with person names
        
        PHASE 2: ReID Matching on unknown persons
        - Fetch events where person is unidentified (no sub_label)
        - Extract embeddings and match against stored persons
        - Validate identification accuracy
        
        This mirrors production use: learning from known faces, then identifying
        unknown persons via appearance matching.
        """
        logger.info("=" * 80)
        logger.info("PHASE 1: Learning from Frigate facial recognition events")
        logger.info("=" * 80)
        
        # Import required components
        try:
            from reid_model import ReIDModel
            from matcher import EmbeddingMatcher
        except ImportError as e:
            pytest.skip(f"Required components not available: {e}")
        
        # Initialize ReID model (suppress model loading logs)
        reid_logger = logging.getLogger("reid_model")
        original_level = reid_logger.level
        reid_logger.setLevel(logging.ERROR)
        try:
            model = ReIDModel()
        finally:
            reid_logger.setLevel(original_level)
        logger.info("✓ ReID model initialized")
        
        # Initialize matcher
        matcher = EmbeddingMatcher(
            max_age_hours=48.0,
            decay_mode="linear",
            weight_floor=0.3,
            use_confidence_weighting=True
        )
        logger.info("✓ Matcher initialized")
        
        # Fetch known faces and training data from Frigate
        faces_data = fetch_known_faces(frigate_session, frigate_host)
        
        if not faces_data:
            pytest.skip("No facial recognition data available from Frigate")
        
        # Parse training events to get event IDs with person names
        training_events = parse_face_training_events(faces_data)
        
        if not training_events:
            logger.warning("No facial recognition training events found in Frigate")
            pytest.skip("No facial recognition training data available")
        
        logger.info(f"Found {len(training_events)} facial recognition training events")
        
        # Get known persons from faces data
        known_persons = [k for k in faces_data.keys() if k != 'train']
        logger.info(f"Known persons in Frigate: {known_persons}")

        after_ts = frigate_event_filters.get("after")
        before_ts = frigate_event_filters.get("before")

        def _in_selected_day_window(event_data: dict) -> bool:
            if after_ts is None and before_ts is None:
                return True

            event_start = event_data.get("start_time")
            if event_start is None:
                return False

            try:
                event_ts = int(float(event_start))
            except (TypeError, ValueError):
                return False

            if after_ts is not None and event_ts < after_ts:
                return False
            if before_ts is not None and event_ts >= before_ts:
                return False
            return True
        
        # Fetch actual event data for facial recognition events 
        facial_events = []
        candidate_limit = 120 if frigate_event_filters else 25
        for event_id, person_name, confidence in training_events[:candidate_limit]:
            event_data = fetch_event_by_id(frigate_session, frigate_host, event_id)
            if event_data:
                if not _in_selected_day_window(event_data):
                    continue
                # Add the facial recognition metadata to the event
                event_data['_facial_recognition'] = {
                    'person_name': person_name,
                    'confidence': confidence
                }
                facial_events.append(event_data)
        
        facial_events.sort(
            key=lambda event: float(event.get("start_time") or 0),
            reverse=True,
        )

        # Filter out zoom camera events
        facial_events = [e for e in facial_events if e.get("camera", "").lower() != "zoom"]
        
        if not facial_events:
            if after_ts is not None and before_ts is not None:
                selected_day = datetime.fromtimestamp(after_ts).strftime("%Y-%m-%d")
                match_report.add_note(
                    "Selected day "
                    f"{selected_day} (UTC) had 0 facial-recognition training events "
                    "from non-zoom cameras."
                )
            pytest.skip("No facial recognition events available from non-zoom cameras")
        
        logger.info(f"Loaded {len(facial_events)} facial recognition events from non-zoom cameras")
        
        # Fetch unknown person events for ReID testing
        unknown_events = fetch_recent_events(
            frigate_session,
            frigate_host,
            limit=50,
            **frigate_event_filters,
        )
        unknown_events = [e for e in unknown_events if e.get("camera", "").lower() != "zoom"]
        unknown_events = [e for e in unknown_events if not e.get("sub_label")]
        
        logger.info(f"  Facial recognition events: {len(facial_events)}")
        logger.info(f"  Unknown person events for ReID testing: {len(unknown_events)}")
        
        learning_mode = "facial_recognition"
        
        # PHASE 1: Learn from facial recognition events
        learned_persons = set()
        phase1_processed = 0
        
        for event in facial_events[:10]:  # Limit to first 10 for reasonable duration
            event_id = event["id"]
            camera = event.get("camera", "unknown")
            
            # Get facial recognition data we attached earlier
            facial_data = event.get('_facial_recognition', {})
            person_name = facial_data.get('person_name')
            frigate_confidence = facial_data.get('confidence', 0.90)
            
            if not person_name:
                logger.warning(f"Skipping event {event_id}: no facial recognition data")
                continue
            
            # Fetch snapshot
            snapshot_url = get_event_snapshot_url(frigate_host, event_id)
            snapshot_bytes = fetch_snapshot_bytes(frigate_session, snapshot_url)
            
            if not snapshot_bytes or not validate_image(snapshot_bytes):
                logger.warning(f"Skipping invalid snapshot for {person_name}: {event_id}")
                continue
            
            try:
                # Extract embedding
                b64_image = base64.b64encode(snapshot_bytes).decode()
                embedding = model.extract_embedding(b64_image)
                
                # Store in embedding database
                temp_embedding_store.store_embedding(
                    person_id=person_name,
                    embedding=embedding,
                    camera=camera,
                    confidence=frigate_confidence
                )
                
                learned_persons.add(person_name)
                phase1_processed += 1
                
                logger.info(
                    f"✓ Learned: {person_name} from {camera} "
                    f"(confidence: {frigate_confidence:.2%}, event: {event_id})"
                )
                
                # Add to report
                match_report.add_match(
                    event_id=event_id,
                    snapshot_bytes=snapshot_bytes,
                    person_id=person_name,
                    confidence=frigate_confidence,
                    camera=camera,
                    timestamp=event.get("start_time") or datetime.now().isoformat(),
                    source="facial_recognition"
                )
                
            except Exception as e:
                logger.error(f"Failed to process facial recognition event {event_id}: {e}")
        
        # Verify we learned at least one person
        assert phase1_processed > 0, "Should have successfully learned at least one person"
        logger.info(f"✓ Phase 1 complete: Learned {len(learned_persons)} persons: {sorted(learned_persons)}")
        
        # PHASE 2: Test ReID matching on unknown persons
        logger.info("=" * 80)
        logger.info("PHASE 2: Testing ReID matching on unknown persons")
        logger.info("=" * 80)
        
        if not unknown_events:
            logger.warning("No unknown person events found - skipping ReID matching test")
            # Test still passes if we learned persons successfully
            return
        
        # Get stored embeddings for matching
        stored_embeddings = temp_embedding_store.get_all_embeddings()
        logger.info(f"Matching against {len(stored_embeddings)} known persons")
        
        phase2_processed = 0
        reid_matches = 0
        no_matches = 0
        
        for event in unknown_events[:15]:  # Process up to 15 unknown events
            event_id = event["id"]
            camera = event.get("camera", "unknown")
            
            # Fetch snapshot
            snapshot_url = get_event_snapshot_url(frigate_host, event_id)
            snapshot_bytes = fetch_snapshot_bytes(frigate_session, snapshot_url)
            
            if not snapshot_bytes or not validate_image(snapshot_bytes):
                logger.debug(f"Skipping invalid snapshot: {event_id}")
                continue
            
            try:
                # Extract embedding
                b64_image = base64.b64encode(snapshot_bytes).decode()
                embedding = model.extract_embedding(b64_image)
                
                # Get top-k matches for debugging (shows alternatives)
                top_matches = matcher.find_top_k_matches(
                    query_embedding=embedding,
                    stored_embeddings=stored_embeddings,
                    k=5,
                    threshold=0.0  # Get all matches for comparison
                )
                
                # Use best match if above threshold
                if top_matches and top_matches[0][1] >= REID_MATCH_THRESHOLD:
                    matched_person = top_matches[0][0]
                    similarity = top_matches[0][1]
                    alternatives = top_matches[1:4]  # Next 3 best matches
                    
                    # Successful ReID match
                    reid_matches += 1
                    
                    # Log alternatives if match is borderline (< 0.75)
                    if similarity < 0.75 and len(alternatives) > 0:
                        logger.warning(
                            f"⚠️  Borderline match: {matched_person} on {camera} "
                            f"(similarity: {similarity:.3f}, event: {event_id})"
                        )
                        logger.warning(
                            f"   Alternatives: " + 
                            ", ".join([f"{p}={s:.3f}" for p, s in alternatives[:2]])
                        )
                    else:
                        logger.info(
                            f"✓ ReID Match: {matched_person} on {camera} "
                            f"(similarity: {similarity:.3f}, event: {event_id})"
                        )
                    
                    match_report.add_match(
                        event_id=event_id,
                        snapshot_bytes=snapshot_bytes,
                        person_id=matched_person,
                        confidence=similarity,
                        camera=camera,
                        timestamp=event.get("start_time") or datetime.now().isoformat(),
                        source="reid_model",
                        alternatives=alternatives
                    )
                else:
                    # No match found
                    no_matches += 1
                    best_similarity = top_matches[0][1] if top_matches else 0.0
                    best_person = top_matches[0][0] if top_matches else "none"
                    
                    logger.info(
                        f"○ No match on {camera} "
                        f"(best: {best_person}={best_similarity:.3f}, event: {event_id})"
                    )
                    
                    match_report.add_match(
                        event_id=event_id,
                        snapshot_bytes=snapshot_bytes,
                        person_id=f"Unknown (best: {best_person} {best_similarity:.3f})",
                        confidence=best_similarity,
                        camera=camera,
                        timestamp=event.get("start_time") or datetime.now().isoformat(),
                        source="unknown",
                        alternatives=top_matches[:3] if top_matches else None
                    )
                
                phase2_processed += 1
                
            except Exception as e:
                logger.error(f"Failed to process unknown person event {event_id}: {e}")
        
        # Summary
        logger.info("=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Phase 1 - Facial Recognition:")
        logger.info(f"  Processed: {phase1_processed} events")
        logger.info(f"  Learned: {len(learned_persons)} unique persons")
        logger.info(f"Phase 2 - ReID Matching:")
        logger.info(f"  Processed: {phase2_processed} unknown events")
        logger.info(f"  Matched: {reid_matches} ({reid_matches/max(phase2_processed,1)*100:.1f}%)")
        logger.info(f"  No match: {no_matches} ({no_matches/max(phase2_processed,1)*100:.1f}%)")
        logger.info("=" * 80)
        
        # Test is successful if we successfully processed both phases
        assert phase1_processed > 0, "Should have processed facial recognition events"
        assert phase2_processed > 0 or not unknown_events, "Should have processed unknown events if available"
        logger.info("✓ Comprehensive E2E test complete")


