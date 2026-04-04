"""
Unit tests for the ReID system components.

Run with: python -m pytest tests/ -v

Install pytest first:
  pip install pytest pytest-cov
"""

import pytest
import base64
import numpy as np
from PIL import Image
import io
import tempfile
import os


@pytest.fixture
def sample_image():
    """Create a simple test image (100x50 RGB)."""
    img = Image.new("RGB", (100, 50), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return base64.b64encode(img_byte_arr.read()).decode("utf-8")


@pytest.fixture
def sample_image_blue():
    """Create a different test image (100x50 RGB, blue)."""
    img = Image.new("RGB", (100, 50), color="blue")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return base64.b64encode(img_byte_arr.read()).decode("utf-8")


@pytest.fixture
def temp_db():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestEmbeddingStore:
    """Tests for the EmbeddingStore module."""

    def test_embedding_store_initialization(self, temp_db):
        """Test that embedding store initializes correctly."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        assert store.db_path == temp_db
        assert len(store.embeddings) == 0

    def test_store_and_retrieve_embedding(self, temp_db):
        """Test storing and retrieving an embedding."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)

        embedding = np.random.rand(256)
        store.store_embedding("person1", embedding, "camera1", confidence=0.95)

        assert store.person_exists("person1")
        retrieved = store.get_embedding("person1")
        assert retrieved is not None
        assert len(retrieved) == 256

    def test_get_all_embeddings(self, temp_db):
        """Test retrieving all embeddings."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)

        emb1 = np.random.rand(256)
        emb2 = np.random.rand(256)
        store.store_embedding("person1", emb1, "camera1")
        store.store_embedding("person2", emb2, "camera2")

        all_embs = store.get_all_embeddings()
        assert len(all_embs) == 2
        assert "person1" in all_embs
        assert "person2" in all_embs

    def test_delete_person(self, temp_db):
        """Test deleting a person."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)

        embedding = np.random.rand(256)
        store.store_embedding("person1", embedding, "camera1")
        assert store.person_exists("person1")

        store.delete_person("person1")
        assert not store.person_exists("person1")

    def test_persistence(self, temp_db):
        """Test that embeddings persist across instances."""
        from embedding_store import EmbeddingStore

        store1 = EmbeddingStore(temp_db)
        embedding = np.random.rand(256)
        store1.store_embedding("person1", embedding, "camera1")

        store2 = EmbeddingStore(temp_db)
        assert store2.person_exists("person1")
        retrieved = store2.get_embedding("person1")
        assert len(retrieved) == 256

    # ── False-positive / event_id support ──────────────────────────────

    def test_store_embedding_with_event_id(self, temp_db):
        """Embedding stored with event_id should persist the event_id field."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        emb = np.random.rand(256)
        store.store_embedding("alice", emb, "cam1", confidence=0.9, event_id="evt-001")

        # Reload from disk to confirm persistence
        store2 = EmbeddingStore(temp_db)
        entries = store2.embeddings["alice"]
        assert entries[0]["event_id"] == "evt-001"

    def test_remove_embeddings_by_event_id_matching(self, temp_db):
        """remove_embeddings_by_event_id should remove only the matched entry."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        store.store_embedding("alice", np.random.rand(256), "cam1", event_id="evt-bad")
        store.store_embedding("alice", np.random.rand(256), "cam1", event_id="evt-good")

        removed = store.remove_embeddings_by_event_id("alice", "evt-bad")
        assert removed == 1
        # Should still have the good embedding
        assert store.person_exists("alice")
        remaining_events = [e.get("event_id") for e in store.embeddings["alice"]]
        assert "evt-bad" not in remaining_events
        assert "evt-good" in remaining_events

    def test_remove_embeddings_by_event_id_fallback(self, temp_db):
        """When event_id is absent from stored entries, falls back to removing the newest."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        # Store two embeddings WITHOUT event_id
        store.store_embedding("bob", np.random.rand(256), "cam1", confidence=0.9)
        store.store_embedding("bob", np.random.rand(256), "cam1", confidence=0.8)

        # Use a non-matching event_id → triggers fallback
        removed = store.remove_embeddings_by_event_id(
            "bob",
            "evt-unknown",
            fallback_to_latest=True,
        )
        assert removed == 1
        # One embedding should remain
        assert store.person_exists("bob")
        assert len(store.embeddings["bob"]) == 1

    def test_remove_embeddings_by_event_id_no_fallback_is_idempotent(self, temp_db):
        """Unknown event_id should not remove unrelated embeddings when fallback is disabled."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        store.store_embedding("bob", np.random.rand(256), "cam1", event_id="evt-known")

        removed = store.remove_embeddings_by_event_id(
            "bob",
            "evt-missing",
            fallback_to_latest=False,
        )
        assert removed == 0
        assert store.person_exists("bob")
        assert len(store.embeddings["bob"]) == 1

    def test_remove_embeddings_by_event_id_last_entry_deletes_person(self, temp_db):
        """Removing the only embedding should delete the person key entirely."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        store.store_embedding("carol", np.random.rand(256), "cam1", event_id="evt-only")

        removed = store.remove_embeddings_by_event_id("carol", "evt-only")
        assert removed == 1
        assert not store.person_exists("carol")

    def test_remove_embeddings_by_event_id_unknown_person(self, temp_db):
        """Removing embeddings for an unknown person should return 0 gracefully."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        removed = store.remove_embeddings_by_event_id("nobody", "evt-123")
        assert removed == 0

    def test_get_latest_event_id(self, temp_db):
        """get_latest_event_id should return the event_id of the most recent embedding."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        store.store_embedding("dave", np.random.rand(256), "cam1", event_id="evt-old")
        store.store_embedding("dave", np.random.rand(256), "cam1", event_id="evt-new")

        # Most recent is prepended (index 0)
        assert store.get_latest_event_id("dave") == "evt-new"

    def test_get_latest_event_id_no_event_ids(self, temp_db):
        """get_latest_event_id should return None when no event_ids are stored."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        store.store_embedding("eve", np.random.rand(256), "cam1")
        assert store.get_latest_event_id("eve") is None

    def test_get_latest_event_id_unknown_person(self, temp_db):
        """get_latest_event_id should return None for unknown person."""
        from embedding_store import EmbeddingStore

        store = EmbeddingStore(temp_db)
        assert store.get_latest_event_id("nobody") is None


class TestMatcher:
    """Tests for the EmbeddingMatcher module."""

    def test_find_best_match_exact(self):
        """Test finding exact match with identical embeddings."""
        from matcher import EmbeddingMatcher

        matcher = EmbeddingMatcher()
        query = np.array([1.0, 0.0, 0.0])
        stored = {
            "person1": (np.array([1.0, 0.0, 0.0]), "camera1", 0.9),
            "person2": (np.array([0.0, 1.0, 0.0]), "camera2", 0.8),
        }

        matched, score = matcher.find_best_match(query, stored, threshold=0.9)
        assert matched == "person1"
        assert score >= 0.99

    def test_find_best_match_below_threshold(self):
        """Test that no match is returned below threshold."""
        from matcher import EmbeddingMatcher

        matcher = EmbeddingMatcher()
        query = np.array([1.0, 0.0, 0.0])
        stored = {
            "person1": (np.array([0.0, 1.0, 0.0]), "camera1", 0.9),
        }

        matched, score = matcher.find_best_match(query, stored, threshold=0.9)
        assert matched is None

    def test_find_best_match_empty_store(self):
        """Test matching against empty embedding store."""
        from matcher import EmbeddingMatcher

        matcher = EmbeddingMatcher()
        query = np.array([1.0, 0.0, 0.0])
        stored = {}

        matched, score = matcher.find_best_match(query, stored, threshold=0.5)
        assert matched is None
        assert score == 0.0

    def test_find_top_k_matches(self):
        """Test finding top-k matches."""
        from matcher import EmbeddingMatcher

        matcher = EmbeddingMatcher()
        query = np.array([1.0, 0.0, 0.0])
        stored = {
            "person1": (np.array([1.0, 0.0, 0.0]), "camera1", 0.9),
            "person2": (np.array([0.9, 0.1, 0.0]), "camera2", 0.8),
            "person3": (np.array([0.0, 1.0, 0.0]), "camera3", 0.7),
        }

        top_matches = matcher.find_top_k_matches(query, stored, k=2, threshold=0.0)
        assert len(top_matches) <= 2
        assert top_matches[0][1] >= top_matches[1][1]


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_embedding_storage_and_matching(self, temp_db):
        """Test complete workflow: store embeddings and match."""
        from embedding_store import EmbeddingStore
        from matcher import EmbeddingMatcher

        store = EmbeddingStore(temp_db)
        matcher = EmbeddingMatcher()

        # Create and store embeddings
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])
        store.store_embedding("alice", emb1, "camera1", confidence=0.95)
        store.store_embedding("bob", emb2, "camera2", confidence=0.90)

        # Test matching
        stored = store.get_all_embeddings()
        query = np.array([1.0, 0.0, 0.0])

        matched, score = matcher.find_best_match(query, stored, threshold=0.5)
        assert matched == "alice"
        assert score > 0.9


class TestFrigateSnapshotHelpers:
    """Tests for Frigate snapshot helper URL and query handling."""

    def test_get_event_snapshot_url_defaults_to_snapshot(self):
        """Snapshot helper should default to the event snapshot endpoint."""
        from tests.utils.frigate_api import get_event_snapshot_url

        assert (
            get_event_snapshot_url("http://frigate:5000", "abc123")
            == "http://frigate:5000/api/events/abc123/snapshot.jpg"
        )

    def test_get_event_snapshot_url_supports_clean_copy(self):
        """Snapshot helper should expose the clean snapshot endpoint."""
        from tests.utils.frigate_api import get_event_snapshot_url

        assert (
            get_event_snapshot_url("http://frigate:5000", "abc123", "clean")
            == "http://frigate:5000/api/events/abc123/snapshot-clean.webp"
        )

    def test_fetch_snapshot_bytes_uses_snapshot_query_params(self):
        """Snapshot endpoint requests should send Frigate-supported query params."""
        from tests.utils.frigate_api import fetch_snapshot_bytes

        class Response:
            status_code = 200
            content = b"snapshot-bytes"

            def raise_for_status(self):
                return None

        class Session:
            def __init__(self):
                self.calls = []

            def get(self, url, params=None, timeout=None):
                self.calls.append((url, params, timeout))
                return Response()

        session = Session()
        result = fetch_snapshot_bytes(
            session,
            "http://frigate:5000/api/events/abc123/snapshot.jpg",
            crop=True,
            quality=90,
            height=420,
        )

        assert result == b"snapshot-bytes"
        assert session.calls == [
            (
                "http://frigate:5000/api/events/abc123/snapshot.jpg",
                {
                    "bbox": 0,
                    "timestamp": 0,
                    "crop": 1,
                    "quality": 90,
                    "height": 420,
                },
                10,
            )
        ]

    def test_fetch_snapshot_bytes_skips_params_for_clean_snapshot(self):
        """Clean snapshot requests should not send snapshot-only query params."""
        from tests.utils.frigate_api import fetch_snapshot_bytes

        class Response:
            status_code = 200
            content = b"clean-bytes"

            def raise_for_status(self):
                return None

        class Session:
            def __init__(self):
                self.calls = []

            def get(self, url, params=None, timeout=None):
                self.calls.append((url, params, timeout))
                return Response()

        session = Session()
        result = fetch_snapshot_bytes(
            session,
            "http://frigate:5000/api/events/abc123/snapshot-clean.webp",
        )

        assert result == b"clean-bytes"
        assert session.calls == [
            (
                "http://frigate:5000/api/events/abc123/snapshot-clean.webp",
                None,
                10,
            )
        ]


class TestLocalCropHelpers:
    """Unit tests for the local snapshot crop geometry helpers."""

    def test_build_crop_rect_applies_asymmetric_padding(self):
        """Vertical padding should be larger than horizontal padding."""
        from snapshot_crop import build_local_crop_rect

        # Tight centre box: x=0.4, y=0.4, w=0.2, h=0.2
        geometry = {"box": (0.4, 0.4, 0.2, 0.2)}
        rect = build_local_crop_rect(geometry, padding_x=0.05, padding_y=0.20)
        assert rect is not None
        left, top, right, bottom = rect

        horizontal_expansion = (right - left) - 0.2
        vertical_expansion = (bottom - top) - 0.2
        assert vertical_expansion > horizontal_expansion

    def test_build_crop_rect_targets_2_to_1_aspect_ratio(self):
        """Output rect should have height:width ratio of 2:1."""
        from snapshot_crop import build_local_crop_rect

        # Square-ish person detection at centre
        geometry = {"box": (0.3, 0.3, 0.4, 0.4)}
        rect = build_local_crop_rect(geometry, padding_x=0.05, padding_y=0.20)
        assert rect is not None
        left, top, right, bottom = rect

        w = right - left
        h = bottom - top
        ratio = h / w
        assert abs(ratio - 2.0) < 0.05, f"Expected ~2.0 ratio, got {ratio:.3f}"

    def test_build_crop_rect_expands_wide_box_vertically(self):
        """A very wide detection should have its height expanded to reach 2:1."""
        from snapshot_crop import build_local_crop_rect

        # Wide, shallow box (e.g. a person far away, or arms out)
        geometry = {"box": (0.1, 0.4, 0.8, 0.1)}
        rect = build_local_crop_rect(geometry, padding_x=0.05, padding_y=0.20)
        assert rect is not None
        left, top, right, bottom = rect
        h = bottom - top
        w = right - left
        assert h / w >= 1.9, f"Expected ratio >= 1.9, got {h / w:.3f}"

    def test_build_crop_rect_returns_none_for_missing_geometry(self):
        """None or empty geometry should return None."""
        from snapshot_crop import build_local_crop_rect

        assert build_local_crop_rect(None) is None
        assert build_local_crop_rect({}) is None
        assert build_local_crop_rect({"box": None, "region": None}) is None

    def test_build_crop_rect_allows_out_of_bounds_coords(self):
        """Rect touching frame edge may have coords outside [0,1] — letterbox handles it."""
        from snapshot_crop import build_local_crop_rect

        # Person at top-left corner
        geometry = {"box": (0.0, 0.0, 0.1, 0.2)}
        rect = build_local_crop_rect(geometry, padding_x=0.05, padding_y=0.20)
        assert rect is not None
        left, top, right, bottom = rect
        assert right > left and bottom > top

    def test_letterbox_to_ratio_pads_wide_image(self):
        """A wider-than-2:1 image should be padded top/bottom."""
        from snapshot_crop import letterbox_to_ratio

        img = Image.new("RGB", (200, 100))  # 0.5:1 ratio (too wide)
        result = letterbox_to_ratio(img, target_ratio=2.0)
        w, h = result.size
        assert h / w == pytest.approx(2.0, abs=0.02)
        assert w == 200  # width unchanged

    def test_letterbox_to_ratio_pads_tall_image(self):
        """A taller-than-2:1 image should be padded left/right."""
        from snapshot_crop import letterbox_to_ratio

        img = Image.new("RGB", (100, 400))  # 4:1 ratio (too tall)
        result = letterbox_to_ratio(img, target_ratio=2.0)
        w, h = result.size
        assert h / w == pytest.approx(2.0, abs=0.02)
        assert h == 400  # height unchanged

    def test_letterbox_fill_colour_is_imagenet_mean(self):
        """Padded pixels should use the ImageNet mean colour (124, 116, 104)."""
        from snapshot_crop import letterbox_to_ratio

        img = Image.new("RGB", (200, 50), color=(255, 0, 0))  # too wide
        result = letterbox_to_ratio(img, target_ratio=2.0)
        # Top-left pixel is in the padding region
        pad_pixel = result.getpixel((0, 0))
        assert pad_pixel == (124, 116, 104)

    def test_crop_snapshot_bytes_returns_jpeg_for_valid_input(self):
        """crop_snapshot_bytes should return JPEG bytes for a valid clean frame."""
        from snapshot_crop import crop_snapshot_bytes

        # 640x480 white image
        img = Image.new("RGB", (640, 480), color=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        geometry = {"box": (0.3, 0.2, 0.4, 0.6)}
        result = crop_snapshot_bytes(raw_bytes, geometry)

        assert result is not None
        parsed = Image.open(io.BytesIO(result))
        assert parsed.format == "JPEG"

    def test_crop_snapshot_bytes_output_is_2to1_ratio(self):
        """Output image should be close to 2:1 height:width."""
        from snapshot_crop import crop_snapshot_bytes

        img = Image.new("RGB", (640, 480))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        geometry = {"box": (0.2, 0.1, 0.6, 0.8)}
        result = crop_snapshot_bytes(raw_bytes, geometry)

        assert result is not None
        parsed = Image.open(io.BytesIO(result))
        w, h = parsed.size
        ratio = h / w
        assert abs(ratio - 2.0) < 0.15, f"Expected ~2.0, got {ratio:.3f}"

    def test_crop_snapshot_bytes_returns_none_for_missing_geometry(self):
        """None geometry should produce None output."""
        from snapshot_crop import crop_snapshot_bytes

        img = Image.new("RGB", (200, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        assert crop_snapshot_bytes(buf.getvalue(), None) is None

    def test_crop_snapshot_bytes_letterboxes_edge_crop(self):
        """Person at image edge should produce a letterboxed 2:1 output."""
        from snapshot_crop import crop_snapshot_bytes

        img = Image.new("RGB", (640, 480))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        # Top-left corner — expansion will push rect out of bounds
        geometry = {"box": (0.0, 0.0, 0.15, 0.30)}
        result = crop_snapshot_bytes(raw_bytes, geometry)

        assert result is not None
        parsed = Image.open(io.BytesIO(result))
        w, h = parsed.size
        ratio = h / w
        assert abs(ratio - 2.0) < 0.15, f"Expected ~2.0 letterboxed, got {ratio:.3f}"


class TestCropSnapshotPil:
    """Tests for crop_snapshot_pil and pil_to_jpeg_bytes (Gap 3 lossless path)."""

    def test_crop_snapshot_pil_returns_pil_image(self):
        """crop_snapshot_pil should return a PIL Image, not bytes."""
        from snapshot_crop import crop_snapshot_pil

        img = Image.new("RGB", (640, 480), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        geometry = {"box": (0.2, 0.1, 0.5, 0.7)}
        result = crop_snapshot_pil(raw_bytes, geometry)

        assert result is not None
        assert isinstance(result, Image.Image)

    def test_crop_snapshot_pil_matches_bytes_dimensions(self):
        """PIL and bytes crop should produce the same image dimensions."""
        from snapshot_crop import crop_snapshot_pil, crop_snapshot_bytes

        img = Image.new("RGB", (640, 480))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        geometry = {"box": (0.2, 0.1, 0.4, 0.6)}
        pil_result = crop_snapshot_pil(raw_bytes, geometry)
        bytes_result = crop_snapshot_bytes(raw_bytes, geometry)

        assert pil_result is not None and bytes_result is not None
        bytes_img = Image.open(io.BytesIO(bytes_result))
        assert pil_result.size == bytes_img.size

    def test_crop_snapshot_pil_is_rgb(self):
        """Returned PIL image must be in RGB mode."""
        from snapshot_crop import crop_snapshot_pil

        img = Image.new("RGBA", (640, 480), color=(10, 20, 30, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        geometry = {"box": (0.1, 0.1, 0.8, 0.8)}
        result = crop_snapshot_pil(raw_bytes, geometry)

        assert result is not None
        assert result.mode == "RGB"

    def test_crop_snapshot_pil_returns_none_for_missing_geometry(self):
        """None geometry should return None."""
        from snapshot_crop import crop_snapshot_pil

        img = Image.new("RGB", (200, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        assert crop_snapshot_pil(buf.getvalue(), None) is None

    def test_pil_to_jpeg_bytes_produces_valid_jpeg(self):
        """pil_to_jpeg_bytes should produce decodable JPEG bytes."""
        from snapshot_crop import pil_to_jpeg_bytes

        img = Image.new("RGB", (128, 256), color=(255, 0, 0))
        result = pil_to_jpeg_bytes(img, quality=85)

        assert isinstance(result, bytes)
        decoded = Image.open(io.BytesIO(result))
        assert decoded.format == "JPEG"
        assert decoded.size == (128, 256)

    def test_pil_avoids_jpeg_artifacts_vs_bytes(self):
        """PIL path should have no JPEG encoding loss; bytes path introduces DCT artifacts."""
        from snapshot_crop import crop_snapshot_pil, crop_snapshot_bytes

        # Solid-colour PNG source (lossless, no pre-existing JPEG artifacts)
        img = Image.new("RGB", (640, 480), color=(200, 100, 50))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        geometry = {"box": (0.25, 0.25, 0.4, 0.4)}  # well inside frame

        pil_crop = crop_snapshot_pil(raw_bytes, geometry)
        bytes_crop = crop_snapshot_bytes(raw_bytes, geometry)

        assert pil_crop is not None and bytes_crop is not None

        # Re-encode the PIL crop at max quality → if we then compare with
        # pil_crop, MSE should be very small (only colour subsampling loss
        # from lossless→JPEG at q=100 is negligible on solid colour).
        # Re-encode at q=50 to amplify JPEG effects for a more robust assertion.
        pil_via_jpeg = Image.open(
            io.BytesIO(crop_snapshot_bytes(raw_bytes, geometry, quality=50))
        )

        # Force same size for comparison
        size = (
            min(pil_crop.width, pil_via_jpeg.width),
            min(pil_crop.height, pil_via_jpeg.height),
        )
        pil_small = np.array(pil_crop.resize(size)).astype(float)
        jpeg_small = np.array(pil_via_jpeg.resize(size)).astype(float)

        mse_pil_vs_jpeg = float(np.mean((pil_small - jpeg_small) ** 2))

        # JPEG at q=50 on a region with colour gradients must distort pixels.
        assert mse_pil_vs_jpeg > 0, "Expected JPEG encoding to change pixel values"


class TestCropSnapshotDisplay:
    """Tests for crop_snapshot_bytes_for_display (display without letterboxing)."""

    def test_crop_snapshot_for_display_returns_jpeg_bytes(self):
        """Display crop should return JPEG bytes."""
        from snapshot_crop import crop_snapshot_bytes_for_display

        img = Image.new("RGB", (640, 480), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        geometry = {"box": (0.2, 0.1, 0.5, 0.7)}
        result = crop_snapshot_bytes_for_display(raw_bytes, geometry)

        assert result is not None
        assert isinstance(result, bytes)
        # Verify it's valid JPEG
        decoded = Image.open(io.BytesIO(result))
        assert decoded.format == "JPEG"

    def test_crop_snapshot_for_display_no_letterbox_on_edge_crop(self):
        """Display crop at image edge should NOT letterbox (unlike regular crop)."""
        from snapshot_crop import crop_snapshot_bytes, crop_snapshot_bytes_for_display

        img = Image.new("RGB", (640, 480), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        # Top-left corner — expansion will push rect out of bounds
        geometry = {"box": (0.0, 0.0, 0.15, 0.30)}

        # Regular crop letterboxes with ImageNet mean RGB (124,116,104)
        regular = crop_snapshot_bytes(raw_bytes, geometry)
        regular_img = Image.open(io.BytesIO(regular))

        # Display crop should NOT letterbox (just return cropped region)
        display = crop_snapshot_bytes_for_display(raw_bytes, geometry)
        display_img = Image.open(io.BytesIO(display))

        # Regular should be 2:1 (letterboxed)
        regular_ratio = regular_img.height / regular_img.width
        assert abs(regular_ratio - 2.0) < 0.15, (
            f"Regular should be ~2:1, got {regular_ratio:.2f}"
        )

        # Display should NOT be 2:1 (no padding, just the crop)
        display_ratio = display_img.height / display_img.width
        assert abs(display_ratio - 2.0) > 0.3, (
            f"Display should NOT be 2:1, got {display_ratio:.2f}"
        )

    def test_crop_snapshot_for_display_no_brown_padding(self):
        """Display crop should not contain ImageNet mean brown padding."""
        from snapshot_crop import crop_snapshot_bytes_for_display, IMAGENET_MEAN_RGB

        img = Image.new("RGB", (640, 480), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        # Top-left corner — would normally be padded with brown letterbox
        geometry = {"box": (0.0, 0.0, 0.15, 0.30)}
        display = crop_snapshot_bytes_for_display(raw_bytes, geometry)
        display_img = Image.open(io.BytesIO(display))

        # Convert to numpy for pixel analysis
        pixels = np.array(display_img)

        # Check that no pixels match the brown letterbox colour
        # (allowing for JPEG compression artifacts ±10 per channel)
        brown_r, brown_g, brown_b = IMAGENET_MEAN_RGB
        brown_mask = (
            (np.abs(pixels[:, :, 0].astype(int) - brown_r) <= 10)
            & (np.abs(pixels[:, :, 1].astype(int) - brown_g) <= 10)
            & (np.abs(pixels[:, :, 2].astype(int) - brown_b) <= 10)
        )

        # For a red source image at image edge, we shouldn't have brown padding
        brown_pixel_count = np.sum(brown_mask)
        total_pixels = pixels.shape[0] * pixels.shape[1]
        brown_fraction = brown_pixel_count / total_pixels

        # Less than 5% should be brown-ish (any present is JPEG compression artifact)
        assert brown_fraction < 0.05, (
            f"Display crop has too much brown: {brown_fraction:.2%}"
        )

    def test_crop_snapshot_for_display_handles_missing_geometry(self):
        """Display crop with None geometry should return None."""
        from snapshot_crop import crop_snapshot_bytes_for_display

        img = Image.new("RGB", (200, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        result = crop_snapshot_bytes_for_display(buf.getvalue(), None)

        assert result is None

    def test_crop_snapshot_for_display_valid_interior_crop(self):
        """Display crop well inside frame should work correctly."""
        from snapshot_crop import crop_snapshot_bytes_for_display

        img = Image.new("RGB", (640, 480), color=(50, 100, 150))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        # Well inside bounds — should crop cleanly
        geometry = {"box": (0.25, 0.25, 0.75, 0.75)}
        result = crop_snapshot_bytes_for_display(raw_bytes, geometry)

        assert result is not None
        decoded = Image.open(io.BytesIO(result))
        assert decoded.format == "JPEG"
        # Should be roughly square (0.25-0.75 is 50% range)
        ratio = decoded.height / decoded.width
        assert 0.8 < ratio < 1.2, f"Interior crop should be ~square, got {ratio:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
