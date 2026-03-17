"""Tests for recency weighting and confidence weighting in EmbeddingMatcher.

These tests verify that the new configurable recency decay modes and
confidence weighting work correctly and interact properly with the
embedding retention policy.
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "frigate_identity_service"
    ),
)

from matcher import EmbeddingMatcher


class TestRecencyWeighting:
    """Test recency weighting with different decay modes."""

    def test_linear_decay_fresh_embedding(self):
        """Test that fresh embeddings get weight near 1.0 with linear decay."""
        matcher = EmbeddingMatcher(
            max_age_hours=48.0,
            decay_mode="linear",
            weight_floor=0.3,
        )

        # Fresh timestamp (now)
        fresh_timestamp = datetime.now().isoformat()
        weight = matcher._recency_weight(fresh_timestamp)

        assert 0.95 <= weight <= 1.0, (
            f"Fresh embedding should have weight near 1.0, got {weight}"
        )

    def test_linear_decay_old_embedding(self):
        """Test that embeddings near max_age get weight near floor with linear decay."""
        matcher = EmbeddingMatcher(
            max_age_hours=48.0,
            decay_mode="linear",
            weight_floor=0.3,
        )

        # Very old timestamp (47 hours ago, near max)
        old_timestamp = (datetime.now() - timedelta(hours=47)).isoformat()
        weight = matcher._recency_weight(old_timestamp)

        assert 0.3 <= weight <= 0.35, (
            f"Old embedding should have weight near floor (0.3), got {weight}"
        )

    def test_linear_decay_mid_age_embedding(self):
        """Test linear decay at midpoint (24h with 48h max should be ~0.65)."""
        matcher = EmbeddingMatcher(
            max_age_hours=48.0,
            decay_mode="linear",
            weight_floor=0.3,
        )

        # Mid-age timestamp (24 hours ago, halfway)
        mid_timestamp = (datetime.now() - timedelta(hours=24)).isoformat()
        weight = matcher._recency_weight(mid_timestamp)

        # Expected: 1.0 - (1.0 - 0.3) * (24 / 48) = 1.0 - 0.7 * 0.5 = 0.65
        expected = 0.65
        assert abs(weight - expected) < 0.05, (
            f"Mid-age embedding should have weight ~{expected}, got {weight}"
        )

    def test_exponential_decay_faster_initial(self):
        """Test that exponential decay drops faster initially than linear."""
        linear_matcher = EmbeddingMatcher(
            max_age_hours=48.0,
            decay_mode="linear",
            weight_floor=0.3,
        )

        exponential_matcher = EmbeddingMatcher(
            max_age_hours=48.0,
            decay_mode="exponential",
            weight_floor=0.3,
        )

        # Check at 8 hours (early in lifecycle)
        timestamp_8h = (datetime.now() - timedelta(hours=8)).isoformat()
        linear_weight = linear_matcher._recency_weight(timestamp_8h)
        exp_weight = exponential_matcher._recency_weight(timestamp_8h)

        assert exp_weight < linear_weight, (
            f"Exponential should decay faster initially: exp={exp_weight} should be < linear={linear_weight}"
        )

    def test_no_decay_mode(self):
        """Test that 'none' decay mode returns constant weight of 1.0."""
        matcher = EmbeddingMatcher(
            max_age_hours=48.0,
            decay_mode="none",
            weight_floor=0.3,
        )

        # Test at various ages - all should be 1.0
        for hours_ago in [0, 12, 24, 36, 47]:
            timestamp = (datetime.now() - timedelta(hours=hours_ago)).isoformat()
            weight = matcher._recency_weight(timestamp)
            assert weight == 1.0, (
                f"'none' decay mode should always return 1.0, got {weight} at {hours_ago}h"
            )

    def test_invalid_timestamp_fallback(self):
        """Test that invalid timestamps return mid-range weight."""
        matcher = EmbeddingMatcher(
            max_age_hours=48.0,
            decay_mode="linear",
            weight_floor=0.3,
        )

        weight = matcher._recency_weight("invalid-timestamp")
        expected_mid = (1.0 + 0.3) / 2  # 0.65
        assert weight == expected_mid, (
            f"Invalid timestamp should return mid-range weight {expected_mid}, got {weight}"
        )

    def test_weight_floor_respected(self):
        """Test that weights never go below configured floor."""
        matcher = EmbeddingMatcher(
            max_age_hours=48.0,
            decay_mode="linear",
            weight_floor=0.2,
        )

        # Test way beyond max_age
        ancient_timestamp = (datetime.now() - timedelta(hours=100)).isoformat()
        weight = matcher._recency_weight(ancient_timestamp)

        assert weight >= 0.2, f"Weight should never go below floor 0.2, got {weight}"


class TestConfidenceWeighting:
    """Test confidence weighting functionality."""

    def test_confidence_disabled_returns_1(self):
        """Test that confidence weighting disabled returns 1.0."""
        matcher = EmbeddingMatcher(use_confidence_weighting=False)

        for confidence in [0.0, 0.5, 0.9, 1.0]:
            weight = matcher._confidence_weight(confidence)
            assert weight == 1.0, (
                f"Disabled confidence weighting should return 1.0, got {weight}"
            )

    def test_confidence_enabled_scaling(self):
        """Test confidence weighting scales correctly from 0.7 to 1.0."""
        matcher = EmbeddingMatcher(use_confidence_weighting=True)

        # Test boundary values
        assert matcher._confidence_weight(0.0) == 0.7, (
            "Confidence 0.0 should map to 0.7"
        )
        assert matcher._confidence_weight(1.0) == 1.0, (
            "Confidence 1.0 should map to 1.0"
        )

        # Test midpoint
        mid_weight = matcher._confidence_weight(0.5)
        expected = 0.7 + (0.3 * 0.5)  # 0.85
        assert abs(mid_weight - expected) < 0.01, (
            f"Confidence 0.5 should map to ~{expected}, got {mid_weight}"
        )

    def test_confidence_clamping(self):
        """Test that confidence values outside [0, 1] are clamped."""
        matcher = EmbeddingMatcher(use_confidence_weighting=True)

        # Test below 0
        assert matcher._confidence_weight(-0.5) == 0.7, (
            "Negative confidence should clamp to 0.7"
        )

        # Test above 1
        assert matcher._confidence_weight(1.5) == 1.0, (
            "Confidence > 1 should clamp to 1.0"
        )


class TestCombinedWeighting:
    """Test interaction between recency and confidence weighting."""

    def test_both_weightings_multiply(self):
        """Test that recency and confidence weights multiply together."""
        matcher = EmbeddingMatcher(
            max_age_hours=48.0,
            decay_mode="linear",
            weight_floor=0.5,
            use_confidence_weighting=True,
        )

        # Create old, low-confidence embedding
        old_timestamp = (datetime.now() - timedelta(hours=40)).isoformat()
        low_confidence = 0.3

        recency_w = matcher._recency_weight(old_timestamp)
        conf_w = matcher._confidence_weight(low_confidence)

        # Both should be < 1.0, so combined weight should be even lower
        assert recency_w < 1.0, "Old embedding should have reduced recency weight"
        assert conf_w < 1.0, "Low confidence should have reduced confidence weight"

        combined = recency_w * conf_w
        assert combined < recency_w, "Combined weight should be less than recency alone"
        assert combined < conf_w, "Combined weight should be less than confidence alone"

    def test_fresh_high_confidence_gets_best_weight(self):
        """Test that fresh, high-confidence embeddings get weight near 1.0."""
        matcher = EmbeddingMatcher(
            max_age_hours=48.0,
            decay_mode="linear",
            weight_floor=0.3,
            use_confidence_weighting=True,
        )

        fresh_timestamp = datetime.now().isoformat()
        high_confidence = 0.95

        recency_w = matcher._recency_weight(fresh_timestamp)
        conf_w = matcher._confidence_weight(high_confidence)
        combined = recency_w * conf_w

        assert combined >= 0.95, (
            f"Fresh high-confidence embedding should have weight near 1.0, got {combined}"
        )


class TestConfigurationValidation:
    """Test that matcher validates configuration properly."""

    def test_invalid_decay_mode_raises(self):
        """Test that invalid decay mode raises ValueError."""
        try:
            EmbeddingMatcher(decay_mode="invalid")
            assert False, "Should have raised ValueError for invalid decay_mode"
        except ValueError as e:
            assert "decay_mode" in str(e).lower()

    def test_valid_decay_modes_accepted(self):
        """Test that all valid decay modes are accepted."""
        for mode in ["linear", "exponential", "none"]:
            matcher = EmbeddingMatcher(decay_mode=mode)
            assert matcher.decay_mode == mode.lower()

    def test_configuration_stored(self):
        """Test that configuration is stored in instance."""
        matcher = EmbeddingMatcher(
            max_age_hours=72.0,
            decay_mode="exponential",
            weight_floor=0.4,
            use_confidence_weighting=True,
        )

        assert matcher.max_age_hours == 72.0
        assert matcher.decay_mode == "exponential"
        assert matcher.weight_floor == 0.4
        assert matcher.use_confidence_weighting is True


if __name__ == "__main__":
    # Run tests manually
    import traceback

    test_classes = [
        TestRecencyWeighting,
        TestConfidenceWeighting,
        TestCombinedWeighting,
        TestConfigurationValidation,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed! ✓")
    sys.exit(0 if failed == 0 else 1)
