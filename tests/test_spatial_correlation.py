"""Unit tests for spatial correlation features."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identity_service import calculate_iou, calculate_correlation_score


class TestCalculateIoU:
    """Test IoU (Intersection over Union) calculation."""
    
    def test_identical_boxes(self):
        """Test IoU of identical boxes returns 1.0."""
        box = [10, 10, 50, 50]
        iou = calculate_iou(box, box)
        assert iou == 1.0
    
    def test_non_overlapping_boxes(self):
        """Test IoU of non-overlapping boxes returns 0.0."""
        box_a = [0, 0, 10, 10]
        box_b = [20, 20, 30, 30]
        iou = calculate_iou(box_a, box_b)
        assert iou == 0.0
    
    def test_partially_overlapping_boxes(self):
        """Test IoU of partially overlapping boxes."""
        box_a = [0, 0, 20, 20]  # Area: 400
        box_b = [10, 10, 30, 30]  # Area: 400
        # Intersection: 10x10 = 100
        # Union: 400 + 400 - 100 = 700
        # IoU: 100/700 ≈ 0.143
        iou = calculate_iou(box_a, box_b)
        assert 0.14 < iou < 0.15
    
    def test_one_box_inside_another(self):
        """Test IoU when one box is completely inside another."""
        box_a = [0, 0, 100, 100]  # Area: 10000
        box_b = [25, 25, 75, 75]  # Area: 2500
        # Intersection: 2500 (box_b is fully inside box_a)
        # Union: 10000 + 2500 - 2500 = 10000
        # IoU: 2500/10000 = 0.25
        iou = calculate_iou(box_a, box_b)
        assert iou == 0.25
    
    def test_none_box_returns_zero(self):
        """Test that None boxes return 0.0."""
        box = [10, 10, 50, 50]
        assert calculate_iou(None, box) == 0.0
        assert calculate_iou(box, None) == 0.0
        assert calculate_iou(None, None) == 0.0
    
    def test_empty_list_returns_zero(self):
        """Test that empty lists return 0.0."""
        box = [10, 10, 50, 50]
        assert calculate_iou([], box) == 0.0
        assert calculate_iou(box, []) == 0.0
    
    def test_invalid_box_length_returns_zero(self):
        """Test that boxes with wrong number of elements return 0.0."""
        box = [10, 10, 50, 50]
        assert calculate_iou([10, 10], box) == 0.0
        assert calculate_iou(box, [10, 10, 50]) == 0.0
        assert calculate_iou([10, 10, 50, 50, 60], box) == 0.0
    
    def test_invalid_values_return_zero(self):
        """Test that invalid box values return 0.0."""
        box = [10, 10, 50, 50]
        # Box with invalid format (x_max < x_min)
        invalid_box = [50, 10, 10, 50]
        # Should handle gracefully
        result = calculate_iou(box, invalid_box)
        assert result == 0.0
    
    def test_touching_boxes_edge_case(self):
        """Test boxes that touch at edges (no overlap)."""
        box_a = [0, 0, 10, 10]
        box_b = [10, 0, 20, 10]  # Touching on right edge
        iou = calculate_iou(box_a, box_b)
        # No intersection area (edges touching doesn't count)
        assert iou == 0.0
    
    def test_high_overlap(self):
        """Test boxes with high overlap."""
        box_a = [0, 0, 100, 100]  # Area: 10000
        box_b = [10, 10, 100, 100]  # Area: 8100
        # Intersection: 90x90 = 8100
        # Union: 10000 + 8100 - 8100 = 10000
        # IoU: 8100/10000 = 0.81
        iou = calculate_iou(box_a, box_b)
        assert 0.80 < iou < 0.82


class TestCalculateCorrelationScore:
    """Test composite correlation score calculation."""
    
    def test_perfect_temporal_and_spatial(self):
        """Test perfect correlation (instant + perfect overlap)."""
        score = calculate_correlation_score(
            temporal_delta=0.0,
            iou=1.0,
            temporal_weight=0.6,
            spatial_weight=0.4,
            max_temporal_window=2.0
        )
        assert score == 1.0
    
    def test_temporal_only_fallback(self):
        """Test fallback to temporal-only when iou=0.0."""
        # Perfect temporal, no spatial data
        score = calculate_correlation_score(
            temporal_delta=0.0,
            iou=0.0,
            temporal_weight=0.6,
            spatial_weight=0.4,
            max_temporal_window=2.0
        )
        assert score == 1.0  # Should return temporal score (1.0)
        
        # Half temporal window, no spatial data
        score = calculate_correlation_score(
            temporal_delta=1.0,
            iou=0.0,
            temporal_weight=0.6,
            spatial_weight=0.4,
            max_temporal_window=2.0
        )
        assert score == 0.5  # Should return temporal score (0.5)
    
    def test_old_detection_low_score(self):
        """Test that old detections get low scores."""
        score = calculate_correlation_score(
            temporal_delta=2.0,  # At the edge of window
            iou=0.0,
            temporal_weight=0.6,
            spatial_weight=0.4,
            max_temporal_window=2.0
        )
        assert score == 0.0
    
    def test_composite_score_calculation(self):
        """Test composite score with both temporal and spatial signals."""
        # Mid temporal (1 sec = 0.5 score) + good spatial (0.8 IoU)
        # Expected: 0.6 * 0.5 + 0.4 * 0.8 = 0.3 + 0.32 = 0.62
        score = calculate_correlation_score(
            temporal_delta=1.0,
            iou=0.8,
            temporal_weight=0.6,
            spatial_weight=0.4,
            max_temporal_window=2.0
        )
        assert 0.61 < score < 0.63
    
    def test_spatial_improves_old_temporal(self):
        """Test that spatial data improves correlation for older detections."""
        # Old temporal (1.5 sec = 0.25 score) but perfect spatial
        # Expected: 0.6 * 0.25 + 0.4 * 1.0 = 0.15 + 0.4 = 0.55
        score = calculate_correlation_score(
            temporal_delta=1.5,
            iou=1.0,
            temporal_weight=0.6,
            spatial_weight=0.4,
            max_temporal_window=2.0
        )
        assert 0.54 < score < 0.56
    
    def test_weight_normalization(self):
        """Test that weights are normalized if they don't sum to 1.0."""
        # Weights sum to 2.0, should be normalized to 0.5 each
        score = calculate_correlation_score(
            temporal_delta=0.0,  # Temporal score = 1.0
            iou=0.0,  # No spatial (should use temporal only)
            temporal_weight=1.0,
            spatial_weight=1.0,
            max_temporal_window=2.0
        )
        # With iou=0.0, should fall back to temporal-only
        assert score == 1.0
    
    def test_score_clamping(self):
        """Test that scores are clamped to [0.0, 1.0]."""
        # Beyond temporal window
        score = calculate_correlation_score(
            temporal_delta=3.0,
            iou=0.0,
            temporal_weight=0.6,
            spatial_weight=0.4,
            max_temporal_window=2.0
        )
        assert 0.0 <= score <= 1.0
        
        # Perfect conditions
        score = calculate_correlation_score(
            temporal_delta=0.0,
            iou=1.0,
            temporal_weight=0.6,
            spatial_weight=0.4,
            max_temporal_window=2.0
        )
        assert 0.0 <= score <= 1.0
    
    def test_different_weights(self):
        """Test with different weight configurations."""
        # Heavy temporal weight
        score_temporal = calculate_correlation_score(
            temporal_delta=0.5,
            iou=0.5,
            temporal_weight=0.9,
            spatial_weight=0.1,
            max_temporal_window=2.0
        )
        
        # Heavy spatial weight
        score_spatial = calculate_correlation_score(
            temporal_delta=0.5,
            iou=0.5,
            temporal_weight=0.1,
            spatial_weight=0.9,
            max_temporal_window=2.0
        )
        
        # Both should be different
        assert score_temporal != score_spatial


class TestMultiPersonScenario:
    """Test that correct person is selected in multi-person scenes."""
    
    def test_select_best_composite_score(self):
        """Test that person with best composite score is selected."""
        # Scenario: Two persons detected
        # Person A: Recent (0.1s ago), moderate overlap (0.5 IoU)
        # Person B: Older (1.0s ago), perfect overlap (1.0 IoU)
        
        score_a = calculate_correlation_score(
            temporal_delta=0.1,
            iou=0.5,
            temporal_weight=0.6,
            spatial_weight=0.4,
            max_temporal_window=2.0
        )
        # Expected: 0.6 * 0.95 + 0.4 * 0.5 = 0.57 + 0.2 = 0.77
        
        score_b = calculate_correlation_score(
            temporal_delta=1.0,
            iou=1.0,
            temporal_weight=0.6,
            spatial_weight=0.4,
            max_temporal_window=2.0
        )
        # Expected: 0.6 * 0.5 + 0.4 * 1.0 = 0.3 + 0.4 = 0.7
        
        # Person A should win (more recent and decent overlap)
        assert score_a > score_b
    
    def test_spatial_helps_distinguish_persons(self):
        """Test that spatial data helps distinguish between two persons at similar times."""
        # Both persons detected at same time, but different positions
        
        score_good_overlap = calculate_correlation_score(
            temporal_delta=0.5,
            iou=0.9,  # Good overlap with snapshot
            temporal_weight=0.6,
            spatial_weight=0.4,
            max_temporal_window=2.0
        )
        
        score_bad_overlap = calculate_correlation_score(
            temporal_delta=0.5,
            iou=0.1,  # Poor overlap with snapshot
            temporal_weight=0.6,
            spatial_weight=0.4,
            max_temporal_window=2.0
        )
        
        # Good overlap should win significantly
        assert score_good_overlap > score_bad_overlap
        # Difference should be significant (spatial weight * IoU difference)
        assert (score_good_overlap - score_bad_overlap) > 0.3


class TestBackwardCompatibility:
    """Test backward compatibility with queue entries without box field."""
    
    def test_missing_box_field_handled(self):
        """Test that missing 'box' field in detection doesn't crash."""
        # Old-style detection record without 'box' field
        old_detection = {
            "person_id": "alice",
            "timestamp": 1000.0,
            "event_id": "test123",
            "zones": ["front_yard"],
            "confidence": 0.95
        }
        
        # Should handle gracefully - box will be None
        box = old_detection.get("box")
        assert box is None
        
        # IoU calculation with None should return 0.0
        other_box = [10, 10, 50, 50]
        iou = calculate_iou(box, other_box)
        assert iou == 0.0
        
        # Correlation score should fall back to temporal-only
        score = calculate_correlation_score(
            temporal_delta=0.5,
            iou=0.0,  # No spatial data
            temporal_weight=0.6,
            spatial_weight=0.4,
            max_temporal_window=2.0
        )
        # Should return temporal score only
        expected_temporal = 1.0 - (0.5 / 2.0)
        assert score == expected_temporal


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
