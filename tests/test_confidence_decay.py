"""Unit tests for confidence decay functionality."""
import pytest
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identity_service import calculate_effective_confidence


class TestCalculateEffectiveConfidence:
    """Test confidence decay calculation."""
    
    def test_fresh_detection_no_decay(self):
        """Test that fresh detection (None timestamp) returns base confidence unchanged."""
        base_confidence = 0.87
        effective, minutes = calculate_effective_confidence(base_confidence, None)
        
        assert effective == base_confidence
        assert minutes == 0.0
    
    def test_no_decay_within_start_window(self):
        """Test no decay occurs within decay_start_minutes."""
        base_confidence = 0.87
        # Last seen 3 minutes ago (within 5 minute start window)
        last_seen = time.time() - (3 * 60)
        
        effective, minutes = calculate_effective_confidence(
            base_confidence, 
            last_seen,
            decay_start_minutes=5.0
        )
        
        assert effective == base_confidence
        assert 2.9 < minutes < 3.1  # approximately 3 minutes
    
    def test_linear_decay_after_start(self):
        """Test correct linear decay after decay_start_minutes."""
        base_confidence = 0.80
        # Last seen 8 minutes ago (3 minutes into decay phase with 5 minute start)
        last_seen = time.time() - (8 * 60)
        
        effective, minutes = calculate_effective_confidence(
            base_confidence,
            last_seen,
            decay_start_minutes=5.0,
            decay_rate_per_minute=0.10
        )
        
        # After 5 minutes no decay, then 3 minutes of 0.10 decay = 0.30 reduction
        expected = 0.80 - (3 * 0.10)  # 0.50
        assert abs(effective - expected) < 0.01
        assert 7.9 < minutes < 8.1
    
    def test_zero_confidence_at_full_decay(self):
        """Test confidence reaches zero at full_decay_minutes."""
        base_confidence = 0.87
        # Last seen 15 minutes ago (at full decay threshold)
        last_seen = time.time() - (15 * 60)
        
        effective, minutes = calculate_effective_confidence(
            base_confidence,
            last_seen,
            decay_start_minutes=5.0,
            full_decay_minutes=15.0
        )
        
        assert effective == 0.0
        assert 14.9 < minutes < 15.1
    
    def test_zero_confidence_after_full_decay(self):
        """Test confidence stays zero after full_decay_minutes."""
        base_confidence = 0.95
        # Last seen 20 minutes ago (past full decay)
        last_seen = time.time() - (20 * 60)
        
        effective, minutes = calculate_effective_confidence(
            base_confidence,
            last_seen,
            decay_start_minutes=5.0,
            full_decay_minutes=15.0
        )
        
        assert effective == 0.0
        assert 19.9 < minutes < 20.1
    
    def test_custom_decay_rate(self):
        """Test that custom decay rates work correctly."""
        base_confidence = 0.90
        # Last seen 7 minutes ago (2 minutes into decay with 5 minute start)
        last_seen = time.time() - (7 * 60)
        
        # Higher decay rate
        effective, _ = calculate_effective_confidence(
            base_confidence,
            last_seen,
            decay_start_minutes=5.0,
            decay_rate_per_minute=0.20  # Double rate
        )
        
        # 2 minutes of 0.20 decay = 0.40 reduction
        expected = 0.90 - (2 * 0.20)  # 0.50
        assert abs(effective - expected) < 0.01
    
    def test_custom_start_time(self):
        """Test that custom decay_start_minutes works correctly."""
        base_confidence = 0.85
        # Last seen 8 minutes ago
        last_seen = time.time() - (8 * 60)
        
        # Longer start window (10 minutes)
        effective, _ = calculate_effective_confidence(
            base_confidence,
            last_seen,
            decay_start_minutes=10.0,
            decay_rate_per_minute=0.10
        )
        
        # Should still be 8 minutes, but no decay yet (within 10 minute window)
        assert effective == base_confidence
    
    def test_custom_full_decay_time(self):
        """Test that custom full_decay_minutes works correctly."""
        base_confidence = 0.90
        # Last seen 10 minutes ago
        last_seen = time.time() - (10 * 60)
        
        # Shorter full decay window
        effective, _ = calculate_effective_confidence(
            base_confidence,
            last_seen,
            decay_start_minutes=2.0,
            full_decay_minutes=10.0
        )
        
        # Should hit full decay at 10 minutes
        assert effective == 0.0
    
    def test_decay_never_goes_negative(self):
        """Test that confidence never goes below 0.0."""
        base_confidence = 0.50
        # Last seen 12 minutes ago with high decay rate
        last_seen = time.time() - (12 * 60)
        
        effective, _ = calculate_effective_confidence(
            base_confidence,
            last_seen,
            decay_start_minutes=5.0,
            decay_rate_per_minute=0.20  # Would decay -0.90 total
        )
        
        assert effective >= 0.0
        assert effective <= base_confidence
    
    def test_partial_decay(self):
        """Test decay in middle of decay range."""
        base_confidence = 1.0
        # Last seen 10 minutes ago
        last_seen = time.time() - (10 * 60)
        
        effective, minutes = calculate_effective_confidence(
            base_confidence,
            last_seen,
            decay_start_minutes=5.0,
            decay_rate_per_minute=0.10,
            full_decay_minutes=15.0
        )
        
        # 5 minutes of decay at 0.10 per minute = 0.50 reduction
        expected = 1.0 - (5 * 0.10)  # 0.50
        assert abs(effective - expected) < 0.01
        assert 9.9 < minutes < 10.1


class TestPersonLastSeenIntegration:
    """Test that person_last_seen dictionary is updated correctly."""
    
    def test_person_last_seen_exists(self):
        """Test that person_last_seen dictionary exists in the module."""
        from identity_service import person_last_seen
        
        assert isinstance(person_last_seen, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
