"""Unit tests for core module."""

import numpy as np
import pytest

from stereo_alignment.core import (
    _compute_features_and_offsets,
    _create_labeled_stereo_images,
    run_stereo_alignment_metric,
)


class TestComputeFeaturesAndOffsets:
    """Tests for _compute_features_and_offsets function."""
    
    def test_raises_on_insufficient_features(self):
        """Test that ValueError is raised when matching fails."""
        img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        with pytest.raises(ValueError):
            _compute_features_and_offsets(img1, img2, max_features=1000, min_features=10)
    
    def test_returns_correct_tuple_length(self):
        """Test that function returns tuple with 5 elements."""
        img1 = create_feature_rich_image()
        img2 = create_feature_rich_image()
        
        result = _compute_features_and_offsets(img1, img2, max_features=1000, min_features=10)
        
        assert len(result) == 5, "Should return 5 elements"
    
    def test_output_types(self):
        """Test that outputs have correct types."""
        img1 = create_feature_rich_image()
        img2 = create_feature_rich_image()
        
        diverged_img, converged_img, pitch, roll, offsets = _compute_features_and_offsets(
            img1, img2, max_features=1000, min_features=10
        )
        
        assert isinstance(diverged_img, np.ndarray), "diverged_img should be ndarray"
        assert isinstance(converged_img, np.ndarray), "converged_img should be ndarray"
        assert isinstance(pitch, float), "pitch should be float"
        assert isinstance(roll, float), "roll should be float"
        assert isinstance(offsets, np.ndarray), "offsets should be ndarray"


class TestCreateLabeledStereoImages:
    """Tests for _create_labeled_stereo_images function."""
    
    def test_creates_labeled_image(self):
        """Test that labeled image is created correctly."""
        img1 = np.zeros((480, 640, 3), dtype=np.uint8)
        img2 = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create dummy concatenated images
        diverged_img = np.hstack([img1, img2])
        converged_img = np.hstack([img1, img2])
        
        pitch_offset = 2.5
        roll_metric = 0.3
        pixel_y_offsets = np.array([2.3, 2.5, 2.7])
        
        result = _create_labeled_stereo_images(
            diverged_img, converged_img, pitch_offset, roll_metric, pixel_y_offsets
        )
        
        # Result should be vertically stacked
        assert result.shape == (960, 1280, 3), "Should be vertically stacked"
        assert not np.array_equal(result, np.zeros_like(result)), "Should have content"
    
    def test_handles_different_metrics(self):
        """Test with different metric values."""
        img1 = np.zeros((480, 640, 3), dtype=np.uint8)
        img2 = np.zeros((480, 640, 3), dtype=np.uint8)
        diverged_img = np.hstack([img1, img2])
        converged_img = np.hstack([img1, img2])
        pixel_y_offsets = np.array([1.0, 2.0, 3.0])
        
        result1 = _create_labeled_stereo_images(
            diverged_img, converged_img, 1.0, 0.1, pixel_y_offsets
        )
        result2 = _create_labeled_stereo_images(
            diverged_img, converged_img, 5.0, 2.0, pixel_y_offsets
        )
        
        assert not np.array_equal(result1, result2), "Different metrics should produce different labels"


class TestRunStereoAlignmentMetric:
    """Tests for run_stereo_alignment_metric function."""
    
    def test_returns_correct_tuple_length(self):
        """Test that function returns tuple with 4 elements."""
        img1 = create_feature_rich_image()
        img2 = create_feature_rich_image()
        
        result = run_stereo_alignment_metric(img1, img2, max_features=1000)
        
        assert len(result) == 4, "Should return 4 elements"
    
    def test_output_types(self):
        """Test that outputs have correct types."""
        img1 = create_feature_rich_image()
        img2 = create_feature_rich_image()
        
        pitch, roll, labeled_img, offsets = run_stereo_alignment_metric(
            img1, img2, max_features=1000
        )
        
        assert isinstance(pitch, float), "pitch should be float"
        assert isinstance(roll, float), "roll should be float"
        assert isinstance(labeled_img, np.ndarray), "labeled_img should be ndarray"
        assert isinstance(offsets, np.ndarray), "offsets should be ndarray"
    
    def test_raises_on_insufficient_features(self):
        """Test that ValueError is raised for blank images."""
        img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        with pytest.raises(ValueError):
            run_stereo_alignment_metric(img1, img2, max_features=1000)
    
    def test_labeled_image_dimensions(self):
        """Test that labeled image has correct dimensions."""
        img1 = create_feature_rich_image()
        img2 = create_feature_rich_image()
        
        _, _, labeled_img, _ = run_stereo_alignment_metric(img1, img2, max_features=1000)
        
        # Should be vertically stacked, horizontally concatenated images
        expected_height = img1.shape[0] * 2
        expected_width = img1.shape[1] * 2
        
        assert labeled_img.shape[0] == expected_height, "Height should be doubled"
        assert labeled_img.shape[1] == expected_width, "Width should be doubled"


# Helper functions

def create_feature_rich_image():
    """Create an image with many detectable features."""
    img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    
    # Add checkerboard pattern for strong features
    square_size = 40
    for i in range(0, 640, square_size):
        for j in range(0, 480, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                img[j:j+square_size, i:i+square_size] = [255, 255, 255]
            else:
                img[j:j+square_size, i:i+square_size] = [0, 0, 0]
    
    return img
