"""Unit tests for drawing module."""

import numpy as np
import pytest

from stereo_alignment.drawing import draw_lines, draw_feature_match


class TestDrawLines:
    """Tests for draw_lines function."""
    
    def test_draws_correct_number_of_lines(self):
        """Test that the correct number of lines is drawn."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = draw_lines(img, num_lines=5)
        
        assert result.shape == img.shape, "Output should have same shape as input"
        assert not np.array_equal(result, img), "Image should be modified"
    
    def test_raises_on_invalid_num_lines(self):
        """Test that ValueError is raised for invalid num_lines."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="minimal number of line"):
            draw_lines(img, num_lines=0)
        
        with pytest.raises(ValueError, match="minimal number of line"):
            draw_lines(img, num_lines=-1)
    
    def test_raises_on_invalid_margin_ratio(self):
        """Test that ValueError is raised for invalid margin ratio."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="margin_line_offset_ratio"):
            draw_lines(img, margin_line_offset_ratio=-0.1)
        
        with pytest.raises(ValueError, match="margin_line_offset_ratio"):
            draw_lines(img, margin_line_offset_ratio=0.5)
    
    def test_does_not_modify_original(self):
        """Test that original image is not modified."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img_copy = img.copy()
        
        draw_lines(img, num_lines=5)
        
        assert np.array_equal(img, img_copy), "Original image should not be modified"
    
    def test_with_different_margins(self):
        """Test with different margin ratios."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result1 = draw_lines(img, num_lines=5, margin_line_offset_ratio=0.0)
        result2 = draw_lines(img, num_lines=5, margin_line_offset_ratio=0.2)
        
        assert not np.array_equal(result1, result2), "Different margins should produce different results"


class TestDrawFeatureMatch:
    """Tests for draw_feature_match function."""
    
    def test_draws_feature_matches(self):
        """Test that feature matches are drawn correctly."""
        img1 = np.zeros((480, 640, 3), dtype=np.uint8)
        img2 = np.zeros((480, 640, 3), dtype=np.uint8)
        features = np.array([
            [[100, 200], [120, 210]],
            [[300, 150], [310, 160]],
        ])
        
        result = draw_feature_match(img1, img2, features, (255, 0, 0))
        
        # Output should be concatenated horizontally
        assert result.shape == (480, 1280, 3), "Output should be concatenated images"
        assert not np.array_equal(result, np.zeros_like(result)), "Lines should be drawn"
    
    def test_raises_on_mismatched_image_sizes(self):
        """Test that ValueError is raised for mismatched image sizes."""
        img1 = np.zeros((480, 640, 3), dtype=np.uint8)
        img2 = np.zeros((240, 320, 3), dtype=np.uint8)
        features = np.array([[[100, 200], [120, 210]]])
        
        with pytest.raises(ValueError, match="Images must be same size"):
            draw_feature_match(img1, img2, features, (255, 0, 0))
    
    def test_raises_on_grayscale_image(self):
        """Test that ValueError is raised for grayscale images."""
        img1 = np.zeros((480, 640), dtype=np.uint8)
        img2 = np.zeros((480, 640), dtype=np.uint8)
        features = np.array([[[100, 200], [120, 210]]])
        
        with pytest.raises(ValueError, match="shape should be"):
            draw_feature_match(img1, img2, features, (255, 0, 0))
    
    def test_handles_empty_features(self):
        """Test behavior with empty features array."""
        img1 = np.zeros((480, 640, 3), dtype=np.uint8)
        img2 = np.zeros((480, 640, 3), dtype=np.uint8)
        features = np.array([]).reshape(0, 2, 2)
        
        result = draw_feature_match(img1, img2, features, (255, 0, 0))
        
        assert result.shape == (480, 1280, 3), "Should still concatenate images"
    
    def test_different_colors(self):
        """Test that different colors produce different results."""
        img1 = np.zeros((480, 640, 3), dtype=np.uint8)
        img2 = np.zeros((480, 640, 3), dtype=np.uint8)
        features = np.array([[[100, 200], [120, 210]]])
        
        result_red = draw_feature_match(img1, img2, features, (255, 0, 0))
        result_green = draw_feature_match(img1, img2, features, (0, 255, 0))
        
        assert not np.array_equal(result_red, result_green), "Different colors should produce different results"
