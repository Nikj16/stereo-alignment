"""Unit tests for CLI module."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest

from stereo_alignment.cli import setup_logging, load_image, main


class TestSetupLogging:
    """Tests for setup_logging function."""
    
    def test_sets_debug_level_when_verbose(self):
        """Test that DEBUG level is set when verbose=True."""
        import logging
        
        # Clear any existing handlers
        logger = logging.getLogger()
        logger.handlers.clear()
        
        setup_logging(verbose=True)
        
        # Check the root logger level after basicConfig
        assert logger.level == logging.DEBUG
    
    def test_sets_info_level_when_not_verbose(self):
        """Test that INFO level is set when verbose=False."""
        import logging
        
        # Clear any existing handlers
        logger = logging.getLogger()
        logger.handlers.clear()
        
        setup_logging(verbose=False)
        
        # Check the root logger level after basicConfig
        assert logger.level == logging.INFO


class TestLoadImage:
    """Tests for load_image function."""
    
    def test_loads_valid_image(self):
        """Test that valid image is loaded correctly."""
        # Create temporary test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(tmp.name, img)
            tmp_path = tmp.name
        
        try:
            result = load_image(tmp_path)
            
            assert isinstance(result, np.ndarray), "Should return ndarray"
            assert result.shape == (100, 100, 3), "Should have correct shape"
        finally:
            Path(tmp_path).unlink()
    
    def test_raises_on_invalid_path(self):
        """Test that ValueError is raised for invalid path."""
        with pytest.raises(ValueError, match="Failed to load image"):
            load_image("/nonexistent/path/image.jpg")
    
    def test_converts_bgr_to_rgb(self):
        """Test that image is converted from BGR to RGB."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            # Use PNG to avoid JPEG compression artifacts
            # Create image with known BGR values
            img_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
            img_bgr[:, :] = [255, 0, 0]  # Blue in BGR
            cv2.imwrite(tmp.name, img_bgr)
            tmp_path = tmp.name
        
        try:
            result = load_image(tmp_path)
            
            # Should be red in RGB (allowing small tolerance for encoding)
            assert result[0, 0, 0] < 10, "Red channel should be near 0"
            assert result[0, 0, 1] < 10, "Green channel should be near 0"
            assert result[0, 0, 2] > 245, "Blue channel should be near 255"
        finally:
            Path(tmp_path).unlink()


class TestMain:
    """Tests for main function."""
    
    @patch('stereo_alignment.cli.cv2.waitKey')
    @patch('stereo_alignment.cli.cv2.imshow')
    @patch('stereo_alignment.cli.cv2.namedWindow')
    @patch('stereo_alignment.cli.run_stereo_alignment_metric')
    def test_successful_execution(self, mock_run, mock_window, mock_imshow, mock_waitkey):
        """Test successful execution of main function."""
        # Create temporary test images
        img1_path = create_temp_test_image()
        img2_path = create_temp_test_image()
        
        try:
            # Mock the stereo alignment function
            mock_run.return_value = (
                2.5,  # pitch
                0.3,  # roll
                np.zeros((960, 1280, 3), dtype=np.uint8),  # labeled image
                np.array([2.3, 2.5, 2.7])  # offsets
            )
            
            with patch('sys.argv', ['cli.py', img1_path, img2_path]):
                result = main()
            
            assert result == 0, "Should return 0 on success"
            mock_run.assert_called_once()
        finally:
            Path(img1_path).unlink()
            Path(img2_path).unlink()
    
    @patch('stereo_alignment.cli.cv2.waitKey')
    @patch('stereo_alignment.cli.cv2.imshow')
    @patch('stereo_alignment.cli.cv2.namedWindow')
    @patch('stereo_alignment.cli.run_stereo_alignment_metric')
    def test_saves_output_when_requested(self, mock_run, mock_window, mock_imshow, mock_waitkey):
        """Test that output is saved when --save-output is provided."""
        img1_path = create_temp_test_image()
        img2_path = create_temp_test_image()
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            mock_run.return_value = (
                2.5, 0.3,
                np.zeros((960, 1280, 3), dtype=np.uint8),
                np.array([2.3, 2.5, 2.7])
            )
            
            with patch('sys.argv', ['cli.py', img1_path, img2_path, '--save-output', output_path]):
                result = main()
            
            assert result == 0, "Should return 0 on success"
            assert Path(output_path).exists(), "Output file should be created"
        finally:
            Path(img1_path).unlink()
            Path(img2_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()
    
    def test_returns_error_on_mismatched_dimensions(self):
        """Test that error is returned for mismatched image dimensions."""
        img1_path = create_temp_test_image(size=(100, 100))
        img2_path = create_temp_test_image(size=(200, 200))
        
        try:
            with patch('sys.argv', ['cli.py', img1_path, img2_path]):
                result = main()
            
            assert result == 1, "Should return 1 on error"
        finally:
            Path(img1_path).unlink()
            Path(img2_path).unlink()
    
    def test_returns_error_on_exception(self):
        """Test that error is returned when exception occurs."""
        with patch('sys.argv', ['cli.py', '/nonexistent1.jpg', '/nonexistent2.jpg']):
            result = main()
        
        assert result == 1, "Should return 1 on error"


# Helper functions

def create_temp_test_image(size=(100, 100)):
    """Create a temporary test image file."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        # Add some features
        for i in range(0, size[1], 20):
            for j in range(0, size[0], 20):
                img[j:j+10, i:i+10] = [255, 255, 255]
        cv2.imwrite(tmp.name, img)
        return tmp.name
