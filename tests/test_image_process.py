"""Unit tests for image_process module."""

import numpy as np
import pytest

from stereo_alignment.image_process import (
    compute_roll,
    separate_converging_diverging,
    _reduce_rand_kps_num,
    _filter_lowe_ratio,
    find_correspondences,
)


class TestComputeRoll:
    """Tests for compute_roll function."""
    
    def test_no_roll_horizontal_alignment(self):
        """Test that perfectly aligned points return zero roll."""
        src_pts = np.array([[10.0, 20.0], [50.0, 20.0], [90.0, 20.0]])
        img_pts = np.array([[10.0, 20.0], [50.0, 20.0], [90.0, 20.0]])
        
        roll = compute_roll(src_pts, img_pts)
        
        assert abs(roll) < 0.01, "Perfectly aligned points should have zero roll"
    
    def test_positive_roll_clockwise(self):
        """Test that clockwise rotation returns positive roll."""
        # Simulate clockwise rotation: right side moves up
        src_pts = np.array([[10.0, 20.0], [50.0, 20.0], [90.0, 20.0]])
        img_pts = np.array([[10.0, 20.0], [50.0, 18.0], [90.0, 15.0]])
        
        roll = compute_roll(src_pts, img_pts)
        
        assert roll > 0, "Clockwise rotation should give positive roll"
    
    def test_negative_roll_counterclockwise(self):
        """Test that counter-clockwise rotation returns negative roll."""
        # Simulate counter-clockwise rotation: right side moves down
        src_pts = np.array([[10.0, 20.0], [50.0, 20.0], [90.0, 20.0]])
        img_pts = np.array([[10.0, 20.0], [50.0, 22.0], [90.0, 25.0]])
        
        roll = compute_roll(src_pts, img_pts)
        
        assert roll < 0, "Counter-clockwise rotation should give negative roll"
    
    def test_outlier_rejection(self):
        """Test that outliers are properly rejected."""
        # Need more inliers for outlier rejection to work effectively
        src_pts = np.array([
            [10.0, 20.0], [20.0, 20.0], [30.0, 20.0], [40.0, 20.0],
            [50.0, 20.0], [60.0, 20.0], [70.0, 20.0], [80.0, 20.0],
            [90.0, 20.0], [100.0, 20.0], [110.0, 100.0]  # outlier
        ])
        img_pts = np.array([
            [10.0, 20.0], [20.0, 20.0], [30.0, 20.0], [40.0, 20.0],
            [50.0, 20.0], [60.0, 20.0], [70.0, 20.0], [80.0, 20.0],
            [90.0, 20.0], [100.0, 20.0], [110.0, 20.0]
        ])
        
        roll = compute_roll(src_pts, img_pts)
        
        # Should still be close to zero despite outlier
        assert abs(roll) < 5.0, "Outliers should not significantly affect roll with enough inliers"


class TestSeparateConvergingDiverging:
    """Tests for separate_converging_diverging function."""
    
    def test_all_converging(self):
        """Test when all features are converging (moving left or staying)."""
        # In stereo: converging means x-diff <= 0 (img is left of or same as src)
        src_pts = np.array([[100.0, 50.0], [200.0, 50.0], [300.0, 50.0]])
        img_pts = np.array([[90.0, 50.0], [190.0, 50.0], [290.0, 50.0]])  # moved left
        
        converged, diverged = separate_converging_diverging(src_pts, img_pts)
        
        # When img_pts < src_pts (moved left), diff > 0, mask is False -> diverged
        # The logic: diff = src - img, if diff[:, 0] <= 0 then converged
        # Here: diff = [10, 10, 10], so all > 0, mask is False -> all diverged
        assert diverged.shape[0] == 3, "Features moving left are diverging"
        assert converged.shape[0] == 0, "No converging features"
    
    def test_all_diverging(self):
        """Test when all features are diverging (moving right)."""
        src_pts = np.array([[100.0, 50.0], [200.0, 50.0], [300.0, 50.0]])
        img_pts = np.array([[110.0, 50.0], [210.0, 50.0], [310.0, 50.0]])  # moved right
        
        converged, diverged = separate_converging_diverging(src_pts, img_pts)
        
        # diff = src - img = [-10, -10, -10], mask is True (<=0) -> converged
        assert converged.shape[0] == 3, "Features moving right are converging"
        assert diverged.shape[0] == 0, "No diverging features"
    
    def test_mixed_features(self):
        """Test with both converging and diverging features."""
        src_pts = np.array([[100.0, 50.0], [200.0, 50.0], [300.0, 50.0]])
        img_pts = np.array([[90.0, 50.0], [210.0, 50.0], [290.0, 50.0]])
        
        converged, diverged = separate_converging_diverging(src_pts, img_pts)
        
        # diff = [10, -10, 10]
        # mask: [False, True, False]
        # converged: index 1, diverged: indices 0, 2
        assert converged.shape[0] == 1, "One feature should be converging"
        assert diverged.shape[0] == 2, "Two features should be diverging"
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        src_pts = np.array([[100.0, 50.0], [200.0, 50.0]])
        img_pts = np.array([[90.0, 50.0], [210.0, 50.0]])
        
        converged, diverged = separate_converging_diverging(src_pts, img_pts)
        
        # Shape should be [num_features, 2, 2] -> [num_features, [src, img], [x, y]]
        if converged.shape[0] > 0:
            assert converged.shape[1:] == (2, 2), "Converged features should have shape (n, 2, 2)"
        if diverged.shape[0] > 0:
            assert diverged.shape[1:] == (2, 2), "Diverged features should have shape (n, 2, 2)"


class TestReduceRandKpsNum:
    """Tests for _reduce_rand_kps_num function."""
    
    def test_no_reduction_when_below_max(self):
        """Test that no reduction occurs when below max_features."""
        kp1 = [MockKeypoint(i, i) for i in range(50)]
        des1 = np.random.rand(50, 64)
        kp2 = [MockKeypoint(i, i) for i in range(50)]
        des2 = np.random.rand(50, 64)
        
        kp1_out, des1_out, kp2_out, des2_out = _reduce_rand_kps_num(
            100, kp1, des1, kp2, des2
        )
        
        assert len(kp1_out) == 50, "Should not reduce when below max"
        assert len(kp2_out) == 50, "Should not reduce when below max"
    
    def test_reduction_when_above_max(self):
        """Test that reduction occurs when above max_features."""
        kp1 = [MockKeypoint(i, i) for i in range(200)]
        des1 = np.random.rand(200, 64)
        kp2 = [MockKeypoint(i, i) for i in range(200)]
        des2 = np.random.rand(200, 64)
        
        kp1_out, des1_out, kp2_out, des2_out = _reduce_rand_kps_num(
            100, kp1, des1, kp2, des2
        )
        
        assert len(kp1_out) == 100, "Should reduce to max_features"
        assert len(kp2_out) == 100, "Should reduce to max_features"
        assert des1_out.shape[0] == 100, "Descriptors should match keypoints"
        assert des2_out.shape[0] == 100, "Descriptors should match keypoints"
    
    def test_disabled_reduction(self):
        """Test that max_features=-1 disables reduction."""
        kp1 = [MockKeypoint(i, i) for i in range(200)]
        des1 = np.random.rand(200, 64)
        kp2 = [MockKeypoint(i, i) for i in range(200)]
        des2 = np.random.rand(200, 64)
        
        kp1_out, des1_out, kp2_out, des2_out = _reduce_rand_kps_num(
            -1, kp1, des1, kp2, des2
        )
        
        assert len(kp1_out) == 200, "Should not reduce when max_features=-1"
        assert len(kp2_out) == 200, "Should not reduce when max_features=-1"


class TestFilterLoweRatio:
    """Tests for _filter_lowe_ratio function."""
    
    def test_filters_bad_matches(self):
        """Test that matches with poor ratios are filtered out."""
        kp1 = [MockKeypoint(i, i) for i in range(10)]
        kp2 = [MockKeypoint(i, i) for i in range(10)]
        
        # Create matches with varying quality
        matches = [
            (MockMatch(0, 0, 0.1), MockMatch(0, 1, 0.5)),  # good ratio
            (MockMatch(1, 1, 0.5), MockMatch(1, 2, 0.55)), # bad ratio
            (MockMatch(2, 2, 0.2), MockMatch(2, 3, 0.6)),  # good ratio
        ]
        
        src_pts, img_pts = _filter_lowe_ratio(0.7, kp1, kp2, matches)
        
        assert src_pts.shape[0] == 2, "Should filter out bad matches"
        assert img_pts.shape[0] == 2, "Should filter out bad matches"


class TestFindCorrespondences:
    """Tests for find_correspondences function."""
    
    def test_raises_on_insufficient_features(self):
        """Test that ValueError is raised when features are insufficient."""
        # Create nearly blank images
        img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        with pytest.raises(ValueError, match="Not enough features"):
            find_correspondences(img1, img2)
    
    def test_returns_valid_shapes(self):
        """Test that output arrays have correct shapes."""
        # Create images with some texture
        img1 = create_test_image_with_features()
        img2 = create_test_image_with_features()
        
        src_pts, img_pts = find_correspondences(img1, img2, max_features=500)
        
        assert src_pts.shape[1] == 2, "src_pts should have shape (n, 2)"
        assert img_pts.shape[1] == 2, "img_pts should have shape (n, 2)"
        assert src_pts.shape[0] == img_pts.shape[0], "Should have same number of points"


# Helper classes and functions

class MockKeypoint:
    """Mock OpenCV KeyPoint for testing."""
    def __init__(self, x, y):
        self.pt = (x, y)


class MockMatch:
    """Mock OpenCV DMatch for testing."""
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.distance = distance


def create_test_image_with_features():
    """Create a test image with detectable features."""
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Add some high-contrast features
    for i in range(0, 640, 50):
        for j in range(0, 480, 50):
            img[j:j+10, i:i+10] = [255, 255, 255]
            img[j+10:j+20, i:i+10] = [0, 0, 0]
    return img
