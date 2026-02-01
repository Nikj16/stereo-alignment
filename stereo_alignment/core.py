"""Code for generating stereo alignment metrics and artifacts.

This file should have limited public functions, that accept and
return only base python / numpy / internal calibration types.

i.e. no ros!
"""

import logging
from typing import Tuple

import cv2
import numpy as np

from stereo_alignment.drawing import draw_feature_match, draw_lines
from stereo_alignment.image_process import compute_roll, find_correspondences, separate_converging_diverging


def _compute_features_and_offsets(
    img_ref: np.ndarray,
    img_follow: np.ndarray,
    max_features: int,
    min_features: int = 10,
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Process stereo alignment and matching feature and measure alignment metric.

    Args:
        img_ref: reference image
        img_follow: follow image
        max_features: maximum number of features kept for knnMatch. Uniform sampling with fixed seed
            Note: max_features disabled when set at default value of -1
        min_features: number of minimum features to compute metrics on

    Returns:
        diverged_feature_img: diverged features image
        converged_feature_img: converged features image
        pitch_offset: value of pitch_offset, to write on image
        roll_metric: value of roll_metric, to write on image
        pixel_y_offsets: pixel y offsets without outliers

    Raises:
        ValueError: There is a failure detecting or matching suffcient features.

    """
    src_pts, img_pts = find_correspondences(img_ref, img_follow, max_features=max_features)
    # Check if valid data
    if img_pts is None:
        s = "No features detected"
        logging.critical(s)
        raise ValueError(s)

    num_features_matched = src_pts.shape[0]
    logging.info(f"Got {num_features_matched} matched features")
    if num_features_matched < min_features:
        s = f"Could not match sufficient features, expected >{min_features}, got {num_features_matched}"
        logging.critical(s)
        raise ValueError(s)

    converged_features, diverged_features = separate_converging_diverging(src_pts, img_pts)
    converged_features_img = draw_feature_match(img_ref, img_follow, converged_features, (255, 0, 0))
    diverged_features_img = draw_feature_match(img_ref, img_follow, diverged_features, (0, 255, 0))

    roll_metric = compute_roll(src_pts, img_pts)
    feature_pixel_difference = src_pts - img_pts
    height_index = 1
    pixel_y_diff = feature_pixel_difference[:, height_index]
    # use median to throw away outliers
    pitch_offset = float(np.median(pixel_y_diff))

    # do some extra outlier rejection on the y-pixel difference
    indices_within_one_std = abs(pixel_y_diff - np.mean(pixel_y_diff)) < 1 * np.std(pixel_y_diff)
    pixel_y_offsets = pixel_y_diff[indices_within_one_std]

    return diverged_features_img, converged_features_img, pitch_offset, roll_metric, pixel_y_offsets


def _create_labeled_stereo_images(
    diverged_features_img: np.ndarray,
    converged_features_img: np.ndarray,
    pitch_offset: float,
    roll_metric: float,
    pixel_y_offsets: np.ndarray,
) -> np.ndarray:
    """Label the stereo images with the pitch and roll metric.

    Labels the two input images with identifiers, pitch_offset and roll_metric and vertically stacks them into a
    single output image.

    Args:
        diverged_features_img: diverged features image
        converged_features_img: converged features image
        pitch_offset: value of pitch_offset, to write on image
        roll_metric: value of roll_metric, to write on image
        pixel_y_offsets: pixel y offset without outliers

    Returns: vertically stacked images with labels and info

    """
    font_name = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3
    font_color = (255, 0, 0)  # red

    img_height = diverged_features_img.shape[0]
    top_corner = (50, 50)
    bottom_corner = np.array([50, img_height - 50], dtype=np.int32)
    up_one_line = np.array([0, -75], dtype=np.int32)
    pixels_y_offset_mean = np.mean(pixel_y_offsets)
    pixels_y_offset_absolute_mean = np.mean(np.abs(pixel_y_offsets))
    converged_features_img = draw_lines(converged_features_img, 10)
    diverged_features_img = draw_lines(diverged_features_img, 10)
    cv2.putText(
        diverged_features_img,
        "Diverged features",
        top_corner,
        font_name,
        fontScale=font_scale,
        color=font_color,
        thickness=font_thickness,
    )
    cv2.putText(
        converged_features_img,
        "Converged features",
        top_corner,
        font_name,
        fontScale=font_scale,
        color=font_color,
        thickness=font_thickness,
    )
    cv2.putText(
        diverged_features_img,
        f"Median pitch offset(px): {pitch_offset:.4f}",
        bottom_corner,
        font_name,
        fontScale=font_scale,
        color=font_color,
        thickness=font_thickness,
    )
    cv2.putText(
        converged_features_img,
        f"Roll metric: {roll_metric:.4f}",
        bottom_corner,
        font_name,
        fontScale=font_scale,
        color=font_color,
        thickness=font_thickness,
    )
    cv2.putText(
        diverged_features_img,
        f"Mean pitch offset(px): {pixels_y_offset_mean:.4f}",
        bottom_corner + up_one_line,
        font_name,
        fontScale=font_scale,
        color=font_color,
        thickness=font_thickness,
    )
    cv2.putText(
        diverged_features_img,
        f"Mean absolute pitch offset(px): {pixels_y_offset_absolute_mean:.4f}",
        bottom_corner + 2 * up_one_line,
        font_name,
        fontScale=font_scale,
        color=font_color,
        thickness=font_thickness,
    )
    img_concat = cv2.vconcat((diverged_features_img, converged_features_img))
    return img_concat


def run_stereo_alignment_metric(
    ref_img: np.ndarray,
    follow_image: np.ndarray,
    max_features: int = 10000,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Process stereo alignment and save the images of stereo alignment.

    Args:
        ref_img: reference image, rgb
        follow_img: follow image, rgb
        max_features: maximum number of features kept for knnMatch. Uniform sampling with fixed seed
            Note: max_features disabled when set at default value of -1

    Returns:
        pitch_offset: value of pitch_offset
        roll_metric: value of roll_metric
        labeled_image: labeled image
        pitch_y_no_outliers: pixel y offsets without outliers
    """
    # create marked up images
    (diverged_features_img, converged_features_img, pitch_offset, roll_metric, pitch_y_no_outliers) = (
        _compute_features_and_offsets(ref_img, follow_image, max_features)
    )

    img_stereo_labeled = _create_labeled_stereo_images(
        diverged_features_img, converged_features_img, pitch_offset, roll_metric, pitch_y_no_outliers
    )

    return pitch_offset, roll_metric, img_stereo_labeled, pitch_y_no_outliers
