"""Functions for image processing."""

import logging
import time
from typing import Tuple

import cv2
import numpy as np
from scipy import stats


def compute_roll(src_pts: np.ndarray, img_pts: np.ndarray) -> float:
    """Compute a value representing roll in degrees between two images.

    Takes arrays of matched key points (x,y) coordinates and processes them to compute
    a value for the roll between the images.

    Args:
        src_pts: np array of key points (x,y) coordinates
        img_pts: np array of key points (x,y) coordinates

    Returns:
        roll_degrees: The roll in degrees of the second image with respect to the reference image. A positive value
            indicates a clockwise rotation of the second image compared to the reference image. A negative
            value represents a counter-clockwise rotation.

    """
    scr_img_pts_paired = np.concatenate((src_pts, img_pts), axis=1)
    # sort along x-axis(left-right) of the first image:
    sorted_pair = scr_img_pts_paired[scr_img_pts_paired[:, 0].argsort()]
    # find the vertical offsets from left to right:
    y_pixel_diff = np.float64(sorted_pair[:, 1] - sorted_pair[:, 3])
    x_pixel = sorted_pair[:, 0]
    # Filter out outliers that are more than 3 standard deviations away from the mean
    # Perform this filtering step 3 times to produce (ideally) a reliable set of points
    for _ in range(3):
        mean = np.mean(y_pixel_diff)
        std = np.std(y_pixel_diff)
        idx = np.where(np.logical_and(y_pixel_diff > mean - 3 * std, y_pixel_diff < mean + 3 * std))[0]
        if len(idx) != 0:
            y_pixel_diff = y_pixel_diff[idx]
            x_pixel = x_pixel[idx]

    # TODO: The real pitch metric should probably be return as a combination of gradient and intercept for
    # better accuracy
    gradient, intercept, _, _, _ = stats.linregress(x_pixel, y_pixel_diff)
    roll_degrees = np.rad2deg(np.arctan(gradient))
    return roll_degrees


def separate_converging_diverging(src_pts: np.ndarray, img_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract converging and diverging features.

    Args:
        src_pts: positions of BRISK points on img1
        img_pts: positions of corresponding BRISK points on img2 [width, height]

    Return:
        converged_feature: List of matched points that shows convergence
        diverged_feature: List of matched points that shows divergence
    """
    diff = src_pts - img_pts

    mask = diff[:, 0] <= 0

    converged_features = np.column_stack([src_pts[mask], img_pts[mask]])
    diverged_features = np.column_stack((src_pts[~mask], img_pts[~mask]))

    return converged_features.reshape(-1, 2, 2), diverged_features.reshape(-1, 2, 2)


# Feature matching code


def _reduce_rand_kps_num(
    max_features: int, kp1: np.ndarray, des1: np.ndarray, kp2: np.ndarray, des2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduces number of keypoints and descriptor randomly.

    Args:
        max_features: Maximum number of features kept for knnMatch
        kp1: Keypoints from reference image
        des1: Descriptors from reference image
        kp2: Keypoints from following image
        des2: Descriptors from following image

    Returns:
        kp1: Randomely reduced keypoints for reference image
        des1: Corresponding descriptors for kp1
        kp2: Randomely reduced keypoints for following image
        des2: Corresponding descriptors for kp2
    """
    features_count_1, features_count_2 = len(kp1), len(kp2)
    logging.info(f"Detected {features_count_1} and {features_count_2} features.")
    if max_features > 0 and max_features < max(features_count_1, features_count_2):
        new_features_count_1 = min(max_features, features_count_1)
        new_features_count_2 = min(max_features, features_count_2)
        logging.info(f"Reducing to random {new_features_count_1} and {new_features_count_2} features.")
        np.random.seed(0)  # fixed seed for determistic results
        kept_indices_1 = np.random.choice(np.arange(features_count_1), new_features_count_1, replace=False)
        kept_indices_2 = np.random.choice(np.arange(features_count_2), new_features_count_2, replace=False)
        kp1, des1 = np.array(kp1)[kept_indices_1], np.array(des1)[kept_indices_1]
        kp2, des2 = np.array(kp2)[kept_indices_2], np.array(des2)[kept_indices_2]
    return kp1, des1, kp2, des2


def _filter_lowe_ratio(
    threshold: float, kp1: np.ndarray, kp2: np.ndarray, matches: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter good matches as per Lowe's ratio test.

    Args:
        threshold: threshold of distance between closest and next closest neighbors
                https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
        kp1: Keypoints from reference image
        kp2: Keypoints from following images
        matches: Matches found from the Flann KNN match algorithm

    Returns:
        src_pts: positions of BRISK points on reference image with the filtered points
        img_pts: positions of corresponding BRISK points on following images [width, height]
    """
    # store all the good matches as per Lowe's ratio test.
    filtered_matches = []
    for _, (closest_match, snd_closest_match) in enumerate(matches):
        if closest_match.distance < threshold * snd_closest_match.distance:
            filtered_matches.append(closest_match)
    src_pts = np.float32([kp1[closest_match.queryIdx].pt for closest_match in filtered_matches]).reshape(-1, 2)
    img_pts = np.float32([kp2[closest_match.trainIdx].pt for closest_match in filtered_matches]).reshape(-1, 2)

    return src_pts, img_pts


def find_correspondences(
    img1: np.ndarray, img2: np.ndarray, threshold: float = 0.7, max_features: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate lists of corresponding points on two images. Modified from the one in fleet/calibration.

    It is recommended to pass gray scale images to this function. Grayscale images tend
    to result in better feature matching.

    Args:
        img1: image1 (sRBG)
        img2: image2 to be compared with image1 to find corresponding pixels (sRBG)
        threshold: threshold of distance between closest and next closest neighbors
                https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
        max_features: maximum number of features kept for knnMatch. Uniform sampling with fixed seed
        Note: max_features disabled when set at default value of -1

    Returns:
        src_pts: x,y positions of BRISK points on img1 in pixels, shape [num_points, 2]
        img_pts: x,y positions of corresponding BRISK points on img2, shape [num_points, 2]

    Raises:
        ValueError: if the features matching failed

    """
    # Find all BRISK points on each image
    cv2.setRNGSeed(0)  # Tree construction has rng, set seed for consistent, testable results.
    feature = cv2.BRISK_create()
    kp1, des1 = feature.detectAndCompute(img1, None)
    kp2, des2 = feature.detectAndCompute(img2, None)
    kp1, des1, kp2, des2 = _reduce_rand_kps_num(max_features, kp1, des1, kp2, des2)

    min_features = 100
    if len(kp1) < min_features or len(kp2) < min_features:
        s = f"Not enough features detetected: {len(kp1)} for img1, {len(kp2)} for img2. Must be >= {min_features}."
        logging.error(s)
        raise ValueError(s)

    # Enum for flann algorithms isn't exposed in cv2. Reference to cpp code:
    # https://github.com/opencv/opencv/blob/852904e1a40df964b25b96a197e0ac7ee2bd9b8f/modules/flann/include/opencv2
    # /flann/defines.h#L70
    flann_index_kdtree = 1
    index_params = {"algorithm": flann_index_kdtree, "trees": 5}
    search_params = {"checks": 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        start_time = time.perf_counter()
        matches = flann.knnMatch(np.float32(des1), np.float32(des2), k=2)
        logging.debug(f"knnMatch took {time.perf_counter() - start_time}")
    except (cv2.error, ValueError, TypeError) as e:
        s = "knnMatch failed."
        logging.error(s)
        raise ValueError(s) from e

    src_pts, img_pts = _filter_lowe_ratio(threshold, kp1, kp2, matches)

    return src_pts, img_pts


# end feature matching code
