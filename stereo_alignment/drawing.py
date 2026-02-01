"""Functions for rendering images as part of qualitative checks."""

from typing import Tuple

import cv2
import numpy as np


def draw_lines(image: np.ndarray, num_lines: int = 10, margin_line_offset_ratio: float = 0.05) -> np.ndarray:
    """Draws `num_lines` equally spaced horizontal lines on `img`.

    Args:
        image: image to draw line on
        num_lines: number of line to draw on the image
        margin_line_offset_ratio: offset ratio of the first and last line. It must be between 0 and 0.3
            Creates extra spacing between the top/bottom line and top/bottom of image. Useful to get
            denser lines closer to the middle of the image.

    Returns:
        img_out: copy of img with the lines drawn on it

    Raise:
        ValueError: If the number of line to draw is not larger than 0
    """
    img_out = image.copy()

    if num_lines < 1:
        s = "minimal number of line to draw is one"
        raise ValueError(s)
    max_margin_line_offset_ratio = 0.3
    if not 0 <= margin_line_offset_ratio <= max_margin_line_offset_ratio:
        s = "margin_line_offset_ratio should be between 0 and 0.3"
        raise ValueError(s)

    height, width = img_out.shape[0], img_out.shape[1]
    # Compute padding for the top and bottom of the image
    margin_line_offset_pixel = np.round(margin_line_offset_ratio * height).astype(int)
    # Compute the spacing between the lines given the padding
    spacing = (height - 2 * margin_line_offset_pixel) // (num_lines - 1)
    current_height = margin_line_offset_pixel

    for _ in range(num_lines):
        start = (0, current_height)
        end = (width - 1, current_height)
        img_out = cv2.line(img_out, start, end, (255, 0, 255), thickness=5)
        current_height += spacing

    return img_out


def draw_feature_match(
    img1: np.ndarray, img2: np.ndarray, features: np.ndarray, color: Tuple[int, int, int]
) -> np.ndarray:
    """Draws matches for converging and diverging features.

    Args:
        img1: Reference image in rgb, shape [H, W, 3]
        img2: Following image in rgb, shape [H, W, 3]
        features: List of matched pairs points for reference and following image,
            shape [num_pairs, 2, 2] --> [num_points, [left, right], [x , y]]
        color: RGB colors (you use plt.imshow to display images)

    Returns:
        img_out: Image with matched features drawn

    """
    if not np.array_equal(img1.shape, img2.shape):
        s = "Images must be same size"
        raise ValueError(s)
    # cheching if we have a colored image
    rgb_size = 3
    if len(img1.shape) != rgb_size or img1.shape[2] != rgb_size:
        s = f"The shape should be (H, W, 3) but got {img1.shape}"
        raise ValueError(s)

    width_offset = img1.shape[1]
    img_out = cv2.hconcat((img1, img2))
    features = features.round().astype(np.int32)

    for pairs in features:
        ref_pt = (pairs[0][0], pairs[0][1])
        follow_pt = (pairs[1][0] + width_offset, pairs[1][1])
        img_out = cv2.line(img_out, ref_pt, follow_pt, color, thickness=2)

    return img_out
