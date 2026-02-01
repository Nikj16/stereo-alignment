"""Main script for stereo alignment analysis."""

import argparse
import logging
import sys

import cv2
import numpy as np

from stereo_alignment import run_stereo_alignment_metric


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application.
    
    Args:
        verbose: If True, set logging level to DEBUG, otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array in RGB format
        
    Raises:
        ValueError: If image cannot be loaded
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def main() -> int:
    """Main function to run stereo alignment analysis.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Stereo alignment analysis - compute roll and pitch metrics for stereo image pairs"
    )
    parser.add_argument(
        "ref_image",
        type=str,
        help="Path to the reference (left) stereo image",
    )
    parser.add_argument(
        "follow_image",
        type=str,
        help="Path to the following (right) stereo image",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10000,
        help="Maximum number of features to use for matching (default: 10000)",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="Optional path to save the output image",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    try:
        # Load images
        logging.info(f"Loading reference image: {args.ref_image}")
        ref_img = load_image(args.ref_image)
        
        logging.info(f"Loading follow image: {args.follow_image}")
        follow_img = load_image(args.follow_image)
        
        # Check that images have the same dimensions
        if ref_img.shape != follow_img.shape:
            logging.error(
                f"Image dimensions don't match: ref={ref_img.shape}, follow={follow_img.shape}"
            )
            return 1
        
        # Run stereo alignment analysis
        logging.info("Running stereo alignment analysis...")
        pitch_offset, roll_metric, labeled_image, pitch_y_no_outliers = run_stereo_alignment_metric(
            ref_img, follow_img, max_features=args.max_features
        )
        
        # Print results
        print("\n" + "=" * 60)
        print("STEREO ALIGNMENT RESULTS")
        print("=" * 60)
        print(f"Pitch Offset (median): {pitch_offset:.4f} pixels")
        print(f"Roll Metric: {roll_metric:.4f} degrees")
        print(f"Pitch Y mean: {np.mean(pitch_y_no_outliers):.4f} pixels")
        print(f"Pitch Y std: {np.std(pitch_y_no_outliers):.4f} pixels")
        print(f"Pitch Y mean (absolute): {np.mean(np.abs(pitch_y_no_outliers)):.4f} pixels")
        print(f"Number of valid features: {len(pitch_y_no_outliers)}")
        print("=" * 60 + "\n")
        
        # Convert RGB back to BGR for cv2 display/save
        labeled_image_bgr = cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR)
        
        # Save output if requested
        if args.save_output:
            cv2.imwrite(args.save_output, labeled_image_bgr)
            logging.info(f"Saved output image to: {args.save_output}")
        
        # Display the result
        window_name = "Stereo Alignment Analysis"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, labeled_image_bgr)
        
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return 0
        
    except Exception as e:
        logging.error(f"Error during processing: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
