# Stereo Alignment Analysis

A Python tool for analyzing stereo image pairs to compute roll and pitch alignment metrics using feature matching techniques.

## Overview

This tool processes stereo image pairs (left/right or reference/follow) and computes alignment metrics including:
- **Roll metric**: Rotation angle between images in degrees
- **Pitch offset**: Vertical misalignment between images in pixels
- Feature matching visualization showing converging and diverging features

The analysis uses BRISK feature detection, FLANN-based matching, and statistical outlier rejection to provide robust alignment measurements.

## Features

- BRISK (Binary Robust Invariant Scalable Keypoints) feature detection
- FLANN (Fast Library for Approximate Nearest Neighbors) based matching
- Lowe's ratio test for filtering good matches
- Automatic outlier rejection using statistical methods
- Visual output with annotated features and metrics
- Support for saving output images
- Configurable maximum number of features for performance tuning

## Installation

### Quick Install

Install directly from source:

```bash
# Clone the repository
git clone <repository-url>
cd stereo-alignment

# Install in development mode
pip install -e .
```

### Using pip (after building)

```bash
# Build the package
python -m build

# Install the built package
pip install dist/stereo_alignment-0.1.0-py3-none-any.whl
```

### Manual Installation

```bash
# Install dependencies only
pip install opencv-python>=4.5.0 numpy>=1.19.0 scipy>=1.5.0
```

### For Development

```bash
# Install with development dependencies
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

After installation, you can use the `stereo-align` command:

```bash
# Using the installed command
stereo-align <reference_image> <follow_image>

# Or run directly with Python
python main.py <reference_image> <follow_image>
```

Basic usage with options:
```bash
stereo-align left.jpg right.jpg --max-features 5000 --save-output result.jpg -v
```

### Command Line Arguments

- `ref_image`: Path to the reference (left) stereo image (required)
- `follow_image`: Path to the following (right) stereo image (required)
- `--max-features`: Maximum number of features to use for matching (default: 10000)
- `--save-output`: Path to save the annotated output image (optional)
- `-v, --verbose`: Enable verbose logging for debugging

### Example

```bash
# Analyze stereo pair with default settings
python main.py images/left_camera.jpg images/right_camera.jpg

# Analyze with fewer features and save result
python main.py images/left_camera.jpg images/right_camera.jpg \
    --max-features 3000 \
    --save-output results/alignment_analysis.jpg \
    --verbose
```

## Output

### Console Output

The tool prints detailed metrics to the console:

```
============================================================
STEREO ALIGNMENT RESULTS
============================================================
Pitch Offset (median): 2.3456 pixels
Roll Metric: 0.1234 degrees
Pitch Y mean: 2.4123 pixels
Pitch Y std: 0.8765 pixels
Pitch Y mean (absolute): 2.5678 pixels
Number of valid features: 487
============================================================
```

### Visual Output

The output image contains:
1. **Top half**: Diverged features (green lines) with pitch metrics
2. **Bottom half**: Converged features (red lines) with roll metric
3. Horizontal reference lines for alignment verification
4. Annotated metrics overlaid on the images

### Display Window

A resizable OpenCV window displays the annotated result. Press any key to close the window.

## Architecture

### Module Structure

```
stereo-alignment/
├── stereo_alignment/           # Main package directory
│   ├── __init__.py            # Package initialization
│   ├── cli.py                 # Command-line interface
│   ├── core.py                # Core alignment processing
│   ├── image_process.py       # Feature detection and matching
│   └── drawing.py             # Visualization functions
├── tests/                     # Test directory
├── pyproject.toml            # Modern Python packaging config
├── setup.cfg                 # Alternative packaging config
└── README.md                 # Documentation
```

### Key Components

#### `stereo_alignment.cli`
- Command-line interface entry point
- Image loading and validation
- Result display and saving
- Logging configuration

#### `stereo_alignment.core`
- `run_stereo_alignment_metric()`: Main processing pipeline
- `_compute_features_and_offsets()`: Feature matching and metric computation
- `_create_labeled_stereo_images()`: Generate annotated output images

#### `stereo_alignment.image_process`
- `find_correspondences()`: BRISK feature detection and FLANN matching
- `compute_roll()`: Calculate roll angle from matched features
- `separate_converging_diverging()`: Classify feature convergence
- `_filter_lowe_ratio()`: Apply Lowe's ratio test
- `_reduce_rand_kps_num()`: Reduce keypoints for performance

#### `stereo_alignment.drawing`
- `draw_lines()`: Draw horizontal reference lines
- `draw_feature_match()`: Visualize matched features

## Algorithm Details

### Feature Detection

1. **BRISK Detection**: Binary feature detector that is robust to scale and rotation
2. **Feature Reduction**: Optional random sampling to limit computational cost
3. **Minimum Features**: Requires at least 100 features per image

### Feature Matching

1. **FLANN Matching**: Fast approximate nearest neighbor search using KD-trees
2. **Lowe's Ratio Test**: Filters matches where closest neighbor is significantly better than second closest (threshold: 0.7)
3. **K=2 Matching**: Finds two nearest neighbors for ratio testing

### Metric Computation

#### Roll Metric
- Sorts matched points by x-coordinate
- Computes vertical offset (y-difference) across horizontal positions
- Applies 3-sigma outlier rejection (3 iterations)
- Fits linear regression to y-offset vs x-position
- Returns arctangent of gradient in degrees

#### Pitch Offset
- Computes vertical pixel differences for all matched features
- Uses median for robust central tendency
- Reports mean and mean absolute values
- Applies 1-sigma outlier rejection for refined statistics

### Convergence Classification

Features are classified as:
- **Converged**: x-offset ≤ 0 (feature moves left or stays put)
- **Diverged**: x-offset > 0 (feature moves right)

## Limitations and Considerations

1. **Image Requirements**:
   - Images must have the same dimensions
   - Sufficient texture for feature detection
   - Reasonable overlap between stereo pair

2. **Performance**:
   - Processing time increases with image size and feature count
   - Use `--max-features` to balance accuracy vs speed

3. **Accuracy**:
   - Results depend on feature matching quality
   - Poor lighting or texture can reduce accuracy
   - Extreme misalignment may cause matching failures

4. **Color Space**:
   - Images are converted to RGB internally
   - Grayscale images recommended for better feature matching

## Troubleshooting

### "Not enough features detected"
- Increase image texture/detail
- Check image quality and lighting
- Reduce `--max-features` constraint

### "Could not match sufficient features"
- Ensure images are from a valid stereo pair
- Check for sufficient overlap
- Verify images are correctly oriented

### "knnMatch failed"
- Usually indicates insufficient features after filtering
- Try different images or adjust parameters

## Contributing

When modifying the code:
1. Follow existing code style and docstring format
2. Add logging for important operations
3. Include error handling with descriptive messages
4. Update this README for new features

## License

MIT License - See LICENSE file for details

## Contact

Nikhil Jaiyam - nikhil.jaiyam@example.com

## Development

### Building the Package

```bash
# Install build tools
pip install build

# Build source distribution and wheel
python -m build
```

### Running Tests

```bash
# Install with test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=stereo_alignment --cov-report=html

# Run specific test file
pytest tests/test_image_process.py

# Run with verbose output
pytest -v
```

### Test Structure

The test suite includes:
- **test_image_process.py**: Tests for feature detection and matching
- **test_drawing.py**: Tests for visualization functions
- **test_core.py**: Tests for core alignment processing
- **test_cli.py**: Tests for command-line interface

Coverage should be above 80% for all modules.
