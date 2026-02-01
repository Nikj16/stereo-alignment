"""Stereo alignment analysis package for computing roll and pitch metrics."""

from stereo_alignment.core import run_stereo_alignment_metric
from stereo_alignment.image_process import (
    compute_roll,
    find_correspondences,
    separate_converging_diverging,
)

__version__ = "0.1.0"
__all__ = [
    "run_stereo_alignment_metric",
    "compute_roll",
    "find_correspondences",
    "separate_converging_diverging",
]
