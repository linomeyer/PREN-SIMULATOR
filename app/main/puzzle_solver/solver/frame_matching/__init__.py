"""
Solver V2 Frame-Matching Module.

This module provides frame contact detection and hypothesis generation:
- compute_frame_contact_features: Compute 7 metrics for segment-to-frame match
- compute_frame_cost: Aggregate metrics to frame cost
- estimate_pose_grob_F: Estimate initial pose from frame contact
- generate_frame_hypotheses: Generate and rank top-N frame hypotheses per piece

See docs/design/03_matching.md Phase 1 for algorithm details.
"""

from .features import compute_frame_contact_features, compute_frame_cost
from .hypotheses import estimate_pose_grob_F, generate_frame_hypotheses

__all__ = [
    "compute_frame_contact_features",
    "compute_frame_cost",
    "estimate_pose_grob_F",
    "generate_frame_hypotheses",
]
