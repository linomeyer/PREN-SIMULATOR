"""
Frame Contact Features Computation.

This module computes raw frame contact metrics and aggregates them to cost.
Metrics: dist_mean/p90/max, coverage_in_band, inlier_ratio, angle_diff, flatness.

See docs/design/04_scoring.md §C for metric definitions.
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import ContourSegment, FrameContactFeatures
    from ..config import MatchingConfig, FrameModel


def compute_frame_contact_features(
    seg: ContourSegment,
    side: str,
    frame: FrameModel,
    config: MatchingConfig
) -> FrameContactFeatures:
    """
    Compute frame contact features for segment-to-frame-edge match.

    Args:
        seg: ContourSegment with points_mm, chord, direction_angle_deg, flatness_error
        side: Frame edge to test ("TOP" | "BOTTOM" | "LEFT" | "RIGHT")
        frame: FrameModel (inner_width_mm=128, inner_height_mm=190)
        config: MatchingConfig with frame_band_mm, frame_angle_deg

    Returns:
        FrameContactFeatures with 7 metrics:
        - dist_mean_mm: Mean distance segment points → frame line
        - dist_p90_mm: 90th percentile distance (robust vs outliers)
        - dist_max_mm: Maximum distance
        - coverage_in_band: Arclength fraction within ±frame_band_mm
        - inlier_ratio: Point fraction within ±frame_band_mm
        - angle_diff_deg: Angle difference segment → frame edge
        - flatness_error_mm: Segment flatness (from seg.flatness_error)

    Algorithm:
        1. Define frame line (TOP: y=190, BOTTOM: y=0, LEFT: x=0, RIGHT: x=128)
        2. Compute distances for all points (perpendicular to frame line)
        3. Distance metrics: mean, p90, max
        4. Coverage in band: arclength-based (linear interpolation at boundaries)
        5. Inlier ratio: point-based (#inliers / #points)
        6. Angle difference: min_angle_diff(seg.direction_angle_deg, expected_angle)
        7. Flatness: copy from seg.flatness_error

    Notes:
        - Coverage uses arclength (more robust than point count)
        - Expected angles: TOP/BOTTOM → 0°/180° (horizontal), LEFT/RIGHT → 90°/-90° (vertical)
        - All distances in mm

    Example:
        >>> seg = ContourSegment(points_mm=[(10, 189.5), (20, 190.2), (30, 189.8)], ...)
        >>> features = compute_frame_contact_features(seg, "TOP", frame, config)
        >>> assert features.dist_p90_mm < 1.0  # Good contact
        >>> assert features.coverage_in_band > 0.8
    """
    from ..models import FrameContactFeatures

    # 1. Define frame line and expected angle
    if side == "TOP":
        frame_line_mm = frame.inner_height_mm  # 190mm
        expected_angle_deg = 0.0  # Horizontal
        axis = 1  # y-axis

    elif side == "BOTTOM":
        frame_line_mm = 0.0
        expected_angle_deg = 0.0  # Horizontal
        axis = 1  # y-axis

    elif side == "LEFT":
        frame_line_mm = 0.0
        expected_angle_deg = 90.0  # Vertical
        axis = 0  # x-axis

    elif side == "RIGHT":
        frame_line_mm = frame.inner_width_mm  # 128mm
        expected_angle_deg = 90.0  # Vertical
        axis = 0  # x-axis

    else:
        raise ValueError(f"Invalid side: {side}. Must be TOP/BOTTOM/LEFT/RIGHT")

    # 2. Compute distances for all points
    distances = np.abs(seg.points_mm[:, axis] - frame_line_mm)

    # 3. Distance metrics
    dist_mean_mm = float(np.mean(distances))
    dist_p90_mm = float(np.percentile(distances, 90))
    dist_max_mm = float(np.max(distances))

    # 4. Coverage in band (arclength-based)
    t = config.frame_band_mm
    L_in_band = 0.0

    for i in range(len(seg.points_mm) - 1):
        p1, p2 = seg.points_mm[i], seg.points_mm[i + 1]
        d1, d2 = distances[i], distances[i + 1]
        seg_len = np.linalg.norm(p2 - p1)

        if d1 <= t and d2 <= t:
            # Both in band → full segment length
            L_in_band += seg_len
        elif d1 <= t or d2 <= t:
            # One in band → proportional length (linear interpolation)
            d_diff = abs(d2 - d1)
            if d_diff > 1e-6:
                # Fraction of segment within band
                if d1 <= t:
                    ratio_in = (t - d1) / d_diff
                else:
                    ratio_in = (t - d2) / d_diff
                L_in_band += seg_len * ratio_in

    coverage_in_band = L_in_band / seg.length_mm if seg.length_mm > 0 else 0.0

    # 5. Inlier ratio (point-based)
    inlier_count = np.sum(distances <= t)
    inlier_ratio = float(inlier_count / len(distances)) if len(distances) > 0 else 0.0

    # 6. Angle difference
    angle_diff_deg = _min_angle_diff(seg.direction_angle_deg, expected_angle_deg)

    # 7. Flatness (copy from segment)
    flatness_error_mm = seg.flatness_error

    return FrameContactFeatures(
        dist_mean_mm=dist_mean_mm,
        dist_p90_mm=dist_p90_mm,
        dist_max_mm=dist_max_mm,
        coverage_in_band=coverage_in_band,
        inlier_ratio=inlier_ratio,
        angle_diff_deg=angle_diff_deg,
        flatness_error_mm=flatness_error_mm
    )


def compute_frame_cost(
    features: FrameContactFeatures,
    config: MatchingConfig
) -> float:
    """
    Aggregate frame contact features to frame cost.

    Args:
        features: FrameContactFeatures with 7 raw metrics
        config: MatchingConfig with frame_band_mm, frame_angle_deg, frame_weights

    Returns:
        Frame cost in [0, inf), lower = better. Typically [0, 1] for good matches.

    Algorithm:
        1. Map raw metrics to costs [0, 1]:
           - cost_dist_p90 = min(dist_p90_mm / frame_band_mm, 1.0)
           - cost_coverage = 1.0 - coverage_in_band
           - cost_angle = min(angle_diff_deg / frame_angle_deg, 1.0)
           - cost_flatness = min(flatness_error_mm / frame_band_mm, 1.0)
        2. Aggregate with weights: cost_frame = Σ w_k * cost_k

    Notes:
        - All costs normalized to [0, 1] (can exceed 1 for very bad matches)
        - Default weights: dist_p90=0.3, coverage=0.3, angle_diff=0.2, flatness=0.2
        - Flatness normalized by frame_flat_ref_mm (independent of frame_band_mm)
        - Coverage/inlier weighting controlled by frame_coverage_vs_inlier_policy

    Example:
        >>> features = FrameContactFeatures(dist_p90_mm=0.5, coverage_in_band=0.9, ...)
        >>> cost = compute_frame_cost(features, config)
        >>> assert 0 <= cost < 1  # Good match
    """
    t = config.frame_band_mm
    alpha = config.frame_angle_deg

    # 1. Cost mappings (normalize to [0, 1])
    weights = config.frame_weights
    cost_frame = 0.0

    # Distance metrics (always active)
    cost_dist_p90 = min(features.dist_p90_mm / t, 1.0)
    cost_frame += weights.get("dist_p90", 0.3) * cost_dist_p90

    # Angle (always active)
    cost_angle = min(features.angle_diff_deg / alpha, 1.0)
    cost_frame += weights.get("angle_diff", 0.2) * cost_angle

    # Flatness (always active, normalized by frame_flat_ref_mm)
    cost_flatness = min(features.flatness_error_mm / config.frame_flat_ref_mm, 1.0)
    cost_frame += weights.get("flatness", 0.2) * cost_flatness

    # 2. Coverage vs Inlier (policy-dependent)
    if config.frame_coverage_vs_inlier_policy == "coverage":
        cost_coverage = 1.0 - features.coverage_in_band
        cost_frame += weights.get("coverage", 0.3) * cost_coverage
    elif config.frame_coverage_vs_inlier_policy == "inlier":
        cost_inlier = 1.0 - features.inlier_ratio
        cost_frame += weights.get("coverage", 0.3) * cost_inlier  # Uses coverage weight
    elif config.frame_coverage_vs_inlier_policy == "balanced":
        cost_coverage = 1.0 - features.coverage_in_band
        cost_inlier = 1.0 - features.inlier_ratio
        # Both with halved weight
        cost_frame += weights.get("coverage", 0.3) * 0.5 * cost_coverage
        cost_frame += weights.get("coverage", 0.3) * 0.5 * cost_inlier

    return cost_frame


def _min_angle_diff(a_deg: float, b_deg: float) -> float:
    """
    Minimum angular distance between two angles (handles wraparound).

    Args:
        a_deg: Angle in degrees [-180, 180) or [0, 360)
        b_deg: Angle in degrees [-180, 180) or [0, 360)

    Returns:
        Minimum angular distance in [0, 180] degrees

    Notes:
        - Handles wraparound: 0° ↔ 359° → 1° difference
        - Symmetric: min_angle_diff(a, b) == min_angle_diff(b, a)

    Examples:
        >>> _min_angle_diff(0, 10)
        10.0
        >>> _min_angle_diff(0, 350)
        10.0
        >>> _min_angle_diff(90, -90)
        180.0
    """
    diff = abs(a_deg - b_deg)
    return min(diff, 360.0 - diff)
