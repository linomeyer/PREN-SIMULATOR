"""
Frame Hypothesis Generation.

This module generates and ranks frame placement hypotheses:
- estimate_pose_grob_F: Estimate initial pose from frame contact
- generate_frame_hypotheses: Test all segment×side combinations, keep top-N per piece

See docs/design/03_matching.md Phase 1 for algorithm details.
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from ..models import ContourSegment, FrameHypothesis, Pose2D
    from ..config import MatchingConfig, FrameModel


def estimate_pose_grob_F(
    seg: ContourSegment,
    side: str,
    frame: FrameModel,
    config: MatchingConfig
) -> tuple[Pose2D, float]:
    """
    Estimate initial pose in Frame coordinate system from frame contact.

    Args:
        seg: ContourSegment with chord, direction_angle_deg
        side: Frame edge ("TOP" | "BOTTOM" | "LEFT" | "RIGHT")
        frame: FrameModel (inner_width_mm=128, inner_height_mm=190)
        config: MatchingConfig with pose_grob_theta_mode

    Returns:
        Tuple (pose_grob_F, uncertainty_mm):
        - pose_grob_F: Pose2D in Frame coordinates (initial estimate)
        - uncertainty_mm: Mode-dependent (5.0mm default, 10.0mm for segment_aligned)

    Algorithm (Option A: Projection + Alignment):
        1. Project segment chord midpoint onto frame edge
        2. Translation:
           - TOP/BOTTOM: x from chord_mid[0], y at frame edge (0 or 190mm)
           - LEFT/RIGHT: y from chord_mid[1], x at frame edge (0 or 128mm)
        3. Rotation: config.pose_grob_theta_mode
           - "zero": theta=0° (no rotation)
           - "side_aligned": theta based on side (0° TOP/BOTTOM, 90° LEFT/RIGHT)
           - "segment_aligned": theta=seg.direction_angle_deg (heuristic only)
        4. Uncertainty: Mode-dependent (higher for segment_aligned)

    Notes:
        - This is **coarse estimate** (grob = coarse)
        - segment_aligned assumes segment coord = piece coord (heuristic only)
        - Refined later in Step 9 (Pose Refinement)
        - uncertainty_mm reserved for future beam weighting (see 00_structure.md L112)

    Example:
        >>> config = MatchingConfig(pose_grob_theta_mode="side_aligned")
        >>> seg = ContourSegment(chord=([50, 189], [60, 190]), direction_angle_deg=45.0, ...)
        >>> pose, unc = estimate_pose_grob_F(seg, "TOP", frame, config)
        >>> assert abs(pose.x_mm - 55) < 5  # Near chord midpoint
        >>> assert abs(pose.y_mm - 190) < 1  # At top edge
        >>> assert pose.theta_deg == 0.0  # side_aligned for TOP
    """
    from ..models import Pose2D

    # 1. Compute chord midpoint
    chord_start, chord_end = seg.chord
    chord_mid = (chord_start + chord_end) / 2

    # 2. Estimate translation based on side
    if side == "TOP":
        x_mm = float(chord_mid[0])
        y_mm = float(frame.inner_height_mm)  # 190mm
    elif side == "BOTTOM":
        x_mm = float(chord_mid[0])
        y_mm = 0.0
    elif side == "LEFT":
        x_mm = 0.0
        y_mm = float(chord_mid[1])
    elif side == "RIGHT":
        x_mm = float(frame.inner_width_mm)  # 128mm
        y_mm = float(chord_mid[1])
    else:
        raise ValueError(f"Invalid side: {side}. Must be TOP/BOTTOM/LEFT/RIGHT")

    # 3. Rotation based on config.pose_grob_theta_mode
    if config.pose_grob_theta_mode == "zero":
        theta_deg = 0.0
    elif config.pose_grob_theta_mode == "side_aligned":
        # Expected orientation based on frame side
        if side in ["TOP", "BOTTOM"]:
            theta_deg = 0.0  # Horizontal aligned
        else:  # LEFT, RIGHT
            theta_deg = 90.0  # Vertical aligned
    elif config.pose_grob_theta_mode == "segment_aligned":
        # Heuristic: Segment direction as piece rotation
        # WARNING: Only valid if segment coord = piece coord
        theta_deg = seg.direction_angle_deg
    else:
        theta_deg = 0.0  # Fallback

    # 4. Create pose
    pose_grob_F = Pose2D(x_mm=x_mm, y_mm=y_mm, theta_deg=theta_deg)

    # 5. Uncertainty (mode-dependent)
    if config.pose_grob_theta_mode == "segment_aligned":
        uncertainty_mm = 10.0  # Higher due to heuristic
    else:
        uncertainty_mm = 5.0

    return pose_grob_F, uncertainty_mm


def generate_frame_hypotheses(
    segments: list[ContourSegment],
    frame: FrameModel,
    config: MatchingConfig
) -> dict[int | str, list[FrameHypothesis]]:
    """
    Generate all frame hypotheses, rank, keep top-N per piece.

    Args:
        segments: List of ContourSegment from all pieces
        frame: FrameModel (128×190mm)
        config: MatchingConfig with frame params and debug_topN_frame_hypotheses_per_piece

    Returns:
        Dict mapping piece_id → list of top-N FrameHypothesis (sorted by cost ascending)

    Algorithm:
        1. For each segment (length >= min_frame_seg_len_mm):
           - Test all 4 frame sides (TOP, BOTTOM, LEFT, RIGHT)
           - Compute features, cost, pose → FrameHypothesis
        2. Group by piece_id
        3. Sort by cost_frame (ascending, lower = better)
        4. Keep top-N per piece (N = debug_topN_frame_hypotheses_per_piece)

    Notes:
        - Brute-force: All segments × 4 sides tested (e.g., 8 segs × 4 = 32 hypotheses/piece)
        - Short segments (< min_frame_seg_len_mm) skipped
        - Top-N kept for diversity (default N=5)
        - Cost function naturally filters bad matches

    Example:
        >>> segments = [seg1, seg2, ...]  # From 4 pieces
        >>> hyps_by_piece = generate_frame_hypotheses(segments, frame, config)
        >>> assert len(hyps_by_piece) == 4  # 4 pieces
        >>> for piece_id, hyps in hyps_by_piece.items():
        >>>     assert len(hyps) <= 5  # Top-5
        >>>     assert hyps[0].cost_frame <= hyps[-1].cost_frame  # Sorted
    """
    from ..models import FrameHypothesis
    from .features import compute_frame_contact_features, compute_frame_cost

    SIDES = ["TOP", "BOTTOM", "LEFT", "RIGHT"]
    hypotheses_by_piece = defaultdict(list)

    # 1. Generate hypotheses for all segment×side combinations
    for seg in segments:
        # Skip short segments
        if seg.length_mm < config.min_frame_seg_len_mm:
            continue

        for side in SIDES:
            # Compute features
            features = compute_frame_contact_features(seg, side, frame, config)

            # Compute cost
            cost_frame = compute_frame_cost(features, config)

            # Estimate pose
            pose_grob_F, uncertainty_mm = estimate_pose_grob_F(seg, side, frame, config)

            # Create hypothesis
            hyp = FrameHypothesis(
                piece_id=seg.piece_id,
                segment_id=seg.segment_id,
                side=side,
                pose_grob_F=pose_grob_F,
                features=features,
                cost_frame=cost_frame,
                is_committed=False
            )

            hypotheses_by_piece[seg.piece_id].append(hyp)

    # 2. Sort and keep top-N per piece
    top_N = config.debug_topN_frame_hypotheses_per_piece
    result = {}
    for piece_id, hyps in hypotheses_by_piece.items():
        # Sort by cost ascending (lower = better)
        hyps_sorted = sorted(hyps, key=lambda h: h.cost_frame)
        result[piece_id] = hyps_sorted[:top_N]

    return result
