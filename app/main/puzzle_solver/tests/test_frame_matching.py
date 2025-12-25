"""
Tests for Frame-Matching Module (Step 4).

Tests frame contact feature computation, cost aggregation, pose estimation,
and hypothesis generation.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.frame_matching import (
    compute_frame_contact_features,
    compute_frame_cost,
    estimate_pose_grob_F,
    generate_frame_hypotheses
)
from solver.frame_matching.features import _min_angle_diff
from solver.models import (
    PuzzlePiece, ContourSegment, FrameContactFeatures, Pose2D
)
from solver.config import MatchingConfig, FrameModel
import numpy as np


def test_min_angle_diff():
    """Test minimum angle difference with wraparound."""
    print("Test 1: min_angle_diff...", end=" ")

    # Same angle
    assert np.isclose(_min_angle_diff(0, 0), 0.0), "Same angle should be 0"

    # Small difference
    assert np.isclose(_min_angle_diff(0, 10), 10.0), "0° → 10° should be 10°"
    assert np.isclose(_min_angle_diff(10, 0), 10.0), "Symmetric"

    # Wraparound cases
    assert np.isclose(_min_angle_diff(0, 350), 10.0), "0° ↔ 350° should be 10°"
    assert np.isclose(_min_angle_diff(350, 0), 10.0), "Symmetric wraparound"
    assert np.isclose(_min_angle_diff(0, -10), 10.0), "0° ↔ -10° should be 10°"

    # 180° difference
    assert np.isclose(_min_angle_diff(90, -90), 180.0), "90° ↔ -90° should be 180°"
    assert np.isclose(_min_angle_diff(0, 180), 180.0), "0° ↔ 180° should be 180°"

    # Edge cases
    assert np.isclose(_min_angle_diff(45, 135), 90.0), "45° → 135° should be 90°"
    assert np.isclose(_min_angle_diff(45, -135), 180.0), "45° ↔ -135° should be 180°"

    print("✓")


def test_features_ideal_top():
    """Test features computation for segment at TOP frame edge (ideal case)."""
    print("Test 2: features (ideal TOP)...", end=" ")

    # Create segment at TOP (y ≈ 190mm)
    points = np.array([
        [10, 189.5],
        [20, 190.2],
        [30, 189.8],
        [40, 190.1]
    ], dtype=float)

    chord = (points[0].copy(), points[-1].copy())
    direction_angle_deg = 0.0  # Horizontal
    flatness_error = 0.2  # Flat segment

    seg = ContourSegment(
        piece_id=1,
        segment_id=0,
        points_mm=points,
        length_mm=30.0,
        chord=chord,
        direction_angle_deg=direction_angle_deg,
        flatness_error=flatness_error,
        profile_1d=None
    )

    frame = FrameModel(inner_width_mm=128.0, inner_height_mm=190.0)
    config = MatchingConfig()

    features = compute_frame_contact_features(seg, "TOP", frame, config)

    # Assertions
    assert features.dist_p90_mm < 1.0, f"Expected dist_p90 < 1.0, got {features.dist_p90_mm}"
    assert features.coverage_in_band > 0.8, f"Expected coverage > 0.8, got {features.coverage_in_band}"
    assert features.angle_diff_deg < 10.0, f"Expected angle_diff < 10°, got {features.angle_diff_deg}"
    assert features.flatness_error_mm == flatness_error, f"Flatness mismatch"

    print(f"✓ (dist_p90={features.dist_p90_mm:.2f}mm, coverage={features.coverage_in_band:.2f})")


def test_features_wrong_side():
    """Test features computation for segment tested against wrong side."""
    print("Test 3: features (wrong side)...", end=" ")

    # Same segment at TOP (y ≈ 190mm)
    points = np.array([
        [10, 189.5],
        [20, 190.2],
        [30, 189.8]
    ], dtype=float)

    chord = (points[0].copy(), points[-1].copy())
    seg = ContourSegment(
        piece_id=1, segment_id=0, points_mm=points, length_mm=20.0,
        chord=chord, direction_angle_deg=0.0, flatness_error=0.2, profile_1d=None
    )

    frame = FrameModel(inner_width_mm=128.0, inner_height_mm=190.0)
    config = MatchingConfig()

    # Test against TOP (correct)
    features_top = compute_frame_contact_features(seg, "TOP", frame, config)

    # Test against BOTTOM (wrong)
    features_bottom = compute_frame_contact_features(seg, "BOTTOM", frame, config)

    # BOTTOM should have worse metrics (segment at y=190, tested against y=0)
    assert features_bottom.dist_p90_mm > features_top.dist_p90_mm, \
        "Wrong side should have higher dist_p90"
    assert features_bottom.coverage_in_band < features_top.coverage_in_band, \
        "Wrong side should have lower coverage"

    print(f"✓ (TOP: {features_top.dist_p90_mm:.1f}mm, BOTTOM: {features_bottom.dist_p90_mm:.1f}mm)")


def test_cost_mapping_perfect():
    """Test cost mapping for perfect features."""
    print("Test 4: cost mapping (perfect)...", end=" ")

    # Perfect features
    features = FrameContactFeatures(
        dist_mean_mm=0.1,
        dist_p90_mm=0.2,
        dist_max_mm=0.5,
        coverage_in_band=1.0,
        inlier_ratio=1.0,
        angle_diff_deg=1.0,
        flatness_error_mm=0.1
    )

    config = MatchingConfig()
    cost = compute_frame_cost(features, config)

    # Perfect features should yield low cost
    assert cost < 0.11, f"Perfect features should have cost < 0.11, got {cost:.3f}"

    print(f"✓ (cost={cost:.3f})")


def test_cost_mapping_bad():
    """Test cost mapping for bad features."""
    print("Test 5: cost mapping (bad)...", end=" ")

    # Bad features
    features = FrameContactFeatures(
        dist_mean_mm=5.0,
        dist_p90_mm=8.0,
        dist_max_mm=10.0,
        coverage_in_band=0.1,
        inlier_ratio=0.2,
        angle_diff_deg=45.0,
        flatness_error_mm=3.0
    )

    config = MatchingConfig()
    cost = compute_frame_cost(features, config)

    # Bad features should yield high cost
    assert cost > 0.8, f"Bad features should have cost > 0.8, got {cost:.3f}"

    print(f"✓ (cost={cost:.3f})")


def test_pose_estimation_top():
    """Test pose estimation for TOP segment (side_aligned mode)."""
    print("Test 6: pose estimation (TOP)...", end=" ")

    # Segment at TOP (y ≈ 190mm)
    points = np.array([[50, 189], [60, 190]], dtype=float)
    chord = (points[0].copy(), points[-1].copy())

    seg = ContourSegment(
        piece_id=1, segment_id=0, points_mm=points, length_mm=10.0,
        chord=chord, direction_angle_deg=45.0,  # Segment has 45° but side_aligned ignores this
        flatness_error=0.1, profile_1d=None
    )

    frame = FrameModel(inner_width_mm=128.0, inner_height_mm=190.0)
    config = MatchingConfig()  # Default: pose_grob_theta_mode="side_aligned"

    pose, unc = estimate_pose_grob_F(seg, "TOP", frame, config)

    # Assertions
    assert abs(pose.x_mm - 55) < 5, f"Expected x ≈ 55 (chord midpoint), got {pose.x_mm}"
    assert abs(pose.y_mm - 190) < 1, f"Expected y ≈ 190 (top edge), got {pose.y_mm}"
    assert pose.theta_deg == 0.0, f"TOP side_aligned should be 0°, got {pose.theta_deg}"
    assert unc == 5.0, f"Expected uncertainty = 5.0mm, got {unc}"

    print(f"✓ (pose=({pose.x_mm:.1f}, {pose.y_mm:.1f}, {pose.theta_deg:.1f}°))")


def test_generate_hypotheses_basic():
    """Test hypothesis generation for multiple pieces."""
    print("Test 7: generate_hypotheses (basic)...", end=" ")

    # Create 2 pieces × 4 segments each = 8 segments
    segments = []
    for piece_id in [1, 2]:
        for seg_id in range(4):
            # Create segments at various positions
            y_offset = seg_id * 50  # 0, 50, 100, 150
            points = np.array([
                [10 + seg_id * 10, y_offset],
                [20 + seg_id * 10, y_offset + 10]
            ], dtype=float)

            chord = (points[0].copy(), points[-1].copy())
            seg = ContourSegment(
                piece_id=piece_id,
                segment_id=seg_id,
                points_mm=points,
                length_mm=15.0,  # > min_frame_seg_len_mm (10mm)
                chord=chord,
                direction_angle_deg=45.0,
                flatness_error=0.5,
                profile_1d=None
            )
            segments.append(seg)

    frame = FrameModel(inner_width_mm=128.0, inner_height_mm=190.0)
    config = MatchingConfig()
    config.debug_topN_frame_hypotheses_per_piece = 5

    hyps_by_piece = generate_frame_hypotheses(segments, frame, config)

    # Assertions
    assert len(hyps_by_piece) == 2, f"Expected 2 pieces, got {len(hyps_by_piece)}"

    for piece_id, hyps in hyps_by_piece.items():
        # Each piece: 4 segments × 4 sides = 16 hypotheses → top-5 kept
        assert len(hyps) <= 5, f"Expected ≤5 hypotheses per piece, got {len(hyps)}"
        assert len(hyps) > 0, f"Expected >0 hypotheses for piece {piece_id}"

        # Check sorted by cost ascending
        for i in range(len(hyps) - 1):
            assert hyps[i].cost_frame <= hyps[i + 1].cost_frame, \
                f"Hypotheses not sorted by cost (piece {piece_id})"

        # Check hypothesis fields
        for hyp in hyps:
            assert hyp.piece_id == piece_id, f"Hypothesis piece_id mismatch"
            assert hyp.side in ["TOP", "BOTTOM", "LEFT", "RIGHT"], f"Invalid side: {hyp.side}"
            assert hyp.cost_frame >= 0, f"Cost should be >= 0"
            assert not hyp.is_committed, f"New hypotheses should not be committed"

    print(f"✓ ({len(hyps_by_piece)} pieces, {len(hyps_by_piece[1])} hyps each)")


def test_short_segment_filtering():
    """Test that segments < min_frame_seg_len_mm are filtered."""
    print("Test 8: short segment filtering...", end=" ")

    # Create short segment (< 10mm)
    points = np.array([[10, 100], [15, 100]], dtype=float)  # 5mm long
    chord = (points[0].copy(), points[-1].copy())

    seg_short = ContourSegment(
        piece_id=1, segment_id=0, points_mm=points, length_mm=5.0,
        chord=chord, direction_angle_deg=0.0, flatness_error=0.1, profile_1d=None
    )

    frame = FrameModel(inner_width_mm=128.0, inner_height_mm=190.0)
    config = MatchingConfig()
    config.min_frame_seg_len_mm = 10.0

    hyps = generate_frame_hypotheses([seg_short], frame, config)

    # Should have no hypotheses (segment too short)
    assert len(hyps) == 0, f"Expected 0 hypotheses for short segment, got {len(hyps)}"

    print("✓ (0 hypotheses for 5mm segment)")


def test_flatness_independent_of_frame_band():
    """Flatness normierung unabhängig von frame_band_mm."""
    print("Test 9: flatness independent...", end=" ")

    # Test with perfect features except flatness
    features = FrameContactFeatures(
        dist_mean_mm=0.0,  # Perfect
        dist_p90_mm=0.0,  # Perfect
        dist_max_mm=0.0,
        coverage_in_band=1.0,  # Perfect
        inlier_ratio=1.0,  # Perfect
        angle_diff_deg=0.0,  # Perfect
        flatness_error_mm=0.5  # Only flatness has cost
    )

    config1 = MatchingConfig()
    config1.frame_band_mm = 1.0
    config1.frame_flat_ref_mm = 1.0

    config2 = MatchingConfig()
    config2.frame_band_mm = 2.0  # Different band
    config2.frame_flat_ref_mm = 1.0  # Same flat ref (flatness should be identical)

    cost1 = compute_frame_cost(features, config1)
    cost2 = compute_frame_cost(features, config2)

    # Costs should be identical (only flatness contributes, uses frame_flat_ref_mm)
    assert abs(cost1 - cost2) < 0.001, \
        f"Flatness cost should be independent of frame_band_mm: {cost1:.3f} vs {cost2:.3f}"

    print(f"✓ (cost1={cost1:.3f}, cost2={cost2:.3f})")


def test_policy_switches_weights():
    """Policy umschalten ändert effektive Gewichte."""
    print("Test 10: policy switches...", end=" ")

    features = FrameContactFeatures(
        dist_mean_mm=0.5,
        dist_p90_mm=0.5,
        dist_max_mm=1.0,
        coverage_in_band=0.8,
        inlier_ratio=0.6,
        angle_diff_deg=5.0,
        flatness_error_mm=0.2
    )

    config_cov = MatchingConfig()
    config_cov.frame_coverage_vs_inlier_policy = "coverage"

    config_inl = MatchingConfig()
    config_inl.frame_coverage_vs_inlier_policy = "inlier"

    config_bal = MatchingConfig()
    config_bal.frame_coverage_vs_inlier_policy = "balanced"

    cost_cov = compute_frame_cost(features, config_cov)
    cost_inl = compute_frame_cost(features, config_inl)
    cost_bal = compute_frame_cost(features, config_bal)

    # Policies should yield different costs
    assert cost_cov != cost_inl, "Coverage vs inlier policy should differ"

    print(f"✓ (coverage={cost_cov:.3f}, inlier={cost_inl:.3f}, balanced={cost_bal:.3f})")


def test_pose_theta_modes():
    """Pose theta basiert auf Mode."""
    print("Test 11: theta modes...", end=" ")

    points = np.array([[50, 189], [60, 190]], dtype=float)
    chord = (points[0].copy(), points[-1].copy())

    seg = ContourSegment(
        piece_id=1, segment_id=0, points_mm=points, length_mm=10.0,
        chord=chord, direction_angle_deg=45.0,  # Segment has 45°
        flatness_error=0.1, profile_1d=None
    )

    frame = FrameModel(inner_width_mm=128.0, inner_height_mm=190.0)

    # Mode: side_aligned (default)
    config_side = MatchingConfig()
    config_side.pose_grob_theta_mode = "side_aligned"

    pose_top, unc_top = estimate_pose_grob_F(seg, "TOP", frame, config_side)
    pose_left, unc_left = estimate_pose_grob_F(seg, "LEFT", frame, config_side)

    assert pose_top.theta_deg == 0.0, f"TOP side_aligned should be 0°, got {pose_top.theta_deg}"
    assert pose_left.theta_deg == 90.0, f"LEFT side_aligned should be 90°, got {pose_left.theta_deg}"
    assert unc_top == 5.0

    # Mode: segment_aligned
    config_seg = MatchingConfig()
    config_seg.pose_grob_theta_mode = "segment_aligned"

    pose_seg, unc_seg = estimate_pose_grob_F(seg, "TOP", frame, config_seg)

    assert pose_seg.theta_deg == 45.0, f"segment_aligned should use seg direction, got {pose_seg.theta_deg}"
    assert unc_seg == 10.0, "segment_aligned should have higher uncertainty"

    # Mode: zero
    config_zero = MatchingConfig()
    config_zero.pose_grob_theta_mode = "zero"

    pose_zero, _ = estimate_pose_grob_F(seg, "TOP", frame, config_zero)

    assert pose_zero.theta_deg == 0.0

    print("✓ (side_aligned, segment_aligned, zero)")


def run_all_tests():
    """Run all frame-matching tests."""
    print("=" * 60)
    print("Step 4: Frame-Matching - Tests")
    print("=" * 60)
    print()

    test_min_angle_diff()
    test_features_ideal_top()
    test_features_wrong_side()
    test_cost_mapping_perfect()
    test_cost_mapping_bad()
    test_pose_estimation_top()
    test_generate_hypotheses_basic()
    test_short_segment_filtering()
    test_flatness_independent_of_frame_band()
    test_policy_switches_weights()
    test_pose_theta_modes()

    print()
    print("=" * 60)
    print("✅ Alle Tests bestanden (11/11)")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
