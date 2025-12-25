"""
Tests for Segmentation Module (Step 3).

Tests contour segmentation algorithm and geometric property computation.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.segmentation import segment_piece
from solver.segmentation.contour_segmenter import (
    _compute_chord,
    _compute_direction_angle,
    _compute_flatness_error,
    _compute_arclength,
)
from solver.models import PuzzlePiece
from solver.config import MatchingConfig
import numpy as np


def test_arclength():
    """Test arclength computation."""
    print("Test 1: Arclength...", end=" ")

    # Straight line: (0,0) → (3,0) → (3,4)
    points = np.array([[0, 0], [3, 0], [3, 4]], dtype=float)
    length = _compute_arclength(points)
    expected = 3.0 + 4.0  # 7.0
    assert np.isclose(length, expected, atol=1e-6), \
        f"Expected {expected}, got {length}"

    # Single point
    points_single = np.array([[1, 1]], dtype=float)
    length_single = _compute_arclength(points_single)
    assert length_single == 0.0, f"Single point should have length 0, got {length_single}"

    print("✓")


def test_chord_computation():
    """Test chord endpoint computation."""
    print("Test 2: Chord computation...", end=" ")

    points = np.array([[10, 20], [15, 25], [20, 30]], dtype=float)
    chord = _compute_chord(points)

    start_pt, end_pt = chord
    assert start_pt.shape == (2,), f"Start point shape mismatch: {start_pt.shape}"
    assert end_pt.shape == (2,), f"End point shape mismatch: {end_pt.shape}"
    assert np.allclose(start_pt, [10, 20]), f"Start point wrong: {start_pt}"
    assert np.allclose(end_pt, [20, 30]), f"End point wrong: {end_pt}"

    print("✓")


def test_direction_angle():
    """Test direction angle computation (edge cases)."""
    print("Test 3: Direction angle...", end=" ")

    # Right (0°)
    chord_right = (np.array([0, 0], dtype=float), np.array([1, 0], dtype=float))
    angle = _compute_direction_angle(chord_right)
    assert np.isclose(angle, 0.0, atol=1e-6), f"Expected 0°, got {angle}"

    # Up (90°)
    chord_up = (np.array([0, 0], dtype=float), np.array([0, 1], dtype=float))
    angle = _compute_direction_angle(chord_up)
    assert np.isclose(angle, 90.0, atol=1e-6), f"Expected 90°, got {angle}"

    # Left (±180°)
    chord_left = (np.array([0, 0], dtype=float), np.array([-1, 0], dtype=float))
    angle = _compute_direction_angle(chord_left)
    assert np.isclose(abs(angle), 180.0, atol=1e-6), f"Expected ±180°, got {angle}"

    # Down (-90°)
    chord_down = (np.array([0, 0], dtype=float), np.array([0, -1], dtype=float))
    angle = _compute_direction_angle(chord_down)
    assert np.isclose(angle, -90.0, atol=1e-6), f"Expected -90°, got {angle}"

    # 45°
    chord_45 = (np.array([0, 0], dtype=float), np.array([1, 1], dtype=float))
    angle = _compute_direction_angle(chord_45)
    assert np.isclose(angle, 45.0, atol=1e-6), f"Expected 45°, got {angle}"

    print("✓")


def test_flatness_error_straight_line():
    """Test flatness error for perfectly straight segment."""
    print("Test 4: Flatness (straight line)...", end=" ")

    # Straight line: points exactly on chord
    points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
    chord = (points[0], points[-1])
    flatness = _compute_flatness_error(points, chord)

    # Should be very close to 0 (within numerical precision)
    assert flatness < 1e-10, f"Straight line should have flatness ≈ 0, got {flatness}"

    print("✓")


def test_flatness_error_curved():
    """Test flatness error for curved segment."""
    print("Test 5: Flatness (curved)...", end=" ")

    # Arc: points deviate from chord
    # Chord: (0,0) → (2,0), midpoint at (1, 0.5) deviates by 0.5mm
    points = np.array([[0, 0], [1, 0.5], [2, 0]], dtype=float)
    chord = (points[0], points[-1])
    flatness = _compute_flatness_error(points, chord)

    # Expected: RMS of perpendicular distances
    # Distances: [0, 0.5, 0] → RMS = sqrt((0² + 0.5² + 0²)/3) = sqrt(0.25/3) ≈ 0.289
    expected = np.sqrt(0.25 / 3)
    assert np.isclose(flatness, expected, atol=1e-3), \
        f"Expected flatness ≈ {expected:.3f}, got {flatness:.3f}"

    # Flatness should be > 0 for non-straight segment
    assert flatness > 0, f"Curved segment should have flatness > 0, got {flatness}"

    print("✓")


def test_flatness_error_degenerate_chord():
    """Test flatness error for degenerate chord (start ≈ end)."""
    print("Test 6: Flatness (degenerate chord)...", end=" ")

    # Degenerate: all points at same location
    points = np.array([[1, 1], [1, 1], [1, 1]], dtype=float)
    chord = (points[0], points[-1])
    flatness = _compute_flatness_error(points, chord)

    # Should return 0.0 (degenerate case)
    assert flatness == 0.0, f"Degenerate chord should have flatness 0, got {flatness}"

    print("✓")


def test_segment_piece_basic():
    """Test basic segment_piece with synthetic data (closed square)."""
    print("Test 7: segment_piece (basic)...", end=" ")

    # Create synthetic square-ish contour (40mm × 40mm)
    # 4 corners create 4 segments initially, but wraparound merge reduces to 3
    corner1 = np.linspace([0, 0], [40, 0], 20)  # Bottom edge
    corner2 = np.linspace([40, 0], [40, 40], 20)  # Right edge
    corner3 = np.linspace([40, 40], [0, 40], 20)  # Top edge
    corner4 = np.linspace([0, 40], [0, 0], 20)  # Left edge (closes to [0, 0])

    contour = np.vstack([corner1, corner2, corner3, corner4])

    piece = PuzzlePiece(
        piece_id=1,
        contour_mm=contour,
        mask=np.zeros((10, 10), dtype=np.uint8),
        bbox_mm=(0.0, 0.0, 40.0, 40.0)
    )

    config = MatchingConfig()
    segments = segment_piece(piece, config)

    # Assertions: Closed square with wraparound merge → 3 segments
    assert len(segments) == 3, f"Expected 3 segments (wraparound merge), got {len(segments)}"

    for i, seg in enumerate(segments):
        assert seg.piece_id == 1, f"Segment {i}: wrong piece_id"
        assert seg.segment_id == i, f"Segment {i}: wrong segment_id"
        assert seg.points_mm.shape[1] == 2, f"Segment {i}: points not 2D"
        assert seg.points_mm.shape[0] >= 2, f"Segment {i}: need at least 2 points"
        assert seg.length_mm > 0, f"Segment {i}: length must be > 0"
        assert seg.flatness_error >= 0, f"Segment {i}: flatness must be >= 0"
        assert seg.chord[0].shape == (2,), f"Segment {i}: chord start wrong shape"
        assert seg.chord[1].shape == (2,), f"Segment {i}: chord end wrong shape"
        assert -180 <= seg.direction_angle_deg < 180, \
            f"Segment {i}: direction angle out of range [{seg.direction_angle_deg}]"
        assert seg.profile_1d is None, f"Segment {i}: profile should be None (lazy)"

    print(f"✓ ({len(segments)} segments)")


def test_segment_piece_min_length():
    """Test that segments respect minimum length constraint."""
    print("Test 8: segment_piece (min length)...", end=" ")

    # Create contour with many small edges (should merge to min_frame_seg_len_mm)
    points = []
    for i in range(20):
        # Small zig-zag pattern (2mm per edge)
        points.append([i * 2, 0 if i % 2 == 0 else 2])

    contour = np.array(points, dtype=float)

    piece = PuzzlePiece(
        piece_id=2,
        contour_mm=contour,
        mask=np.zeros((10, 10), dtype=np.uint8),
        bbox_mm=(0.0, 0.0, 40.0, 2.0)
    )

    config = MatchingConfig()
    segments = segment_piece(piece, config)

    # Most segments should be >= min_frame_seg_len_mm (10.0mm)
    # Allow some flexibility (e.g., last segment might be shorter)
    long_segments = [seg for seg in segments if seg.length_mm >= config.min_frame_seg_len_mm]
    assert len(long_segments) >= len(segments) - 1, \
        f"Most segments should be >= {config.min_frame_seg_len_mm}mm"

    print(f"✓ ({len(long_segments)}/{len(segments)} segments >= {config.min_frame_seg_len_mm}mm)")


def test_segment_piece_simple_contour():
    """Test segmentation with very simple contour (triangle)."""
    print("Test 9: segment_piece (triangle)...", end=" ")

    # Simple triangle: 3 corners
    # Note: Creates 3-4 segments depending on curvature detection
    edge1 = np.linspace([0, 0], [30, 0], 15)  # Bottom edge
    edge2 = np.linspace([30, 0], [15, 25], 15)  # Right edge
    edge3 = np.linspace([15, 25], [0, 0.05], 15)  # Left edge (closes to [0, 0])

    contour = np.vstack([edge1, edge2, edge3])

    piece = PuzzlePiece(
        piece_id=3,
        contour_mm=contour,
        mask=np.zeros((10, 10), dtype=np.uint8),
        bbox_mm=(0.0, 0.0, 30.0, 25.0)
    )

    config = MatchingConfig()
    segments = segment_piece(piece, config)

    # Triangle: should create 2-4 segments (depends on splits + wraparound merge)
    assert 2 <= len(segments) <= 4, f"Expected 2-4 segments, got {len(segments)}"

    print(f"✓ ({len(segments)} segments)")


def test_segment_piece_high_threshold():
    """Test segmentation with high curvature threshold (fewer splits)."""
    print("Test 10: segment_piece (high threshold)...", end=" ")

    # Create square contour (same as Test 7)
    corner1 = np.linspace([0, 0], [40, 0], 20)
    corner2 = np.linspace([40, 0], [40, 40], 20)
    corner3 = np.linspace([40, 40], [0, 40], 20)
    corner4 = np.linspace([0, 40], [0, 0], 20)
    contour = np.vstack([corner1, corner2, corner3, corner4])

    piece = PuzzlePiece(
        piece_id=4,
        contour_mm=contour,
        mask=np.zeros((10, 10), dtype=np.uint8),
        bbox_mm=(0.0, 0.0, 40.0, 40.0)
    )

    # High threshold: fewer splits
    config = MatchingConfig()
    config.curvature_angle_threshold_deg = 60.0  # High threshold

    segments = segment_piece(piece, config)

    # With high threshold (60°), square corners might not split
    # Expect fewer segments than with default threshold (30°)
    assert len(segments) <= 3, \
        f"High threshold should create ≤3 segments, got {len(segments)}"

    print(f"✓ ({len(segments)} segments with threshold={config.curvature_angle_threshold_deg}°)")


def test_wraparound_merge_closed_contour():
    """Test wraparound merge for closed contours."""
    print("Test 11: wraparound merge (closed)...", end=" ")

    # Create closed square (first pt = last pt within 1mm)
    corner1 = np.linspace([0, 0], [20, 0], 10)
    corner2 = np.linspace([20, 0], [20, 20], 10)
    corner3 = np.linspace([20, 20], [0, 20], 10)
    corner4 = np.linspace([0, 20], [0, 0.1], 10)  # Close to [0, 0] (0.1mm apart)
    contour = np.vstack([corner1, corner2, corner3, corner4])

    piece_closed = PuzzlePiece(
        piece_id=5,
        contour_mm=contour,
        mask=np.zeros((10, 10), dtype=np.uint8),
        bbox_mm=(0.0, 0.0, 20.0, 20.0)
    )

    # Enable wraparound merge
    config_wrap = MatchingConfig()
    config_wrap.enable_wraparound_merge = True
    config_wrap.wraparound_merge_dist_mm = 1.0  # 0.1mm < 1.0mm → closed

    segments_wrap = segment_piece(piece_closed, config_wrap)

    # Disable wraparound merge
    config_nowrap = MatchingConfig()
    config_nowrap.enable_wraparound_merge = False

    segments_nowrap = segment_piece(piece_closed, config_nowrap)

    # With wraparound: should have fewer segments (last + first merged)
    assert len(segments_wrap) < len(segments_nowrap), \
        f"Wraparound should reduce segments: {len(segments_wrap)} vs {len(segments_nowrap)}"

    print(f"✓ ({len(segments_nowrap)} → {len(segments_wrap)} segments with wraparound)")


def run_all_tests():
    """Run all segmentation tests."""
    print("=" * 60)
    print("Step 3: Segmentierung + Flatness - Tests")
    print("=" * 60)
    print()

    test_arclength()
    test_chord_computation()
    test_direction_angle()
    test_flatness_error_straight_line()
    test_flatness_error_curved()
    test_flatness_error_degenerate_chord()
    test_segment_piece_basic()
    test_segment_piece_min_length()
    test_segment_piece_simple_contour()
    test_segment_piece_high_threshold()
    test_wraparound_merge_closed_contour()

    print()
    print("=" * 60)
    print("✅ Alle Tests bestanden (11/11)")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
