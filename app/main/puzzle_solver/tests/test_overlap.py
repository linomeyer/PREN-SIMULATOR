"""
Tests for Overlap/Collision Detection (Step 7).

Test Spec: docs/test_spec/07_overlap_test_spec.md
Implements Tests 1-16 for penetration_depth() and penetration_depth_max().

SAT/MTV implementation will be in solver/overlap/collision.py (Phase 2).
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from solver.beam_solver.state import SolverState
from solver.models import PuzzlePiece, Pose2D

# Tolerances from Spec §1
EPS = 1e-9
TOL_MM = 1e-3

# Skip tests if collision.py not implemented yet
pytest.importorskip("solver.overlap.collision", reason="collision.py not implemented yet")

from solver.overlap.collision import penetration_depth, penetration_depth_max


# ========== Test Polygons (Spec §2) ==========

def sq10():
    """Square 10x10mm at origin, CCW"""
    return np.array([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10]
    ], dtype=np.float64)


def sq10_shift(dx, dy):
    """Square 10x10mm shifted by (dx, dy)"""
    return sq10() + np.array([dx, dy])


def rect20x10():
    """Rectangle 20x10mm at origin, CCW"""
    return np.array([
        [0, 0],
        [20, 0],
        [20, 10],
        [0, 10]
    ], dtype=np.float64)


def thin():
    """Very thin polygon 0.1x10mm, CCW"""
    return np.array([
        [0, 0],
        [0.1, 0],
        [0.1, 10],
        [0, 10]
    ], dtype=np.float64)


def l_shape():
    """L-shaped polygon (non-convex), CCW"""
    return np.array([
        [0, 0],
        [10, 0],
        [10, 3],
        [3, 3],
        [3, 10],
        [0, 10]
    ], dtype=np.float64)


# ========== Unit Tests: penetration_depth(poly_a, poly_b) ==========
# Spec §3: Tests 1-11 (Convex Polygons)

def test_01_no_overlap_separated():
    """Test 1: No Overlap (separated squares)"""
    a = sq10()
    b = sq10_shift(20, 0)
    depth = penetration_depth(a, b)
    assert abs(depth - 0.0) <= TOL_MM


def test_02_tangential_edge_contact():
    """Test 2: Tangential edge contact (touching, no overlap)"""
    a = sq10()
    b = sq10_shift(10, 0)  # touches at x=10
    depth = penetration_depth(a, b)
    assert abs(depth - 0.0) <= TOL_MM


def test_03_tangential_corner_contact():
    """Test 3: Tangential corner contact (touching at one corner)"""
    a = sq10()
    b = sq10_shift(10, 10)  # touches at (10,10)
    depth = penetration_depth(a, b)
    assert abs(depth - 0.0) <= TOL_MM


def test_04_small_overlap():
    """Test 4: Small Overlap (0.5mm)"""
    a = sq10()
    b = sq10_shift(9.5, 0)  # overlap in x: 0.5
    depth = penetration_depth(a, b)
    assert 0.49 <= depth <= 0.51


def test_05_larger_overlap():
    """Test 5: Larger Overlap (3.0mm)"""
    a = sq10()
    b = sq10_shift(7.0, 0)  # overlap in x: 3.0
    depth = penetration_depth(a, b)
    assert 2.99 <= depth <= 3.01


def test_06_full_containment():
    """Test 6: Full Containment (inner square inside outer square)"""
    outer = np.array([
        [0, 0],
        [20, 0],
        [20, 20],
        [0, 20]
    ], dtype=np.float64)
    inner = np.array([
        [5, 5],
        [15, 5],
        [15, 15],
        [5, 15]
    ], dtype=np.float64)

    # MTV minimal: 5.0mm (shortest distance to make disjoint)
    depth_outer_inner = penetration_depth(outer, inner)
    depth_inner_outer = penetration_depth(inner, outer)

    assert 4.99 <= depth_outer_inner <= 5.01
    assert 4.99 <= depth_inner_outer <= 5.01  # Symmetry


def test_07_identical_polygons():
    """Test 7: Identical polygons (maximal overlap case)"""
    a = sq10()
    b = sq10()  # identical
    depth = penetration_depth(a, b)

    # MTV minimal is 10.0mm (along x or y to make disjoint)
    assert 9.99 <= depth <= 10.01


def test_08_symmetry_property():
    """Test 8: Symmetry property (random shift overlap)"""
    a = rect20x10()
    b = np.array([
        [15, 2],
        [35, 2],
        [35, 12],
        [15, 12]
    ], dtype=np.float64)  # Overlap in x: 5, in y: 8

    depth_ab = penetration_depth(a, b)
    depth_ba = penetration_depth(b, a)

    # Symmetry
    assert abs(depth_ab - depth_ba) <= TOL_MM

    # Min overlap is 5.0mm
    assert 4.99 <= depth_ab <= 5.01


def test_09_translation_invariance():
    """Test 9: Translation invariance"""
    a1 = sq10()
    b1 = sq10_shift(9.5, 0)  # depth ~0.5

    a2 = sq10_shift(100, -50)
    b2 = sq10_shift(109.5, -50)

    depth1 = penetration_depth(a1, b1)
    depth2 = penetration_depth(a2, b2)

    assert abs(depth1 - depth2) <= TOL_MM
    assert 0.49 <= depth1 <= 0.51  # Same as Test 4
    assert 0.49 <= depth2 <= 0.51


def test_10_robustness_cw_vs_ccw():
    """Test 10: Robustness to polygon orientation (CW vs CCW)"""
    a_ccw = sq10()
    a_cw = np.flipud(sq10())  # Reversed order (CW)
    b = sq10_shift(9.5, 0)

    depth_ccw = penetration_depth(a_ccw, b)
    depth_cw = penetration_depth(a_cw, b)

    # Both should give ~0.5mm (same as Test 4)
    assert 0.49 <= depth_ccw <= 0.51
    assert 0.49 <= depth_cw <= 0.51


def test_11_thin_polygon_stability():
    """Test 11: Very thin polygon stability (no overlap)"""
    a = thin()
    b = thin() + np.array([1.0, 0])  # Separated by 1.0mm (width=0.1)

    depth = penetration_depth(a, b)
    assert abs(depth - 0.0) <= TOL_MM  # No NaNs/Exceptions


# ========== Unit Tests: Non-convex Polygons ==========
# Spec §4: Tests 12-13 (Strategy-agnostic)

def test_12_nonconvex_no_overlap():
    """Test 12: Non-convex no overlap"""
    a = l_shape()
    b = sq10_shift(30, 0)
    depth = penetration_depth(a, b)
    assert abs(depth - 0.0) <= TOL_MM


def test_13_nonconvex_clear_overlap():
    """Test 13: Non-convex clear overlap (must be > 0)"""
    a = l_shape()
    b = np.array([
        [2, 2],
        [6, 2],
        [6, 6],
        [2, 6]
    ], dtype=np.float64)  # Inside L-shape, clear intersection

    depth = penetration_depth(a, b)

    # Strategy-agnostic bounds
    assert depth > 0.1  # Clearly > 0
    assert depth < 4.0  # Upper bound (sanity check)


# ========== Unit Tests: penetration_depth_max(state, pieces) ==========
# Spec §5: Tests 14-16 (State-Level)

def test_14_max_depth_single_overlap_pair():
    """Test 14: Max depth over 3 pieces (1 overlap pair)"""
    # Setup pieces
    pieces = {
        1: PuzzlePiece(piece_id=1, contour_mm=sq10()),
        2: PuzzlePiece(piece_id=2, contour_mm=sq10()),
        3: PuzzlePiece(piece_id=3, contour_mm=sq10())
    }

    # Setup state
    state = SolverState(all_piece_ids={1, 2, 3})
    state.placed_pieces = {1, 2, 3}
    state.poses_F = {
        1: Pose2D(0, 0, 0),
        2: Pose2D(9.5, 0, 0),  # overlap 0.5mm with p1
        3: Pose2D(30, 0, 0)    # no overlap
    }

    depth_max = penetration_depth_max(state, pieces)
    assert 0.49 <= depth_max <= 0.51


def test_15_max_depth_returns_maximum():
    """Test 15: Max depth returns the maximum (two overlapping pairs)"""
    pieces = {
        1: PuzzlePiece(piece_id=1, contour_mm=sq10()),
        2: PuzzlePiece(piece_id=2, contour_mm=sq10()),
        3: PuzzlePiece(piece_id=3, contour_mm=sq10())
    }

    state = SolverState(all_piece_ids={1, 2, 3})
    state.placed_pieces = {1, 2, 3}
    state.poses_F = {
        1: Pose2D(0, 0, 0),
        2: Pose2D(9.5, 0, 0),  # overlap 0.5mm with p1
        3: Pose2D(-7.0, 0, 0)  # overlap 3.0mm with p1
    }

    depth_max = penetration_depth_max(state, pieces)
    assert 2.99 <= depth_max <= 3.01  # Max is 3.0mm


def test_16_ignores_unplaced_pieces():
    """Test 16: Ignores unplaced pieces (pose missing)"""
    pieces = {
        1: PuzzlePiece(piece_id=1, contour_mm=sq10()),
        2: PuzzlePiece(piece_id=2, contour_mm=sq10()),
        3: PuzzlePiece(piece_id=3, contour_mm=sq10())
    }

    # Only p1, p2 placed; p3 unplaced
    state = SolverState(all_piece_ids={1, 2, 3})
    state.placed_pieces = {1, 2}
    state.poses_F = {
        1: Pose2D(0, 0, 0),
        2: Pose2D(9.5, 0, 0)  # overlap 0.5mm
    }
    # p3 not in poses_F

    depth_max = penetration_depth_max(state, pieces)
    assert 0.49 <= depth_max <= 0.51  # No exception from p3
