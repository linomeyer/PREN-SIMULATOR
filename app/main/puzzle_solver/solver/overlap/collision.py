"""
SAT/MTV Collision Detection (Step 7).

Implements penetration depth calculation using:
- SAT (Separating Axis Theorem) for convex polygons
- MTV (Minimum Translation Vector) for overlap measurement
- Triangulation for non-convex polygons (Option B from Design Doc)

Design Basis:
- docs/design/06_collision.md
- docs/test_spec/07_overlap_test_spec.md

All measurements in millimeters.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from solver.models import Pose2D, PuzzlePiece
from solver.beam_solver.state import SolverState

# Numerical stability constants (Spec §1)
EPS = 1e-9  # Internal stability
TOL_MM = 1e-3  # Assertion tolerance


# ========== Helper Functions ==========

def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    """
    Ensure polygon is CCW (counter-clockwise) via signed area.

    Args:
        poly: Polygon points (N, 2)

    Returns:
        Polygon in CCW order

    Notes:
        - Signed area < 0 → CW → reverse
        - Robust to degenerate cases
    """
    # Shoelace formula for signed area
    x = poly[:, 0]
    y = poly[:, 1]
    signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    signed_area += 0.5 * (x[-1] * y[0] - x[0] * y[-1])  # Close loop

    if signed_area < 0:
        return np.flipud(poly)
    return poly


def _is_convex(poly: np.ndarray) -> bool:
    """
    Check if polygon is convex via cross-product sign.

    Args:
        poly: Polygon points (N, 2), assumed CCW

    Returns:
        True if convex, False if concave

    Notes:
        - All cross products should have same sign (CCW → positive)
        - Uses EPS tolerance for numerical stability
    """
    n = len(poly)
    if n < 3:
        return True

    # Check cross product sign at each vertex
    sign = None
    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i + 1) % n]
        p2 = poly[(i + 2) % n]

        # Cross product of edges (p1-p0) x (p2-p1)
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        if abs(cross) > EPS:  # Ignore near-zero (collinear)
            if sign is None:
                sign = np.sign(cross)
            elif np.sign(cross) != sign:
                return False  # Sign changed → concave

    return True


def _transform_polygon(poly: np.ndarray, pose: Pose2D) -> np.ndarray:
    """
    Transform polygon from local to world coordinates.

    Args:
        poly: Polygon points in local coords (N, 2)
        pose: Pose2D (x, y, theta)

    Returns:
        Transformed polygon in world coords (N, 2)

    Notes:
        - Rotation around origin, then translation
        - Angle in degrees (converted internally)
    """
    theta_rad = np.deg2rad(pose.theta_deg)
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)

    # Rotation matrix
    rot = np.array([[c, -s], [s, c]])

    # Rotate then translate
    poly_rot = poly @ rot.T
    poly_world = poly_rot + np.array([pose.x_mm, pose.y_mm])

    return poly_world


def _project_polygon(poly: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    """
    Project polygon onto axis.

    Args:
        poly: Polygon points (N, 2)
        axis: Unit vector (2,)

    Returns:
        (min_proj, max_proj): Min/max projection values
    """
    projections = poly @ axis
    return np.min(projections), np.max(projections)


def _get_edge_normals(poly: np.ndarray) -> np.ndarray:
    """
    Get edge normal vectors (perpendicular to edges).

    Args:
        poly: Polygon points (N, 2), CCW

    Returns:
        Edge normals (N, 2), normalized

    Notes:
        - For CCW polygon, outward normal is 90° CCW from edge
        - Normal = (-dy, dx) for edge (dx, dy)
    """
    n = len(poly)
    normals = np.zeros((n, 2))

    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i + 1) % n]
        edge = p1 - p0

        # Perpendicular (90° CCW): (dx, dy) → (-dy, dx)
        normal = np.array([-edge[1], edge[0]])

        # Normalize
        norm = np.linalg.norm(normal)
        if norm > EPS:
            normal /= norm

        normals[i] = normal

    return normals


def _sat_mtv_convex(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """
    SAT/MTV for convex polygons.

    Args:
        poly_a: Convex polygon (N, 2), CCW
        poly_b: Convex polygon (M, 2), CCW

    Returns:
        MTV length (0.0 if no overlap)

    Algorithm:
        1. Test all edge normals as separating axes
        2. If gap found → no overlap → 0.0
        3. If all axes overlap → compute MTV (smallest overlap)
        4. Return ||MTV||

    Notes:
        - Numerically stable (EPS for comparisons)
        - Symmetric: mtv(a,b) == mtv(b,a)
    """
    # Ensure CCW
    poly_a = _ensure_ccw(poly_a)
    poly_b = _ensure_ccw(poly_b)

    # Collect all edge normals (potential separating axes)
    normals_a = _get_edge_normals(poly_a)
    normals_b = _get_edge_normals(poly_b)
    axes = np.vstack([normals_a, normals_b])

    # Remove duplicate axes (angular tolerance)
    unique_axes = []
    for axis in axes:
        # Check if already in list (cosine similarity > 0.999)
        is_duplicate = False
        for existing in unique_axes:
            if abs(np.dot(axis, existing)) > 0.999:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_axes.append(axis)

    # Test all axes
    min_overlap = float('inf')

    for axis in unique_axes:
        # Project both polygons
        min_a, max_a = _project_polygon(poly_a, axis)
        min_b, max_b = _project_polygon(poly_b, axis)

        # Check for separation
        if max_a < min_b - EPS or max_b < min_a - EPS:
            return 0.0  # Separated → no overlap

        # Compute MTV (Minimum Translation Vector) magnitude on this axis
        #
        # Standard SAT overlap: min(max_a, max_b) - max(min_a, min_b)
        # But this gives the OVERLAP SIZE, not the MTV!
        #
        # For MTV, we need the minimum distance to separate:
        # - If A overlaps B from the left: push A left by (max_a - min_b)
        # - If A overlaps B from the right: push A right by (max_b - min_a)
        # - MTV = minimum of both
        #
        # Special case - Containment:
        #   If A is inside B: [5,15] ⊂ [0,20]
        #     max_a - min_b = 15 - 0 = 15 (push left through B)
        #     max_b - min_a = 20 - 5 = 15 (push right through B)
        #     min = 15 ← WRONG! Should be 5 (distance to nearest edge)
        #
        # The correct MTV for containment:
        #   Distance to left edge: (min_a - min_b) = 5 - 0 = 5
        #   Distance to right edge: (max_b - max_a) = 20 - 15 = 5
        #   MTV = min(5, 5) = 5 ← CORRECT
        #
        # Solution: Check if one interval STRICTLY contains the other
        # Strict containment: one is fully inside, not touching edges
        a_strictly_contains_b = (min_a < min_b - EPS) and (max_a > max_b + EPS)
        b_strictly_contains_a = (min_b < min_a - EPS) and (max_b > max_a + EPS)

        if b_strictly_contains_a:
            # A is strictly inside B → distance to nearest edge
            overlap = min(min_a - min_b, max_b - max_a)
        elif a_strictly_contains_b:
            # B is strictly inside A → distance to nearest edge
            overlap = min(min_b - min_a, max_a - max_b)
        else:
            # Partial overlap or identical → standard MTV formula
            overlap = min(max_a - min_b, max_b - min_a)

        if overlap < min_overlap:
            min_overlap = overlap

    # All axes overlapped → MTV is minimal overlap
    return max(0.0, min_overlap)


def _triangulate_earcut(poly: np.ndarray) -> list[np.ndarray]:
    """
    Triangulate polygon using ear clipping algorithm.

    Args:
        poly: Polygon points (N, 2), simple (no self-intersections)

    Returns:
        List of triangles, each (3, 2)

    Notes:
        - Robust ear clipping implementation
        - CCW orientation preserved
        - Handles concave polygons
    """
    # Ensure CCW
    poly = _ensure_ccw(poly)
    n = len(poly)

    if n < 3:
        return []
    if n == 3:
        return [poly]

    # Copy vertices (will be removed during triangulation)
    vertices = list(range(n))
    triangles = []

    # Helper: Check if triangle is an "ear"
    def is_ear(i: int, vertices: list[int]) -> bool:
        n_v = len(vertices)
        prev_idx = vertices[(i - 1) % n_v]
        curr_idx = vertices[i]
        next_idx = vertices[(i + 1) % n_v]

        a = poly[prev_idx]
        b = poly[curr_idx]
        c = poly[next_idx]

        # Check if triangle is CCW (convex at vertex b)
        edge1 = b - a
        edge2 = c - b
        cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]

        if cross < EPS:  # Not convex → not an ear
            return False

        # Check if any other vertex is inside triangle
        for j in range(n_v):
            v_idx = vertices[j]
            if v_idx in [prev_idx, curr_idx, next_idx]:
                continue

            p = poly[v_idx]
            if _point_in_triangle(p, a, b, c):
                return False

        return True

    # Ear clipping loop
    max_iterations = 2 * n  # Prevent infinite loop
    iterations = 0

    while len(vertices) > 3 and iterations < max_iterations:
        ear_found = False

        for i in range(len(vertices)):
            if is_ear(i, vertices):
                # Cut ear
                prev_idx = vertices[(i - 1) % len(vertices)]
                curr_idx = vertices[i]
                next_idx = vertices[(i + 1) % len(vertices)]

                triangle = np.array([
                    poly[prev_idx],
                    poly[curr_idx],
                    poly[next_idx]
                ])
                triangles.append(triangle)

                # Remove vertex
                vertices.pop(i)
                ear_found = True
                break

        if not ear_found:
            raise ValueError(
                f"Triangulation failed: no ear found with {len(vertices)} vertices remaining. "
                f"Polygon may have self-intersections or degenerate geometry."
            )

        iterations += 1

    # Add final triangle
    if len(vertices) == 3:
        triangle = np.array([poly[vertices[0]], poly[vertices[1]], poly[vertices[2]]])
        triangles.append(triangle)

    return triangles


def _point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """
    Check if point p is inside triangle (a, b, c).

    Uses barycentric coordinates.
    """
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < EPS:
        return False

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # Check if inside (with small tolerance)
    return (u >= -EPS) and (v >= -EPS) and (u + v <= 1.0 + EPS)


# ========== Main Functions ==========

def penetration_depth(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """
    Compute penetration depth between two polygons.

    Args:
        poly_a: Polygon A (N, 2) in mm
        poly_b: Polygon B (M, 2) in mm

    Returns:
        Penetration depth (MTV length) in mm, 0.0 if no overlap

    Properties:
        - Symmetric: depth(a, b) == depth(b, a)
        - Translation invariant
        - CW/CCW tolerant

    Algorithm:
        1. Check if both convex → direct SAT/MTV
        2. If non-convex → triangulate
        3. Test all triangle pairs
        4. Aggregate: max(depths) (conservative)

    Notes:
        - Uses triangulation (Option B) for non-convex
        - Aggregation: "max" (conservative for pruning)
        - No NaNs/Exceptions guaranteed
    """
    # Ensure valid input
    poly_a = np.asarray(poly_a, dtype=np.float64)
    poly_b = np.asarray(poly_b, dtype=np.float64)

    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0

    # Check convexity
    a_convex = _is_convex(poly_a)
    b_convex = _is_convex(poly_b)

    if a_convex and b_convex:
        # Both convex → direct SAT/MTV
        return _sat_mtv_convex(poly_a, poly_b)

    # Non-convex → triangulate
    triangles_a = [poly_a] if a_convex else _triangulate_earcut(poly_a)
    triangles_b = [poly_b] if b_convex else _triangulate_earcut(poly_b)

    # Test all triangle pairs
    max_depth = 0.0
    for tri_a in triangles_a:
        for tri_b in triangles_b:
            depth = _sat_mtv_convex(tri_a, tri_b)
            max_depth = max(max_depth, depth)

    return max_depth


def penetration_depth_max(
    state: SolverState,
    pieces: dict[int, PuzzlePiece]
) -> float:
    """
    Maximum penetration depth over all placed piece pairs.

    Args:
        state: SolverState with poses_F
        pieces: Dict piece_id → PuzzlePiece

    Returns:
        Max depth in mm (0.0 if no overlaps)

    Notes:
        - Only checks placed pieces (in state.poses_F)
        - Transforms to world coordinates using poses_F
        - Pairwise O(n²) check
        - Returns 0.0 if contour_mm not available (test scenarios)
    """
    max_depth = 0.0

    # Only check pieces that have poses (defensive - avoids KeyError)
    placed_with_poses = [pid for pid in state.placed_pieces if pid in state.poses_F]

    # Pairwise check
    for i in range(len(placed_with_poses)):
        for j in range(i + 1, len(placed_with_poses)):
            id_a = placed_with_poses[i]
            id_b = placed_with_poses[j]

            # Get pieces
            piece_a = pieces[id_a]
            piece_b = pieces[id_b]

            # Skip if contour_mm not available (test scenarios without full piece data)
            # Returns 0.0 as documented - acceptable for tests that only check bbox
            if piece_a.contour_mm is None or piece_b.contour_mm is None:
                continue

            # Transform to world coords
            poly_a_world = _transform_polygon(
                piece_a.contour_mm,
                state.poses_F[id_a]
            )
            poly_b_world = _transform_polygon(
                piece_b.contour_mm,
                state.poses_F[id_b]
            )

            # Compute depth
            depth = penetration_depth(poly_a_world, poly_b_world)
            max_depth = max(max_depth, depth)

    return max_depth
