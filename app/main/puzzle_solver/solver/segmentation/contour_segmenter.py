"""
Contour Segmentation for Puzzle Pieces.

This module implements adaptive contour segmentation via curvature-based splitting
and length-based merging. Segments are stable (consistent IDs) and include geometric
properties (chord, direction, flatness).

Algorithm:
1. Split: Find curvature maxima (angle change > threshold)
2. Merge: Combine segments < min_frame_seg_len_mm
3. Target: 4-12 segments per piece
4. Compute: chord, direction_angle_deg, flatness_error

See docs/design/03_matching.md Phase 1 for specification.
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import PuzzlePiece, ContourSegment
    from ..config import MatchingConfig


def segment_piece(piece: PuzzlePiece, config: MatchingConfig) -> List[ContourSegment]:
    """
    Segment piece contour into stable segments.

    Args:
        piece: PuzzlePiece with contour_mm (N, 2) in mm
        config: MatchingConfig with segmentation parameters

    Returns:
        List of ContourSegment with:
        - segment_id: 0, 1, 2, ... (sequential)
        - length_mm ≥ min_frame_seg_len_mm (if possible)
        - chord, direction_angle_deg, flatness_error computed
        - profile_1d: None (lazy, computed in Step 5)

    Algorithm:
        1. Split at curvature maxima (angle change > 30°)
        2. Merge segments < min_frame_seg_len_mm
        3. Target count: 4-12 segments
        4. Compute geometric properties for each segment

    Notes:
        - Coordinate system: Inherits from piece.contour_mm (typically M or F)
        - Closed contours: First point ≈ last point assumed
        - Edge cases: Accept < 4 or > 12 segments if merge constraints violated

    Raises:
        ValueError: If piece.contour_mm is None (must run convert_pieces_px_to_mm() first)

    Example:
        >>> piece = PuzzlePiece(piece_id=1, contour_mm=..., ...)
        >>> config = MatchingConfig()
        >>> segments = segment_piece(piece, config)
        >>> assert 4 <= len(segments) <= 12
    """
    from ..models import ContourSegment

    # Guard: Ensure contour_mm is present
    if piece.contour_mm is None:
        raise ValueError(
            f"Piece {piece.piece_id}: contour_mm is None. "
            f"Run convert_pieces_px_to_mm() before segmentation."
        )

    contour_mm = piece.contour_mm
    target_range = config.target_seg_count_range
    min_len_mm = config.min_frame_seg_len_mm

    # Step 1: Find split candidates via curvature maxima
    split_indices = _find_split_candidates(
        contour_mm,
        angle_threshold_deg=config.curvature_angle_threshold_deg,
        window_size=config.curvature_window_pts
    )

    # Step 2: Create initial segments from splits
    segments_raw = _create_segments_from_splits(contour_mm, split_indices)

    # Step 3: Merge short segments to reach target count and min length
    segments_merged = _merge_short_segments(segments_raw, min_len_mm, target_range)

    # Step 3b: Wraparound merge for closed contours (if enabled)
    if config.enable_wraparound_merge and len(segments_merged) > 1:
        segments_merged = _wraparound_merge_if_closed(
            segments_merged,
            contour_mm,
            config.wraparound_merge_dist_mm,
            min_len_mm
        )

    # Step 4: Create ContourSegment objects with all properties
    segments = []
    for seg_id, seg_points in enumerate(segments_merged):
        chord = _compute_chord(seg_points)
        direction_angle_deg = _compute_direction_angle(chord)
        flatness_error = _compute_flatness_error(seg_points, chord)
        length_mm = _compute_arclength(seg_points)

        seg = ContourSegment(
            piece_id=piece.piece_id,
            segment_id=seg_id,
            points_mm=seg_points,
            length_mm=length_mm,
            chord=chord,
            direction_angle_deg=direction_angle_deg,
            flatness_error=flatness_error,
            profile_1d=None  # Lazy (Step 5: Inner Matching)
        )
        segments.append(seg)

    return segments


def _find_split_candidates(
    contour_mm: np.ndarray,
    angle_threshold_deg: float,
    window_size: int
) -> List[int]:
    """
    Find split candidates via curvature maxima (angle change detection).

    Args:
        contour_mm: Contour points (N, 2) in mm
        angle_threshold_deg: Angle change threshold for split (from config)
        window_size: Tangent angle window size (from config, should be odd)

    Returns:
        List of split indices (local curvature maxima)

    Algorithm:
        - Compute local tangent angles at each point (window size from config)
        - Find points with angle change > threshold
        - Return indices of curvature maxima

    Notes:
        - Window size: From config (default 5 points = ±2 around center)
        - Angle change: Absolute difference between successive tangents
        - Edge handling: Skip first/last half_window points
    """
    N = len(contour_mm)
    min_points = window_size * 2
    if N < min_points:  # Too few points for meaningful segmentation
        return []

    half_window = window_size // 2
    split_indices = []

    # Compute tangent angles at each point
    angles = []
    for i in range(half_window, N - half_window):
        # Local tangent: direction from prev to next point
        prev_idx = i - half_window
        next_idx = i + half_window
        vec = contour_mm[next_idx] - contour_mm[prev_idx]
        angle_rad = np.arctan2(vec[1], vec[0])
        angles.append(angle_rad)

    # Find curvature maxima (large angle changes)
    for i in range(1, len(angles)):
        # Angle difference (handle wrapping around ±π)
        angle_diff_rad = angles[i] - angles[i - 1]
        # Normalize to [-π, π]
        angle_diff_rad = np.arctan2(np.sin(angle_diff_rad), np.cos(angle_diff_rad))
        angle_diff_deg = np.abs(np.rad2deg(angle_diff_rad))

        if angle_diff_deg > angle_threshold_deg:
            # Map back to original contour index
            split_idx = i + half_window
            split_indices.append(split_idx)

    return split_indices


def _create_segments_from_splits(contour_mm: np.ndarray, split_indices: List[int]) -> List[np.ndarray]:
    """
    Create initial segments from split indices.

    Args:
        contour_mm: Contour points (N, 2) in mm
        split_indices: List of split point indices

    Returns:
        List of segment point arrays

    Notes:
        - Segments: [0:split[0]], [split[0]:split[1]], ..., [split[-1]:N]
        - If no splits: Return entire contour as single segment
        - Closed contours: Last segment may wrap around (handled by merge)
    """
    if not split_indices:
        # No splits: return entire contour as single segment
        return [contour_mm.copy()]

    # Ensure splits are sorted and unique
    splits = sorted(set(split_indices))

    segments = []
    start_idx = 0
    for split_idx in splits:
        seg_points = contour_mm[start_idx:split_idx + 1].copy()
        if len(seg_points) >= 2:  # Valid segment
            segments.append(seg_points)
        start_idx = split_idx

    # Last segment: from last split to end
    if start_idx < len(contour_mm):
        seg_points = contour_mm[start_idx:].copy()
        if len(seg_points) >= 2:
            segments.append(seg_points)

    return segments


def _merge_short_segments(
    segments: List[np.ndarray],
    min_len_mm: float,
    target_range: tuple[int, int]
) -> List[np.ndarray]:
    """
    Merge short segments to reach target count and minimum length.

    Args:
        segments: List of segment point arrays
        min_len_mm: Minimum segment length threshold (mm)
        target_range: Target segment count range (min, max)

    Returns:
        List of merged segment point arrays

    Algorithm:
        - Greedy: Iteratively merge shortest segment with neighbor
        - Stop when: count in target range AND all segments ≥ min_len
        - Or: No more segments < min_len (accept violation if unavoidable)

    Strategy:
        - Find shortest segment < min_len
        - Merge with shorter neighbor (left or right)
        - Repeat until target reached or no short segments
    """
    target_min, target_max = target_range
    merged = [seg.copy() for seg in segments]

    while len(merged) > target_min:
        # Compute lengths
        lengths = [_compute_arclength(seg) for seg in merged]

        # Find shortest segment < min_len
        short_indices = [i for i, length in enumerate(lengths) if length < min_len_mm]

        if not short_indices:
            # No short segments left
            break

        if len(merged) <= target_max and all(length >= min_len_mm for length in lengths):
            # In target range and all segments long enough
            break

        # Find shortest segment
        shortest_idx = min(short_indices, key=lambda i: lengths[i])

        # Merge with shorter neighbor
        if shortest_idx == 0:
            # First segment: merge with next
            neighbor_idx = 1
        elif shortest_idx == len(merged) - 1:
            # Last segment: merge with previous
            neighbor_idx = shortest_idx - 1
        else:
            # Middle segment: merge with shorter neighbor
            left_len = lengths[shortest_idx - 1]
            right_len = lengths[shortest_idx + 1]
            neighbor_idx = shortest_idx - 1 if left_len < right_len else shortest_idx + 1

        # Merge segments
        if neighbor_idx < shortest_idx:
            # Merge neighbor + shortest
            merged_seg = np.vstack([merged[neighbor_idx], merged[shortest_idx]])
            merged[neighbor_idx] = merged_seg
            merged.pop(shortest_idx)
        else:
            # Merge shortest + neighbor
            merged_seg = np.vstack([merged[shortest_idx], merged[neighbor_idx]])
            merged[shortest_idx] = merged_seg
            merged.pop(neighbor_idx)

    return merged


def _wraparound_merge_if_closed(
    segments: List[np.ndarray],
    contour_mm: np.ndarray,
    max_dist_mm: float,
    min_len_mm: float
) -> List[np.ndarray]:
    """
    Merge first and last segments if contour is closed (wraparound merge).

    Args:
        segments: List of segment point arrays
        contour_mm: Original contour points (N, 2) in mm
        max_dist_mm: Maximum distance between first and last point to consider closed
        min_len_mm: Minimum segment length (for merge decision)

    Returns:
        List of segments with wraparound merge applied (if applicable)

    Algorithm:
        1. Check if contour is closed: distance(contour[0], contour[-1]) < max_dist_mm
        2. If closed AND len(segments) > 1:
           - Merge last segment + first segment → new first segment
           - Remove last segment
        3. Return updated segments

    Notes:
        - Only merges if contour is closed (first ≈ last point)
        - Preserves segment order (last merged into first)
        - Useful for closed contours (e.g., outer puzzle piece edges)
    """
    if len(segments) <= 1:
        return segments

    # Check if contour is closed
    first_pt = contour_mm[0]
    last_pt = contour_mm[-1]
    dist = np.linalg.norm(last_pt - first_pt)

    if dist >= max_dist_mm:
        # Not closed, no merge
        return segments

    # Contour is closed: merge last + first segment
    # New first segment: last segment points + first segment points
    merged_first = np.vstack([segments[-1], segments[0]])

    # Replace first segment with merged, remove last
    result = [merged_first] + segments[1:-1]

    return result


def _compute_chord(points_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute chord endpoints (start, end).

    Args:
        points_mm: Segment points (M, 2) in mm

    Returns:
        Tuple (start_pt, end_pt) where each is np.ndarray shape (2,)

    Notes:
        - Start: First point in segment
        - End: Last point in segment
        - Format: (start_pt.copy(), end_pt.copy()) for immutability
    """
    start_pt = points_mm[0].copy()
    end_pt = points_mm[-1].copy()
    return (start_pt, end_pt)


def _compute_direction_angle(chord: tuple[np.ndarray, np.ndarray]) -> float:
    """
    Compute chord direction angle in degrees [-180, 180).

    Args:
        chord: Tuple (start_pt, end_pt) in mm

    Returns:
        Direction angle in degrees, counterclockwise positive

    Formula:
        angle = atan2(dy, dx) where dx = end_x - start_x, dy = end_y - start_y

    Notes:
        - Range: [-180, 180) degrees
        - 0°: Right (positive X)
        - 90°: Up (positive Y)
        - ±180°: Left (negative X)
    """
    start_pt, end_pt = chord
    dx = end_pt[0] - start_pt[0]
    dy = end_pt[1] - start_pt[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.rad2deg(angle_rad)

    # Normalize to [-180, 180)
    if angle_deg >= 180.0:
        angle_deg -= 360.0

    return float(angle_deg)


def _compute_flatness_error(points_mm: np.ndarray, chord: tuple[np.ndarray, np.ndarray]) -> float:
    """
    Compute RMS perpendicular distance from points to chord line.

    Args:
        points_mm: Segment points (M, 2) in mm
        chord: Tuple (start_pt, end_pt) in mm

    Returns:
        RMS flatness error in mm

    Algorithm:
        1. Define chord line: L(t) = start + t*(end - start)
        2. For each point: compute perpendicular distance to line
        3. Return RMS: sqrt(mean(distances²))

    Formula:
        - Chord vector: v = end - start
        - Point P distance to line: ||perp_component(P - start, v)||
        - Perpendicular component: (P - start) - proj(P - start, v)

    Notes:
        - Degenerate chord (start ≈ end): return 0.0
        - Lower error = straighter segment (better for frame contact)
        - Used in frame contact scoring (see docs/design/04_scoring.md)
    """
    start_pt, end_pt = chord
    chord_vec = end_pt - start_pt
    chord_len = np.linalg.norm(chord_vec)

    if chord_len < 1e-6:  # Degenerate chord (start ≈ end)
        return 0.0

    chord_dir = chord_vec / chord_len

    # Compute perpendicular distances
    distances = []
    for pt in points_mm:
        vec_to_pt = pt - start_pt
        # Project onto chord direction
        proj_len = np.dot(vec_to_pt, chord_dir)
        # Perpendicular component
        perp = vec_to_pt - proj_len * chord_dir
        dist = np.linalg.norm(perp)
        distances.append(dist)

    # RMS
    rms = np.sqrt(np.mean(np.array(distances) ** 2))
    return float(rms)


def _compute_arclength(points_mm: np.ndarray) -> float:
    """
    Compute total arclength of point sequence.

    Args:
        points_mm: Segment points (M, 2) in mm

    Returns:
        Total arclength in mm

    Formula:
        arclength = sum(||p[i+1] - p[i]||) for i in range(M-1)

    Notes:
        - Euclidean distance between consecutive points
        - Piecewise linear approximation of curve
    """
    if len(points_mm) < 2:
        return 0.0

    diffs = np.diff(points_mm, axis=0)  # (M-1, 2)
    distances = np.linalg.norm(diffs, axis=1)  # (M-1,)
    total_length = np.sum(distances)

    return float(total_length)
