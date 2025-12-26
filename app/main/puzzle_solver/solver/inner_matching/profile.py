"""
1D Profile Extraction for Inner Edge Matching.

This module extracts 1D signed chord distance profiles from contour segments:
- extract_1d_profile: Main API for profile extraction

See docs/design/03_matching.md Phase 3 for algorithm details.
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import ContourSegment
    from ..config import MatchingConfig


def extract_1d_profile(
    seg: ContourSegment,
    config: MatchingConfig
) -> np.ndarray:
    """
    Extract 1D signed chord distance profile from segment.

    Args:
        seg: ContourSegment with points_mm, chord
        config: MatchingConfig with profile_samples_N

    Returns:
        profile: (N,) array of signed perpendicular distances to chord (mm)

    Algorithm:
        1. Resample segment points to N samples along arclength
        2. For each resampled point, compute signed perpendicular distance to chord
           - Positive: right of chord (right-hand rule)
           - Negative: left of chord
        3. Optional smoothing (currently disabled)

    Notes:
        - Resampling ensures consistent N across all segments
        - Signed distance preserves orientation information
        - Profile is invariant to segment start/end swap (symmetric)
        - Lazy computation: Sets seg.profile_1d in-place
        - Smoothing: config.profile_smoothing_window currently ignored

    Example:
        >>> seg = ContourSegment(points_mm=np.array([[0,0], [10,1], [20,0]]), ...)
        >>> config = MatchingConfig(profile_samples_N=128)
        >>> profile = extract_1d_profile(seg, config)
        >>> assert profile.shape == (128,)
        >>> assert np.max(np.abs(profile)) < 2.0  # Low flatness → small profile
    """
    N = config.profile_samples_N

    # 1. Compute cumulative arclength
    points = seg.points_mm
    M = len(points)

    if M < 2:
        # Degenerate segment: return zero profile
        seg.profile_1d = np.zeros(N)
        return seg.profile_1d

    # Cumulative lengths: [0, dist(p0→p1), dist(p0→p2), ...]
    diffs = np.diff(points, axis=0)  # (M-1, 2)
    seg_lengths = np.linalg.norm(diffs, axis=1)  # (M-1,)
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])  # (M,)

    total_length = cumulative_lengths[-1]

    if total_length < 1e-6:
        # Degenerate: all points identical
        seg.profile_1d = np.zeros(N)
        return seg.profile_1d

    # 2. Resample points at N uniform arclength positions
    target_lengths = np.linspace(0, total_length, N)  # (N,)

    # Interpolate x and y separately
    resampled_x = np.interp(target_lengths, cumulative_lengths, points[:, 0])
    resampled_y = np.interp(target_lengths, cumulative_lengths, points[:, 1])
    resampled_points = np.stack([resampled_x, resampled_y], axis=1)  # (N, 2)

    # 3. Compute signed perpendicular distance to chord
    chord_start, chord_end = seg.chord
    chord_vec = chord_end - chord_start  # (2,)
    chord_len = np.linalg.norm(chord_vec)

    if chord_len < 1e-6:
        # Degenerate chord: start == end
        seg.profile_1d = np.zeros(N)
        return seg.profile_1d

    chord_vec_normalized = chord_vec / chord_len

    # Perpendicular vector (90° CCW rotation, right-hand rule)
    chord_perp = np.array([-chord_vec_normalized[1], chord_vec_normalized[0]])

    # For each resampled point: signed distance = dot(p - start, perp)
    vec_to_points = resampled_points - chord_start  # (N, 2)
    signed_distances = np.dot(vec_to_points, chord_perp)  # (N,)

    # 4. Optional smoothing (currently disabled)
    # if config.profile_smoothing_window > 1:
    #     from scipy.ndimage import uniform_filter1d
    #     signed_distances = uniform_filter1d(signed_distances, config.profile_smoothing_window)

    # 5. Store and return
    seg.profile_1d = signed_distances
    return signed_distances
