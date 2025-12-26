"""
Inner Match Candidate Generation.

This module generates and ranks inner edge match candidates:
- compute_ncc: Normalized cross-correlation for profile similarity
- generate_inner_candidates: Main API for candidate generation

See docs/design/03_matching.md Phase 3 and 04_scoring.md §B for algorithm details.
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from ..models import ContourSegment, InnerMatchCandidate
    from ..config import MatchingConfig


def compute_ncc(
    profile_a: np.ndarray,
    profile_b: np.ndarray
) -> float:
    """
    Compute normalized cross-correlation (Pearson correlation) between two profiles.

    DEPRECATED: Use compute_ncc_with_flip() for sign-flip detection.
    Kept for backward compatibility.

    Args:
        profile_a: 1D profile array (N,)
        profile_b: 1D profile array (N,)

    Returns:
        NCC in [-1, 1], higher = more similar

    Notes:
        - Returns 0.0 if either std = 0 (degenerate/flat profile)
        - Does NOT handle sign-flip (opposite-side profiles)
    """
    if len(profile_a) != len(profile_b):
        raise ValueError(f"Profiles must have same length: {len(profile_a)} vs {len(profile_b)}")

    # Normalize: subtract mean
    a_norm = profile_a - np.mean(profile_a)
    b_norm = profile_b - np.mean(profile_b)

    # Compute standard deviations
    std_a = np.std(profile_a)
    std_b = np.std(profile_b)

    # Degenerate case: flat profile (std ≈ 0)
    if std_a < 1e-6 or std_b < 1e-6:
        return 0.0

    # NCC formula
    ncc = np.dot(a_norm, b_norm) / (std_a * std_b * len(profile_a))
    return float(np.clip(ncc, -1.0, 1.0))


def compute_ncc_with_flip(
    profile_a: np.ndarray,
    profile_b: np.ndarray,
    allow_reversal: bool = True
) -> tuple[float, bool, bool]:
    """
    Compute NCC with sign-flip and reversal detection.

    Args:
        profile_a: 1D profile array (N,)
        profile_b: 1D profile array (N,)
        allow_reversal: If True, test reversed profile_b

    Returns:
        Tuple (best_ncc, reversal_used, sign_flip_used):
        - best_ncc: Best NCC value (absolute value maximized)
        - reversal_used: True if profile_b was reversed
        - sign_flip_used: True if profile_b was sign-flipped (negated)

    Algorithm:
        1. Test forward: NCC(a, b) and NCC(a, -b)
        2. If allow_reversal: Test reversed: NCC(a, b[::-1]) and NCC(a, -b[::-1])
        3. Return combination with max |NCC|

    Notes:
        - Sign-flip detects opposite-side orientations (profile on other side of chord)
        - Critical for signed profiles where NCC ≈ -1 indicates perfect match on opposite side
        - Returns best_ncc with its original sign (not abs value)

    Example:
        >>> a = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        >>> b = np.array([0.0, -1.0, -2.0, -1.0, 0.0])  # Opposite side
        >>> ncc, rev, flip = compute_ncc_with_flip(a, b, allow_reversal=False)
        >>> assert abs(ncc) > 0.99  # High |NCC|
        >>> assert flip == True  # Detected sign flip
    """
    # Test forward
    ncc_fwd = compute_ncc(profile_a, profile_b)
    ncc_fwd_flip = compute_ncc(profile_a, -profile_b)

    # Track best
    best_ncc = ncc_fwd
    best_abs_ncc = abs(ncc_fwd)
    reversal = False
    sign_flip = False

    # Prefer sign-flip if |NCC| is better OR equal (to detect opposite-side matches)
    if abs(ncc_fwd_flip) > best_abs_ncc or (abs(ncc_fwd_flip) == best_abs_ncc and ncc_fwd_flip > best_ncc):
        best_ncc = ncc_fwd_flip
        best_abs_ncc = abs(ncc_fwd_flip)
        sign_flip = True

    # Test reversal if enabled
    if allow_reversal:
        ncc_rev = compute_ncc(profile_a, profile_b[::-1])
        ncc_rev_flip = compute_ncc(profile_a, -profile_b[::-1])

        # Prefer reversal without flip if |NCC| better OR equal with better sign
        if abs(ncc_rev) > best_abs_ncc or (abs(ncc_rev) == best_abs_ncc and ncc_rev > best_ncc):
            best_ncc = ncc_rev
            best_abs_ncc = abs(ncc_rev)
            reversal = True
            sign_flip = False

        # Prefer reversal with flip if |NCC| better OR equal with better sign
        if abs(ncc_rev_flip) > best_abs_ncc or (abs(ncc_rev_flip) == best_abs_ncc and ncc_rev_flip > best_ncc):
            best_ncc = ncc_rev_flip
            best_abs_ncc = abs(ncc_rev_flip)
            reversal = True
            sign_flip = True

    return best_ncc, reversal, sign_flip


def _prefilter_candidates(
    seg_ref: ContourSegment,
    all_segments: list[ContourSegment],
    config: MatchingConfig
) -> list[ContourSegment]:
    """
    Prefilter candidate segments for inner matching.

    Args:
        seg_ref: Reference segment
        all_segments: All segments from all pieces
        config: MatchingConfig with prefilter parameters

    Returns:
        List of candidate segments after prefiltering

    Filters:
        1. Piece ID: Exclude same piece (no self-matching)
        2. Length: |len_a - len_b| / max(len_a, len_b) <= length_tolerance_ratio
        3. Flatness (optional): |flatness_a - flatness_b| <= flatness_tolerance_mm
        4. Frame-likelihood (optional): Prefer lower frame_cost (inner segments)

    Notes:
        - Flatness/Frame-likelihood filters currently always active (no None check)
        - Length tolerance: 0.15 (15%) default, strict
        - Reduces NCC computation cost (O(n²) → O(n*k) where k << n)

    Example:
        >>> seg_ref = ContourSegment(piece_id=1, length_mm=20.0, ...)
        >>> candidates = _prefilter_candidates(seg_ref, all_segments, config)
        >>> assert all(seg.piece_id != 1 for seg in candidates)  # No self-matches
    """
    candidates = []

    for seg in all_segments:
        # Filter 1: Piece ID (no self-matching)
        if seg.piece_id == seg_ref.piece_id:
            continue

        # Filter 2: Length tolerance
        len_a = seg_ref.length_mm
        len_b = seg.length_mm
        max_len = max(len_a, len_b)

        if max_len < 1e-6:
            # Degenerate: both zero length
            continue

        len_ratio = min(len_a, len_b) / max_len
        len_diff = 1.0 - len_ratio

        if len_diff > config.length_tolerance_ratio:
            continue

        # Filter 3: Flatness tolerance (optional, currently always active)
        flatness_diff = abs(seg_ref.flatness_error - seg.flatness_error)
        if flatness_diff > config.flatness_tolerance_mm:
            continue

        # Filter 4: Frame-likelihood (optional, currently always active)
        # Note: Requires seg to have frame_cost_min attribute (not in current ContourSegment)
        # TODO: Add frame_cost_min to ContourSegment or use separate lookup
        # For now: skip this filter (no attribute available)

        candidates.append(seg)

    return candidates


def _compute_length_cost(
    len_a: float,
    len_b: float,
    tolerance_ratio: float
) -> float:
    """
    Compute length compatibility cost.

    Args:
        len_a: Segment A length (mm)
        len_b: Segment B length (mm)
        tolerance_ratio: Acceptable mismatch ratio (e.g., 0.15 = 15%)

    Returns:
        cost in [0, 1], lower = more compatible

    Formula:
        len_ratio = min(len_a, len_b) / max(len_a, len_b)
        len_diff = 1 - len_ratio
        cost = clamp(len_diff / tolerance_ratio, 0, 1)

    Notes:
        - Perfect match (same length): cost = 0.0
        - Tolerance threshold: cost = 1.0
        - Beyond tolerance: clamped to 1.0

    Examples:
        >>> _compute_length_cost(10.0, 10.0, 0.15)
        0.0  # Perfect
        >>> _compute_length_cost(10.0, 11.5, 0.15)
        0.87  # 15% diff → 0.15/0.15 = 1.0 → clamped
        >>> _compute_length_cost(10.0, 12.0, 0.15)
        1.0  # >15% diff → clamped
    """
    max_len = max(len_a, len_b)

    if max_len < 1e-6:
        # Both zero length: perfect match
        return 0.0

    len_ratio = min(len_a, len_b) / max_len
    len_diff = 1.0 - len_ratio

    cost = len_diff / tolerance_ratio
    return min(cost, 1.0)


def _compute_icp_fit_cost(
    seg_a: ContourSegment,
    seg_b: ContourSegment,
    config: MatchingConfig
) -> float:
    """
    Compute ICP geometric fit cost.

    Args:
        seg_a: Reference segment
        seg_b: Candidate segment
        config: MatchingConfig with enable_icp flag

    Returns:
        fit_cost in [0, 1], lower = better fit

    Notes:
        - **Placeholder**: Currently returns 0.0 (ICP not implemented)
        - TODO: Step 9 (Pose Refinement) - Implement ICP
        - If enable_icp=False: always returns 0.0
        - If enable_icp=True: placeholder returns 0.0 (stub)

    Algorithm (TODO):
        1. Align segment B to segment A (initial pose from chord overlap)
        2. Iterative Closest Point (ICP) refinement
        3. Compute RMS distance after alignment
        4. Normalize: fit_cost = clamp(rms_mm / fit_ref_mm, 0, 1)

    Example:
        >>> config = MatchingConfig(enable_icp=False)
        >>> cost = _compute_icp_fit_cost(seg_a, seg_b, config)
        >>> assert cost == 0.0  # Disabled
    """
    # TODO: Step 9 (Pose Refinement) - Implement ICP alignment
    # For now: return 0.0 regardless of enable_icp flag
    return 0.0


def _compute_inner_cost(
    seg_a: ContourSegment,
    seg_b: ContourSegment,
    config: MatchingConfig
) -> tuple[float, dict, bool, bool]:
    """
    Compute aggregated inner match cost between two segments.

    Args:
        seg_a: Reference segment
        seg_b: Candidate segment
        config: MatchingConfig with inner_weights, profile extraction params

    Returns:
        Tuple (cost_inner, features_dict, reversal_used, sign_flip_used):
        - cost_inner: Aggregated cost in [0, 1]
        - features_dict: Raw feature values (profile_cost, length_cost, fit_cost)
        - reversal_used: True if profile_b was reversed for better match
        - sign_flip_used: True if profile_b was sign-flipped (negated)

    Algorithm:
        1. Extract profiles (if not already cached)
        2. Compute NCC with sign-flip and reversal detection
        3. Compute costs: profile, length, fit (clamped to [0, 1])
        4. Aggregate: cost_inner = Σ w_k * cost_k

    Cost Formula:
        cost_inner = w_profile * (1 - |NCC|)
                   + w_length * length_cost
                   + w_fit * fit_cost

    Notes:
        - Weights from config.inner_weights (default: profile=0.6, length=0.2, fit=0.2)
        - Sign-flip detects opposite-side orientations (critical for signed profiles)
        - profile_cost clamped to [0, 1] (was [0, 2] before clamp)
        - cost_inner ∈ [0, 1] for good matches

    Example:
        >>> cost, features, rev, flip = _compute_inner_cost(seg_a, seg_b, config)
        >>> assert 0.0 <= cost <= 1.0  # Clamped range
        >>> assert "profile_cost" in features
    """
    from .profile import extract_1d_profile

    # 1. Extract profiles (lazy)
    if seg_a.profile_1d is None:
        extract_1d_profile(seg_a, config)
    if seg_b.profile_1d is None:
        extract_1d_profile(seg_b, config)

    profile_a = seg_a.profile_1d
    profile_b = seg_b.profile_1d

    # 2. Compute NCC with sign-flip and reversal detection
    ncc_best, reversal_used, sign_flip_used = compute_ncc_with_flip(
        profile_a, profile_b, allow_reversal=True
    )

    # 3. Compute costs (with clamp)
    # Profile cost: 1 - |NCC|, clamped to [0, 1]
    profile_cost = 1.0 - abs(ncc_best)
    profile_cost = max(0.0, min(1.0, profile_cost))

    length_cost = _compute_length_cost(
        seg_a.length_mm,
        seg_b.length_mm,
        config.length_tolerance_ratio
    )

    fit_cost = _compute_icp_fit_cost(seg_a, seg_b, config)

    # 4. Aggregate with weights
    weights = config.inner_weights
    cost_inner = (
        weights.get("profile", 0.6) * profile_cost +
        weights.get("length", 0.2) * length_cost +
        weights.get("fit", 0.2) * fit_cost
    )

    features = {
        "profile_cost": profile_cost,
        "length_cost": length_cost,
        "fit_cost": fit_cost,
        "ncc_best": ncc_best,
        "sign_flip_used": sign_flip_used,
    }

    return cost_inner, features, reversal_used, sign_flip_used


def generate_inner_candidates(
    segments: list[ContourSegment],
    config: MatchingConfig
) -> dict[tuple[int | str, int], list[InnerMatchCandidate]]:
    """
    Generate all inner match candidates, rank, keep top-k per segment.

    Args:
        segments: List of ContourSegment from all pieces
        config: MatchingConfig with inner matching parameters

    Returns:
        Dict mapping (piece_id, segment_id) → list of top-k InnerMatchCandidate
        (sorted by cost_inner ascending)

    Algorithm:
        1. For each segment A:
           - Prefilter candidates (length, flatness, piece ID)
           - For each candidate B:
               * Compute inner cost (NCC, length, fit)
               * Create InnerMatchCandidate
           - Sort by cost_inner ascending
           - Keep top-k (k = config.topk_per_segment, default 10)
        2. Return dict

    Notes:
        - Prefiltering reduces NCC computations from O(n²) to O(n*k)
        - Top-k ensures diversity (multiple plausible matches per segment)
        - Segments without candidates → empty list in dict

    Example:
        >>> segments = [seg1, seg2, ...]  # From 4 pieces
        >>> candidates = generate_inner_candidates(segments, config)
        >>> assert len(candidates) <= len(segments)  # Some may have 0 candidates
        >>> for seg_ref, cands in candidates.items():
        ...     assert len(cands) <= 10  # Top-k = 10
        ...     assert cands[0].cost_inner <= cands[-1].cost_inner  # Sorted
    """
    from ..models import InnerMatchCandidate
    import warnings

    # Validate weight normalization (should sum to 1.0 for cost_inner ∈ [0,1])
    weight_sum = sum(config.inner_weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        warnings.warn(
            f"inner_weights sum={weight_sum:.3f} ≠ 1.0. "
            f"cost_inner range will be [0, {weight_sum:.3f}] instead of [0, 1]. "
            f"Beam-Solver ranking still works, but thresholds may need adjustment."
        )

    result = defaultdict(list)
    top_k = config.topk_per_segment

    for seg_a in segments:
        # 1. Prefilter candidates
        candidates_b = _prefilter_candidates(seg_a, segments, config)

        if not candidates_b:
            # No candidates after prefiltering
            continue

        # 2. Compute costs for all candidates
        candidate_list = []

        for seg_b in candidates_b:
            cost_inner, features, reversal_used, sign_flip_used = _compute_inner_cost(seg_a, seg_b, config)

            # Determine variant name for debug
            if reversal_used and sign_flip_used:
                variant = "rev_flip"
            elif reversal_used:
                variant = "rev"
            elif sign_flip_used:
                variant = "fwd_flip"
            else:
                variant = "fwd"

            # Create InnerMatchCandidate
            cand = InnerMatchCandidate(
                seg_a_ref=(seg_a.piece_id, seg_a.segment_id),
                seg_b_ref=(seg_b.piece_id, seg_b.segment_id),
                cost_inner=cost_inner,
                profile_cost=features["profile_cost"],
                length_cost=features["length_cost"],
                fit_cost=features["fit_cost"],
                reversal_used=reversal_used,
                sign_flip_used=sign_flip_used,
                ncc_best=features["ncc_best"],
                best_variant=variant
            )

            candidate_list.append(cand)

        # 3. Sort by cost_inner ascending (lower = better)
        candidate_list.sort(key=lambda c: c.cost_inner)

        # 4. Keep top-k
        result[(seg_a.piece_id, seg_a.segment_id)] = candidate_list[:top_k]

    return dict(result)
