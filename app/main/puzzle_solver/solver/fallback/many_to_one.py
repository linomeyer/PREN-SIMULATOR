"""
Many-to-One Fallback Strategy

Implements composite segment creation and extended candidate generation
for low-confidence solutions (Step 8).

Key functions:
- compute_confidence(): exp(-k_conf × cost_total)
- create_composite_segments(): Generate adjacent segment chains (k=2, k=3)
- extend_inner_candidates(): Add composite ↔ atomic match candidates
- should_trigger_fallback(): Check confidence threshold
"""

import numpy as np
import copy
from typing import Optional

from ..models import ContourSegment, InnerMatchCandidate
from ..config import MatchingConfig
from ..segmentation.contour_segmenter import (
    _compute_arclength,
    _compute_flatness_error,
    _compute_direction_angle
)
from ..inner_matching.profile import extract_1d_profile
from ..inner_matching.candidates import _compute_inner_cost


# ========== Public API ==========

def compute_confidence(cost_total: float, config: MatchingConfig) -> float:
    """
    Compute confidence from total cost.

    Formula: confidence = exp(-k_conf × cost_total)

    Args:
        cost_total: Total solution cost (≥0)
        config: MatchingConfig with k_conf parameter

    Returns:
        Confidence in [0, 1], higher = better solution

    Edge cases:
        - cost < 0: ValueError
        - cost = 0: returns 1.0
        - cost = inf: returns 0.0

    Tests: test_fallback.py Tests 1-6
    """
    if cost_total < 0:
        raise ValueError(f"cost_total must be >= 0, got {cost_total}")

    if np.isinf(cost_total):
        return 0.0

    return float(np.exp(-config.k_conf * cost_total))


def should_trigger_fallback(solution, config: MatchingConfig) -> tuple[bool, Optional[str]]:
    """
    Check if fallback should be triggered.

    Logic: confidence < threshold AND enable_many_to_one_fallback = True

    Args:
        solution: PuzzleSolution or float confidence [0, 1]
        config: MatchingConfig with fallback parameters

    Returns:
        (trigger, reason) tuple:
        - trigger: True if fallback should run
        - reason: "confidence_below_threshold" or None

    Tests: test_fallback.py Tests 13-17
    """
    # Check if fallback is enabled
    if not config.many_to_one_enable:
        return False, None

    # Extract confidence value (handle both float and PuzzleSolution)
    if isinstance(solution, float):
        confidence = solution
    else:
        # Assume PuzzleSolution object with .confidence attribute
        confidence = solution.confidence

    if confidence < config.fallback_conf_threshold:
        return True, "confidence_below_threshold"

    return False, None


def create_composite_segments(
    segments: list[ContourSegment],
    config: MatchingConfig
) -> list[ContourSegment]:
    """
    Create composite segments from adjacent chains.

    Generates k=2 and optionally k=3 composite segments from adjacent
    segment pairs/triples within each piece.

    Args:
        segments: All atomic segments (all pieces)
        config: MatchingConfig with:
            - many_to_one_max_chain_len: 2 or 3
            - max_composites_per_piece_k2: Cap for k=2
            - max_composites_per_piece_k3: Cap for k=3

    Returns:
        List of composite ContourSegments with offset segment_ids:
        - k=2 composites: segment_id = 2000, 2001, ...
        - k=3 composites: segment_id = 3000, 3001, ...

    Logic:
        1. Group segments by piece_id
        2. For each piece:
           - Generate k=2 pairs: (seg_i + seg_{i+1})
           - Apply cap: keep first max_composites_per_piece_k2
           - Optional k=3: (seg_i + seg_{i+1} + seg_{i+2})
        3. Build each composite via _build_composite_segment()

    Tests: test_fallback.py Tests 7-12
    """
    # 1. Group by piece_id
    pieces_segments = {}
    for seg in segments:
        pid = seg.piece_id
        if pid not in pieces_segments:
            pieces_segments[pid] = []
        pieces_segments[pid].append(seg)

    # Sort segments within each piece by segment_id
    for pid in pieces_segments:
        pieces_segments[pid] = sorted(pieces_segments[pid], key=lambda s: s.segment_id)

    # 2. Generate composites
    composites = []
    composite_index = 0

    for pid, piece_segs in pieces_segments.items():
        n_segs = len(piece_segs)

        # k=2 composites (adjacent pairs)
        k2_chains = []
        for i in range(n_segs - 1):
            k2_chains.append([i, i + 1])

        # Apply k=2 cap (first N)
        k2_cap = min(len(k2_chains), config.max_composites_per_piece_k2)
        k2_chains = k2_chains[:k2_cap]

        # Build k=2 composites
        for chain_indices in k2_chains:
            composite = _build_composite_segment(
                piece_segs,
                chain_indices,
                composite_index,
                config
            )
            composites.append(composite)
            composite_index += 1

        # k=3 composites (optional)
        if config.many_to_one_max_chain_len >= 3:
            k3_chains = []
            for i in range(n_segs - 2):
                k3_chains.append([i, i + 1, i + 2])

            # Apply k=3 cap
            k3_cap = min(len(k3_chains), config.max_composites_per_piece_k3)
            k3_chains = k3_chains[:k3_cap]

            # Build k=3 composites
            for chain_indices in k3_chains:
                composite = _build_composite_segment(
                    piece_segs,
                    chain_indices,
                    composite_index,
                    config
                )
                composites.append(composite)
                composite_index += 1

    return composites


def extend_inner_candidates(
    original_candidates: dict[tuple, list[InnerMatchCandidate]],
    atomic_segments: list[ContourSegment],
    composite_segments: list[ContourSegment],
    config: MatchingConfig
) -> dict[tuple, list[InnerMatchCandidate]]:
    """
    Extend inner match candidates with composite segment matches.

    Adds bidirectional composite ↔ atomic candidates to original candidates.
    Applies prefilters (length ±20%) and caps (topk, max_total).

    Args:
        original_candidates: Existing atomic ↔ atomic candidates dict[seg_ref → list]
        atomic_segments: All atomic segments
        composite_segments: Generated composite segments
        config: MatchingConfig with:
            - penalty_composite_used: Cost penalty per composite match
            - topk_per_segment_fallback: Top-k per segment
            - max_total_candidates_fallback: Hard cap

    Returns:
        Extended candidate dict: {(piece_id, segment_id): [InnerMatchCandidate, ...]}

    Logic:
        1. Copy original candidates (deep copy dict)
        2. Generate composite ↔ atomic candidates (bidirectional)
        3. Add composite penalty to cost_inner
        4. Apply length prefilter (±20%)
        5. Apply topk_per_segment_fallback cap (per segment)
        6. Apply max_total_candidates_fallback hard cap

    Exclusions:
        - composite ↔ composite (V1)
        - Same piece (seg_a.piece_id == seg_b.piece_id)

    Tests: test_fallback.py Tests 18-21
    """
    # 1. Copy original candidates (dict format - SSOT)
    extended = copy.deepcopy(original_candidates)

    # 2. Generate composite ↔ atomic candidates
    new_candidates = []

    # Direction 1: composite (seg_a) ↔ atomic (seg_b)
    for comp_seg in composite_segments:
        for atom_seg in atomic_segments:
            # Skip same piece
            if comp_seg.piece_id == atom_seg.piece_id:
                continue

            # Length filter ±20%
            if not _length_filter(comp_seg.length_mm, atom_seg.length_mm, tolerance=0.2):
                continue

            # Compute cost
            cost_inner, features, reversal, sign_flip = _compute_inner_cost(
                comp_seg, atom_seg, config
            )

            # Add composite penalty
            cost_inner += config.penalty_composite_used

            # Create candidate
            cand = InnerMatchCandidate(
                seg_a_ref=(comp_seg.piece_id, comp_seg.segment_id),
                seg_b_ref=(atom_seg.piece_id, atom_seg.segment_id),
                cost_inner=cost_inner,
                profile_cost=features["profile_cost"],
                length_cost=features["length_cost"],
                fit_cost=features["fit_cost"],
                reversal_used=reversal,
                sign_flip_used=sign_flip,
                ncc_best=features.get("ncc_best", 0.0),
                best_variant=features.get("best_variant", "fwd")
            )

            new_candidates.append(cand)

    # Direction 2: atomic (seg_a) ↔ composite (seg_b)
    for atom_seg in atomic_segments:
        for comp_seg in composite_segments:
            # Skip same piece
            if atom_seg.piece_id == comp_seg.piece_id:
                continue

            # Length filter ±20%
            if not _length_filter(atom_seg.length_mm, comp_seg.length_mm, tolerance=0.2):
                continue

            # Compute cost
            cost_inner, features, reversal, sign_flip = _compute_inner_cost(
                atom_seg, comp_seg, config
            )

            # Add composite penalty
            cost_inner += config.penalty_composite_used

            # Create candidate
            cand = InnerMatchCandidate(
                seg_a_ref=(atom_seg.piece_id, atom_seg.segment_id),
                seg_b_ref=(comp_seg.piece_id, comp_seg.segment_id),
                cost_inner=cost_inner,
                profile_cost=features["profile_cost"],
                length_cost=features["length_cost"],
                fit_cost=features["fit_cost"],
                reversal_used=reversal,
                sign_flip_used=sign_flip,
                ncc_best=features.get("ncc_best", 0.0),
                best_variant=features.get("best_variant", "fwd")
            )

            new_candidates.append(cand)

    # 3. Merge new candidates into extended dict
    for cand in new_candidates:
        key = cand.seg_a_ref
        if key not in extended:
            extended[key] = []
        extended[key].append(cand)

    # 4. Apply topk_per_segment_fallback cap (per seg_a_ref)
    topk = config.topk_per_segment_fallback
    for key in extended:
        extended[key] = sorted(extended[key], key=lambda c: c.cost_inner)[:topk]

    # 5. Apply max_total_candidates_fallback hard cap
    all_candidates = [c for cand_list in extended.values() for c in cand_list]
    if len(all_candidates) > config.max_total_candidates_fallback:
        # Keep best (lowest cost) candidates globally
        all_candidates = sorted(all_candidates, key=lambda c: c.cost_inner)[:config.max_total_candidates_fallback]

        # Rebuild dict from capped list
        extended = {}
        for cand in all_candidates:
            key = cand.seg_a_ref
            if key not in extended:
                extended[key] = []
            extended[key].append(cand)

    return extended


# ========== Helper Functions ==========

def _build_composite_segment(
    segments: list[ContourSegment],
    indices: list[int],
    composite_index: int,
    config: MatchingConfig
) -> ContourSegment:
    """
    Build single composite segment from adjacent segment chain.

    Args:
        segments: Segments from one piece (ordered by segment_id)
        indices: Indices into segments list (e.g., [0, 1] for seg_0 + seg_1)
        composite_index: Global composite index (for segment_id assignment)
        config: MatchingConfig

    Returns:
        Composite ContourSegment with:
        - Concatenated points (duplicate junction removed)
        - Recomputed features (length, flatness, chord, direction)
        - Offset segment_id (e.g., 2000 + composite_index for k=2)
        - Extracted profile (N=128)
    """
    # Concatenate points
    points_list = [segments[i].points_mm for i in indices]
    points_composite = _concatenate_points(points_list)

    # Compute chord (new endpoints)
    chord = (points_composite[0], points_composite[-1])

    # Recompute features
    length_mm = _compute_arclength(points_composite)
    flatness_error = _compute_flatness_error(points_composite, chord)
    direction_angle_deg = _compute_direction_angle(chord)

    # Assign segment_id (offset-based)
    chain_len = len(indices)
    segment_id_offset = 1000 * chain_len  # k=2 → 2000, k=3 → 3000
    segment_id = segment_id_offset + composite_index

    # Create composite segment
    composite = ContourSegment(
        piece_id=segments[indices[0]].piece_id,
        segment_id=segment_id,
        points_mm=points_composite,
        length_mm=length_mm,
        chord=chord,
        direction_angle_deg=direction_angle_deg,
        flatness_error=flatness_error,
        profile_1d=None  # Will compute next
    )

    # Explicitly extract profile (N=128 resampling)
    extract_1d_profile(composite, config)

    return composite


def _concatenate_points(points_list: list[np.ndarray]) -> np.ndarray:
    """
    Concatenate segment points, removing duplicate junction points.

    Example:
        [0,5,10] + [10,15,20] → [0,5,10,15,20] (5 points, not 6)

    Args:
        points_list: List of point arrays (each (M, 2))

    Returns:
        Concatenated points (N, 2) with duplicates removed
    """
    if len(points_list) == 0:
        return np.empty((0, 2))

    result = points_list[0].copy()

    for points_next in points_list[1:]:
        # Check if last point of result == first point of next
        if np.linalg.norm(result[-1] - points_next[0]) < 1e-6:
            # Duplicate junction → skip first point of next, keep last of result
            result = np.vstack([result, points_next[1:]])
        else:
            # No duplicate → concat directly
            result = np.vstack([result, points_next])

    return result


def _length_filter(len_a: float, len_b: float, tolerance: float = 0.2) -> bool:
    """
    Check if segment lengths are within ±tolerance (default ±20%).

    Args:
        len_a, len_b: Segment lengths (mm)
        tolerance: Relative tolerance (0.2 = ±20%)

    Returns:
        True if compatible, False otherwise

    Formula:
        ratio = min(len_a, len_b) / max(len_a, len_b)
        accept if ratio >= (1 - tolerance)
    """
    if len_a <= 0 or len_b <= 0:
        return False

    ratio = min(len_a, len_b) / max(len_a, len_b)
    return ratio >= (1 - tolerance)


def run_fallback_iteration(*args, **kwargs):
    """
    Stub for backward compatibility.

    Actual fallback logic is in solver.beam_solver.solver._attempt_fallback_rerun()
    This function is not used by tests directly.
    """
    raise NotImplementedError(
        "run_fallback_iteration() is a stub. "
        "Use _attempt_fallback_rerun() in solver.beam_solver.solver instead."
    )
