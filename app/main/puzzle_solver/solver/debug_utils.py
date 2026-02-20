"""
Debug utilities for Step 9 (Edge Cases & Failure Modes).

This module provides helpers for creating and populating DebugBundle structures.

See docs/test_spec/09_edgecases_test_spec.md §4 for debug requirements.
"""

from typing import Optional
import math
from .models import DebugBundle, SolutionStatus, PuzzlePiece, FrameHypothesis, InnerMatchCandidate
from .config import MatchingConfig
from .beam_solver.state import SolverState


def create_debug_bundle(
    status: SolutionStatus,
    config: MatchingConfig,
    pieces: dict,  # dict[int, PuzzlePiece]
    frame_hypotheses: dict,  # dict[int, list[FrameHypothesis]]
    inner_candidates: list,  # list[InnerMatchCandidate]
    solver_summary: dict,
    last_best_state: Optional[SolverState] = None,
    fallback_info: Optional[dict] = None,
    failure_reason: Optional[str] = None,
    affected_pieces: Optional[list] = None
) -> DebugBundle:
    """
    Create complete DebugBundle for PuzzleSolution.

    Args:
        status: Solution status
        config: Matching configuration
        pieces: Dict mapping piece_id -> PuzzlePiece
        frame_hypotheses: Frame hypotheses per piece (top-N already filtered)
        inner_candidates: Inner match candidates (already filtered)
        solver_summary: Solver statistics dict with keys:
            - prune_counts: dict[str, int]
            - beam_exhausted: bool
            - max_expansions_reached: bool
            - frontier_mode: str (optional)
        last_best_state: Best state seen (for NO_SOLUTION)
        fallback_info: Fallback statistics (if fallback triggered)
        failure_reason: Reason string (for INVALID_INPUT)
        affected_pieces: Piece IDs affected (for INVALID_INPUT)

    Returns:
        DebugBundle with all mandatory fields populated

    Notes:
        - config_dump: Convert config to dict via to_dict()
        - area_score: Computed from best state or None
        - frame_hypotheses: Top-N per config.debug_topN_frame_hypotheses_per_piece
        - inner_candidates: Top-N per config.debug_topN_inner_candidates_per_segment
    """
    # 1. Config dump
    config_dump = config.to_dict() if hasattr(config, 'to_dict') else vars(config)

    # 2. Area score (from last_best_state if available)
    area_score = None
    if last_best_state is not None:
        area_score = _compute_area_score(last_best_state)

    # 3. Serialize last_best_state (DBG-05)
    last_best_state_dict = None
    if last_best_state is not None:
        last_best_state_dict = _serialize_solver_state(last_best_state)

    # 4. Build debug bundle
    debug = DebugBundle(
        status=status.value,
        config_dump=config_dump,
        n_pieces=len(pieces),
        area_score=area_score,
        frame_hypotheses=_filter_frame_hypotheses_topn(frame_hypotheses, config),
        inner_candidates=_filter_inner_candidates_topn(inner_candidates, config),
        solver_summary=solver_summary,
        last_best_state=last_best_state_dict,
        fallback=fallback_info,
        refinement=None,  # TODO: Step 10 refinement
        collision=None,  # Populated by collision module
        failure_reason=failure_reason,
        affected_pieces=affected_pieces
    )

    return debug


def _compute_area_score(state: SolverState) -> Optional[float]:
    """
    Compute area coverage score from SolverState.

    Area score = fraction of pieces placed [0..1]

    Args:
        state: SolverState

    Returns:
        Area score [0..1] or None if not computable
    """
    try:
        if state.all_piece_ids is None or len(state.all_piece_ids) == 0:
            return None

        placed_count = len(state.placed)
        total_count = len(state.all_piece_ids)

        return placed_count / total_count
    except Exception:
        return None


def _serialize_solver_state(state: SolverState) -> dict:
    """
    Serialize SolverState to dict for debug export.

    Args:
        state: SolverState to serialize

    Returns:
        Dict with state fields (poses, cost, placed pieces, etc.)

    Notes:
        - Must be JSON-serializable (no numpy arrays)
        - Includes cost_total, placed pieces, open_edges count
    """
    placed_dict = {}
    for piece_id, pose in state.placed.items():
        placed_dict[str(piece_id)] = {
            "x_mm": _sanitize_float(pose.x_mm),
            "y_mm": _sanitize_float(pose.y_mm),
            "theta_deg": _sanitize_float(pose.theta_deg)
        }

    return {
        "cost_total": _sanitize_float(state.cost_total),
        "placed": placed_dict,
        "unplaced_count": len(state.unplaced),
        "open_edges_count": len(state.open_edges),
        "is_complete": state.is_complete(),
    }


def _filter_frame_hypotheses_topn(frame_hypotheses: dict, config: MatchingConfig) -> dict:
    """
    Filter frame hypotheses to top-N per config (DBG-03).

    Args:
        frame_hypotheses: Dict mapping piece_id -> list[FrameHypothesis]
        config: MatchingConfig with debug_topN_frame_hypotheses_per_piece

    Returns:
        Filtered dict with max N hypotheses per piece
    """
    topn = config.debug_topN_frame_hypotheses_per_piece
    filtered = {}

    for piece_id, hyps in frame_hypotheses.items():
        # Take top N (already sorted by cost in frame_matching)
        filtered[piece_id] = [_serialize_frame_hypothesis(h) for h in hyps[:topn]]

    return filtered


def _filter_inner_candidates_topn(inner_candidates: list, config: MatchingConfig) -> dict:
    """
    Filter inner candidates to top-N per segment pair (DBG-03).

    Args:
        inner_candidates: List of InnerMatchCandidate
        config: MatchingConfig with debug_topN_inner_candidates_per_segment

    Returns:
        Dict mapping (seg_a, seg_b) -> list of top-N candidates
    """
    topn = config.debug_topN_inner_candidates_per_segment

    # Group by segment pair
    grouped = {}
    for cand in inner_candidates:
        key = (str(cand.seg_a_ref), str(cand.seg_b_ref))
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(cand)

    # Sort by cost and take top N
    filtered = {}
    for key, cands in grouped.items():
        sorted_cands = sorted(cands, key=lambda c: c.cost_inner)[:topn]
        filtered[key] = [_serialize_inner_candidate(c) for c in sorted_cands]

    return filtered


def _serialize_frame_hypothesis(hyp: FrameHypothesis) -> dict:
    """Serialize FrameHypothesis to dict."""
    return {
        "piece_id": hyp.piece_id,
        "segment_id": hyp.segment_id,
        "side": hyp.side,
        "cost_frame": _sanitize_float(hyp.cost_frame),
        "pose_grob_F": {
            "x_mm": _sanitize_float(hyp.pose_grob_F.x_mm),
            "y_mm": _sanitize_float(hyp.pose_grob_F.y_mm),
            "theta_deg": _sanitize_float(hyp.pose_grob_F.theta_deg)
        }
    }


def _serialize_inner_candidate(cand: InnerMatchCandidate) -> dict:
    """Serialize InnerMatchCandidate to dict."""
    return {
        "seg_a_ref": str(cand.seg_a_ref),
        "seg_b_ref": str(cand.seg_b_ref),
        "cost_inner": _sanitize_float(cand.cost_inner),
        "profile_cost": _sanitize_float(cand.profile_cost),
        "length_cost": _sanitize_float(cand.length_cost),
        "reversal_used": cand.reversal_used,
        "ncc_best": _sanitize_float(cand.ncc_best)
    }


def _sanitize_float(value: float) -> float | str:
    """
    Sanitize float for JSON export (DBG-02).

    Converts NaN/Inf to strings for safe JSON serialization.

    Args:
        value: Float value to sanitize

    Returns:
        Original value, or "NaN"/"Infinity"/"-Infinity" string
    """
    if math.isnan(value):
        return "NaN"
    elif math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    else:
        return value


def validate_puzzle_input(
    pieces: list[PuzzlePiece],
    frame,  # FrameModel
    config: MatchingConfig
) -> Optional[tuple[str, list]]:
    """
    Validate puzzle input (INVALID_INPUT check).

    Args:
        pieces: List of PuzzlePiece
        frame: FrameModel
        config: MatchingConfig

    Returns:
        None if valid, or (failure_reason, affected_pieces) tuple if invalid

    Invalid cases (FM-12/13/14):
        - n not in {4, 5, 6}
        - Piece missing contour_mm
        - Piece with empty contour
        - Frame dimensions invalid
    """
    # Check piece count
    n = len(pieces)
    if not (4 <= n <= 6):
        return (f"Invalid piece count: {n} (must be 4-6)", [])

    # Check each piece
    affected = []
    for i, piece in enumerate(pieces):
        # Check contour_mm exists
        if piece.contour_mm is None:
            affected.append(piece.piece_id)
            continue

        # Check contour not empty
        if len(piece.contour_mm) == 0:
            affected.append(piece.piece_id)

    if affected:
        return (f"Pieces missing or empty contour_mm: {affected}", affected)

    # Check frame
    if frame.inner_width_mm <= 0 or frame.inner_height_mm <= 0:
        return ("Invalid frame dimensions", [])

    return None  # Valid input
