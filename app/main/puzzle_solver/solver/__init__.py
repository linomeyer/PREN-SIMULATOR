"""
Solver V2: Multi-Hypothesis Beam Search Puzzle Solver.

This is the main entry point for the new puzzle solver implementation.
See docs/design/ for complete design documentation.
See docs/implementation/00_structure.md for implementation roadmap.

Main API:
    solve_puzzle(pieces, frame, config) -> PuzzleSolution

Status: Step 1 complete (Fundament: Config + Models)
        Steps 2-9 in progress (see CLAUDE.md for current status)
"""

from typing import List
import numpy as np

from .config import MatchingConfig, FrameModel, Transform2D
from .models import (
    Pose2D,
    PuzzlePiece,
    ContourSegment,
    FrameContactFeatures,
    FrameHypothesis,
    InnerMatchCandidate,
    SolutionStatus,
    DebugBundle,
    PuzzleSolution,
)
from .fallback.many_to_one import compute_confidence, should_trigger_fallback
from .beam_solver.solver import beam_search, _attempt_fallback_rerun
from .segmentation.contour_segmenter import segment_piece
from .frame_matching.hypotheses import generate_frame_hypotheses
from .inner_matching.candidates import generate_inner_candidates
from .debug_utils import validate_puzzle_input, create_debug_bundle


__all__ = [
    # Main API
    "solve_puzzle",
    # Config
    "MatchingConfig",
    "FrameModel",
    "Transform2D",
    # Models
    "Pose2D",
    "PuzzlePiece",
    "ContourSegment",
    "FrameContactFeatures",
    "FrameHypothesis",
    "InnerMatchCandidate",
    "SolutionStatus",
    "DebugBundle",
    "PuzzleSolution",
]


def _flatten_inner_candidates(candidates_dict: dict) -> List[InnerMatchCandidate]:
    """
    Flatten inner candidates dict to flat list for beam_search().

    Args:
        candidates_dict: Dict from generate_inner_candidates()
                        Format: {(piece_id, seg_id): [InnerMatchCandidate, ...]}

    Returns:
        Flat list of all InnerMatchCandidate objects

    Notes:
        - beam_search() expects flat list (global pool)
        - generate_inner_candidates() returns dict grouped by segment
        - This helper bridges the format mismatch
    """
    flat_list = []
    for cand_list in candidates_dict.values():
        flat_list.extend(cand_list)
    return flat_list


def solve_puzzle(pieces: List[PuzzlePiece], frame: FrameModel, config: MatchingConfig) -> PuzzleSolution:
    """
    Solve puzzle using multi-hypothesis beam search.

    This is the main entry point for the solver. It takes extracted puzzle pieces
    and attempts to find a consistent solution within the given frame.

    Args:
        pieces: List of PuzzlePiece objects (from solver.models.PuzzlePiece)
                Each piece must have:
                - contour_mm: np.ndarray[(N,2)] contour points in mm
                - mask: binary mask
                - bbox_mm: bounding box in mm
                - Additional fields (image, center_mm) optional
        frame: FrameModel defining the A5 frame (128x190mm)
               - Includes optional T_MF transform if machine coordinates needed
        config: MatchingConfig with all solver parameters
                - See MatchingConfig docstring for parameter groups

    Returns:
        PuzzleSolution with:
            - poses_F: Piece poses in Frame coordinate system (F)
            - poses_M: Piece poses in Machine coordinate system (M) if T_MF set
            - matches: Match constraints used
            - total_cost: Solution cost
            - confidence: Solution confidence = exp(-k_conf * total_cost)
            - status: Solution status (OK, OK_WITH_FALLBACK, ...)
            - debug: Debug bundle (if config.export_debug_json=True)

    Raises:
        NotImplementedError: Implementation in progress (Steps 2-9)
        ValueError: Invalid input (e.g., pieces empty, invalid frame dimensions)

    Notes:
        - Input coordinates converted to mm internally
        - All processing in Frame coordinate system (F)
        - Output poses in F (and optionally M if T_MF set)
        - See docs/design/00_overview.md for algorithm overview
        - See docs/implementation/00_structure.md for implementation steps

    Implementation Status:
        [] Step 1: Config + Models (current)
        [ ] Step 2: Einheiten & KS
        [ ] Step 3: Segmentierung + Flatness
        [ ] Step 4: Frame-Matching
        [ ] Step 5: Inner-Matching
        [ ] Step 6: Beam-Solver V1
        [ ] Step 7: Overlap-Modul
        [ ] Step 8: Confidence + Fallback
        [ ] Step 9: Pose-Refinement
        [ ] Step 10: Integration + Tests
    """
    # Step 10: Pipeline integration

    # ========== 1. INPUT VALIDATION (Step 9) ==========
    error = validate_puzzle_input(pieces, frame, config)
    if error:
        failure_reason, failure_message, affected_pieces = error
        # Build minimal debug bundle for INVALID_INPUT
        debug = create_debug_bundle(
            status=SolutionStatus.INVALID_INPUT,
            config=config,
            pieces={p.piece_id: p for p in pieces},
            frame_hypotheses={},
            inner_candidates=[],
            solver_summary={"prune_counts": {}},
            last_best_state=None,
            fallback_info=None,
            failure_reason=failure_reason,
            failure_message=failure_message,
            affected_pieces=affected_pieces
        )
        return PuzzleSolution(
            poses_F={},
            status=SolutionStatus.INVALID_INPUT,
            debug=debug
        )

    # ========== 2. BUILD DATA STRUCTURES ==========
    all_pieces = {p.piece_id: p for p in pieces}
    all_segments = {}
    segments_flat = []

    # Step 3: Segmentation
    for piece in pieces:
        segments = segment_piece(piece, config)
        all_segments[piece.piece_id] = segments
        segments_flat.extend(segments)

    # ========== 3. GENERATE HYPOTHESES & CANDIDATES ==========

    # Step 4: Frame matching
    frame_hypotheses = generate_frame_hypotheses(segments_flat, frame, config)

    # Step 5: Inner matching
    inner_candidates_dict = generate_inner_candidates(segments_flat, config)

    # Flatten candidates dict → list (beam_search expects flat list)
    inner_candidates_flat = _flatten_inner_candidates(inner_candidates_dict)

    # ========== 4. BEAM SEARCH (Step 6) ==========
    states = beam_search(
        all_pieces=all_pieces,
        all_segments=all_segments,
        frame_hypotheses=frame_hypotheses,
        inner_candidates=inner_candidates_flat,
        config=config,
        frame=frame
    )

    # Check if solution found
    if not states:
        # F1: Beam exhausted - no states at all
        solver_summary = {
            "prune_counts": {},  # Placeholder (TODO: track in beam_search)
            "beam_exhausted": True,
            "max_expansions_reached": False
        }
        debug = create_debug_bundle(
            status=SolutionStatus.NO_SOLUTION,
            config=config,
            pieces=all_pieces,
            frame_hypotheses=frame_hypotheses,
            inner_candidates=inner_candidates_flat,
            solver_summary=solver_summary,
            last_best_state=None
        )
        return PuzzleSolution(
            poses_F={},
            status=SolutionStatus.NO_SOLUTION,
            debug=debug
        )

    best_state = states[0]  # Best (lowest cost)

    # Check if complete solution
    if not best_state.is_complete():
        # F1: NO_SOLUTION - only partial states found
        solver_summary = {
            "prune_counts": {},  # Placeholder
            "beam_exhausted": False,
            "max_expansions_reached": False  # Could be true, but we don't track yet
        }
        debug = create_debug_bundle(
            status=SolutionStatus.NO_SOLUTION,
            config=config,
            pieces=all_pieces,
            frame_hypotheses=frame_hypotheses,
            inner_candidates=inner_candidates_flat,
            solver_summary=solver_summary,
            last_best_state=best_state
        )
        return PuzzleSolution(
            poses_F=best_state.poses_F,
            total_cost=best_state.cost_total,
            cost_breakdown=best_state.cost_breakdown,
            status=SolutionStatus.NO_SOLUTION,
            debug=debug
        )

    # ========== 5. CONFIDENCE & FALLBACK (Step 8) ==========
    confidence = compute_confidence(best_state.cost_total, config)
    trigger_fallback, reason = should_trigger_fallback(confidence, config)

    fallback_debug = None
    if trigger_fallback:
        # Attempt fallback rerun with composite segments
        fallback_state, fallback_debug = _attempt_fallback_rerun(
            original_state=best_state,
            all_pieces=all_pieces,
            all_segments=all_segments,
            frame_hypotheses=frame_hypotheses,
            config=config,
            frame=frame
        )

        if fallback_state and fallback_state.is_complete():
            # Compute fallback confidence
            fallback_conf = compute_confidence(fallback_state.cost_total, config)

            if fallback_conf >= config.fallback_conf_threshold:
                # F2: OK_WITH_FALLBACK - fallback improved confidence
                best_state = fallback_state
                confidence = fallback_conf
                status = SolutionStatus.OK_WITH_FALLBACK
            else:
                # F2: LOW_CONFIDENCE_SOLUTION - fallback didn't reach threshold
                # Use fallback if better cost
                if fallback_state.cost_total < best_state.cost_total:
                    best_state = fallback_state
                    confidence = fallback_conf
                status = SolutionStatus.LOW_CONFIDENCE_SOLUTION
        else:
            # Fallback failed or incomplete - keep original, mark LOW_CONFIDENCE
            status = SolutionStatus.LOW_CONFIDENCE_SOLUTION
    else:
        # No fallback needed
        status = SolutionStatus.OK

    # ========== 6. POSE TRANSFORM (if T_MF set) ==========
    poses_M = None
    if frame.T_MF is not None:
        poses_M = {}
        for piece_id, pose_F in best_state.poses_F.items():
            # Transform position
            point_F = np.array([[pose_F.x_mm, pose_F.y_mm]])
            point_M = frame.T_MF.apply(point_F)[0]
            # Transform rotation
            theta_M = pose_F.theta_deg + frame.T_MF.theta_deg
            poses_M[piece_id] = Pose2D(point_M[0], point_M[1], theta_M)

    # ========== 7. BUILD DEBUG BUNDLE ==========
    debug = None
    if config.export_debug_json:
        solver_summary = {
            "prune_counts": {},  # Placeholder (TODO: implement stats tracking)
            "beam_exhausted": False,
            "max_expansions_reached": False,
            "frontier_mode": getattr(config, 'frontier_mode', 'auto')
        }
        debug = create_debug_bundle(
            status=status,
            config=config,
            pieces=all_pieces,
            frame_hypotheses=frame_hypotheses,
            inner_candidates=inner_candidates_flat,
            solver_summary=solver_summary,
            last_best_state=best_state,
            fallback_info=fallback_debug
        )

    # ========== 8. RETURN SOLUTION ==========
    return PuzzleSolution(
        poses_F=best_state.poses_F,
        poses_M=poses_M,
        matches=best_state.matches,
        total_cost=best_state.cost_total,
        cost_breakdown=best_state.cost_breakdown,
        confidence=confidence,
        status=status,
        overlap_violations=0,  # TODO: Count from state or collision module
        debug=debug
    )
