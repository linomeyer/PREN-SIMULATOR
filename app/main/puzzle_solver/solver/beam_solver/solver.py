"""
Beam-Solver Main Loop (Step 6).

This module implements beam_search() - the multi-hypothesis state-space search.

Key Concepts:
- Hybrid seeding: empty state + top frame hypotheses
- Beam expansion loop: expand, extract complete, prune
- D1: Completion = all_placed AND open_edges==0
- D5: NO_SOLUTION = return best partial state (not empty list)

See docs/design/05_solver.md for solver algorithm.
See docs/implementation/06_beam_solver_test_spec.md for test specification.
"""

from __future__ import annotations
from typing import Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .state import SolverState
from .expansion import expand_state
from ..models import PuzzlePiece, FrameHypothesis, InnerMatchCandidate
from ..config import MatchingConfig, FrameModel
from ..fallback.many_to_one import (
    create_composite_segments,
    extend_inner_candidates
)
from ..inner_matching.candidates import generate_inner_candidates
import copy


def beam_search(
    all_pieces: dict[int, PuzzlePiece],
    all_segments: dict[int, list],  # ContourSegment
    frame_hypotheses: dict[int, list[FrameHypothesis]],
    inner_candidates: list[InnerMatchCandidate],
    config: MatchingConfig,
    frame: FrameModel
) -> list[SolverState]:
    """
    Multi-hypothesis beam search for puzzle solving.

    Args:
        all_pieces: Dict mapping piece_id → PuzzlePiece
        all_segments: Dict mapping piece_id → list[ContourSegment]
        frame_hypotheses: Dict mapping piece_id → list[FrameHypothesis] (Top-N)
        inner_candidates: List of all InnerMatchCandidate (global pool)
        config: MatchingConfig with solver parameters
        frame: FrameModel for boundary checks

    Returns:
        List of SolverState sorted by cost_total (ascending, best first)
        - Complete states prioritized
        - If no complete: best partial states (D5 decision)

    Algorithm:
        1. Seeding: Hybrid (empty state + top frame hypotheses per piece)
        2. Expansion loop:
           - Expand all states in beam
           - Extract complete states
           - Prune incomplete states to beam_width
           - Repeat until beam empty or max_expansions reached
        3. Return: Complete states if any, else best partial (D5)

    Design Decisions:
        - D1: is_complete() = all_placed AND open_edges==0 (strict)
        - D5: Return best partial if no complete (debugfreundlich)

    Example:
        >>> solutions = beam_search(pieces, segments, frame_hyps,
        ...                         inner_cands, config, frame)
        >>> best_solution = solutions[0]  # Lowest cost
        >>> if best_solution.is_complete():
        ...     print("Complete solution found!")
    """
    # 1. Seeding (Hybrid: empty + frame hypotheses)
    all_piece_ids = set(all_pieces.keys())
    beam = _create_initial_beam(all_piece_ids, frame_hypotheses, config)

    # 2. Expansion loop
    all_complete_states = []
    all_partial_states = beam.copy()  # Track all partial states for D5
    expansions = 0

    while beam and expansions < config.max_expansions:
        # a. Expand all states in beam
        successors = []
        for state in beam:
            new_states = expand_state(
                state, all_pieces, all_segments,
                frame_hypotheses, inner_candidates,
                config, frame
            )
            successors.extend(new_states)

        # b. Extract complete states (D1: all_placed AND open_edges==0)
        complete = [s for s in successors if s.is_complete()]
        all_complete_states.extend(complete)

        # c. Prune to beam_width (keep incomplete states only)
        incomplete = [s for s in successors if not s.is_complete()]
        beam = sorted(incomplete, key=lambda s: s.cost_total)[:config.beam_width]

        # Track all partial states for D5 (beam collapse fallback)
        all_partial_states.extend(beam)

        expansions += 1

    # 3. Return (D5: best partial if no complete)
    if all_complete_states:
        # Return all complete states, sorted by cost
        return sorted(all_complete_states, key=lambda s: s.cost_total)
    else:
        # D5 Decision: Return best partial states (not empty list)
        # Return best states from all tracked partials
        return sorted(all_partial_states, key=lambda s: s.cost_total)[:config.beam_width]


def _create_initial_beam(
    all_piece_ids: set[int],
    frame_hypotheses: dict[int, list[FrameHypothesis]],
    config: MatchingConfig
) -> list[SolverState]:
    """
    Create initial beam (Hybrid seeding).

    Args:
        all_piece_ids: Set of all piece IDs
        frame_hypotheses: Dict mapping piece_id → list[FrameHypothesis]
        config: MatchingConfig

    Returns:
        List of SolverState sorted by cost_total (ascending)

    Seeding Strategy (B1 test):
        - Empty state (cost 0.0)
        - Top-1 frame hypothesis per piece (if available)
        - Total: 1 + n_pieces states (up to beam_width)

    Notes:
        - Empty state allows inner-only solutions (robust)
        - Frame seeds give fast convergence if frame good
        - Sorted by cost → empty state first (cost 0.0)

    Example:
        >>> beam = _create_initial_beam({1,2,3}, frame_hyps, config)
        >>> len(beam) == 4  # Empty + 3 seeds (if beam_width >= 4)
        >>> beam[0].cost_total == 0.0  # Empty state first
    """
    beam = []

    # Seed A: Empty state (cost 0.0)
    empty_state = SolverState(all_piece_ids=all_piece_ids)
    beam.append(empty_state)

    # Seed B: Top-1 frame hypothesis per piece
    for piece_id in all_piece_ids:
        if piece_id in frame_hypotheses and len(frame_hypotheses[piece_id]) > 0:
            # Take best (first) frame hypothesis
            best_hyp = frame_hypotheses[piece_id][0]

            # Create seed state with this piece placed
            seed_state = SolverState.seed_with_frame_hypothesis(
                all_piece_ids=all_piece_ids,
                hyp=best_hyp
            )
            beam.append(seed_state)

    # Sort by cost and limit to beam_width
    beam = sorted(beam, key=lambda s: s.cost_total)[:config.beam_width]

    return beam


def _attempt_fallback_rerun(
    original_state: SolverState,
    all_pieces: dict[int, PuzzlePiece],
    all_segments: dict[int, list],
    frame_hypotheses: dict[int, list[FrameHypothesis]],
    config: MatchingConfig,
    frame: FrameModel
) -> tuple[SolverState | None, dict]:
    """
    Attempt many-to-one fallback rerun with composite segments.

    Steps:
    1. Create composite segments from atomic segments (k=2, optionally k=3)
    2. Generate original atomic candidates
    3. Extend candidates with composite ↔ atomic matches
    4. Rerun beam_search with extended candidates and fallback config
    5. Return best fallback state + debug info

    Args:
        original_state: Best state from initial beam_search
        all_pieces: All puzzle pieces
        all_segments: All atomic segments (grouped by piece_id)
        frame_hypotheses: Frame hypotheses (same as initial run)
        config: MatchingConfig (will be copied and modified for fallback)
        frame: FrameModel for boundary checks

    Returns:
        (fallback_state, debug_info) tuple:
        - fallback_state: Best state from fallback run (or None if failed)
        - debug_info: Dict with composite counts, candidate counts, etc.

    Max iterations: 1 (no recursive fallback)

    Tests: test_fallback.py Tests 22-25
    """
    # 1. Extract all segments (flat list)
    segments_flat = []
    for seg_list in all_segments.values():
        segments_flat.extend(seg_list)

    # 2. Create composite segments
    composite_segments = create_composite_segments(segments_flat, config)

    # 3. Count composites per piece (for debug)
    composites_per_piece = {}
    for comp in composite_segments:
        pid = comp.piece_id
        composites_per_piece[pid] = composites_per_piece.get(pid, 0) + 1

    # 4. Generate original atomic candidates
    original_candidates = generate_inner_candidates(segments_flat, config)

    # 5. Extend candidates with composites
    extended_candidates_dict = extend_inner_candidates(
        original_candidates,
        atomic_segments=segments_flat,
        composite_segments=composite_segments,
        config=config
    )

    # 6. Flatten dict to list for beam_search
    # beam_search expects list[InnerMatchCandidate] (global pool)
    inner_candidates_flat = []
    for cand_list in extended_candidates_dict.values():
        inner_candidates_flat.extend(cand_list)

    # 7. Create fallback config (override beam_width, topk)
    fallback_config = copy.deepcopy(config)
    fallback_config.beam_width = config.beam_width_fallback
    fallback_config.topk_per_segment = config.topk_per_segment_fallback

    # 8. Rerun beam_search with extended candidates
    fallback_states = beam_search(
        all_pieces=all_pieces,
        all_segments=all_segments,
        frame_hypotheses=frame_hypotheses,
        inner_candidates=inner_candidates_flat,  # Extended with composites
        config=fallback_config,
        frame=frame
    )

    # 9. Extract best fallback state
    if not fallback_states:
        return None, {}

    best_fallback = min(fallback_states, key=lambda s: s.cost_total)

    # 10. Build debug info
    debug_info = {
        "composites_created_per_piece": composites_per_piece,
        "total_composites_created": len(composite_segments),
        "total_candidates_extended": len(inner_candidates_flat),
        # "composite_matches_used": []  # TODO: Extract from best_fallback matches
    }

    return best_fallback, debug_info
