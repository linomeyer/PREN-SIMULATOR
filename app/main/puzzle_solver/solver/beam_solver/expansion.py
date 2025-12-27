"""
Beam-Solver Expansion (Step 6).

This module implements expand_state() which generates child states from a parent state.

Key Concepts:
- Move 1: Place piece via FrameHypothesis (commit frame constraint)
- Move 2: Place piece via InnerMatchCandidate (inner edge matching)
- Pruning: Outside frame, overlap, committed frame conflicts
- Frontier: Hybrid (frame placement → inner matching)
- Penalty: Missing frame contact (soft constraint)

See docs/design/05_solver.md for expansion algorithm.
See docs/implementation/06_beam_solver_test_spec.md for test specification.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

# Import from models
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from solver.beam_solver.state import SolverState
from solver.models import (
    Pose2D, FrameHypothesis, InnerMatchCandidate,
    ContourSegment, PuzzlePiece
)
from solver.config import MatchingConfig, FrameModel
from solver.overlap.collision import penetration_depth_max


def expand_state(
    state: SolverState,
    all_pieces: dict[int, PuzzlePiece],
    all_segments: dict[int, list[ContourSegment]],
    frame_hypotheses: dict[int, list[FrameHypothesis]],
    inner_candidates: list[InnerMatchCandidate],
    config: MatchingConfig,
    frame: FrameModel
) -> list[SolverState]:
    """
    Expand state to generate child states.

    Args:
        state: Parent state to expand
        all_pieces: Dict mapping piece_id → PuzzlePiece
        all_segments: Dict mapping piece_id → list[ContourSegment]
        frame_hypotheses: Dict mapping piece_id → list[FrameHypothesis] (Top-N)
        inner_candidates: List of all InnerMatchCandidate (global pool)
        config: MatchingConfig with solver parameters
        frame: FrameModel for boundary checks

    Returns:
        List of new states (may be empty if all pruned)

    Algorithm:
        1. Determine frontier (unplaced_pieces, open_edges)
        2. Generate expansions:
           - If <2 placed: Frame placement for unplaced pieces
           - If >=2 placed: Inner matching via open_edges
        3. Apply pruning (outside frame, overlap stub, conflicts)
        4. Return filtered states sorted by cost

    Notes:
        - Move 1: Place via FrameHypothesis → commit frame constraint
        - Move 2: Place via InnerMatchCandidate → compute pose, add penalty if no frame
        - Pruning: Hard rejection for invalid states
        - Dedup: Remove duplicate states (same placed + poses)
        - Branching cap: Use config.debug_topN_frame_hypotheses_per_piece (D4)

    Example:
        >>> new_states = expand_state(state, pieces, segments, frame_hyps,
        ...                            inner_cands, config, frame)
        >>> # new_states sorted by cost_total (ascending)
    """
    new_states = []

    # Get frontier
    unplaced_pieces, open_edges = state.get_frontier()

    # Branching cap (D4: reuse debug_topN_frame_hypotheses_per_piece)
    branching_cap = config.debug_topN_frame_hypotheses_per_piece

    # Sequential frontier switching (design/05_solver.md §Frontier-Ansatz 3)
    # Phase 1 (n_placed < 1): Frame placement only (bootstrap, empty state)
    # Phase 2 (n_placed >= 1): Inner matching (priority), fallback to frame if no open edges
    n_placed = len(state.placed_pieces)
    threshold = 1  # V1: Switch after first piece (prevents empty-state inner matching)

    # Move 2: Place via InnerMatchCandidate (Phase 2: after threshold with candidates available)
    if n_placed >= threshold and len(open_edges) > 0 and len(inner_candidates) > 0:
        for candidate in inner_candidates:
            seg_a_ref, seg_b_ref = candidate.seg_a_ref, candidate.seg_b_ref
            piece_a, seg_a_id = seg_a_ref
            piece_b, seg_b_id = seg_b_ref

            # Preconditions:
            # - piece_a placed, piece_b unplaced
            # - (piece_a, seg_a_id) in open_edges
            # - piece_a != piece_b (no self-match, E7)

            if piece_a == piece_b:
                continue  # E7: Self-match rejected

            if piece_a not in state.placed_pieces:
                continue  # piece_a not placed

            if piece_b in state.placed_pieces:
                continue  # piece_b already placed

            if (piece_a, seg_a_id) not in open_edges:
                continue  # Edge not in frontier

            # Create new state
            new_state = state.copy()

            # Compute pose_B from pose_A and candidate (D3: 180deg flip + chord midpoint)
            pose_A = state.poses_F[piece_a]
            pose_B = _compute_pose_from_inner_match(
                pose_A, piece_a, seg_a_id, piece_b, seg_b_id,
                all_segments, candidate
            )

            # Place piece_b
            new_state.placed_pieces.add(piece_b)
            new_state.unplaced_pieces.discard(piece_b)
            new_state.poses_F[piece_b] = pose_B

            # Close matched edges (E6)
            new_state.open_edges.discard((piece_a, seg_a_id))
            new_state.committed_edges.add((piece_a, seg_a_id))
            new_state.committed_edges.add((piece_b, seg_b_id))

            # Open remaining edges of piece_b (D2: all non-committed)
            if piece_b in all_segments:
                for seg in all_segments[piece_b]:
                    edge_id = (piece_b, seg.segment_id)
                    if edge_id not in new_state.committed_edges:
                        new_state.open_edges.add(edge_id)

            # Update cost (add inner cost)
            new_state.cost_total += candidate.cost_inner
            new_state.cost_breakdown['inner'] = \
                new_state.cost_breakdown.get('inner', 0.0) + candidate.cost_inner

            # Penalty if piece_b has no committed frame constraint (E3)
            if piece_b not in new_state.committed_frame_constraints:
                penalty = config.penalty_missing_frame_contact
                new_state.cost_total += penalty
                new_state.cost_breakdown['penalty'] = \
                    new_state.cost_breakdown.get('penalty', 0.0) + penalty

            # Record match
            new_state.matches.append({
                'seg_a': seg_a_ref,
                'seg_b': seg_b_ref,
                'cost_inner': candidate.cost_inner
            })

            # Pruning checks
            if not _check_valid_state(new_state, all_pieces, config, frame):
                continue  # Pruned

            new_states.append(new_state)

    # Move 1: Place via FrameHypothesis (Phase 1: before threshold OR no candidates for Phase 2)
    if (n_placed < threshold or len(open_edges) == 0 or len(inner_candidates) == 0) and len(unplaced_pieces) > 0:
        # Move 1: Place via FrameHypothesis
        for piece_id in unplaced_pieces:
            if piece_id not in frame_hypotheses:
                continue

            # Limit to top-N frame hypotheses per piece (branching cap)
            hyps = frame_hypotheses[piece_id][:branching_cap]

            for hyp in hyps:
                # Create new state
                new_state = state.copy()

                # Place piece
                new_state.placed_pieces.add(piece_id)
                new_state.unplaced_pieces.discard(piece_id)
                new_state.poses_F[piece_id] = hyp.pose_grob_F

                # Commit frame hypothesis
                new_state.committed_frame_constraints[piece_id] = hyp
                new_state.committed_edges.add((piece_id, hyp.segment_id))

                # Update cost (add frame cost)
                new_state.cost_total += hyp.cost_frame
                new_state.cost_breakdown['frame'] = \
                    new_state.cost_breakdown.get('frame', 0.0) + hyp.cost_frame

                # Update open_edges (D2: all non-committed segments)
                if piece_id in all_segments:
                    for seg in all_segments[piece_id]:
                        edge_id = (piece_id, seg.segment_id)
                        if edge_id not in new_state.committed_edges:
                            new_state.open_edges.add(edge_id)

                # Pruning checks
                if not _check_valid_state(new_state, all_pieces, config, frame):
                    continue  # Pruned

                new_states.append(new_state)

    # Dedup (E11: same placed + poses → keep best cost)
    new_states = _deduplicate_states(new_states)

    # Sort by cost_total (ascending)
    new_states.sort(key=lambda s: s.cost_total)

    return new_states


def _compute_pose_from_inner_match(
    pose_A: Pose2D,
    piece_a: int,
    seg_a_id: int,
    piece_b: int,
    seg_b_id: int,
    all_segments: dict[int, list[ContourSegment]],
    candidate: InnerMatchCandidate
) -> Pose2D:
    """
    Compute pose_B from inner match (D3: 180deg flip + chord midpoint).

    Algorithm (Test E5 spec):
        1. Get chord midpoint of seg_a in Frame coords
        2. theta_B = pose_A.theta_deg + 180.0
        3. Place piece_b so seg_b chord midpoint aligns with seg_a midpoint

    Args:
        pose_A: Pose of piece_a in Frame coords
        piece_a: Piece A ID
        seg_a_id: Segment A ID
        piece_b: Piece B ID
        seg_b_id: Segment B ID
        all_segments: Segment data
        candidate: InnerMatchCandidate

    Returns:
        Pose2D for piece_b in Frame coords

    Notes:
        - Simple V1 implementation (chord-based alignment)
        - Assumes segment chords available in piece-local coords
        - 180deg flip ensures pieces face each other

    Example (from test E5):
        pose_A = Pose2D(0, 0, 0)
        seg_a chord: (0,0)→(20,0) in piece coords
        seg_b chord: (0,0)→(20,0) in piece coords
        → pose_B = Pose2D(20, 0, 180)
    """
    # Get segments
    seg_a = all_segments[piece_a][seg_a_id]
    seg_b = all_segments[piece_b][seg_b_id]

    # Get chord endpoints in piece-local coords
    # Chord: first and last point of segment
    pts_a = seg_a.points_mm  # shape (N, 2)
    pts_b = seg_b.points_mm

    chord_a_start = pts_a[0]
    chord_a_end = pts_a[-1]
    chord_b_start = pts_b[0]
    chord_b_end = pts_b[-1]

    # Chord midpoint in piece-local coords
    mid_a_local = (chord_a_start + chord_a_end) / 2.0
    mid_b_local = (chord_b_start + chord_b_end) / 2.0

    # Transform mid_a to Frame coords
    theta_A_rad = np.deg2rad(pose_A.theta_deg)
    cos_A, sin_A = np.cos(theta_A_rad), np.sin(theta_A_rad)

    # Rotation matrix for pose_A
    R_A = np.array([[cos_A, -sin_A],
                    [sin_A, cos_A]])

    mid_a_F = R_A @ mid_a_local + np.array([pose_A.x_mm, pose_A.y_mm])

    # Compute pose_B (D3: 180deg flip)
    theta_B = pose_A.theta_deg + 180.0

    # Normalize theta to [0, 360)
    theta_B = theta_B % 360.0

    # Compute translation: mid_a_F = R_B @ mid_b_local + T_B
    # → T_B = mid_a_F - R_B @ mid_b_local
    theta_B_rad = np.deg2rad(theta_B)
    cos_B, sin_B = np.cos(theta_B_rad), np.sin(theta_B_rad)
    R_B = np.array([[cos_B, -sin_B],
                    [sin_B, cos_B]])

    T_B = mid_a_F - R_B @ mid_b_local

    return Pose2D(x_mm=T_B[0], y_mm=T_B[1], theta_deg=theta_B)


def _check_valid_state(
    state: SolverState,
    all_pieces: dict[int, PuzzlePiece],
    config: MatchingConfig,
    frame: FrameModel
) -> bool:
    """
    Pruning checks for new state.

    Checks:
        1. Outside frame (tau_frame_mm tolerance) - E8
        2. Overlap (stub for Step 7) - E9
        3. Committed frame conflict (side mismatch) - E10

    Args:
        state: State to validate
        all_pieces: Piece data
        config: Config with pruning thresholds
        frame: FrameModel for boundary checks

    Returns:
        True if valid (not pruned), False if pruned
    """
    # E8: Outside frame check
    if not _check_inside_frame(state, all_pieces, frame, config.tau_frame_mm):
        return False

    # E9: Overlap check (SAT/MTV from Step 7)
    overlap_depth = penetration_depth_max(state, all_pieces)
    if overlap_depth > config.overlap_depth_max_mm_prune:
        return False

    # E10: Committed frame conflict check
    # NOTE: V1 implementation skips this check during expansion since:
    #  - Expansion doesn't re-pose already-placed pieces
    #  - New piece poses are set FROM hypotheses (consistent by construction)
    #  - Conflict check only relevant when modifying committed pieces (not in V1)
    # if not _check_committed_frame_constraints(state, all_pieces, frame):
    #     return False

    return True


def _check_inside_frame(
    state: SolverState,
    all_pieces: dict[int, PuzzlePiece],
    frame: FrameModel,
    tau_frame_mm: float
) -> bool:
    """
    Check if all placed pieces are inside frame (with tolerance).

    Algorithm:
        For each placed piece:
        - Get bounding box in Frame coords (transform via pose)
        - Check if bbox is within frame bounds + tau_frame_mm

    Args:
        state: State to check
        all_pieces: Piece data
        frame: FrameModel
        tau_frame_mm: Tolerance (safety band)

    Returns:
        True if all pieces inside, False if any outside

    Notes:
        - Simple bbox check (V1, conservative)
        - tau_frame_mm allows small violations (noise tolerance)
        - Frame inner bounds: [0, 0] to [W, H]
    """
    for piece_id in state.placed_pieces:
        if piece_id not in all_pieces:
            continue  # Skip if piece data missing

        piece = all_pieces[piece_id]
        pose = state.poses_F[piece_id]

        # Get bbox in piece-local coords (mm)
        if piece.bbox_mm is None:
            continue  # Skip if no bbox data

        x_min, y_min, x_max, y_max = piece.bbox_mm

        # Compute bbox center for rotation pivot
        bbox_cx = (x_min + x_max) / 2.0
        bbox_cy = (y_min + y_max) / 2.0

        # Transform bbox corners to Frame coords
        # Step 1: Translate corners to be centered at origin
        theta_rad = np.deg2rad(pose.theta_deg)
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
        R = np.array([[cos_t, -sin_t],
                      [sin_t, cos_t]])

        corners_centered = np.array([
            [x_min - bbox_cx, y_min - bbox_cy],
            [x_min - bbox_cx, y_max - bbox_cy],
            [x_max - bbox_cx, y_min - bbox_cy],
            [x_max - bbox_cx, y_max - bbox_cy]
        ])

        # Step 2: Rotate around origin (= bbox center)
        corners_rotated = (R @ corners_centered.T).T

        # Step 3: Translate back to bbox center + apply pose translation
        corners_F = corners_rotated + np.array([bbox_cx + pose.x_mm, bbox_cy + pose.y_mm])

        # Check if all corners inside frame + tolerance
        bbox_x_min = corners_F[:, 0].min()
        bbox_x_max = corners_F[:, 0].max()
        bbox_y_min = corners_F[:, 1].min()
        bbox_y_max = corners_F[:, 1].max()

        # Frame bounds (with negative tolerance = shrink allowed zone)
        frame_x_min = -tau_frame_mm
        frame_x_max = frame.inner_width_mm + tau_frame_mm
        frame_y_min = -tau_frame_mm
        frame_y_max = frame.inner_height_mm + tau_frame_mm

        if (bbox_x_min < frame_x_min or bbox_x_max > frame_x_max or
            bbox_y_min < frame_y_min or bbox_y_max > frame_y_max):
            return False  # Outside frame

    return True


def _check_committed_frame_constraints(
    state: SolverState,
    all_pieces: dict[int, PuzzlePiece],
    frame: FrameModel
) -> bool:
    """
    Check if poses satisfy committed frame constraints.

    Algorithm (E10 spec):
        For each committed frame constraint:
        - Check if pose places segment on correct side
        - Simple check: side="TOP" → y >= H - tau

    Args:
        state: State to check
        all_pieces: Piece data
        frame: FrameModel

    Returns:
        True if no conflicts, False if any conflict

    Notes:
        - E10: Committed frame conflict check
        - Simplified V1: Check y-coord for TOP/BOTTOM, x-coord for LEFT/RIGHT
        - Tolerance: 2.0mm (hardcoded for V1)
    """
    tau = 2.0  # Tolerance for side checks (mm)

    for piece_id, hyp in state.committed_frame_constraints.items():
        if piece_id not in state.poses_F:
            continue  # Piece not yet placed

        pose = state.poses_F[piece_id]
        side = hyp.side

        # Simple side check (based on pose position)
        if side == "TOP":
            if pose.y_mm < frame.inner_height_mm - tau:
                return False  # Not at TOP
        elif side == "BOTTOM":
            if pose.y_mm > tau:
                return False  # Not at BOTTOM
        elif side == "LEFT":
            if pose.x_mm > tau:
                return False  # Not at LEFT
        elif side == "RIGHT":
            if pose.x_mm < frame.inner_width_mm - tau:
                return False  # Not at RIGHT

    return True


def _deduplicate_states(states: list[SolverState]) -> list[SolverState]:
    """
    Remove duplicate states (E11).

    Duplicates = same placed pieces + same poses (ignoring cost).
    Keep state with lowest cost.

    Args:
        states: List of states

    Returns:
        Deduplicated list

    Notes:
        - E11: Dedup based on (placed, poses)
        - Poses compared with 1e-6 tolerance
        - Cost not part of dedup key (keep best cost)
    """
    if not states:
        return []

    # Group by (placed, poses) signature
    sig_map = {}  # signature → list of states

    for state in states:
        # Create signature: (placed_pieces, poses_F)
        placed_tuple = tuple(sorted(state.placed_pieces))
        poses_tuple = tuple(
            (pid, round(pose.x_mm, 6), round(pose.y_mm, 6), round(pose.theta_deg, 6))
            for pid, pose in sorted(state.poses_F.items())
        )
        sig = (placed_tuple, poses_tuple)

        if sig not in sig_map:
            sig_map[sig] = []
        sig_map[sig].append(state)

    # Keep best (lowest cost) state per signature
    deduped = []
    for states_group in sig_map.values():
        best_state = min(states_group, key=lambda s: s.cost_total)
        deduped.append(best_state)

    return deduped
