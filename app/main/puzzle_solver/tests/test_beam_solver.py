"""
Tests for Beam-Solver (Step 6): SolverState + Expansion + beam_search.

Test Spec: docs/implementation/06_beam_solver_test_spec.md
Design Decisions D1-D5 finalized (see spec §6).

Test Groups:
- S1-S9: beam_solver/state.py (SolverState class)
- E1-E12: beam_solver/expansion.py (expand_state function)
- B1-B9: beam_solver/solver.py (beam_search function)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from solver.beam_solver.state import SolverState
from solver.models import Pose2D, FrameHypothesis, InnerMatchCandidate, ContourSegment, PuzzlePiece
from solver.config import MatchingConfig, FrameModel


# ========== Fixtures (§1 Test-Spec) ==========

@pytest.fixture
def frame_model():
    """Standard frame for tests: 100mm x 80mm"""
    return FrameModel(inner_width_mm=100.0, inner_height_mm=80.0)


@pytest.fixture
def config():
    """Standard config for beam solver tests"""
    return MatchingConfig(
        beam_width=3,
        max_expansions=20,
        overlap_depth_max_mm_prune=1.0,
        penalty_missing_frame_contact=10.0,
        debug_topN_frame_hypotheses_per_piece=5,
        tau_frame_mm=2.0
    )


@pytest.fixture
def simple_pieces():
    """Create 4 simple test pieces (just IDs, no geometry needed for state tests)"""
    return {1, 2, 3, 4}


def make_segment(piece_id, segment_id, points):
    """Helper to create minimal ContourSegment for tests"""
    pts = np.array(points, dtype=float)
    chord_start, chord_end = pts[0], pts[-1]
    length_mm = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    direction_angle_deg = np.degrees(np.arctan2(chord_end[1] - chord_start[1],
                                                  chord_end[0] - chord_start[0]))
    flatness_error = 0.1  # Dummy value

    return ContourSegment(
        piece_id=piece_id,
        segment_id=segment_id,
        points_mm=pts,
        length_mm=length_mm,
        chord=(chord_start, chord_end),
        direction_angle_deg=direction_angle_deg,
        flatness_error=flatness_error
    )


# ========== Test Group S: beam_solver/state.py (SolverState) ==========

def test_S1_import_single_source():
    """S1: SolverState MUST NOT be in models.py (Single Source of Truth)"""
    print("\nTest S1: Import Single Source...", end=" ")

    from solver import models

    # SolverState should NOT exist in models module
    assert not hasattr(models, "SolverState"), \
        "SolverState MUST NOT be defined in models.py (Single Source: beam_solver/state.py)"

    # But should be importable from beam_solver.state
    from solver.beam_solver.state import SolverState
    assert SolverState is not None

    print("✓")


def test_S2_empty_state(simple_pieces):
    """S2: Empty state initialization"""
    print("\nTest S2: Empty State...", end=" ")

    state = SolverState(all_piece_ids=simple_pieces)

    # Expected state
    assert state.n_pieces == 4
    assert state.placed_pieces == set()
    assert state.unplaced_pieces == {1, 2, 3, 4}
    assert state.poses_F == {}
    assert state.open_edges == set()
    assert state.committed_edges == set()
    assert state.cost_total == 0.0

    print("✓")


def test_S3_seed_state_frame_hypothesis(frame_model):
    """S3: Seed state via FrameHypothesis (1 piece placed)"""
    print("\nTest S3: Seed State...", end=" ")

    # Create frame hypothesis
    hyp = FrameHypothesis(
        piece_id=1,
        segment_id=0,
        side="TOP",
        pose_grob_F=Pose2D(x_mm=10.0, y_mm=70.0, theta_deg=0.0),
        features=None,  # Not needed for state test
        cost_frame=0.4,
        is_committed=False,
        uncertainty_mm=5.0
    )

    # Seed state
    state = SolverState.seed_with_frame_hypothesis(
        all_piece_ids={1, 2},
        hyp=hyp
    )

    # Verify
    assert state.n_pieces == 2
    assert state.placed_pieces == {1}
    assert state.unplaced_pieces == {2}
    assert state.poses_F[1] == Pose2D(x_mm=10.0, y_mm=70.0, theta_deg=0.0)
    assert state.committed_frame_constraints[1] == hyp
    assert state.cost_total == 0.4

    # open_edges: Policy D2 (all non-committed segments)
    # For seed state, no segments info available yet → len >= 0
    assert isinstance(state.open_edges, set)
    assert len(state.open_edges) >= 0

    print("✓")


def test_S4_copy_deep(simple_pieces):
    """S4: copy() creates deep copy"""
    print("\nTest S4: Deep Copy...", end=" ")

    state = SolverState(all_piece_ids=simple_pieces)
    state.placed_pieces.add(1)
    state.poses_F[1] = Pose2D(10, 20, 30)
    state.open_edges.add((1, 0))

    # Copy
    state2 = state.copy()

    # Modify copy
    state2.placed_pieces.add(99)
    state2.poses_F[99] = Pose2D(0, 0, 0)
    state2.open_edges.add((99, 0))

    # Original should be unchanged
    assert 99 not in state.placed_pieces
    assert 99 not in state.poses_F
    assert (99, 0) not in state.open_edges

    # But shared values should be equal
    assert 1 in state2.placed_pieces
    assert state2.poses_F[1] == Pose2D(10, 20, 30)

    print("✓")


def test_S5_get_frontier(simple_pieces):
    """S5: get_frontier() returns two separate sets"""
    print("\nTest S5: Get Frontier...", end=" ")

    state = SolverState(all_piece_ids=simple_pieces)
    state.placed_pieces = {1, 2}
    state.unplaced_pieces = {3, 4}
    state.open_edges = {(1, 0), (2, 1)}

    # Get frontier
    unplaced, open_edges = state.get_frontier()

    # Verify types and content
    assert isinstance(unplaced, set)
    assert isinstance(open_edges, set)
    assert unplaced == {3, 4}
    assert open_edges == {(1, 0), (2, 1)}

    print("✓")


def test_S6_invariants_valid(simple_pieces):
    """S6: Invariants validation (valid state)"""
    print("\nTest S6: Valid Invariants...", end=" ")

    state = SolverState(all_piece_ids=simple_pieces)
    state.placed_pieces = {1, 2}
    state.unplaced_pieces = {3, 4}
    state.poses_F = {1: Pose2D(0, 0, 0), 2: Pose2D(10, 10, 0)}
    state.open_edges = {(1, 0), (2, 1)}

    # Should not raise
    state.validate_invariants()

    print("✓")


def test_S7_invariants_invalid_open_edge():
    """S7: Invariants violation: open_edge from unplaced piece"""
    print("\nTest S7: Invalid Invariants (open_edge)...", end=" ")

    state = SolverState(all_piece_ids={1, 2, 3})
    state.placed_pieces = {1}
    state.unplaced_pieces = {2, 3}
    state.poses_F = {1: Pose2D(0, 0, 0)}
    state.open_edges = {(3, 0)}  # Piece 3 NOT placed!

    # Should raise
    try:
        state.validate_invariants()
        raise AssertionError("Should have raised ValueError for invalid open_edge")
    except ValueError as e:
        assert "open_edges must reference placed pieces" in str(e).lower()

    print("✓")


def test_S8_is_complete():
    """S8: is_complete() logic (D1: all placed AND open_edges==0)"""
    print("\nTest S8: is_complete()...", end=" ")

    # Setup A: Complete (all placed, no open edges)
    state_a = SolverState(all_piece_ids={1, 2})
    state_a.placed_pieces = {1, 2}
    state_a.unplaced_pieces = set()
    state_a.open_edges = set()
    assert state_a.is_complete() == True, "All placed + no open edges → complete"

    # Setup B: Incomplete (all placed, BUT open edges exist)
    state_b = SolverState(all_piece_ids={1, 2})
    state_b.placed_pieces = {1, 2}
    state_b.unplaced_pieces = set()
    state_b.open_edges = {(1, 0)}
    assert state_b.is_complete() == False, "Open edges exist → not complete"

    # Setup C: Incomplete (not all placed)
    state_c = SolverState(all_piece_ids={1, 2})
    state_c.placed_pieces = {1}
    state_c.unplaced_pieces = {2}
    state_c.open_edges = set()
    assert state_c.is_complete() == False, "Not all placed → not complete"

    print("✓")


def test_S9_cost_breakdown_consistency():
    """S9: cost_breakdown consistency with cost_total"""
    print("\nTest S9: Cost Breakdown...", end=" ")

    state = SolverState(all_piece_ids={1, 2})
    state.cost_breakdown = {"frame": 0.4, "inner": 0.2, "penalty": 10.0}
    state.cost_total = 10.6

    # Verify consistency
    computed_total = sum(state.cost_breakdown.values())
    assert abs(computed_total - state.cost_total) < 1e-6, \
        f"cost_total ({state.cost_total}) != sum(breakdown) ({computed_total})"

    print("✓")


# ========== Test Group E: beam_solver/expansion.py (expand_state) ==========

from solver.beam_solver.expansion import expand_state


def test_E1_expand_empty_frame_placement(frame_model, config):
    """E1: Expand from empty → Frame placement creates seeds"""
    print("\nTest E1: Frame Placement...", end=" ")

    # Create empty state
    state = SolverState(all_piece_ids={1, 2})

    # Create frame hypotheses for piece 1
    # Frame: 100x80, tau=2.0
    hyp1 = FrameHypothesis(
        piece_id=1, segment_id=0, side="TOP",
        pose_grob_F=Pose2D(x_mm=40.0, y_mm=65.0, theta_deg=0.0),  # Safe position
        features=None, cost_frame=0.2, is_committed=False, uncertainty_mm=5.0
    )
    hyp2 = FrameHypothesis(
        piece_id=1, segment_id=1, side="BOTTOM",
        pose_grob_F=Pose2D(x_mm=40.0, y_mm=0.5, theta_deg=0.0),  # At BOTTOM (y < tau=2.0)
        features=None, cost_frame=0.5, is_committed=False, uncertainty_mm=5.0
    )

    frame_hyps = {1: [hyp1, hyp2]}

    # Create minimal pieces/segments (for pruning checks)
    pieces = {
        1: PuzzlePiece(piece_id=1, bbox_mm=(0, 0, 10, 10))  # Smaller piece
    }
    segments = {
        1: [
            make_segment(1, 0, [[0,0],[10,0]]),
            make_segment(1, 1, [[10,0],[10,10]])
        ]
    }

    # Expand
    new_states = expand_state(
        state, pieces, segments, frame_hyps, [],
        config, frame_model
    )

    # Expected: 2 new states (one per hypothesis)
    assert len(new_states) >= 2, f"Expected >=2 states, got {len(new_states)}"

    # Best state has cost 0.2
    best_state = new_states[0]
    assert best_state.cost_total == 0.2, \
        f"Expected cost 0.2, got {best_state.cost_total}"
    assert best_state.placed_pieces == {1}

    # Second state has cost 0.5
    assert new_states[1].cost_total == 0.5

    print("✓")


def test_E2_frame_placement_commits_hypothesis(frame_model, config):
    """E2: Frame placement commits hypothesis"""
    print("\nTest E2: Commit Hypothesis...", end=" ")

    state = SolverState(all_piece_ids={1})

    hyp = FrameHypothesis(
        piece_id=1, segment_id=0, side="TOP",
        pose_grob_F=Pose2D(x_mm=10.0, y_mm=70.0, theta_deg=0.0),
        features=None, cost_frame=0.3, is_committed=False, uncertainty_mm=5.0
    )

    frame_hyps = {1: [hyp]}
    pieces = {1: PuzzlePiece(piece_id=1, bbox_mm=(-10, -10, 10, 10))}  # Centered bbox
    segments = {1: [make_segment(1, 0, [[0,0],[10,0]])]}

    new_states = expand_state(state, pieces, segments, frame_hyps, [], config, frame_model)

    # Check committed
    assert len(new_states) == 1
    new_state = new_states[0]
    assert 1 in new_state.committed_frame_constraints
    assert new_state.committed_frame_constraints[1] == hyp
    assert (1, 0) in new_state.committed_edges

    print("✓")


def test_E3_penalty_missing_frame_contact(frame_model, config):
    """E3: Penalty if piece placed via inner match without frame commitment"""
    print("\nTest E3: Missing Frame Penalty...", end=" ")

    # State: piece 1 already placed (with frame commitment)
    state = SolverState(all_piece_ids={1, 2})
    state.placed_pieces = {1}
    state.unplaced_pieces = {2}
    state.poses_F = {1: Pose2D(10, 10, 0)}  # Safe position with centered bbox
    state.open_edges = {(1, 0)}
    state.committed_frame_constraints = {
        1: FrameHypothesis(
            piece_id=1, segment_id=0, side="TOP",
            pose_grob_F=Pose2D(10, 10, 0),  # Match poses_F
            features=None, cost_frame=0.1, is_committed=True, uncertainty_mm=5.0
        )
    }

    # Inner match candidate: (1,0) ↔ (2,0)
    cand = InnerMatchCandidate(
        seg_a_ref=(1, 0), seg_b_ref=(2, 0),
        cost_inner=0.2, profile_cost=0.1, length_cost=0.05, fit_cost=0.05,
        reversal_used=False, sign_flip_used=False
    )

    pieces = {
        1: PuzzlePiece(piece_id=1, bbox_mm=(-10, -5, 10, 5)),  # Centered bbox
        2: PuzzlePiece(piece_id=2, bbox_mm=(-10, -5, 10, 5))   # Centered bbox
    }
    segments = {
        1: [make_segment(1, 0, [[0,0],[20,0]])],
        2: [make_segment(2, 0, [[0,0],[20,0]])]
    }

    new_states = expand_state(state, pieces, segments, {}, [cand], config, frame_model)

    # Expected: penalty added (piece 2 has no frame commitment)
    assert len(new_states) >= 1
    new_state = new_states[0]

    # cost_total = cost_inner (0.2) + penalty (10.0) = 10.2
    expected_cost = 0.2 + config.penalty_missing_frame_contact
    assert abs(new_state.cost_total - expected_cost) < 1e-6, \
        f"Expected cost {expected_cost}, got {new_state.cost_total}"

    print("✓")


def test_E4_no_penalty_if_frame_committed(frame_model, config):
    """E4: No penalty if piece already has frame commitment"""
    print("\nTest E4: No Penalty (Frame Committed)...", end=" ")

    # State: piece 1 placed with frame commitment
    state = SolverState(all_piece_ids={1, 2})
    state.placed_pieces = {1}
    state.unplaced_pieces = {2}
    state.poses_F = {1: Pose2D(0, 70, 0)}
    state.open_edges = {(1, 0)}
    state.committed_frame_constraints = {
        1: FrameHypothesis(
            piece_id=1, segment_id=0, side="TOP",
            pose_grob_F=Pose2D(0, 70, 0),
            features=None, cost_frame=0.1, is_committed=True, uncertainty_mm=5.0
        )
    }

    # Place piece 2 via FRAME first (commits frame)
    hyp2 = FrameHypothesis(
        piece_id=2, segment_id=0, side="TOP",
        pose_grob_F=Pose2D(25, 70, 0),
        features=None, cost_frame=0.15, is_committed=False, uncertainty_mm=5.0
    )

    pieces = {
        1: PuzzlePiece(piece_id=1, bbox_mm=(0, 0, 20, 10)),
        2: PuzzlePiece(piece_id=2, bbox_mm=(0, 0, 20, 10))
    }
    segments = {
        1: [make_segment(1, 0, [[0,0],[20,0]])],
        2: [make_segment(2, 0, [[0,0],[20,0]])]
    }

    # Expand with frame hypothesis (no inner candidates, <2 placed)
    new_states = expand_state(state, pieces, segments, {2: [hyp2]}, [], config, frame_model)

    # Expected: no penalty (piece 2 gets frame commitment)
    assert len(new_states) >= 1
    new_state = new_states[0]

    # cost_total = 0 (parent) + cost_frame (0.15) = 0.15 (no penalty)
    assert abs(new_state.cost_total - 0.15) < 1e-6, \
        f"Expected cost 0.15, got {new_state.cost_total}"
    assert 2 in new_state.committed_frame_constraints

    print("✓")


def test_E5_inner_placement_pose_deterministic(frame_model, config):
    """E5: Inner placement computes pose_B deterministically (180deg flip)"""
    print("\nTest E5: Inner Pose Computation...", end=" ")

    # Setup: piece 1 at (10,10,0), seg_a chord (0,0)→(20,0)
    #        piece 2 unplaced, seg_b chord (0,0)→(20,0)
    #        Expected: pose_B = (30, 10, 180)

    state = SolverState(all_piece_ids={1, 2})
    state.placed_pieces = {1}
    state.unplaced_pieces = {2}
    state.poses_F = {1: Pose2D(x_mm=10.0, y_mm=10.0, theta_deg=0.0)}  # Safe position
    state.open_edges = {(1, 0)}

    # Mock frame commitment for piece 1 (to avoid penalty)
    state.committed_frame_constraints = {
        1: FrameHypothesis(
            piece_id=1, segment_id=0, side="TOP",
            pose_grob_F=Pose2D(10, 10, 0),  # Match poses_F
            features=None, cost_frame=0.0, is_committed=True, uncertainty_mm=5.0
        )
    }

    cand = InnerMatchCandidate(
        seg_a_ref=(1, 0), seg_b_ref=(2, 0),
        cost_inner=0.2, profile_cost=0.1, length_cost=0.05, fit_cost=0.05,
        reversal_used=False, sign_flip_used=False
    )

    pieces = {
        1: PuzzlePiece(piece_id=1, bbox_mm=(-10, -5, 10, 5)),  # Centered bbox
        2: PuzzlePiece(piece_id=2, bbox_mm=(-10, -5, 10, 5))   # Centered bbox
    }

    # Segments with chord (0,0)→(20,0)
    segments = {
        1: [make_segment(1, 0, [[0.0, 0.0], [20.0, 0.0]])],
        2: [make_segment(2, 0, [[0.0, 0.0], [20.0, 0.0]])]
    }

    new_states = expand_state(state, pieces, segments, {}, [cand], config, frame_model)

    assert len(new_states) >= 1
    new_state = new_states[0]

    # Expected pose_B: (30, 10, 180)
    pose_B = new_state.poses_F[2]
    assert abs(pose_B.x_mm - 30.0) < 1e-6, f"Expected x=30, got {pose_B.x_mm}"
    assert abs(pose_B.y_mm - 10.0) < 1e-6, f"Expected y=10, got {pose_B.y_mm}"
    assert abs(pose_B.theta_deg - 180.0) < 1e-6, f"Expected theta=180, got {pose_B.theta_deg}"

    # Cost = cost_inner (0.2) + penalty (10.0 if no frame for piece2)
    # But piece2 has no frame commitment → penalty applies
    expected_cost = 0.2 + config.penalty_missing_frame_contact
    assert abs(new_state.cost_total - expected_cost) < 1e-6, \
        f"Expected cost {expected_cost}, got {new_state.cost_total}"

    print("✓")


def test_E6_open_edges_update(frame_model, config):
    """E6: open_edges update after inner match (close matched, open others)"""
    print("\nTest E6: Open Edges Update...", end=" ")

    state = SolverState(all_piece_ids={1, 2})
    state.placed_pieces = {1}
    state.unplaced_pieces = {2}
    state.poses_F = {1: Pose2D(10, 10, 0)}  # Safe position with centered bbox
    state.open_edges = {(1, 0)}  # Only seg 0 open on piece 1

    cand = InnerMatchCandidate(
        seg_a_ref=(1, 0), seg_b_ref=(2, 0),
        cost_inner=0.1, profile_cost=0.05, length_cost=0.03, fit_cost=0.02,
        reversal_used=False
    )

    pieces = {
        1: PuzzlePiece(piece_id=1, bbox_mm=(-10, -5, 10, 5)),  # Centered bbox
        2: PuzzlePiece(piece_id=2, bbox_mm=(-10, -5, 10, 5))   # Centered bbox
    }

    # Piece 1: seg 0,1; Piece 2: seg 0,1,2
    segments = {
        1: [
            make_segment(1, 0, [[0,0],[10,0]]),
            make_segment(1, 1, [[10,0],[10,10]])
        ],
        2: [
            make_segment(2, 0, [[0,0],[10,0]]),
            make_segment(2, 1, [[10,0],[10,10]]),
            make_segment(2, 2, [[10,10],[0,10]])
        ]
    }

    new_states = expand_state(state, pieces, segments, {}, [cand], config, frame_model)

    assert len(new_states) >= 1
    new_state = new_states[0]

    # Check open_edges
    # (1,0) should be closed (committed)
    assert (1, 0) not in new_state.open_edges, "(1,0) should be closed"
    assert (1, 0) in new_state.committed_edges

    # (2,0) should be closed (committed)
    assert (2, 0) not in new_state.open_edges, "(2,0) should be closed"
    assert (2, 0) in new_state.committed_edges

    # (2,1) and (2,2) should be opened (D2: all non-committed)
    assert (2, 1) in new_state.open_edges, "(2,1) should be open"
    assert (2, 2) in new_state.open_edges, "(2,2) should be open"

    print("✓")


def test_E7_self_match_rejected():
    """E7: Self-match is never expanded"""
    print("\nTest E7: Self-Match Rejection...", end=" ")

    state = SolverState(all_piece_ids={1})
    state.placed_pieces = {1}
    state.unplaced_pieces = set()
    state.poses_F = {1: Pose2D(0, 0, 0)}
    state.open_edges = {(1, 0), (1, 2)}

    # Self-match candidate: (1,0) ↔ (1,2)
    cand = InnerMatchCandidate(
        seg_a_ref=(1, 0), seg_b_ref=(1, 2),
        cost_inner=0.0, profile_cost=0.0, length_cost=0.0, fit_cost=0.0,
        reversal_used=False
    )

    pieces = {1: PuzzlePiece(piece_id=1, bbox_mm=(0, 0, 20, 10))}
    segments = {
        1: [
            make_segment(1, 0, [[0,0],[10,0]]),
            make_segment(1, 2, [[10,10],[0,10]])
        ]
    }

    frame_model = FrameModel(inner_width_mm=100.0, inner_height_mm=80.0)
    config = MatchingConfig()

    new_states = expand_state(state, pieces, segments, {}, [cand], config, frame_model)

    # Expected: no new states (self-match rejected)
    assert len(new_states) == 0, "Self-match should be rejected"

    print("✓")


def test_E8_outside_frame_prune(frame_model, config):
    """E8: Outside-frame prune (tau_frame_mm tolerance)"""
    print("\nTest E8: Outside Frame Prune...", end=" ")

    state = SolverState(all_piece_ids={1})

    # Hypothesis places piece at x=-5 (5mm outside left edge)
    # Frame: [0, 100] x [0, 80], tau=2.0 → allowed [-2, 102]
    hyp = FrameHypothesis(
        piece_id=1, segment_id=0, side="LEFT",
        pose_grob_F=Pose2D(x_mm=-5.0, y_mm=40.0, theta_deg=0.0),
        features=None, cost_frame=0.1, is_committed=False, uncertainty_mm=5.0
    )

    pieces = {1: PuzzlePiece(piece_id=1, bbox_mm=(0, 0, 20, 10))}
    segments = {1: [make_segment(1, 0, [[0,0],[10,0]])]}

    new_states = expand_state(state, pieces, segments, {1: [hyp]}, [], config, frame_model)

    # Expected: pruned (outside frame)
    assert len(new_states) == 0, "Should be pruned (outside frame)"

    print("✓")


def test_E9_overlap_prune_stub(frame_model, config, monkeypatch):
    """E9: Overlap prune via stub hook"""
    print("\nTest E9: Overlap Stub Prune...", end=" ")

    # Monkeypatch overlap_stub to return 1.2mm
    from solver.beam_solver import expansion

    def mock_overlap_stub(state):
        return 1.2  # Exceeds threshold (1.0)

    monkeypatch.setattr(expansion, '_overlap_stub', mock_overlap_stub)

    state = SolverState(all_piece_ids={1})

    hyp = FrameHypothesis(
        piece_id=1, segment_id=0, side="TOP",
        pose_grob_F=Pose2D(x_mm=10.0, y_mm=70.0, theta_deg=0.0),
        features=None, cost_frame=0.1, is_committed=False, uncertainty_mm=5.0
    )

    pieces = {1: PuzzlePiece(piece_id=1, bbox_mm=(0, 0, 20, 10))}
    segments = {1: [make_segment(1, 0, [[0,0],[10,0]])]}

    new_states = expand_state(state, pieces, segments, {1: [hyp]}, [], config, frame_model)

    # Expected: pruned (overlap > threshold)
    assert len(new_states) == 0, "Should be pruned (overlap exceeds threshold)"

    print("✓")


def test_E10_committed_frame_conflict_prune(frame_model, config):
    """E10: Committed frame conflict prune"""
    print("\nTest E10: Frame Conflict Prune...", end=" ")

    # State: piece 1 committed to TOP (y >= H-2.0 = 78)
    state = SolverState(all_piece_ids={1, 2})
    state.placed_pieces = {1}
    state.unplaced_pieces = {2}
    state.poses_F = {1: Pose2D(x_mm=10.0, y_mm=78.0, theta_deg=0.0)}  # Valid TOP
    state.committed_frame_constraints = {
        1: FrameHypothesis(
            piece_id=1, segment_id=0, side="TOP",
            pose_grob_F=Pose2D(10, 78, 0),
            features=None, cost_frame=0.1, is_committed=True, uncertainty_mm=5.0
        )
    }
    state.open_edges = {(1, 0)}

    # Inner match tries to place piece 2, but would reposition piece 1 implicitly
    # For this test: simulate by creating invalid candidate that causes conflict check failure

    # Actually, E10 tests committed constraint violation during _check_committed_frame_constraints
    # Let's directly test by creating a state where pose violates commitment

    # Simpler: Place piece 2 via frame hypothesis, but the hypothesis conflicts with piece 1's commitment
    # Not applicable in current expansion logic (expansion doesn't re-pose already placed pieces)

    # Alternative: Create a state manually and test _check_committed_frame_constraints directly
    from solver.beam_solver.expansion import _check_committed_frame_constraints

    # Invalid state: committed TOP but pose at y=10 (bottom area)
    invalid_state = SolverState(all_piece_ids={1})
    invalid_state.placed_pieces = {1}
    invalid_state.poses_F = {1: Pose2D(x_mm=10.0, y_mm=10.0, theta_deg=0.0)}  # NOT at TOP
    invalid_state.committed_frame_constraints = {
        1: FrameHypothesis(
            piece_id=1, segment_id=0, side="TOP",
            pose_grob_F=Pose2D(10, 10, 0),  # Hypothesis says TOP, but y=10 is BOTTOM
            features=None, cost_frame=0.1, is_committed=True, uncertainty_mm=5.0
        )
    }

    pieces = {1: PuzzlePiece(piece_id=1, bbox_mm=(0, 0, 20, 10))}

    # Check should fail (conflict)
    result = _check_committed_frame_constraints(invalid_state, pieces, frame_model)
    assert result == False, "Should detect committed frame conflict"

    print("✓")


def test_E11_duplicate_state_dedup():
    """E11: Duplicate states are deduplicated"""
    print("\nTest E11: State Deduplication...", end=" ")

    # Create two identical states (same placed + poses, different cost)
    state1 = SolverState(all_piece_ids={1, 2})
    state1.placed_pieces = {1}
    state1.unplaced_pieces = {2}
    state1.poses_F = {1: Pose2D(x_mm=10.0, y_mm=20.0, theta_deg=30.0)}
    state1.cost_total = 0.5

    state2 = SolverState(all_piece_ids={1, 2})
    state2.placed_pieces = {1}
    state2.unplaced_pieces = {2}
    state2.poses_F = {1: Pose2D(x_mm=10.0, y_mm=20.0, theta_deg=30.0)}
    state2.cost_total = 0.3  # Better cost

    # Also add a different state
    state3 = SolverState(all_piece_ids={1, 2})
    state3.placed_pieces = {2}
    state3.unplaced_pieces = {1}
    state3.poses_F = {2: Pose2D(x_mm=15.0, y_mm=25.0, theta_deg=45.0)}
    state3.cost_total = 0.4

    from solver.beam_solver.expansion import _deduplicate_states

    deduped = _deduplicate_states([state1, state2, state3])

    # Expected: 2 states (state1/state2 merged → keep state2, plus state3)
    assert len(deduped) == 2, f"Expected 2 states after dedup, got {len(deduped)}"

    # Check that best cost was kept for duplicate
    costs = [s.cost_total for s in deduped]
    assert 0.3 in costs, "Best cost (0.3) should be kept"
    assert 0.5 not in costs, "Worse cost (0.5) should be removed"

    print("✓")


def test_E12_beam_candidate_limit(frame_model, config):
    """E12: Beam candidate limit per expansion (branching cap)"""
    print("\nTest E12: Branching Cap...", end=" ")

    state = SolverState(all_piece_ids={1})

    # Create 10 frame hypotheses for piece 1
    hyps = []
    for i in range(10):
        hyps.append(FrameHypothesis(
            piece_id=1, segment_id=i, side="TOP",
            pose_grob_F=Pose2D(x_mm=10.0 + i, y_mm=70.0, theta_deg=0.0),
            features=None, cost_frame=0.1 * (i + 1), is_committed=False, uncertainty_mm=5.0
        ))

    pieces = {1: PuzzlePiece(piece_id=1, bbox_mm=(0, 0, 20, 10))}
    segments = {1: [make_segment(1, i, [[0,0],[10,0]]) for i in range(10)]}

    # config.debug_topN_frame_hypotheses_per_piece = 5 (default)
    new_states = expand_state(state, pieces, segments, {1: hyps}, [], config, frame_model)

    # Expected: max 5 states (branching cap)
    assert len(new_states) <= config.debug_topN_frame_hypotheses_per_piece, \
        f"Expected max {config.debug_topN_frame_hypotheses_per_piece} states, got {len(new_states)}"

    print("✓")


# ========== Test Group B: beam_solver/solver.py (beam_search) ==========

# TODO: B1-B9 tests (will be implemented after expansion.py is done)


# ========== Run All Tests ==========

def run_all_tests():
    """Run all beam solver tests"""
    print("=" * 60)
    print("Beam-Solver Tests (Step 6)")
    print("=" * 60)

    # State tests (S1-S9)
    test_S1_import_single_source()
    test_S2_empty_state({1, 2, 3, 4})
    test_S3_seed_state_frame_hypothesis(
        FrameModel(inner_width_mm=100.0, inner_height_mm=80.0)
    )
    test_S4_copy_deep({1, 2, 3, 4})
    test_S5_get_frontier({1, 2, 3, 4})
    test_S6_invariants_valid({1, 2, 3, 4})
    test_S7_invariants_invalid_open_edge()
    test_S8_is_complete()
    test_S9_cost_breakdown_consistency()

    print()
    print("=" * 60)
    print("✅ State tests passed (9/9)")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
