"""
Solver State for Beam Search (Step 6).

This module defines SolverState - the core data structure for beam search.

IMPORTANT: SolverState is defined ONLY here (Single Source of Truth).
           DO NOT import or re-define in models.py.

Design Decisions (finalized):
- D1: Completion = all_placed AND open_edges==0 (strict)
- D2: open_edges = all non-committed segments

See docs/implementation/06_beam_solver_test_spec.md for specification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import copy as copy_module

# Import from models (but SolverState stays here!)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from solver.models import Pose2D, FrameHypothesis


@dataclass
class SolverState:
    """
    Beam search state for puzzle solver.

    Represents a partial or complete puzzle solution during beam search.

    Attributes:
        n_pieces: Total number of pieces in puzzle
        placed_pieces: Set of piece IDs that have been placed
        unplaced_pieces: Set of piece IDs not yet placed (Frontier part 1)
        poses_F: Dict mapping piece_id → Pose2D in Frame coordinates (mm)
        committed_edges: Set of EdgeID that have been matched (closed)
        open_edges: Set of EdgeID that are available for matching (Frontier part 2)
        cost_total: Total cost of this state (lower = better)
        cost_breakdown: Optional breakdown of cost components
        committed_frame_constraints: Dict mapping piece_id → FrameHypothesis
        matches: List of match records (edges that were matched)

    Invariants (validated by validate_invariants()):
        - I1: placed_pieces ⊆ all_piece_ids
        - I2: unplaced_pieces = all_piece_ids - placed_pieces
        - I3: set(poses_F.keys()) == placed_pieces
        - I4: open_edges only reference placed pieces
        - I5: committed_edges disjoint from open_edges
        - I6: cost_total >= 0.0

    EdgeID Format:
        EdgeID = tuple[PieceID, int] = (piece_id, segment_id)
        MUST be tuple (hashable for sets/dicts)

    Coordinate System:
        All poses in Frame coordinates (F), mm units

    Example:
        >>> state = SolverState(all_piece_ids={1, 2, 3})
        >>> state.placed_pieces == set()
        >>> state.unplaced_pieces == {1, 2, 3}
        >>> state.is_complete() == False
    """

    # Core state (required)
    n_pieces: int
    placed_pieces: set[int]  # PieceID
    unplaced_pieces: set[int]  # Frontier part 1
    poses_F: dict[int, Pose2D]  # piece_id → Pose in Frame coords (mm)
    committed_edges: set[tuple[int, int]]  # EdgeID = (piece_id, segment_id)
    open_edges: set[tuple[int, int]]  # Frontier part 2 (available for matching)

    # Cost tracking
    cost_total: float = 0.0
    cost_breakdown: dict[str, float] = field(default_factory=dict)

    # Constraints & history
    committed_frame_constraints: dict[int, FrameHypothesis] = field(default_factory=dict)
    matches: list = field(default_factory=list)  # Match records (optional structure)

    # Internal tracking (for all_piece_ids)
    _all_piece_ids: set[int] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self):
        """Initialize derived fields"""
        # Store all_piece_ids for validation
        self._all_piece_ids = self.placed_pieces | self.unplaced_pieces

        # Validate n_pieces
        if self.n_pieces != len(self._all_piece_ids):
            raise ValueError(
                f"n_pieces ({self.n_pieces}) != len(placed | unplaced) ({len(self._all_piece_ids)})"
            )

    def __init__(self, all_piece_ids: set[int], **kwargs):
        """
        Create SolverState.

        Args:
            all_piece_ids: Set of all piece IDs in puzzle
            **kwargs: Optional overrides for fields

        Example:
            >>> state = SolverState(all_piece_ids={1, 2, 3})  # Empty state
            >>> state.n_pieces == 3
            >>> state.placed_pieces == set()
        """
        # Initialize with empty defaults
        self.n_pieces = len(all_piece_ids)
        self.placed_pieces = kwargs.get('placed_pieces', set())
        self.unplaced_pieces = kwargs.get('unplaced_pieces', all_piece_ids - self.placed_pieces)
        self.poses_F = kwargs.get('poses_F', {})
        self.committed_edges = kwargs.get('committed_edges', set())
        self.open_edges = kwargs.get('open_edges', set())
        self.cost_total = kwargs.get('cost_total', 0.0)
        self.cost_breakdown = kwargs.get('cost_breakdown', {})
        self.committed_frame_constraints = kwargs.get('committed_frame_constraints', {})
        self.matches = kwargs.get('matches', [])

        # Store all_piece_ids
        self._all_piece_ids = all_piece_ids

    def copy(self) -> SolverState:
        """
        Create deep copy of state.

        Returns:
            New SolverState with deep-copied sets/dicts

        Notes:
            - Sets and dicts are deep-copied (mutations isolated)
            - Immutable values (int, float, Pose2D) are shared (safe)

        Example:
            >>> state2 = state.copy()
            >>> state2.placed_pieces.add(99)
            >>> 99 not in state.placed_pieces  # Original unchanged
        """
        return SolverState(
            all_piece_ids=self._all_piece_ids.copy(),
            placed_pieces=self.placed_pieces.copy(),
            unplaced_pieces=self.unplaced_pieces.copy(),
            poses_F=self.poses_F.copy(),
            committed_edges=self.committed_edges.copy(),
            open_edges=self.open_edges.copy(),
            cost_total=self.cost_total,
            cost_breakdown=self.cost_breakdown.copy(),
            committed_frame_constraints=self.committed_frame_constraints.copy(),
            matches=self.matches.copy()
        )

    def is_complete(self) -> bool:
        """
        Check if puzzle is complete.

        Returns:
            True if all pieces placed AND no open edges

        Design Decision D1 (strict):
            Complete = (len(placed_pieces) == n_pieces) AND (len(open_edges) == 0)

        Notes:
            - D1 ensures "wasserdicht" completion (all edges matched)
            - If open_edges exist → still has unmatched edges → incomplete

        Example:
            >>> state.placed_pieces = {1, 2, 3}  # All placed
            >>> state.open_edges = set()  # No open edges
            >>> state.is_complete() == True
        """
        return (len(self.placed_pieces) == self.n_pieces and
                len(self.open_edges) == 0)

    def get_frontier(self) -> tuple[set[int], set[tuple[int, int]]]:
        """
        Get frontier for expansion.

        Returns:
            Tuple (unplaced_pieces, open_edges)

        Notes:
            - Two separate sets (NOT Union type)
            - unplaced_pieces: Pieces available for frame placement
            - open_edges: Edges available for inner matching

        Example:
            >>> unplaced, open_edges = state.get_frontier()
            >>> isinstance(unplaced, set)  # True
            >>> isinstance(open_edges, set)  # True
        """
        return (self.unplaced_pieces, self.open_edges)

    def validate_invariants(self):
        """
        Validate state invariants.

        Raises:
            ValueError: If any invariant is violated

        Invariants:
            - I1: placed_pieces ⊆ all_piece_ids
            - I2: unplaced_pieces = all_piece_ids - placed_pieces
            - I3: set(poses_F.keys()) == placed_pieces
            - I4: open_edges only reference placed pieces
            - I5: committed_edges disjoint from open_edges
            - I6: cost_total >= 0.0

        Notes:
            - Called automatically in expansion (optional)
            - Helps catch bugs during development

        Example:
            >>> state.validate_invariants()  # No error → valid state
        """
        # I1: placed ⊆ all
        if not self.placed_pieces.issubset(self._all_piece_ids):
            raise ValueError(
                f"I1 violated: placed_pieces {self.placed_pieces} not subset of {self._all_piece_ids}"
            )

        # I2: unplaced = all - placed
        expected_unplaced = self._all_piece_ids - self.placed_pieces
        if self.unplaced_pieces != expected_unplaced:
            raise ValueError(
                f"I2 violated: unplaced_pieces {self.unplaced_pieces} != expected {expected_unplaced}"
            )

        # I3: poses_F.keys() == placed
        if set(self.poses_F.keys()) != self.placed_pieces:
            raise ValueError(
                f"I3 violated: poses_F.keys() {set(self.poses_F.keys())} != placed {self.placed_pieces}"
            )

        # I4: open_edges reference placed pieces only
        for edge_id in self.open_edges:
            piece_id, seg_id = edge_id
            if piece_id not in self.placed_pieces:
                raise ValueError(
                    f"I4 violated: open_edges must reference placed pieces. "
                    f"Edge {edge_id} references unplaced piece {piece_id}"
                )

        # I5: committed_edges disjoint from open_edges
        overlap = self.committed_edges & self.open_edges
        if overlap:
            raise ValueError(
                f"I5 violated: committed_edges and open_edges must be disjoint. Overlap: {overlap}"
            )

        # I6: cost_total >= 0
        if self.cost_total < 0.0:
            raise ValueError(
                f"I6 violated: cost_total ({self.cost_total}) must be >= 0.0"
            )

    @classmethod
    def seed_with_frame_hypothesis(
        cls,
        all_piece_ids: set[int],
        hyp: FrameHypothesis
    ) -> SolverState:
        """
        Create seed state from frame hypothesis.

        Args:
            all_piece_ids: Set of all piece IDs
            hyp: FrameHypothesis to seed with

        Returns:
            New SolverState with piece placed via frame hypothesis

        Notes:
            - Places piece at hyp.pose_grob_F
            - Commits frame hypothesis (stores in committed_frame_constraints)
            - Sets cost_total = hyp.cost_frame
            - open_edges initialized per D2 (currently empty, segments unknown)

        Example:
            >>> hyp = FrameHypothesis(piece_id=1, ..., cost_frame=0.4)
            >>> state = SolverState.seed_with_frame_hypothesis({1, 2}, hyp)
            >>> state.placed_pieces == {1}
            >>> state.cost_total == 0.4
        """
        state = cls(all_piece_ids=all_piece_ids)

        # Place piece
        piece_id = hyp.piece_id
        state.placed_pieces.add(piece_id)
        state.unplaced_pieces.discard(piece_id)
        state.poses_F[piece_id] = hyp.pose_grob_F

        # Commit frame hypothesis
        state.committed_frame_constraints[piece_id] = hyp
        state.committed_edges.add((piece_id, hyp.segment_id))

        # Set cost
        state.cost_total = hyp.cost_frame
        state.cost_breakdown['frame'] = hyp.cost_frame

        # open_edges: D2 policy (all non-committed segments)
        # For seed state, we don't have segment info yet → empty
        # Will be populated by expansion.py when placing subsequent pieces
        state.open_edges = set()

        return state
