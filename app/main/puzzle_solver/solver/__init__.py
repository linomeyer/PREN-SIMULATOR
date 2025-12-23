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
from .config import MatchingConfig, FrameModel, Transform2D
from .models import (
    Pose2D,
    ContourSegment,
    FrameContactFeatures,
    FrameHypothesis,
    InnerMatchCandidate,
    SolutionStatus,
    PuzzleSolution,
)


__all__ = [
    # Main API
    "solve_puzzle",
    # Config
    "MatchingConfig",
    "FrameModel",
    "Transform2D",
    # Models
    "Pose2D",
    "ContourSegment",
    "FrameContactFeatures",
    "FrameHypothesis",
    "InnerMatchCandidate",
    "SolutionStatus",
    "PuzzleSolution",
]


def solve_puzzle(pieces: List, frame: FrameModel, config: MatchingConfig) -> PuzzleSolution:
    """
    Solve puzzle using multi-hypothesis beam search.

    This is the main entry point for the solver. It takes extracted puzzle pieces
    and attempts to find a consistent solution within the given frame.

    Args:
        pieces: List of PuzzlePiece objects from piece_extraction module
                Each piece must have:
                - contour: np.ndarray[(N,2)] contour points
                - Additional fields (mask, bbox, image) optional
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
    raise NotImplementedError(
        "Solver V2 implementation in progress.\n"
        "Current status: Step 1 complete (Config + Models)\n"
        "See docs/implementation/00_structure.md for roadmap (Steps 2-9)\n"
        "See docs/design/CLAUDE.md for current implementation status"
    )
