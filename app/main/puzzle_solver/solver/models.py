"""
Solver V2 Data Models.

This module defines all data structures used in the puzzle solver:
- Pose2D: 2D pose (position + orientation)
- ContourSegment: Segmented contour piece
- FrameContactFeatures: Frame contact metrics
- FrameHypothesis: Frame-first placement hypothesis
- InnerMatchCandidate: Inner edge match candidate
- SolutionStatus: Solution status enum
- PuzzleSolution: Final puzzle solution

All measurements in millimeters unless stated otherwise.
Angles in degrees, range [-180, 180), counterclockwise positive.

NOTE: SolverState is NOT defined here. It is defined in beam_solver/state.py
      (Single Source of Truth, see docs/implementation/00_structure.md §2)
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any
import numpy as np


@dataclass
class Pose2D:
    """
    2D pose: position + orientation.

    Attributes:
        x_mm: X position in mm
        y_mm: Y position in mm
        theta_deg: Orientation angle in degrees [-180, 180), ccw positive

    Notes:
        - Used for piece poses in coordinate systems (Frame F or Machine M)
        - Coordinate system must be tagged in context (see FrameHypothesis, PuzzleSolution)
    """
    x_mm: float
    y_mm: float
    theta_deg: float


@dataclass
class PuzzlePiece:
    """
    Puzzle piece supporting both extraction (px) and solver (mm) coordinates.

    Workflow:
        1. Extraction: Creates with contour/bbox/center (px)
        2. Conversion: convert_pieces_px_to_mm() fills contour_mm/bbox_mm/center_mm
        3. Solver: Uses contour_mm/bbox_mm/center_mm

    Attributes:
        piece_id: Unique piece identifier
        contour: Contour points in px (from extraction), shape (N, 2)
        bbox: Bounding box in px (x_min, y_min, x_max, y_max)
        center: Center of mass in px, shape (2,)
        contour_mm: Contour points in mm (filled by conversion), shape (N, 2), KS M
        bbox_mm: Bounding box in mm (filled by conversion), KS M
        center_mm: Center of mass in mm (filled by conversion), shape (2,), KS M
        mask: Binary mask, shape (H, W)
        image: Optional texture image (RGBA)

    Notes:
        - Input from piece_extraction module with px coordinates
        - convert_pieces_px_to_mm() converts px → mm
        - Solver uses only _mm fields
        - Coordinate system M (Machine) or intermediate extraction system
        - Converted to Frame system (F) internally by solver
        - See docs/implementation/00_structure.md §1.1 for integration
    """
    piece_id: int | str

    # Extraction output (px) - optional:
    contour: Optional[np.ndarray] = None  # (N, 2) in px
    bbox: Optional[tuple[float, float, float, float]] = None  # px
    center: Optional[np.ndarray] = None  # (2,) in px

    # Solver input (mm) - filled by convert_pieces_px_to_mm():
    contour_mm: Optional[np.ndarray] = None  # (N, 2) in mm, KS M
    bbox_mm: Optional[tuple[float, float, float, float]] = None  # mm, KS M
    center_mm: Optional[np.ndarray] = None  # (2,) in mm, KS M

    # Metadata:
    mask: Optional[np.ndarray] = None  # (H, W) binary
    image: Optional[np.ndarray] = None  # Texture (optional)


@dataclass
class ContourSegment:
    """
    Segmented piece of a contour.

    Attributes:
        piece_id: Unique piece identifier (int or str)
        segment_id: Segment index within piece contour
        points_mm: Segment points in mm, shape (M, 2)
        length_mm: Arclength of segment in mm
        chord: Chord endpoints (start, end), each shape (2,) in mm
        direction_angle_deg: Chord direction angle in degrees [-180, 180)
        flatness_error: RMS point-to-chord distance in mm
        profile_1d: Optional 1D profile (lazy-loaded), shape (N,)

    Notes:
        - All coordinates in mm
        - Coordinate system depends on context (typically Frame F)
        - flatness_error: lower = straighter segment
        - profile_1d computed on demand in Step 5 (Inner Matching)
        - Segment orientation: Normalized via chord direction (end - start)
        - Profile orientation: Consistent with chord direction (perpendicular right-hand)
        - reversal_used in InnerMatchCandidate indicates profile was flipped for matching
    """
    piece_id: int | str
    segment_id: int
    points_mm: np.ndarray  # (M, 2)
    length_mm: float
    chord: tuple[np.ndarray, np.ndarray]  # (start_pt, end_pt), each (2,) in mm
    direction_angle_deg: float
    flatness_error: float
    profile_1d: Optional[np.ndarray] = None  # (N,), lazy


@dataclass
class FrameContactFeatures:
    """
    Raw metrics for frame contact hypothesis.

    Attributes:
        dist_mean_mm: Mean distance segment points to frame line (mm)
        dist_p90_mm: 90th percentile distance (mm, robust against outliers)
        dist_max_mm: Maximum distance (mm)
        coverage_in_band: Fraction of segment length within tolerance band [0..1]
        inlier_ratio: Fraction of points within tolerance band [0..1]
        angle_diff_deg: Absolute angle difference segment to frame edge (degrees)
        flatness_error_mm: Segment flatness (RMS point-to-chord distance in mm)

    Notes:
        - These are RAW metrics, NOT costs
        - Costs computed via MatchingConfig.frame_weights in Step 4
        - See docs/design/04_scoring.md §A for metric definitions
    """
    dist_mean_mm: float
    dist_p90_mm: float
    dist_max_mm: float
    coverage_in_band: float  # [0..1]
    inlier_ratio: float  # [0..1]
    angle_diff_deg: float
    flatness_error_mm: float


@dataclass
class FrameHypothesis:
    """
    Frame-first placement hypothesis for a piece.

    Attributes:
        piece_id: Piece identifier
        segment_id: Segment index on piece
        side: Frame side ("TOP" | "BOTTOM" | "LEFT" | "RIGHT")
        pose_grob_F: Initial pose estimate in Frame coordinate system (F)
        features: Raw frame contact metrics
        cost_frame: Aggregated frame contact cost
        is_committed: Whether this hypothesis is committed in solver state
        uncertainty_mm: Pose uncertainty estimate in mm (default 5.0mm, pessimistic)

    Notes:
        - pose_grob_F: Initial estimate from estimate_pose_grob_F()
        - Coordinate system: Frame (F), origin at lower-left inner corner
        - cost_frame = Σ w_k * cost_k(features_k) via MatchingConfig.frame_weights
        - is_committed: False initially, set True in solver when used
        - uncertainty_mm: Estimated pose uncertainty (used if use_pose_uncertainty_in_solver=True)
    """
    piece_id: int | str
    segment_id: int
    side: str  # "TOP" | "BOTTOM" | "LEFT" | "RIGHT"
    pose_grob_F: Pose2D
    features: FrameContactFeatures
    cost_frame: float
    is_committed: bool = False
    uncertainty_mm: float = 5.0


@dataclass
class InnerMatchCandidate:
    """
    Inner edge match candidate between two segments.

    Attributes:
        seg_a_ref: Reference to segment A as (piece_id, segment_id)
        seg_b_ref: Reference to segment B as (piece_id, segment_id)
        cost_inner: Aggregated inner match cost
        profile_cost: 1D profile similarity cost (1 - |NCC|)
        length_cost: Length compatibility cost
        fit_cost: Geometric fit cost (ICP if enabled, else 0)
        reversal_used: Whether segment B was reversed for matching
        sign_flip_used: Whether segment B profile was sign-flipped (opposite side of chord)

    Notes:
        - cost_inner = w_profile * profile_cost + w_length * length_cost + w_fit * fit_cost
        - Weights from MatchingConfig.inner_weights
        - reversal_used: True if profile_b was reversed for better NCC
        - sign_flip_used: True if profile_b was negated (opposite-side orientation)
        - See docs/design/03_matching.md Phase 3 for matching algorithm
    """
    seg_a_ref: tuple[int | str, int]  # (piece_id, segment_id)
    seg_b_ref: tuple[int | str, int]
    cost_inner: float
    profile_cost: float
    length_cost: float
    fit_cost: float
    reversal_used: bool
    sign_flip_used: bool = False


class SolutionStatus(Enum):
    """
    Status of puzzle solution.

    Values:
        OK: Solution found, all constraints satisfied
        OK_WITH_FALLBACK: Solution found after many-to-one fallback
        LOW_CONFIDENCE: Solution found but confidence below threshold
        NO_SOLUTION: No complete solution found (beam exhausted or pruned)
        REFINEMENT_FAILED: Solution found but refinement failed (overlap > threshold)
        INVALID_INPUT: Invalid input (e.g., n not in {4,5,6})

    Notes:
        - See docs/design/09_edgecases.md §C for status definitions
    """
    OK = "OK"
    OK_WITH_FALLBACK = "OK_WITH_FALLBACK"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NO_SOLUTION = "NO_SOLUTION"
    REFINEMENT_FAILED = "REFINEMENT_FAILED"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass
class PuzzleSolution:
    """
    Final puzzle solution.

    Attributes:
        poses_F: Piece poses in Frame coordinate system (F), dict[piece_id -> Pose2D]
        poses_M: Optional piece poses in Machine coordinate system (M), only if T_MF set
        matches: List of match constraints used in solution (type TBD in later steps)
        total_cost: Total solution cost (sum of frame, inner, penalties, overlap)
        confidence: Solution confidence = exp(-k_conf * total_cost)
        status: Solution status (see SolutionStatus enum)
        overlap_violations: Number of piece pairs with overlap > threshold (for A/B testing)
        debug: Optional debug bundle (type TBD in Step 9)

    Notes:
        - Primary output: poses_F (Frame coordinate system)
        - poses_M computed if FrameModel.T_MF is set
        - Coordinate system Frame (F): origin at lower-left inner corner of frame
        - See docs/design/02_datamodels.md §PuzzleSolution for details
    """
    poses_F: dict[int | str, Pose2D]
    poses_M: Optional[dict[int | str, Pose2D]] = None
    matches: list = None  # Type TBD in later steps (MatchConstraint)
    total_cost: float = 0.0
    confidence: float = 0.0
    status: SolutionStatus = SolutionStatus.NO_SOLUTION
    overlap_violations: int = 0  # For A/B test comparison
    debug: Optional[Any] = None  # DebugBundle type TBD in Step 9
