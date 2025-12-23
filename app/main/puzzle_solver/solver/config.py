"""
Solver V2 Configuration Models.

This module defines the configuration structures for the puzzle solver:
- Transform2D: 2D transformations (translation + rotation)
- FrameModel: A5 frame geometry
- MatchingConfig: All solver parameters (7 groups)

All measurements in millimeters unless stated otherwise.
Angles in degrees, range [-180, 180), counterclockwise positive.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Optional


@dataclass
class Transform2D:
    """
    2D transformation: translation + rotation.

    Attributes:
        x_mm: Translation in x (mm)
        y_mm: Translation in y (mm)
        theta_deg: Rotation angle (degrees, [-180, 180), ccw positive)

    Notes:
        - Used for coordinate system transformations (Frame ↔ Machine)
        - Rotation around origin, then translation
    """
    x_mm: float = 0.0
    y_mm: float = 0.0
    theta_deg: float = 0.0

    def to_matrix(self) -> np.ndarray:
        """
        Convert to 3x3 homogeneous transformation matrix.

        Returns:
            3x3 numpy array: [[cos -sin tx]
                              [sin  cos ty]
                              [0    0   1 ]]

        Notes:
            TODO: Implement in Step 2 (Einheiten & KS)
        """
        raise NotImplementedError("Transform2D.to_matrix() - Step 2")

    @classmethod
    def from_matrix(cls, mat: np.ndarray) -> Transform2D:
        """
        Create Transform2D from 3x3 homogeneous matrix.

        Args:
            mat: 3x3 homogeneous transformation matrix

        Returns:
            Transform2D instance

        Notes:
            TODO: Implement in Step 2
        """
        raise NotImplementedError("Transform2D.from_matrix() - Step 2")

    def compose(self, other: Transform2D) -> Transform2D:
        """
        Compose this transform with another: result = self ∘ other.

        Args:
            other: Transform to compose with

        Returns:
            Composed transform

        Notes:
            TODO: Implement in Step 2
        """
        raise NotImplementedError("Transform2D.compose() - Step 2")

    def inverse(self) -> Transform2D:
        """
        Compute inverse transform.

        Returns:
            Inverse transform

        Notes:
            TODO: Implement in Step 2
        """
        raise NotImplementedError("Transform2D.inverse() - Step 2")

    def apply(self, points: np.ndarray) -> np.ndarray:
        """
        Apply transform to points.

        Args:
            points: (N, 2) array of points in mm

        Returns:
            (N, 2) transformed points in mm

        Notes:
            TODO: Implement in Step 2
        """
        raise NotImplementedError("Transform2D.apply() - Step 2")


@dataclass
class FrameModel:
    """
    A5 puzzle frame geometry.

    Attributes:
        inner_width_mm: Inner width (fixed: 128mm)
        inner_height_mm: Inner height (fixed: 190mm)
        corner_radius_mm: Corner radius of inner contour (bewusst offen)
        T_MF: Transform from Frame (F) to Machine (M) coordinate system (bewusst offen)

    Notes:
        - Frame coordinate system (F): origin at lower-left inner corner
        - Machine coordinate system (M): mechanically defined (not yet built)
        - Corner radius and T_MF are placeholders until physically determined
        - See docs/design/01_coordinates.md for coordinate system definitions
    """
    inner_width_mm: float = 128.0
    inner_height_mm: float = 190.0
    corner_radius_mm: Optional[float] = None  # bewusst offen (not yet defined)
    T_MF: Optional[Transform2D] = None  # bewusst offen (machine not built)


@dataclass
class MatchingConfig:
    """
    Complete solver configuration (all parameters).

    Organized in 7 groups:
    1. Frame-first: Frame contact parameters
    2. Segmentation: Contour segmentation parameters
    3. Profile: 1D profile extraction parameters
    4. Inner matching: Inner edge matching parameters
    5. Solver: Beam search parameters
    6. Overlap: Collision detection parameters
    7. Confidence/Fallback: Confidence mapping and fallback parameters
    8. Debug: Debug output parameters

    Notes:
        - All measurements in mm unless stated otherwise
        - Parameters marked with "TODO: Tuning" are initial estimates
        - See docs/design/03_matching.md for parameter semantics
    """

    # ========== 1. Frame-first ==========
    frame_band_mm: float = 1.0
    """Bandwidth around frame edge for contact band (mm). TODO: Tuning"""

    frame_angle_deg: float = 10.0
    """Angular tolerance for segment-to-frame alignment (degrees). TODO: Tuning"""

    min_frame_seg_len_mm: float = 10.0
    """Minimum segment length to be considered for frame contact (mm)"""

    tau_frame_mm: float = 2.0
    """Tolerance band for inside/outside frame checks (mm, for robustness against noise)"""

    frame_weights: dict[str, float] = field(default_factory=lambda: {
        "dist_p90": 0.3,
        "coverage": 0.3,
        "angle_diff": 0.2,
        "flatness": 0.2
    })
    """
    Weights for frame cost aggregation.
    Keys: dist_p90, coverage, angle_diff, flatness
    TODO: Tuning
    """

    penalty_missing_frame_contact: float = 10.0
    """Penalty if a piece has no acceptable frame hypothesis. TODO: Tuning"""

    # ========== 2. Segmentation ==========
    target_seg_count_range: tuple[int, int] = (4, 12)
    """Target range for number of segments per piece"""

    # ========== 3. Profile ==========
    profile_samples_N: int = 128
    """Number of samples for 1D profile resampling. TODO: Tuning"""

    profile_smoothing_window: int = 3
    """Smoothing window for profile (3/5/adaptive). TODO: Tuning"""

    # ========== 4. Inner matching ==========
    topk_per_segment: int = 10
    """Number of top candidates to keep per segment"""

    enable_icp: bool = False
    """Enable ICP fit cost (computationally expensive). TODO: Evaluate performance vs quality"""

    inner_weights: dict[str, float] = field(default_factory=lambda: {
        "profile": 0.6,
        "length": 0.2,
        "fit": 0.2
    })
    """
    Weights for inner match cost aggregation.
    Keys: profile, length, fit
    TODO: Tuning
    """

    # ========== 5. Solver ==========
    beam_width: int = 20
    """Maximum number of hypotheses per beam search iteration. TODO: Tuning"""

    max_expansions: int = 1000
    """Maximum number of state expansions (safety limit)"""

    # ========== 6. Overlap ==========
    overlap_depth_max_mm_prune: float = 1.0
    """Maximum penetration depth for pruning during search (mm). Annahme, TODO: Tuning"""

    overlap_depth_max_mm_final: float = 0.1
    """Maximum penetration depth for final solution (mm). Annahme, TODO: Tuning"""

    polygon_nonconvex_strategy: str = "triangulation"
    """
    Strategy for non-convex polygon overlap (Option B: triangulation).
    Performance critical! See docs/implementation/00_structure.md §3 Step 7.
    """

    nonconvex_aggregation: str = "max"
    """
    Aggregation method for triangle-pair penetration depths.
    - "max": recommended for pruning (conservative)
    - "mean"/"p90": only for diagnosis/ranking experiments
    TODO: Tuning
    """

    # ========== 7. Confidence/Fallback ==========
    k_conf: float = 1.0
    """
    Confidence mapping parameter: conf = exp(-k_conf * cost_total).
    TODO: Tuning (data-driven after initial runs)
    """

    fallback_conf_threshold: float = 0.5
    """Confidence threshold for triggering many-to-one fallback"""

    enable_many_to_one_fallback: bool = True
    """Enable many-to-one matching fallback"""

    many_to_one_max_chain_len: int = 2
    """Maximum chain length for composite segments (2 or 3)"""

    # ========== 8. Debug ==========
    debug_topN_frame_hypotheses_per_piece: int = 5
    """Number of top frame hypotheses to include in debug output per piece"""

    debug_topN_inner_candidates_per_segment: int = 5
    """Number of top inner match candidates to include in debug output per segment"""

    export_debug_json: bool = True
    """Export debug bundle as JSON after each run"""
