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
from typing import Optional, Literal


@dataclass
class Transform2D:
    """
    2D transformation: translation + rotation.

    Attributes:
        x_mm: Translation in x (mm)
        y_mm: Translation in y (mm)
        theta_deg: Rotation angle in degrees (CCW, counter-clockwise), range [-180, 180)

    Rotation convention:
        - CCW (positive theta = counter-clockwise)
        - Example: theta=90° rotates point (1,0) to (0,1)
        - Consistent with standard rotation matrix

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
            3x3 numpy array: [[cos(θ) -sin(θ) tx]
                              [sin(θ)  cos(θ) ty]
                              [0       0      1 ]]

        Notes:
            - Rotation around origin, then translation
            - Angle in degrees, converted to radians internally
            - Standard 2D transformation matrix (rotation + translation)
        """
        theta_rad = np.deg2rad(self.theta_deg)
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)

        return np.array([
            [c, -s, self.x_mm],
            [s,  c, self.y_mm],
            [0,  0, 1]
        ], dtype=float)

    @classmethod
    def from_matrix(cls, mat: np.ndarray) -> Transform2D:
        """
        Create Transform2D from 3x3 homogeneous matrix.

        Args:
            mat: 3x3 homogeneous transformation matrix

        Returns:
            Transform2D instance

        Notes:
            - Extracts translation from mat[:2, 2]
            - Extracts rotation via arctan2(mat[1,0], mat[0,0])
            - Angle normalized to [-180, 180) degrees
        """
        x_mm = float(mat[0, 2])
        y_mm = float(mat[1, 2])
        theta_rad = np.arctan2(mat[1, 0], mat[0, 0])
        theta_deg = np.rad2deg(theta_rad)
        # Normalize to [-180, 180)
        theta_deg = (theta_deg + 180) % 360 - 180
        return cls(x_mm, y_mm, theta_deg)

    def compose(self, other: Transform2D) -> Transform2D:
        """
        Compose this transform with another: result = self ∘ other.

        Args:
            other: Transform to compose with

        Returns:
            Composed transform

        Notes:
            - Matrix multiplication: mat_result = mat_self @ mat_other
            - Order matters: self applied first, then other
        """
        mat_self = self.to_matrix()
        mat_other = other.to_matrix()
        mat_result = mat_self @ mat_other
        return Transform2D.from_matrix(mat_result)

    def inverse(self) -> Transform2D:
        """
        Compute inverse transform.

        Returns:
            Inverse transform

        Notes:
            - T^-1: Reverses the transformation
            - If T transforms points from A to B, T^-1 transforms from B to A
            - Computed via matrix inversion
        """
        mat = self.to_matrix()
        mat_inv = np.linalg.inv(mat)
        return Transform2D.from_matrix(mat_inv)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """
        Apply transform to points.

        Args:
            points: (N, 2) array of points in mm

        Returns:
            (N, 2) transformed points in mm

        Notes:
            - Converts to homogeneous coordinates (N, 3)
            - Applies transformation matrix
            - Returns Cartesian coordinates (N, 2)
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Expected (N, 2) array, got shape {points.shape}")

        # Convert to homogeneous coordinates (N, 3)
        N = points.shape[0]
        points_h = np.hstack([points, np.ones((N, 1))])

        # Apply transformation
        mat = self.to_matrix()
        transformed_h = (mat @ points_h.T).T

        # Return Cartesian coordinates
        return transformed_h[:, :2]


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

    Coordinate Systems:
        - Frame (F): Origin at lower-left inner corner, x-axis right, y-axis up
        - Machine (M): Physical machine coordinate system (camera/gripper reference)
        - Transform T_MF: Maps points from Frame to Machine: p_M = T_MF.apply(p_F)

    Placeholder Values:
        - T_MF = None: Use when working purely in Frame coordinates (solver internals)
        - T_MF can be set for simulator visualization (outside A5 frame area)
        - T_MF default for visualization: Transform2D(200, 200, 0)
          → Places Frame lower-left corner at (200mm, 200mm) outside origin
          → Use utils.conversion.get_default_T_MF() for this default
        - T_MF will be calibrated when physical machine is built
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
    """Tolerance band for inside/outside frame checks (mm, for robustness against noise).

    Reserved for step 6 (frame validation).
    Not used in step 4 (frame feature computation)."""

    frame_weights: dict[str, float] = field(default_factory=lambda: {
        "dist_p90": 0.3,
        "coverage": 0.3,
        "angle_diff": 0.2,
        "flatness": 0.2
    })
    """
    Weights for frame cost aggregation.
    Keys: dist_p90, coverage, angle_diff, flatness
    Note: coverage/inlier weights overridden by frame_coverage_vs_inlier_policy
    TODO: Tuning
    """

    frame_flat_ref_mm: float = 1.0
    """Referenz zur Normierung von flatness_error_mm im Frame-Cost.
       Unabhängig von frame_band_mm (Kontakt-Toleranz).
       TODO: Tuning"""

    frame_coverage_vs_inlier_policy: Literal["coverage", "inlier", "balanced"] = "coverage"
    """Welche Metrik gewichtet wird.
       - 'coverage': Nur coverage_in_band, inlier_ratio ignoriert
       - 'inlier': Nur inlier_ratio, coverage_in_band ignoriert
       - 'balanced': Beide gleichgewichtet
       TODO: Tuning"""

    pose_grob_theta_mode: Literal["zero", "side_aligned", "segment_aligned"] = "side_aligned"
    """Wie initiale theta geschätzt wird.
       - 'zero': theta=0 (keine Rotation)
       - 'side_aligned': theta basierend auf Frame-Side (0° TOP/BOTTOM, 90° LEFT/RIGHT)
       - 'segment_aligned': theta=seg.direction_angle (nur als Heuristik)
       TODO: Tuning"""

    frame_distance_mode: Literal["abs", "signed"] = "abs"
    """Wie dist_* in FrameFeatures berechnet wird.
       - 'abs': Absolute Distanz
       - 'signed': Signed (positiv nach innen)"""

    use_pose_uncertainty_in_solver: bool = False
    """Wenn True: uncertainty_mm beeinflusst Seed-Ranking/Pruning.
       Default False (uncertainty nur für Debug)."""

    pose_uncertainty_penalty_weight: float = 0.0
    """Weight für uncertainty in Ranking (wenn use_pose_uncertainty_in_solver=True)."""

    penalty_missing_frame_contact: float = 10.0
    """Penalty for pieces without frame contact (reserved for step 6).

    Not used in step 4 (hypothesis generation).
    TODO: Tuning"""

    # ========== 2. Segmentation ==========
    target_seg_count_range: tuple[int, int] = (4, 12)
    """Target range for number of segments per piece"""

    curvature_angle_threshold_deg: float = 30.0
    """Split-Kriterium: lokaler Winkelwechsel > threshold. TODO: Tuning"""

    curvature_window_pts: int = 5
    """Fenster für Tangentenwinkel (ungerade Zahl). TODO: Tuning"""

    enable_wraparound_merge: bool = True
    """Erlaubt Merge zwischen letztem und erstem Segment (zyklisch)."""

    wraparound_merge_dist_mm: float = 1.0
    """Max Distanz zwischen contour[0] und contour[-1] für 'geschlossen'."""

    # ========== 3. Profile ==========
    profile_samples_N: int = 128
    """Number of samples for 1D profile resampling. TODO: Tuning"""

    profile_smoothing_window: int = 3
    """Window size for profile smoothing.

    UNUSED IN V1: Smoothing disabled (raw resampled profile returned).
    Reserved for noise robustness (moving average/Gaussian).
    TODO: Implement smoothing for noisy contours"""

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
    """Weights for inner cost aggregation.
    Keys: profile, length, fit

    ASSUMPTION: Weights should sum to 1.0 for cost_inner ∈ [0,1].
    Default: {profile: 0.6, length: 0.2, fit: 0.2} → sum = 1.0 ✓

    If custom weights don't sum to 1.0:
    - cost_inner range becomes [0, sum(weights)]
    - Beam-Solver ranking still works (relative costs)
    - Runtime warning logged if |sum - 1.0| > 0.01

    TODO: Tuning"""

    length_tolerance_ratio: float = 0.15
    """Length mismatch tolerance for prefiltering (15% strict). TODO: Tuning"""

    flatness_tolerance_mm: float = 2.0
    """Flatness mismatch tolerance for prefiltering (mm). TODO: Tuning"""

    frame_likelihood_threshold: float = 0.5
    """Threshold for frame-likelihood prefilter.

    UNUSED IN V1: Prefilter not implemented (reserved for step 6/future).
    Purpose: Prefer inner edges over frame edges in candidate generation.
    TODO: Implement in generate_candidates() as Filter 4"""

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
    Limits: Max 50 triangles per piece (guard against explosion).
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

    penalty_composite_used: float = 5.0
    """
    Penalty for using composite segments in many-to-one matching.
    Small penalty to avoid unnecessary many-to-one (prefer direct 1:1).
    TODO: Tuning
    """

    # ========== 8. Debug ==========
    debug_topN_frame_hypotheses_per_piece: int = 5
    """Number of top frame hypotheses per piece.

    IMPORTANT: Controls BOTH:
    - Algorithm: Beam-solver seeds (step 6)
    - Debug: Logged hypotheses

    Reducing this value affects solver quality.
    Default 5 balances quality vs performance.
    TODO: Tune based on puzzle complexity."""

    debug_topN_inner_candidates_per_segment: int = 5
    """Number of top inner match candidates to include in debug output per segment"""

    export_debug_json: bool = True
    """Export debug bundle as JSON after each run"""
