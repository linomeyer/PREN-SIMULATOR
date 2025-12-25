"""
Unit Conversion and Coordinate System Utilities.

This module provides conversion functions between pixel and millimeter units,
and between coordinate systems.

All functions assume:
- Input in pixels (px) from image/camera coordinate system
- Output in millimeters (mm) in Machine coordinate system (M)

⚠ ASSUMPTIONS:
- Isotropic scaling (sx = sy): Same scale factor for x and y axes
- Axis-aligned pixel coordinate system (no rotation)
- Anisotropic or rotated pixel-KS not supported in V1

See docs/design/01_coordinates.md for coordinate system definitions.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import PuzzlePiece
    from ..config import Transform2D


class PlaceholderScaleError(RuntimeError):
    """
    Raised when placeholder scale is used in production code path.

    This exception prevents accidental use of the uncalibrated placeholder
    scale (0.1 mm/px) in production solver code where it would produce
    systematically wrong results for scoring and pruning.

    See get_default_scale_simulator() for usage.
    """
    pass


def convert_points_px_to_mm(
    points_px: np.ndarray,
    scale_px_to_mm: float,
    origin_offset_mm: Optional[tuple[float, float]] = None
) -> np.ndarray:
    """
    Convert points from pixels to millimeters.

    Args:
        points_px: Points in pixels, shape (N, 2)
        scale_px_to_mm: Scale factor (mm per pixel)
        origin_offset_mm: Optional offset to apply after scaling (x_mm, y_mm)

    Returns:
        Points in mm, shape (N, 2)

    Notes:
        - Formula: points_mm = points_px * scale + offset
        - Coordinate system: Input assumed in image/camera, output in Machine (M)
        - No rotation applied (assumes axes aligned)

    Example:
        >>> points_px = np.array([[100, 200], [150, 250]])
        >>> scale = 0.5  # 0.5mm per pixel
        >>> points_mm = convert_points_px_to_mm(points_px, scale)
        >>> # Result: [[50, 100], [75, 125]]
    """
    if points_px.ndim != 2 or points_px.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) array, got shape {points_px.shape}")

    points_mm = points_px * scale_px_to_mm

    if origin_offset_mm is not None:
        points_mm += np.array(origin_offset_mm, dtype=float)

    return points_mm


def convert_contour_px_to_mm(
    contour_px: np.ndarray,
    scale_px_to_mm: float,
    origin_offset_mm: Optional[tuple[float, float]] = None
) -> np.ndarray:
    """
    Convert contour from pixels to millimeters.

    Args:
        contour_px: Contour points in pixels, shape (N, 2)
        scale_px_to_mm: Scale factor (mm per pixel)
        origin_offset_mm: Optional offset to apply after scaling (x_mm, y_mm)

    Returns:
        Contour in mm, shape (N, 2)

    Notes:
        - Wrapper around convert_points_px_to_mm for clarity
        - Used for PuzzlePiece.contour_mm conversion
        - See docs/implementation/00_structure.md §1.1 for integration

    Example:
        >>> # From piece_extraction module
        >>> contour_px = extracted_piece.contour  # (N, 2) in pixels
        >>> scale = calibration.get_scale()  # mm/px from camera calibration
        >>> contour_mm = convert_contour_px_to_mm(contour_px, scale)
    """
    return convert_points_px_to_mm(contour_px, scale_px_to_mm, origin_offset_mm)


def convert_bbox_px_to_mm(
    bbox_px: tuple[float, float, float, float],
    scale_px_to_mm: float,
    origin_offset_mm: Optional[tuple[float, float]] = None
) -> tuple[float, float, float, float]:
    """
    Convert bounding box from pixels to millimeters.

    Args:
        bbox_px: Bounding box in pixels (x_min, y_min, x_max, y_max)
        scale_px_to_mm: Scale factor (mm per pixel)
        origin_offset_mm: Optional offset to apply after scaling (x_mm, y_mm)

    Returns:
        Bounding box in mm (x_min, y_min, x_max, y_max)

    Notes:
        - Used for PuzzlePiece.bbox_mm conversion
        - Preserves bounding box structure (min, max)

    Example:
        >>> bbox_px = (100, 200, 300, 400)  # (x_min, y_min, x_max, y_max)
        >>> scale = 0.5  # 0.5mm per pixel
        >>> bbox_mm = convert_bbox_px_to_mm(bbox_px, scale)
        >>> # Result: (50.0, 100.0, 150.0, 200.0)
    """
    x_min_px, y_min_px, x_max_px, y_max_px = bbox_px

    # Convert corners
    corners_px = np.array([
        [x_min_px, y_min_px],
        [x_max_px, y_max_px]
    ])
    corners_mm = convert_points_px_to_mm(corners_px, scale_px_to_mm, origin_offset_mm)

    return (
        float(corners_mm[0, 0]),  # x_min
        float(corners_mm[0, 1]),  # y_min
        float(corners_mm[1, 0]),  # x_max
        float(corners_mm[1, 1]),  # y_max
    )


def get_default_scale_simulator(allow_in_production: bool = False) -> float:
    """
    Get default scale for simulator environment.

    Args:
        allow_in_production: If False (default), raises PlaceholderScaleError
                            to prevent accidental use in scoring/pruning.
                            Set True ONLY for visual tests/debugging.

    Returns:
        Scale factor in mm/px for simulator (0.1 mm/px)

    Raises:
        PlaceholderScaleError: If allow_in_production=False (default behavior)

    Notes:
        - Placeholder for simulator integration
        - TODO: Determine from simulator camera calibration
        - See docs/design/01_coordinates.md §3.2 for calibration requirements

    ⚠ WARNING:
        Placeholder scale (0.1 mm/px) is for visual integration ONLY.
        Do NOT use for scoring/pruning - all thresholds (1.0/2.0/0.1 mm)
        will be systematically wrong.
        Replace with calibrated scale before production use.

    Example:
        >>> # For production code (raises error)
        >>> scale = get_default_scale_simulator()  # PlaceholderScaleError
        >>> # For visual debugging only
        >>> scale = get_default_scale_simulator(allow_in_production=True)  # 0.1
    """
    if not allow_in_production:
        raise PlaceholderScaleError(
            "Placeholder scale (0.1 mm/px) must not be used in production. "
            "Set allow_in_production=True only for visual tests, "
            "or provide calibrated scale."
        )
    # Placeholder: Assume 1px = 0.1mm (to be calibrated)
    return 0.1


def get_machine_origin_offset_placeholder() -> tuple[float, float]:
    """
    Get placeholder origin offset for Machine coordinate system.

    Returns:
        Offset (x_mm, y_mm) from pixel (0,0) to Machine origin

    Notes:
        - Placeholder until machine is built
        - TODO: Determine from physical setup (camera position relative to machine)
        - Related to FrameModel.T_MF (Frame to Machine transform)
        - See docs/design/01_coordinates.md §2.2 for Machine coordinate system
    """
    # Placeholder: No offset (Machine origin = Camera (0, 0) scaled)
    return (0.0, 0.0)


def extract_scale_from_metadata(metadata: dict) -> float:
    """
    Extract scale from image metadata (placeholder).

    Args:
        metadata: Dictionary with calibration data (expected key: 'scale_px_to_mm')

    Returns:
        Scale factor in mm/px

    Raises:
        NotImplementedError: Implementation reserved for Step 10 (Integration)

    Notes:
        - Placeholder for camera calibration integration
        - TODO: Implement metadata extraction from camera/image EXIF or calibration file
        - Related to get_default_scale_simulator() (returns hardcoded default)
        - See docs/design/01_coordinates.md §3.2 for calibration requirements
    """
    raise NotImplementedError(
        "extract_scale_from_metadata() - Reserved for Step 10 (Integration)\n"
        "TODO: Implement camera calibration metadata extraction"
    )


def get_default_T_MF() -> Transform2D:
    """
    Get default T_MF transform for simulator visualization.

    Returns:
        Transform2D(200, 200, 0): Places Frame center at (200mm, 200mm), outside origin

    Notes:
        - Default placement for simulator: Frame offset from (0, 0) to avoid overlap
        - Frame inner dimensions: 128×190mm, so center at (64, 95) in Frame coords
        - Transform (200, 200, 0) places lower-left corner at (200, 200) in Machine coords
        - No rotation (theta_deg = 0): Frame axes aligned with Machine axes
        - See docs/design/01_coordinates.md §2.2 for coordinate system definitions

    Example:
        >>> from solver.config import FrameModel
        >>> frame = FrameModel(T_MF=get_default_T_MF())
        >>> # Frame now positioned for visualization outside origin
    """
    from ..config import Transform2D
    return Transform2D(x_mm=200.0, y_mm=200.0, theta_deg=0.0)


def validate_pieces_format(
    pieces: List[PuzzlePiece],
    max_mm_extent: float = 1000.0
) -> None:
    """
    Validate PuzzlePiece format (raises ValueError on invalid input).

    Args:
        pieces: List of PuzzlePiece objects to validate
        max_mm_extent: Maximum plausible extent in mm (Frame-KS).
                      Default 1000mm = 1m (far beyond typical frame).
                      Adjust for different coordinate systems if needed.

    Raises:
        ValueError: If any piece has invalid format:
            - contour_mm not (N, 2) with N >= 3
            - bbox_mm not tuple of 4 floats
            - center_mm not (2,) if present
            - Suspicious values (likely still in pixels: > max_mm_extent)

    Notes:
        - Called before solver processing to fail-fast on invalid inputs
        - mm-plausibility check: Prevents accidental use of pixel coordinates
        - See docs/design/02_datamodels.md §PuzzlePiece for format specification

    Example:
        >>> validate_pieces_format(pieces)  # Default: max 1000mm
        >>> validate_pieces_format(pieces, max_mm_extent=500.0)  # Custom limit
    """
    from ..models import PuzzlePiece

    if not pieces:
        raise ValueError("Empty pieces list")

    for i, piece in enumerate(pieces):
        # Check contour_mm shape
        if not isinstance(piece.contour_mm, np.ndarray):
            raise ValueError(f"Piece {piece.piece_id}: contour_mm must be ndarray, got {type(piece.contour_mm)}")

        if piece.contour_mm.ndim != 2 or piece.contour_mm.shape[1] != 2:
            raise ValueError(
                f"Piece {piece.piece_id}: contour_mm must have shape (N, 2), "
                f"got {piece.contour_mm.shape}"
            )

        if piece.contour_mm.shape[0] < 3:
            raise ValueError(
                f"Piece {piece.piece_id}: contour_mm must have at least 3 points, "
                f"got {piece.contour_mm.shape[0]}"
            )

        # Check bbox_mm format
        if not isinstance(piece.bbox_mm, tuple) or len(piece.bbox_mm) != 4:
            raise ValueError(
                f"Piece {piece.piece_id}: bbox_mm must be tuple of 4 floats, "
                f"got {type(piece.bbox_mm)} with length {len(piece.bbox_mm) if isinstance(piece.bbox_mm, tuple) else 'N/A'}"
            )

        x_min, y_min, x_max, y_max = piece.bbox_mm
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"Piece {piece.piece_id}: bbox_mm invalid bounds "
                f"(x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max})"
            )

        # Check center_mm if present
        if piece.center_mm is not None:
            if not isinstance(piece.center_mm, np.ndarray):
                raise ValueError(
                    f"Piece {piece.piece_id}: center_mm must be ndarray or None, "
                    f"got {type(piece.center_mm)}"
                )
            if piece.center_mm.shape != (2,):
                raise ValueError(
                    f"Piece {piece.piece_id}: center_mm must have shape (2,), "
                    f"got {piece.center_mm.shape}"
                )

        # Sanity check: values should be in mm (not pixels)
        # Puzzle pieces should not exceed max_mm_extent
        max_val = np.max(np.abs(piece.contour_mm))
        if max_val > max_mm_extent:
            raise ValueError(
                f"Piece {piece.piece_id}: Suspicious contour_mm values (max={max_val:.1f}mm > {max_mm_extent}mm). "
                f"Still in pixels? Use convert_pieces_px_to_mm() first."
            )


def convert_pieces_px_to_mm(
    pieces: List[PuzzlePiece],
    scale_px_to_mm: float,
    origin_offset_mm: Optional[tuple[float, float]] = None
) -> List[PuzzlePiece]:
    """
    Convert list of PuzzlePiece from pixels to millimeters (batch conversion).

    Args:
        pieces: List of PuzzlePiece with contour/bbox/center in pixels
        scale_px_to_mm: Scale factor (mm per pixel)
        origin_offset_mm: Optional offset to apply after scaling (x_mm, y_mm)

    Returns:
        New list of PuzzlePiece with contour_mm/bbox_mm/center_mm in mm

    Notes:
        - Creates new PuzzlePiece instances (immutable transformation)
        - Preserves piece_id, mask, image (no unit conversion needed)
        - Uses convert_contour_px_to_mm() and convert_bbox_px_to_mm() internally
        - See docs/implementation/00_structure.md §1.1 for integration with piece_extraction

    Example:
        >>> # From piece_extraction module
        >>> pieces_px = extract_pieces(image)  # contours in pixels
        >>> scale = calibration.get_scale()  # 0.5 mm/px
        >>> pieces_mm = convert_pieces_px_to_mm(pieces_px, scale)
        >>> # Now ready for solver input
    """
    from ..models import PuzzlePiece

    converted_pieces = []

    for piece in pieces:
        # Convert contour
        contour_mm = convert_contour_px_to_mm(
            piece.contour_mm,
            scale_px_to_mm,
            origin_offset_mm
        )

        # Convert bbox
        bbox_mm = convert_bbox_px_to_mm(
            piece.bbox_mm,
            scale_px_to_mm,
            origin_offset_mm
        )

        # Convert center if present
        center_mm = None
        if piece.center_mm is not None:
            center_px = piece.center_mm.reshape(1, 2)  # (2,) -> (1, 2)
            center_mm_arr = convert_points_px_to_mm(
                center_px,
                scale_px_to_mm,
                origin_offset_mm
            )
            center_mm = center_mm_arr[0]  # (1, 2) -> (2,)

        # Create new PuzzlePiece with converted values
        converted_piece = PuzzlePiece(
            piece_id=piece.piece_id,
            contour_mm=contour_mm,
            mask=piece.mask,  # No conversion (pixel-based)
            bbox_mm=bbox_mm,
            image=piece.image,  # No conversion (pixel-based)
            center_mm=center_mm
        )

        converted_pieces.append(converted_piece)

    return converted_pieces
