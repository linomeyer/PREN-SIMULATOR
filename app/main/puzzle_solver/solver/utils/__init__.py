"""
Solver V2 Utility Functions.

This module provides utility functions for the puzzle solver:
- conversion: Pixel to mm conversions, coordinate system transformations
"""

from .conversion import (
    convert_contour_px_to_mm,
    convert_bbox_px_to_mm,
    convert_points_px_to_mm,
    extract_scale_from_metadata,
    convert_pieces_px_to_mm,
    validate_pieces_format,
    get_default_T_MF,
    PlaceholderScaleError,
)

__all__ = [
    "convert_contour_px_to_mm",
    "convert_bbox_px_to_mm",
    "convert_points_px_to_mm",
    "extract_scale_from_metadata",
    "convert_pieces_px_to_mm",
    "validate_pieces_format",
    "get_default_T_MF",
    "PlaceholderScaleError",
]
