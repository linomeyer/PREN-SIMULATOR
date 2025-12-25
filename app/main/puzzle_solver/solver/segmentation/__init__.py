"""
Solver V2 Segmentation Module.

This module provides contour segmentation for puzzle pieces:
- segment_piece: Main API for piece contour segmentation

See docs/design/03_matching.md Phase 1 for algorithm details.
"""

from .contour_segmenter import segment_piece

__all__ = ["segment_piece"]
