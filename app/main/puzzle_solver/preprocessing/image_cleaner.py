"""
Local image cleaning module for individual puzzle pieces.

This module performs piece-level cleaning operations that should be applied
AFTER piece extraction and BEFORE edge detection. It addresses local effects:
- Shadow removal on individual pieces
- Noise reduction (Gaussian, salt & pepper, color noise)
- Contour recalculation and smoothing

Operates on PuzzlePiece objects.
"""

import cv2
import numpy as np
from typing import List
from scipy.ndimage import gaussian_filter1d


class ImageCleaner:
    """
    Cleans individual puzzle pieces after extraction.

    Process:
    1. Remove shadows from piece.image (RGBA)
    2. Reduce noise (bilateral filter)
    3. Recalculate contour from cleaned image
    4. Smooth contour
    5. Update piece object with cleaned data
    """

    # Configuration as class attributes
    SHADOW_REMOVAL_CLAHE_CLIP = 39.0
    SHADOW_REMOVAL_CLAHE_TILE = 4
    DENOISE_BILATERAL_D = 5
    DENOISE_BILATERAL_SIGMA_COLOR = 10
    DENOISE_BILATERAL_SIGMA_SPACE = 10
    CONTOUR_SMOOTHING_SIGMA = 0.01
    CONTOUR_MIN_AREA = 100

    def __init__(self):
        """Initialize ImageCleaner."""
        pass

    def clean_piece(self, piece):
        """
        Clean a single puzzle piece.

        Args:
            piece: PuzzlePiece object with image, contour, mask, bbox

        Returns:
            Modified piece object (modified in place)
        """
        # Store original contour for comparison
        piece.contour_original = piece.contour.copy()

        # 1. Clean the piece image (RGBA)
        cleaned_image = self._clean_image(piece.image)

        # 2. Recalculate contour from cleaned image
        new_contour, new_mask = self._recalculate_contour(cleaned_image, piece.bbox)

        # 3. Smooth the contour
        smoothed_contour = self._smooth_contour(new_contour)

        # 4. Update piece with cleaned data
        piece.image = cleaned_image
        piece.contour = smoothed_contour
        # Update bounding box to match new contour
        piece.bbox = cv2.boundingRect(smoothed_contour)
        # Keep original mask for now, or update if significant changes
        # piece.mask = new_mask  # Optional

        return piece

    def clean_pieces(self, pieces: List) -> List:
        """
        Clean multiple puzzle pieces.

        Args:
            pieces: List of PuzzlePiece objects

        Returns:
            List of cleaned pieces (modified in place)
        """
        for piece in pieces:
            self.clean_piece(piece)

        return pieces

    def _clean_image(self, rgba_image: np.ndarray) -> np.ndarray:
        """
        Clean RGBA piece image: remove shadows and noise.

        Args:
            rgba_image: RGBA image (H, W, 4) with transparency

        Returns:
            Cleaned RGBA image
        """
        # Extract RGB and alpha
        rgb = rgba_image[:, :, :3].copy()
        alpha = rgba_image[:, :, 3].copy()

        # 1. Shadow removal using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        rgb_deshadowed = self._remove_shadows(rgb, alpha)

        # 2. Denoise using bilateral filter (preserves edges)
        rgb_denoised = self._denoise_image(rgb_deshadowed)

        # 3. Recombine with alpha channel
        cleaned = np.dstack([rgb_denoised, alpha])

        return cleaned

    def _remove_shadows(self, rgb_image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Remove shadows from RGB image using CLAHE.

        CLAHE normalizes local contrast, which effectively removes shadow variations.

        Args:
            rgb_image: RGB image (H, W, 3)
            alpha: Alpha channel (H, W) - mask of the piece

        Returns:
            Shadow-removed RGB image
        """
        # Convert to LAB color space (better for shadow removal)
        lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)

        # Split channels
        l, a, b = cv2.split(lab)

        gamma = 0.6 # < 1 = hebt dunkle Bereiche an


        # Apply CLAHE only to L channel (lightness)
        clahe = cv2.createCLAHE(
            clipLimit=self.SHADOW_REMOVAL_CLAHE_CLIP,
            tileGridSize=(self.SHADOW_REMOVAL_CLAHE_TILE, self.SHADOW_REMOVAL_CLAHE_TILE)
        )
        l_clahe = clahe.apply(l)

        gamma = 0.6
        l_float = l_clahe.astype(np.float32) / 255.0
        l_corr = np.power(l_float, gamma)
        l_corr = np.clip(l_corr * 255, 0, 255).astype(np.uint8)

        # Merge back
        lab_clahe = cv2.merge([l_corr, a, b])

        # Convert back to RGB
        rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

        # Only apply to non-transparent regions
        mask_3ch = np.stack([alpha > 0] * 3, axis=-1)
        result = np.where(mask_3ch, rgb_clahe, rgb_image)

        return result

    def _denoise_image(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Denoise RGB image using bilateral filter.

        Bilateral filter reduces noise while preserving edges - perfect for puzzle pieces.

        Args:
            rgb_image: RGB image (H, W, 3)

        Returns:
            Denoised RGB image
        """
        denoised = cv2.bilateralFilter(
            rgb_image,
            d=self.DENOISE_BILATERAL_D,
            sigmaColor=self.DENOISE_BILATERAL_SIGMA_COLOR,
            sigmaSpace=self.DENOISE_BILATERAL_SIGMA_SPACE
        )

        return denoised

    def _recalculate_contour(self, rgba_image: np.ndarray, bbox: tuple) -> tuple:
        """
        Recalculate contour from cleaned RGBA image.

        Args:
            rgba_image: Cleaned RGBA image (H, W, 4)
            bbox: Original bounding box (x, y, w, h)

        Returns:
            Tuple of (contour, mask) in original image coordinates
        """
        # Extract alpha channel as mask
        alpha = rgba_image[:, :, 3]

        # Threshold alpha to binary mask
        _, binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to close small gaps caused by noise
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # No contour found - return original bbox as rectangle
            x, y, w, h = bbox
            contour = np.array([
                [[0, 0]], [[w, 0]], [[w, h]], [[0, h]]
            ], dtype=np.int32)
        else:
            # Take largest contour
            contour = max(contours, key=cv2.contourArea)

        # Convert contour from ROI coordinates to full image coordinates
        x, y, w, h = bbox
        contour_full = contour.copy()
        contour_full[:, 0, 0] += x  # Add x offset
        contour_full[:, 0, 1] += y  # Add y offset

        # Create full-size mask (optional)
        # For now, return None for mask
        mask = None

        return contour_full, mask

    def _smooth_contour(self, contour: np.ndarray) -> np.ndarray:
        """
        Smooth contour to remove noise-induced irregularities.

        Uses Gaussian smoothing on contour coordinates.

        Args:
            contour: Contour array (N, 1, 2)

        Returns:
            Smoothed contour array
        """
        if len(contour) < 3:
            return contour  # Too few points to smooth

        # Extract x and y coordinates
        points = contour[:, 0, :]  # Shape: (N, 2)
        x = points[:, 0]
        y = points[:, 1]

        # Apply Gaussian smoothing
        # Use circular padding to maintain closed contour
        x_smooth = gaussian_filter1d(x, sigma=self.CONTOUR_SMOOTHING_SIGMA, mode='wrap')
        y_smooth = gaussian_filter1d(y, sigma=self.CONTOUR_SMOOTHING_SIGMA, mode='wrap')

        # Reconstruct contour
        smoothed_points = np.column_stack([x_smooth, y_smooth])
        smoothed_contour = smoothed_points[:, np.newaxis, :].astype(np.int32)

        return smoothed_contour

    def visualize_cleaning_effect(self, piece, original_image: np.ndarray) -> dict:
        """
        Create visualization showing cleaning effect.

        Args:
            piece: PuzzlePiece object (after cleaning)
            original_image: Original full image for drawing

        Returns:
            Dictionary with visualization images:
            - 'contour_comparison': Original vs cleaned contours
            - 'piece_before': Original piece image
            - 'piece_after': Cleaned piece image
        """
        vis = {}

        # Draw contour comparison
        h, w = original_image.shape[:2]
        comparison = original_image.copy()

        # Draw original contour in red
        if hasattr(piece, 'contour_original'):
            cv2.drawContours(comparison, [piece.contour_original], -1, (255, 0, 0), 2)

        # Draw cleaned contour in green
        cv2.drawContours(comparison, [piece.contour], -1, (0, 255, 0), 2)

        vis['contour_comparison'] = comparison

        # Piece images (if available)
        # For now, just return the comparison
        # Could be extended to show piece.image before/after

        return vis
