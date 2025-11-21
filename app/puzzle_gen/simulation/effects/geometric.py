"""Geometric distortions and lighting effects for camera simulation."""
import cv2
import numpy as np
from .base import CameraEffect


class VignetteEffect(CameraEffect):
    """Apply radial vignette effect (lighting gradient/falloff)."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply vignette effect to image.

        Args:
            image: Input image array (H, W, 3) RGB uint8

        Returns:
            Image with vignette applied (darker at edges)
        """
        if self._should_skip():
            return image

        h, w = image.shape[:2]

        # Create radial gradient
        center_x, center_y = w / 2, h / 2

        # Create coordinate grids
        y, x = np.ogrid[:h, :w]

        # Calculate distance from center
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # Normalize to 0-1
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist = dist / max_dist

        # Create vignette mask (darker at edges)
        vignette = 1 - (dist ** 2) * self.strength

        # Apply to image
        vignette = vignette[:, :, np.newaxis]  # Add channel dimension
        vignetted = (image.astype(np.float32) * vignette).astype(np.uint8)

        return vignetted


class PerspectiveDistortionEffect(CameraEffect):
    """Apply perspective distortion to simulate camera viewing angle."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply perspective distortion to image.

        Args:
            image: Input image array (H, W, 3) RGB uint8

        Returns:
            Image with perspective distortion applied
        """
        if self._should_skip():
            return image

        h, w = image.shape[:2]

        # Define source points (corners of image)
        src_points = np.float32([
            [0, 0],           # top-left
            [w, 0],           # top-right
            [w, h],           # bottom-right
            [0, h]            # bottom-left
        ])

        # Calculate offset based on strength
        offset = int(w * self.strength * 0.05)  # Max 5% of width

        # Define destination points (slightly skewed)
        dst_points = np.float32([
            [offset, offset],        # top-left moved right/down
            [w - offset, 0],         # top-right moved left
            [w, h - offset],         # bottom-right moved up
            [0, h]                   # bottom-left unchanged
        ])

        # Get perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply perspective transformation
        result = cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

        return result
