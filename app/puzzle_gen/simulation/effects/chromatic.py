"""Chromatic aberration and color effects for camera simulation."""
import cv2
import numpy as np
from .base import CameraEffect


class ChromaticAberrationEffect(CameraEffect):
    """Apply chromatic aberration (color fringing from lens imperfections)."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply chromatic aberration to image.

        Args:
            image: Input image array (H, W, 3) RGB uint8

        Returns:
            Image with chromatic aberration (RGB channels misaligned)
        """
        if self._should_skip():
            return image

        h, w = image.shape[:2]

        # Split channels
        b, g, r = cv2.split(image)

        # Calculate shift amount (in pixels)
        shift = int(self.strength * 10)

        if shift < 1:
            return image

        # Create transformation matrices for slight scaling/shifting
        # Red channel: slightly expand
        M_r = cv2.getRotationMatrix2D((w/2, h/2), 0, 1 + self.strength * 0.01)
        r_shifted = cv2.warpAffine(r, M_r, (w, h))

        # Blue channel: slightly contract
        M_b = cv2.getRotationMatrix2D((w/2, h/2), 0, 1 - self.strength * 0.01)
        b_shifted = cv2.warpAffine(b, M_b, (w, h))

        # Merge channels
        aberrated = cv2.merge([b_shifted, g, r_shifted])

        return aberrated


class PurpleFringingEffect(CameraEffect):
    """Apply purple fringing at high-contrast edges (typical for cheap cameras)."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply purple fringing to image.

        Args:
            image: Input image array (H, W, 3) RGB uint8

        Returns:
            Image with purple/magenta fringing at edges
        """
        if self._should_skip():
            return image

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect edges using Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)

        # Normalize edges to 0-1
        edges = edges / (edges.max() + 1e-6)

        # Threshold to get strong edges only
        edge_mask = (edges > 0.3).astype(np.float32)

        # Blur the mask for softer fringing
        edge_mask = cv2.GaussianBlur(edge_mask, (5, 5), 2)

        # Create purple/magenta color (stronger in R and B channels)
        purple_intensity = self.strength * 40
        purple_overlay = np.zeros_like(image, dtype=np.float32)
        purple_overlay[:, :, 0] = purple_intensity * 0.8  # Red
        purple_overlay[:, :, 1] = purple_intensity * 0.2  # Green (low)
        purple_overlay[:, :, 2] = purple_intensity * 1.0  # Blue

        # Apply purple fringing only at edges
        edge_mask_3d = edge_mask[:, :, np.newaxis]
        result = image.astype(np.float32) + purple_overlay * edge_mask_3d

        # Clip to valid range
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result


class ColorNoiseEffect(CameraEffect):
    """Apply color noise to simulate color camera capturing BW scene."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color noise to image.

        Args:
            image: Input image array (H, W, 3) RGB uint8

        Returns:
            Image with color noise added
        """
        if self._should_skip():
            return image

        # Generate color noise
        noise = np.random.normal(0, self.strength * 15, image.shape)

        # Apply noise
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return noisy
