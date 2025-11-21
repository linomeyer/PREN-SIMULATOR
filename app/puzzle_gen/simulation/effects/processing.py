"""Software post-processing effects for camera simulation."""
import cv2
import numpy as np
from .base import CameraEffect


class OversharpeningEffect(CameraEffect):
    """Apply aggressive software sharpening (typical for smartphone post-processing)."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply oversharpening to image.

        Args:
            image: Input image array (H, W, 3) RGB uint8

        Returns:
            Oversharpened image with halos and artifacts
        """
        if self._should_skip():
            return image

        # Create unsharp mask
        # 1. Blur the image
        blurred = cv2.GaussianBlur(image, (5, 5), 1.5)

        # 2. Calculate sharpening mask (original - blurred)
        mask = image.astype(np.float32) - blurred.astype(np.float32)

        # 3. Add mask back to original with aggressive strength
        sharpened = image.astype(np.float32) + mask * self.strength

        # Clip to valid range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        return sharpened


class NoiseReductionEffect(CameraEffect):
    """Apply aggressive noise reduction causing detail loss (typical for smartphones)."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply aggressive noise reduction to image.

        Args:
            image: Input image array (H, W, 3) RGB uint8

        Returns:
            Image with noise reduced but details lost (plastic/smooth look)
        """
        if self._should_skip():
            return image

        # Calculate filter parameters based on strength
        d = int(9 * self.strength)  # Diameter (max 9)
        sigma_color = 75 * self.strength  # Color sigma (max 75)
        sigma_space = 75 * self.strength  # Space sigma (max 75)

        # Ensure minimum values
        d = max(3, d)
        if d % 2 == 0:
            d += 1  # Must be odd

        # Apply bilateral filter (preserves edges while smoothing)
        result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        return result
