"""Optical and lens-based camera effects."""
import cv2
import numpy as np
from .base import CameraEffect


class BarrelDistortionEffect(CameraEffect):
    """Apply barrel (fisheye) distortion from cheap wide-angle lens."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply barrel distortion to image.

        Args:
            image: Input image array (H, W, 3) RGB uint8

        Returns:
            Image with barrel distortion applied
        """
        if self._should_skip():
            return image

        h, w = image.shape[:2]

        # Create camera matrix
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Distortion coefficients (k1, k2, p1, p2, k3)
        # Positive k1 creates barrel distortion
        k1 = self.strength * 0.5
        dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float32)

        # Apply distortion
        distorted = cv2.undistort(image, camera_matrix, -dist_coeffs)

        return distorted


class LensSoftnessEffect(CameraEffect):
    """Apply lens softness/blur at image edges (cheap lens quality)."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply lens softness to image.

        Args:
            image: Input image array (H, W, 3) RGB uint8

        Returns:
            Image with soft/blurry edges
        """
        if self._should_skip():
            return image

        h, w = image.shape[:2]

        # Create radial mask (1 at center, 0 at edges)
        center_x, center_y = w / 2, h / 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist_norm = dist / max_dist

        # Create sharpness mask (sharp in center, blurry at edges)
        sharpness_mask = 1 - (dist_norm ** 2) * self.strength

        # Blur the entire image
        blurred = cv2.GaussianBlur(image, (15, 15), 5)

        # Blend based on mask
        sharpness_mask_3d = sharpness_mask[:, :, np.newaxis]
        result = (image.astype(np.float32) * sharpness_mask_3d +
                  blurred.astype(np.float32) * (1 - sharpness_mask_3d))

        result = result.astype(np.uint8)

        return result
