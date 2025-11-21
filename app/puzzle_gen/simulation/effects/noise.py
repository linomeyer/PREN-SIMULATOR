"""Sensor noise effects for camera simulation."""
import numpy as np
from .base import CameraEffect


class GaussianNoiseEffect(CameraEffect):
    """Apply Gaussian noise to simulate sensor imperfections."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian noise to image.

        Args:
            image: Input image array (H, W, 3) RGB uint8

        Returns:
            Image with Gaussian noise applied
        """
        if self._should_skip():
            return image

        # Generate Gaussian noise
        noise = np.random.normal(0, self.strength * 25, image.shape)

        # Add noise to image
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return noisy


class SaltPepperNoiseEffect(CameraEffect):
    """Apply salt and pepper noise (dead/hot pixels)."""

    def __init__(self, strength: float, amount: float = 0.002):
        """
        Initialize salt and pepper noise effect.

        Args:
            strength: Noise strength (0-1)
            amount: Fraction of pixels to affect (default 0.002)
        """
        super().__init__(strength)
        self.amount = amount

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply salt and pepper noise to image.

        Args:
            image: Input image array (H, W, 3) RGB uint8

        Returns:
            Image with salt and pepper noise applied
        """
        if self._should_skip():
            return image

        output = image.copy()
        num_salt = int(self.amount * self.strength * image.size * 0.5)
        num_pepper = int(self.amount * self.strength * image.size * 0.5)

        # Add salt (white pixels)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        output[coords[0], coords[1], :] = 255

        # Add pepper (black pixels)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        output[coords[0], coords[1], :] = 0

        return output
