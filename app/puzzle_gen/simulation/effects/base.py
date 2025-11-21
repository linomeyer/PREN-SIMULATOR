"""Base class for camera simulation effects."""
from abc import ABC, abstractmethod
import numpy as np


class CameraEffect(ABC):
    """Abstract base class for camera simulation effects."""

    def __init__(self, strength: float):
        """
        Initialize camera effect.

        Args:
            strength: Effect strength/intensity (0-1 typically)
        """
        self.strength = strength

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply effect to image.

        Args:
            image: Input image array (H, W, 3) RGB uint8

        Returns:
            Image with effect applied
        """
        pass

    def _should_skip(self) -> bool:
        """
        Check if effect should be skipped based on strength.

        Returns:
            True if effect should be skipped (strength <= 0)
        """
        return self.strength <= 0
