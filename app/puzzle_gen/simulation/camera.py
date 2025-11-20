"""Camera simulation effects using OpenCV."""
import cv2
import numpy as np
from typing import Tuple


class CameraSimulator:
    """Simulates RasPi Camera Module v3 artifacts and imperfections."""

    def __init__(
        self,
        fisheye_strength: float = 0.15,
        noise_amount: float = 0.03,
        vignette_strength: float = 0.2,
        color_aberration: float = 0.02,
        color_noise: float = 0.01
    ):
        """
        Initialize camera simulator.

        Args:
            fisheye_strength: Barrel distortion strength (0-1)
            noise_amount: Gaussian + salt-pepper noise (0-1)
            vignette_strength: Radial lighting gradient (0-1)
            color_aberration: Chromatic aberration (0-1)
            color_noise: Color noise in BW scene (0-1)
        """
        self.fisheye_strength = fisheye_strength
        self.noise_amount = noise_amount
        self.vignette_strength = vignette_strength
        self.color_aberration = color_aberration
        self.color_noise = color_noise

    def _apply_barrel_distortion(self, image: np.ndarray) -> np.ndarray:
        """
        Apply barrel (fisheye) distortion.

        Args:
            image: Input image array

        Returns:
            Distorted image
        """
        if self.fisheye_strength <= 0:
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
        k1 = self.fisheye_strength * 0.5
        dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float32)

        # Apply distortion
        distorted = cv2.undistort(image, camera_matrix, -dist_coeffs)

        return distorted

    def _apply_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian noise.

        Args:
            image: Input image array

        Returns:
            Noisy image
        """
        if self.noise_amount <= 0:
            return image

        # Generate Gaussian noise
        noise = np.random.normal(0, self.noise_amount * 25, image.shape)

        # Add noise
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return noisy

    def _apply_salt_pepper_noise(self, image: np.ndarray, amount: float = 0.002) -> np.ndarray:
        """
        Apply salt and pepper noise.

        Args:
            image: Input image array
            amount: Fraction of pixels to affect

        Returns:
            Noisy image
        """
        if self.noise_amount <= 0:
            return image

        output = image.copy()
        num_salt = int(amount * self.noise_amount * image.size * 0.5)
        num_pepper = int(amount * self.noise_amount * image.size * 0.5)

        # Add salt (white pixels)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        output[coords[0], coords[1], :] = 255

        # Add pepper (black pixels)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        output[coords[0], coords[1], :] = 0

        return output

    def _apply_vignette(self, image: np.ndarray) -> np.ndarray:
        """
        Apply radial vignette effect (lighting gradient).

        Args:
            image: Input image array

        Returns:
            Image with vignette
        """
        if self.vignette_strength <= 0:
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
        vignette = 1 - (dist ** 2) * self.vignette_strength

        # Apply to image
        vignette = vignette[:, :, np.newaxis]  # Add channel dimension
        vignetted = (image.astype(np.float32) * vignette).astype(np.uint8)

        return vignetted

    def _apply_chromatic_aberration(self, image: np.ndarray) -> np.ndarray:
        """
        Apply chromatic aberration (color fringing).

        Args:
            image: Input image array (RGB)

        Returns:
            Image with chromatic aberration
        """
        if self.color_aberration <= 0:
            return image

        h, w = image.shape[:2]

        # Split channels
        b, g, r = cv2.split(image)

        # Calculate shift amount (in pixels)
        shift = int(self.color_aberration * 10)

        if shift < 1:
            return image

        # Create transformation matrices for slight scaling/shifting
        # Red channel: slightly expand
        M_r = cv2.getRotationMatrix2D((w/2, h/2), 0, 1 + self.color_aberration * 0.01)
        r_shifted = cv2.warpAffine(r, M_r, (w, h))

        # Blue channel: slightly contract
        M_b = cv2.getRotationMatrix2D((w/2, h/2), 0, 1 - self.color_aberration * 0.01)
        b_shifted = cv2.warpAffine(b, M_b, (w, h))

        # Merge channels
        aberrated = cv2.merge([b_shifted, g, r_shifted])

        return aberrated

    def _apply_color_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color noise to simulate color camera capturing BW scene.

        Args:
            image: Input image array

        Returns:
            Image with color noise
        """
        if self.color_noise <= 0:
            return image

        # Generate color noise
        noise = np.random.normal(0, self.color_noise * 15, image.shape)

        # Apply noise
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return noisy

    def simulate(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all camera simulation effects.

        Args:
            image: Input image array (H, W, 3) RGB

        Returns:
            Simulated camera image
        """
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Apply effects in sequence
        result = image.copy()

        # 1. Barrel distortion (fisheye)
        result = self._apply_barrel_distortion(result)

        # 2. Chromatic aberration
        result = self._apply_chromatic_aberration(result)

        # 3. Vignette
        result = self._apply_vignette(result)

        # 4. Gaussian noise
        result = self._apply_gaussian_noise(result)

        # 5. Salt and pepper noise
        result = self._apply_salt_pepper_noise(result)

        # 6. Color noise
        result = self._apply_color_noise(result)

        return result

    @classmethod
    def from_config(cls, config_params: dict) -> 'CameraSimulator':
        """
        Create simulator from configuration dictionary.

        Args:
            config_params: Dictionary with effect parameters

        Returns:
            CameraSimulator instance
        """
        return cls(
            fisheye_strength=config_params.get('fisheye', 0.15),
            noise_amount=config_params.get('noise', 0.03),
            vignette_strength=config_params.get('vignette', 0.2),
            color_aberration=config_params.get('aberration', 0.02),
            color_noise=config_params.get('color_noise', 0.01)
        )
