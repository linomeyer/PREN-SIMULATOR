"""
Global image cleaning module for correcting camera-induced distortions.

This module corrects global (full-image) distortions that must be addressed
before piece extraction, including:
- Barrel distortion (fisheye effect)
- Perspective distortion
- Vignette (radial lighting falloff)
- Chromatic aberration

Uses calibration-based approach: parameters are determined once from test images,
then saved and reused for production.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime


class GlobalCleaner:
    """
    Corrects global camera distortions on full puzzle images.

    Workflow:
    1. Calibrate once with test images → saves parameters
    2. Load calibration parameters
    3. Apply corrections to production images
    """

    # Default calibration file location
    CALIBRATION_FILE = Path(__file__).parent.parent.parent / "calibration_params.json"

    # Default barrel distortion coefficients (typical for wide-angle camera)
    DEFAULT_BARREL_K1 = -0.3  # Radial distortion coefficient (negative = barrel)
    DEFAULT_BARREL_K2 = 0.05  # Secondary radial distortion
    DEFAULT_BARREL_K3 = 0.0   # Tertiary radial distortion

    # Default vignette strength
    DEFAULT_VIGNETTE_STRENGTH = 0.5

    def __init__(self, calibration_path: Optional[Path] = None):
        """
        Initialize GlobalCleaner.

        Args:
            calibration_path: Path to calibration file. Uses default if None.
        """
        self.calibration_path = calibration_path or self.CALIBRATION_FILE
        self.params = {}
        self.is_calibrated = False

        # Try to load existing calibration
        if self.calibration_path.exists():
            self.load_calibration()

    def calibrate(self, test_images: List[np.ndarray]) -> Dict:
        """
        Calibrate correction parameters from test images.

        This method analyzes test images to determine optimal correction parameters.
        Test images should be:
        - High quality reference images with known geometry
        - Same resolution as production images
        - Contain straight lines/patterns for distortion detection

        Args:
            test_images: List of test images (numpy arrays)

        Returns:
            Dictionary of calibration parameters
        """
        print(f"Calibrating with {len(test_images)} test image(s)...")

        params = {
            'calibration_date': datetime.now().isoformat(),
            'image_shape': test_images[0].shape[:2] if test_images else None,
            'barrel_distortion': self._calibrate_barrel_distortion(test_images),
            'perspective': self._calibrate_perspective(test_images),
            'vignette': self._calibrate_vignette(test_images),
            'chromatic_aberration': self._calibrate_chromatic(test_images),
        }

        self.params = params
        self.is_calibrated = True

        # Save calibration
        self.save_calibration(params)

        print(f"Calibration complete. Saved to {self.calibration_path}")
        return params

    def _calibrate_barrel_distortion(self, images: List[np.ndarray]) -> Dict:
        """
        Calibrate barrel distortion parameters.

        Uses edge detection and line fitting to estimate distortion coefficients.
        """
        if not images:
            return {'k1': 0.0, 'k2': 0.0, 'k3': 0.0}

        # For now, use default values based on typical camera characteristics
        # In production, this should use cv2.calibrateCamera with checkerboard patterns
        img = images[0]
        h, w = img.shape[:2]

        # Use default distortion coefficients (typical for wide-angle cameras)
        # TODO: Implement proper calibration with cv2.calibrateCamera
        k1 = self.DEFAULT_BARREL_K1  # Radial distortion coefficient
        k2 = self.DEFAULT_BARREL_K2  # Secondary radial distortion
        k3 = self.DEFAULT_BARREL_K3  # Tertiary radial distortion

        # Camera matrix (approximate)
        fx = fy = max(h, w)  # Focal length estimate
        cx, cy = w / 2, h / 2  # Principal point (image center)

        return {
            'k1': k1,
            'k2': k2,
            'k3': k3,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'image_size': [w, h]
        }

    def _calibrate_perspective(self, images: List[np.ndarray]) -> Dict:
        """
        Calibrate perspective correction parameters.

        Detects corners and calculates perspective transform to frontal view.
        """
        if not images:
            return {'enabled': False}

        # Detect if there's significant perspective distortion
        # For now, return disabled - would need known reference points
        return {
            'enabled': False,
            'transform_matrix': None
        }

    def _calibrate_vignette(self, images: List[np.ndarray]) -> Dict:
        """
        Calibrate vignette correction parameters.

        Analyzes radial intensity falloff from image center.
        """
        if not images:
            return {'strength': self.DEFAULT_VIGNETTE_STRENGTH, 'enabled': True}

        img = images[0]
        h, w = img.shape[:2]

        # Convert to grayscale for analysis
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Measure intensity at center vs corners
        center_val = gray[h//2-10:h//2+10, w//2-10:w//2+10].mean()
        corners_val = (
            gray[0:20, 0:20].mean() +
            gray[0:20, w-20:w].mean() +
            gray[h-20:h, 0:20].mean() +
            gray[h-20:h, w-20:w].mean()
        ) / 4.0

        # Calculate vignette strength
        if center_val > 0:
            falloff_ratio = corners_val / center_val
            strength = 1.0 - falloff_ratio  # 0 = no vignette, 1 = strong vignette
            strength = np.clip(strength, 0, 1)
        else:
            strength = self.DEFAULT_VIGNETTE_STRENGTH

        return {
            'strength': float(strength),
            'enabled': True if strength > 0.05 else False,
            'center': [w//2, h//2],
            'radius': max(h, w) / 2
        }

    def _calibrate_chromatic(self, images: List[np.ndarray]) -> Dict:
        """
        Calibrate chromatic aberration correction.

        Analyzes RGB channel misalignment.
        """
        if not images or len(images[0].shape) != 3:
            return {'enabled': False}

        # For now, use default parameters
        # Real calibration would analyze edge color fringing
        return {
            'enabled': False,  # Chromatic aberration correction is complex
            'r_offset': [0, 0],
            'b_offset': [0, 0]
        }

    def save_calibration(self, params: Optional[Dict] = None):
        """Save calibration parameters to JSON file."""
        params = params or self.params

        # Create parent directory if needed
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.calibration_path, 'w') as f:
            json.dump(params, f, indent=2)

        print(f"Calibration saved to {self.calibration_path}")

    def load_calibration(self) -> Dict:
        """Load calibration parameters from JSON file."""
        if not self.calibration_path.exists():
            print(f"No calibration file found at {self.calibration_path}")
            self.is_calibrated = False
            return {}

        with open(self.calibration_path, 'r') as f:
            self.params = json.load(f)

        self.is_calibrated = True
        print(f"Calibration loaded from {self.calibration_path}")
        print(f"Calibrated on: {self.params.get('calibration_date', 'unknown')}")

        return self.params

    def clean(self, image: np.ndarray) -> np.ndarray:
        """
        Apply global corrections to full image.

        Args:
            image: Input image (numpy array, BGR or grayscale)

        Returns:
            Corrected image
        """
        if not self.is_calibrated:
            print("Warning: No calibration loaded. Applying default corrections.")
            return self._apply_default_corrections(image)

        corrected = image.copy()

        # Apply corrections in order
        corrected = self._correct_barrel_distortion(corrected)
        corrected = self._correct_perspective(corrected)
        corrected = self._correct_vignette(corrected)
        corrected = self._correct_chromatic(corrected)

        return corrected

    def _correct_barrel_distortion(self, image: np.ndarray) -> np.ndarray:
        """Correct barrel (fisheye) distortion."""
        params = self.params.get('barrel_distortion', {})
        '''
        k1 = params.get('k1', 0.5)
        k2 = params.get('k2', 0.5)
        k3 = params.get('k3', 0.5)
        '''
        k1 = 0.4
        k2 = 0.4
        k3 = 0.4

        # If no distortion, return original
        if abs(k1) < 1e-6 and abs(k2) < 1e-6 and abs(k3) < 1e-6:
            print("no distortion found, return original image")
            return image

        h, w = image.shape[:2]

        # Camera matrix
        fx = params.get('fx', max(h, w))
        fy = params.get('fy', max(h, w))
        cx = params.get('cx', w / 2)
        cy = params.get('cy', h / 2)

        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        dist_coeffs = np.array([k1, k2, 0, 0, k3], dtype=np.float32)

        # Undistort image
        corrected = cv2.undistort(image, camera_matrix, dist_coeffs)

        return corrected

    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """Correct perspective distortion."""
        params = self.params.get('perspective', {})

        if not params.get('enabled', False):
            return image

        transform_matrix = params.get('transform_matrix')
        if transform_matrix is None:
            return image

        h, w = image.shape[:2]
        transform_matrix = np.array(transform_matrix, dtype=np.float32)

        corrected = cv2.warpPerspective(image, transform_matrix, (w, h))

        return corrected

    def _correct_vignette(self, image: np.ndarray) -> np.ndarray:
        """Correct vignette (radial lighting falloff)."""
        params = self.params.get('vignette', {})

        if not params.get('enabled', False):
            return image

        strength = params.get('strength', self.DEFAULT_VIGNETTE_STRENGTH)
        if strength < 0.05:  # Negligible vignette
            return image

        h, w = image.shape[:2]
        center_x = params.get('center', [w//2, h//2])[0]
        center_y = params.get('center', [w//2, h//2])[1]
        radius = params.get('radius', max(h, w) / 2)

        # Create vignette correction mask
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # Normalize distance
        dist_normalized = dist_from_center / radius
        dist_normalized = np.clip(dist_normalized, 0, 1)

        # Create correction curve (inverse of typical vignette falloff)
        # Vignette typically follows: intensity = max * (1 - strength * dist^2)
        # So correction is: multiply by (1 + strength * dist^2)
        correction = 1.0 + strength * (dist_normalized ** 2)

        # Apply correction
        if len(image.shape) == 3:
            correction = correction[:, :, np.newaxis]

        corrected = image.astype(np.float32) * correction
        corrected = np.clip(corrected, 0, 255).astype(image.dtype)

        return corrected

    def _correct_chromatic(self, image: np.ndarray) -> np.ndarray:
        """Correct chromatic aberration."""
        params = self.params.get('chromatic_aberration', {})

        if not params.get('enabled', False) or len(image.shape) != 3:
            return image

        # Extract channels
        b, g, r = cv2.split(image)

        # Get offsets
        r_offset = params.get('r_offset', [0, 0])
        b_offset = params.get('b_offset', [0, 0])

        # Create shift matrices
        h, w = image.shape[:2]
        M_r = np.float32([[1, 0, r_offset[0]], [0, 1, r_offset[1]]])
        M_b = np.float32([[1, 0, b_offset[0]], [0, 1, b_offset[1]]])

        # Shift channels
        r_shifted = cv2.warpAffine(r, M_r, (w, h))
        b_shifted = cv2.warpAffine(b, M_b, (w, h))

        # Merge back
        corrected = cv2.merge([b_shifted, g, r_shifted])

        return corrected

    def _apply_default_corrections(self, image: np.ndarray) -> np.ndarray:
        """Apply default corrections when no calibration is available."""
        # Apply gentle vignette correction with default strength
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        radius = max(h, w) / 2

        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        dist_normalized = np.clip(dist_from_center / radius, 0, 1)

        correction = 1.0 + self.DEFAULT_VIGNETTE_STRENGTH * (dist_normalized ** 2)

        if len(image.shape) == 3:
            correction = correction[:, :, np.newaxis]

        corrected = image.astype(np.float32) * correction
        corrected = np.clip(corrected, 0, 255).astype(image.dtype)

        return corrected

    def get_calibration_info(self) -> str:
        """Get human-readable calibration information."""
        if not self.is_calibrated:
            return "No calibration loaded"

        info = []
        info.append(f"Calibration Date: {self.params.get('calibration_date', 'unknown')}")
        info.append(f"Image Shape: {self.params.get('image_shape', 'unknown')}")

        barrel = self.params.get('barrel_distortion', {})
        info.append(f"Barrel Distortion: k1={barrel.get('k1', 0):.4f}")

        vignette = self.params.get('vignette', {})
        info.append(f"Vignette: strength={vignette.get('strength', 0):.2f}, enabled={vignette.get('enabled', False)}")

        return "\n".join(info)
