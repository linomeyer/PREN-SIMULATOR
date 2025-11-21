"""Camera simulation orchestrator using modular effects."""
import numpy as np
from typing import List

from .effects import (
    BarrelDistortionEffect,
    PerspectiveDistortionEffect,
    ChromaticAberrationEffect,
    PurpleFringingEffect,
    VignetteEffect,
    OversharpeningEffect,
    GaussianNoiseEffect,
    ColorNoiseEffect,
    SaltPepperNoiseEffect,
    NoiseReductionEffect,
    LensSoftnessEffect,
    CameraEffect,
)


class CameraSimulator:
    """Orchestrates camera simulation effects in realistic pipeline order."""

    def __init__(
        self,
        fisheye_strength: float = 0.5,
        perspective_strength: float = 0.4,
        lens_softness: float = 0.5,
        noise_amount: float = 0.25,
        vignette_strength: float = 0.5,
        color_aberration: float = 0.25,
        color_noise: float = 0.35,
        purple_fringing_intensity: float = 0.5,
        oversharpening_amount: float = 1.5,
        noise_reduction_strength: float = 0.5
    ):
        """
        Initialize camera simulator with effect pipeline.

        Args:
            fisheye_strength: Barrel distortion strength (0-1)
            perspective_strength: Perspective distortion from camera angle (0-1)
            lens_softness: Edge softness/blur (0-1)
            noise_amount: Gaussian + salt-pepper noise (0-1)
            vignette_strength: Radial lighting gradient (0-1)
            color_aberration: Chromatic aberration (0-1)
            color_noise: Color noise in BW scene (0-1)
            purple_fringing_intensity: Purple fringing at edges (0-1)
            oversharpening_amount: Aggressive software sharpening (0-2)
            noise_reduction_strength: Detail loss from noise reduction (0-1)
        """
        # Create effect pipeline in realistic order
        # Order simulates cheap smartphone camera:
        # 1. Optical distortions (lens) → 2. Chromatic effects →
        # 3. Lighting → 4. Processing (sharpen) → 5. Sensor noise →
        # 6. Processing (denoise) → 7. Final optical (softness)

        self.effects: List[CameraEffect] = [
            # 1. Barrel distortion (fisheye) - lens optical effect
            BarrelDistortionEffect(fisheye_strength),

            # 2. Perspective distortion - camera angle
            PerspectiveDistortionEffect(perspective_strength),

            # 3. Chromatic aberration - lens optical effect
            ChromaticAberrationEffect(color_aberration),

            # 4. Purple fringing - cheap lens artifact at high-contrast edges
            PurpleFringingEffect(purple_fringing_intensity),

            # 5. Vignette - lighting falloff
            VignetteEffect(vignette_strength),

            # 6. Oversharpening - aggressive software processing
            OversharpeningEffect(oversharpening_amount),

            # 7. Gaussian noise - sensor noise
            GaussianNoiseEffect(noise_amount),

            # 8. Color noise - color sensor imperfections
            ColorNoiseEffect(color_noise),

            # 9. Salt and pepper noise - dead/hot pixels
            SaltPepperNoiseEffect(noise_amount, amount=0.002),

            # 10. Aggressive noise reduction - software artifact causing detail loss
            NoiseReductionEffect(noise_reduction_strength),

            # 11. Lens softness - cheap lens quality, blurry edges
            LensSoftnessEffect(lens_softness),
        ]

    def simulate(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all camera simulation effects in pipeline order.

        Args:
            image: Input image array (H, W, 3) RGB uint8

        Returns:
            Simulated camera image with all effects applied
        """
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Apply effects sequentially through pipeline
        result = image.copy()
        for effect in self.effects:
            result = effect.apply(result)

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
            fisheye_strength=config_params.get('fisheye', 0.65),
            perspective_strength=config_params.get('perspective', 0.45),
            lens_softness=config_params.get('lens_softness', 0.6),
            noise_amount=config_params.get('noise', 0.25),
            vignette_strength=config_params.get('vignette', 0.5),
            color_aberration=config_params.get('aberration', 0.25),
            color_noise=config_params.get('color_noise', 0.35),
            purple_fringing_intensity=config_params.get('purple_fringing', 0.75),
            oversharpening_amount=config_params.get('oversharpening', 1.8),
            noise_reduction_strength=config_params.get('noise_reduction', 0.7)
        )
