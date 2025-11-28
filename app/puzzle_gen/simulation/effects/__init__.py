"""Camera simulation effects package."""

# Base class
from .base import CameraEffect

# Optical effects
from .optical import (
    BarrelDistortionEffect,
    LensSoftnessEffect,
)

# Chromatic effects
from .chromatic import (
    ChromaticAberrationEffect,
    PurpleFringingEffect,
    ColorNoiseEffect,
)

# Noise effects
from .noise import (
    CombinedNoiseEffect,
    GaussianNoiseEffect,
    SaltPepperNoiseEffect,
)

# Geometric effects
from .geometric import (
    VignetteEffect,
    PerspectiveDistortionEffect,
)

# Processing effects
from .processing import (
    OversharpeningEffect,
    NoiseReductionEffect,
)

__all__ = [
    # Base
    'CameraEffect',
    # Optical
    'BarrelDistortionEffect',
    'LensSoftnessEffect',
    # Chromatic
    'ChromaticAberrationEffect',
    'PurpleFringingEffect',
    'ColorNoiseEffect',
    # Noise
    'CombinedNoiseEffect',
    'GaussianNoiseEffect',
    'SaltPepperNoiseEffect',
    # Geometric
    'VignetteEffect',
    'PerspectiveDistortionEffect',
    # Processing
    'OversharpeningEffect',
    'NoiseReductionEffect',
]
