"""Configuration dataclasses for puzzle generation."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


class Layout(Enum):
    """Puzzle layout options."""
    LAYOUT_2X3 = "2x3"  # 2 rows, 3 columns
    LAYOUT_3X2 = "3x2"  # 3 rows, 2 columns


class CutType(Enum):
    """Available cut types for puzzle edges."""
    WAVY = "wavy"
    KNOB_SQUARE = "knob_square"
    KNOB_TRIANGULAR = "knob_triangular"
    KNOB_WAVY = "knob_wavy"
    CHEVRON = "chevron"
    SINGLE_WAVE = "single_wave"
    PARTIAL_WAVE = "partial_wave"
    DOUBLE_CHEVRON = "double_chevron"
    KNOB_SQUARE_NEW = "knob_square_new"
    KNOB_TRAPEZOID = "knob_trapezoid"
    KNOB_ROUND = "knob_round"
    VAR_AMP_INCREASING = "var_amp_increasing"
    VAR_AMP_CHAOTIC = "var_amp_chaotic"


@dataclass
class PuzzleConfig:
    """Puzzle geometry and piece configuration."""
    # A5 dimensions at 300 DPI
    base_width_mm: float = 210.0  # A5 width in mm
    base_height_mm: float = 148.0  # A5 height in mm
    dpi: int = 300  # DPI for mm to pixel conversion

    # Puzzle structure
    layout: Layout = Layout.LAYOUT_2X3
    piece_count: int = 6  # Total pieces (must match layout)

    # Randomization
    seed: int | None = None  # Random seed for reproducibility

    @property
    def base_width_px(self) -> int:
        """Base width in pixels."""
        return int(self.base_width_mm / 25.4 * self.dpi)

    @property
    def base_height_px(self) -> int:
        """Base height in pixels."""
        return int(self.base_height_mm / 25.4 * self.dpi)

    @property
    def grid_rows(self) -> int:
        """Number of rows in the grid."""
        return int(self.layout.value.split('x')[0])

    @property
    def grid_cols(self) -> int:
        """Number of columns in the grid."""
        return int(self.layout.value.split('x')[1])


@dataclass
class RenderConfig:
    """Configuration for image rendering."""
    # Spacing and margins
    margin_px: int = 100  # Margin around entire image
    piece_spacing_px: int = 75  # Minimum spacing between pieces

    # Background
    background_color: Tuple[int, int, int] = (255, 255, 255)  # White
    background_noise_intensity: float = 0.02  # Subtle texture (0-1)

    # Puzzle pieces
    piece_color: Tuple[int, int, int] = (0, 0, 0)  # Black

    # Cut parameters
    cut_depth_ratio: float = 0.075  # Cut depth as ratio of edge length (5-10%)
    wave_frequency: int = 2  # Number of waves for wavy cuts


@dataclass
class CameraConfig:
    """Configuration for camera simulation effects."""
    # Overall intensity control (0-1, where 1 is realistic RasPi v3)
    intensity: float = 1.0

    # Individual effect parameters (0-1 each)
    fisheye_strength: float = 0.15  # Barrel distortion strength
    noise_amount: float = 0.08  # Gaussian + salt-pepper noise (increased from 0.03)
    vignette_strength: float = 0.2  # Radial lighting gradient
    color_aberration: float = 0.02  # Chromatic aberration
    color_noise: float = 0.03  # Color noise in BW scene (increased from 0.01)

    def get_scaled_params(self) -> dict[str, float]:
        """Get effect parameters scaled by overall intensity."""
        return {
            'fisheye': self.fisheye_strength * self.intensity,
            'noise': self.noise_amount * self.intensity,
            'vignette': self.vignette_strength * self.intensity,
            'aberration': self.color_aberration * self.intensity,
            'color_noise': self.color_noise * self.intensity,
        }


@dataclass
class OutputConfig:
    """Configuration for output files."""
    output_dir: str = "./output"
    max_width: int = 4608  # Max resolution width (RasPi Camera v3)
    max_height: int = 2592  # Max resolution height
    image_format: str = "png"  # Output format (png or jpg)
    jpeg_quality: int = 70  # JPEG quality if format is jpg (70 = good compromise)

    # File naming
    filename_prefix: str = "puzzle"

    def get_clean_filename(self, seed: int) -> str:
        """Generate filename for clean image."""
        return f"{self.filename_prefix}_seed{seed}_clean.{self.image_format}"

    def get_noisy_filename(self, seed: int) -> str:
        """Generate filename for noisy image."""
        return f"{self.filename_prefix}_seed{seed}_noisy.{self.image_format}"

    def get_solution_filename(self, seed: int) -> str:
        """Generate filename for solution image."""
        return f"{self.filename_prefix}_seed{seed}_solution.{self.image_format}"


@dataclass
class GeneratorConfig:
    """Complete configuration for puzzle generation."""
    puzzle: PuzzleConfig = field(default_factory=PuzzleConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
