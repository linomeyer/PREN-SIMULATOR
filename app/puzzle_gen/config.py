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

    # Shadows (realistic 3D effect)
    shadow_enabled: bool = True  # Enable/disable shadow rendering
    shadow_offset_x: int = 30  # Shadow offset in X direction (pixels) - INCREASED
    shadow_offset_y: int = 30  # Shadow offset in Y direction (pixels) - INCREASED
    shadow_blur_radius: int = 35  # Gaussian blur radius for soft shadows - INCREASED
    shadow_opacity: float = 0.7  # Shadow opacity (0-1) - INCREASED
    shadow_render_scale: float = 0.5  # Scale for shadow rendering (0.25-1.0, lower=faster)

    # Cut parameters
    cut_depth_ratio: float = 0.075  # Cut depth as ratio of edge length (5-10%)
    wave_frequency: int = 2  # Number of waves for wavy cuts


@dataclass
class CameraConfig:
    """Configuration for camera simulation effects (cheap smartphone camera - EXTREME MODE)."""
    # Overall intensity control (0-1, where 1 is realistic cheap smartphone)
    intensity: float = 1.0

    # Performance optimization
    use_optimized_noise: bool = True  # Use combined noise effect (faster)

    # Distortions - DRAMATICALLY INCREASED
    fisheye_strength: float = 0.65  # Barrel distortion strength (EXTREME: was 0.35)
    perspective_strength: float = 0.45  # Perspective distortion from camera angle (EXTREME: was 0.15)
    lens_softness: float = 0.6  # Edge softness/blur (EXTREME: was 0.2)

    # Chromatic effects - DRAMATICALLY INCREASED
    color_aberration: float = 0.25  # Chromatic aberration (EXTREME: was 0.08)
    color_noise: float = 0.35  # Color noise in BW scene (EXTREME: was 0.12)
    purple_fringing_intensity: float = 0.75  # Purple fringing at high-contrast edges (EXTREME: was 0.25)

    # Noise - DRAMATICALLY INCREASED
    noise_amount: float = 0.25  # Gaussian + salt-pepper noise (EXTREME: was 0.08)
    vignette_strength: float = 0.5  # Radial lighting gradient (EXTREME: was 0.2)

    # Smartphone-specific processing artifacts - DRAMATICALLY INCREASED
    oversharpening_amount: float = 1.8  # Aggressive software sharpening (EXTREME: was 0.8, max 2.0)
    noise_reduction_strength: float = 0.7  # Detail loss from noise reduction (EXTREME: was 0.3)

    def get_scaled_params(self) -> dict[str, float]:
        """Get effect parameters scaled by overall intensity."""
        return {
            # Distortions
            'fisheye': self.fisheye_strength * self.intensity,
            'perspective': self.perspective_strength * self.intensity,
            'lens_softness': self.lens_softness * self.intensity,
            # Chromatic
            'aberration': self.color_aberration * self.intensity,
            'color_noise': self.color_noise * self.intensity,
            'purple_fringing': self.purple_fringing_intensity * self.intensity,
            # Noise
            'noise': self.noise_amount * self.intensity,
            'vignette': self.vignette_strength * self.intensity,
            # Smartphone processing
            'oversharpening': self.oversharpening_amount * self.intensity,
            'noise_reduction': self.noise_reduction_strength * self.intensity,
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


# Performance monitoring global flag (outside dataclass to make it a true class variable)
GeneratorConfig.enable_performance_logging = False
