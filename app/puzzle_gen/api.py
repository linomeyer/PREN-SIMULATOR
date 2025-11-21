"""Puzzle Generator API - Core business logic for Flask integration."""
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image

from .config import GeneratorConfig, CameraConfig
from .geometry.puzzle import PuzzleGenerator, PuzzlePiece
from .rendering.renderer import PuzzleRenderer
from .rendering.watermark import Watermark
from .simulation.camera import CameraSimulator


@dataclass
class PuzzleGenerationResult:
    """Result of puzzle generation containing all generated images and metadata."""

    # Metadata
    seed: int
    piece_count: int
    layout: str
    cut_types: list[str]
    resolution: tuple[int, int]  # (width, height)

    # Generated images (as numpy arrays)
    clean_image: np.ndarray
    solution_image: np.ndarray
    debug_image: Optional[np.ndarray] = None
    noisy_image: Optional[np.ndarray] = None

    # Debug information
    debug_info: dict = field(default_factory=dict)

    def save_all(
        self,
        output_dir: Path | str,
        image_format: str = 'png',
        jpeg_quality: int = 70,
        save_debug: bool = True
    ) -> dict[str, Path]:
        """
        Save all generated images to disk.

        Args:
            output_dir: Directory to save images to
            image_format: 'png' or 'jpg'
            jpeg_quality: JPEG quality if format is jpg
            save_debug: Whether to save debug image

        Returns:
            Dictionary mapping image type to saved file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Helper function to save image
        def save_image(array: np.ndarray, filename: str) -> Path:
            img = Image.fromarray(array)
            filepath = output_dir / filename

            if image_format == 'jpg':
                img = img.convert('RGB')
                img.save(filepath, 'JPEG', quality=jpeg_quality, optimize=True)
            else:
                # Convert to palette mode to reduce file size
                img = img.convert('P', palette=Image.ADAPTIVE, colors=256)
                img.save(filepath, 'PNG', optimize=True, compress_level=9)

            return filepath

        # Save clean image
        saved_files['clean'] = save_image(
            self.clean_image,
            f"puzzle_seed{self.seed}_clean.{image_format}"
        )

        # Save solution image
        saved_files['solution'] = save_image(
            self.solution_image,
            f"puzzle_seed{self.seed}_solution.{image_format}"
        )

        # Save noisy image if available
        if self.noisy_image is not None:
            saved_files['noisy'] = save_image(
                self.noisy_image,
                f"puzzle_seed{self.seed}_noisy.{image_format}"
            )

        # Save debug image if requested and available
        if save_debug and self.debug_image is not None:
            saved_files['debug'] = save_image(
                self.debug_image,
                f"puzzle_seed{self.seed}_debug.png"
            )

        return saved_files


def generate_puzzle_geometry(
    config: GeneratorConfig,
    cut_types: list[str]
) -> tuple[list[PuzzlePiece], int, PuzzleGenerator, dict]:
    """
    Generate puzzle geometry and pieces.

    Args:
        config: Generator configuration
        cut_types: List of cut type names to use

    Returns:
        Tuple of (pieces, seed, generator, debug_info)
    """
    # Set seed
    seed = config.puzzle.seed if config.puzzle.seed is not None else random.randint(0, 999999)

    # Create puzzle generator
    generator = PuzzleGenerator(
        width_px=config.puzzle.base_width_px,
        height_px=config.puzzle.base_height_px,
        rows=config.puzzle.grid_rows,
        cols=config.puzzle.grid_cols,
        cut_depth_ratio=config.render.cut_depth_ratio,
        seed=seed
    )

    # Generate puzzle pieces
    pieces, debug_info = generator.generate(
        cut_types=cut_types,
        canvas_width=config.output.max_width,
        canvas_height=config.output.max_height,
        margin_px=config.render.margin_px,
        spacing_px=config.render.piece_spacing_px,
        wave_frequency=config.render.wave_frequency
    )

    return pieces, seed, generator, debug_info


def render_clean_image(
    pieces: list[PuzzlePiece],
    config: GeneratorConfig,
    debug_info: dict,
    apply_watermark: bool = True,
    seed: int = 0,
    layout: str = '2x3',
    cut_types: list[str] = None
) -> np.ndarray:
    """
    Render clean puzzle image.

    Args:
        pieces: List of puzzle pieces
        config: Generator configuration
        debug_info: Debug information from puzzle generation
        apply_watermark: Whether to apply watermark
        seed: Random seed used for generation
        layout: Puzzle layout string
        cut_types: List of cut types used

    Returns:
        Rendered image as numpy array
    """
    renderer = PuzzleRenderer(
        width=config.output.max_width,
        height=config.output.max_height,
        background_color=config.render.background_color,
        piece_color=config.render.piece_color,
        background_noise=config.render.background_noise_intensity,
        shadow_enabled=config.render.shadow_enabled,
        shadow_offset_x=config.render.shadow_offset_x,
        shadow_offset_y=config.render.shadow_offset_y,
        shadow_blur_radius=config.render.shadow_blur_radius,
        shadow_opacity=config.render.shadow_opacity
    )

    scale_factor = debug_info.get('scale_factor', 1.0)
    clean_array = renderer.get_image_array(pieces, scale_factor=scale_factor)

    # Apply watermark if requested
    if apply_watermark:
        watermark = Watermark()
        clean_array = watermark.apply_to_array(
            clean_array,
            seed=seed,
            piece_count=config.puzzle.piece_count,
            layout=layout,
            cut_types=cut_types or []
        )

    return clean_array


def render_solution_image(
    generator: PuzzleGenerator,
    config: GeneratorConfig,
    apply_watermark: bool = True,
    seed: int = 0,
    layout: str = '2x3',
    cut_types: list[str] = None
) -> np.ndarray:
    """
    Render solution puzzle image (assembled).

    Args:
        generator: Puzzle generator instance
        config: Generator configuration
        apply_watermark: Whether to apply watermark
        seed: Random seed used for generation
        layout: Puzzle layout string
        cut_types: List of cut types used

    Returns:
        Rendered solution image as numpy array
    """
    # Generate solution pieces (assembled)
    solution_pieces = generator.generate_solution(
        canvas_width=config.output.max_width,
        canvas_height=config.output.max_height,
        gap_px=15  # Small gap between pieces to see the solution clearly
    )

    # Create renderer
    renderer = PuzzleRenderer(
        width=config.output.max_width,
        height=config.output.max_height,
        background_color=config.render.background_color,
        piece_color=config.render.piece_color,
        background_noise=config.render.background_noise_intensity,
        shadow_enabled=config.render.shadow_enabled,
        shadow_offset_x=config.render.shadow_offset_x,
        shadow_offset_y=config.render.shadow_offset_y,
        shadow_blur_radius=config.render.shadow_blur_radius,
        shadow_opacity=config.render.shadow_opacity
    )

    # Render solution
    solution_array = renderer.get_image_array(solution_pieces)

    # Apply watermark if requested
    if apply_watermark:
        watermark = Watermark()
        solution_array = watermark.apply_to_array(
            solution_array,
            seed=seed,
            piece_count=config.puzzle.piece_count,
            layout=layout,
            cut_types=cut_types or []
        )

    return solution_array


def render_debug_image(
    pieces: list[PuzzlePiece],
    config: GeneratorConfig,
    debug_info: dict
) -> np.ndarray:
    """
    Render debug image with piece boundaries and information.

    Args:
        pieces: List of puzzle pieces
        config: Generator configuration
        debug_info: Debug information from puzzle generation

    Returns:
        Rendered debug image as numpy array
    """
    renderer = PuzzleRenderer(
        width=config.output.max_width,
        height=config.output.max_height,
        background_color=config.render.background_color,
        piece_color=config.render.piece_color,
        background_noise=config.render.background_noise_intensity,
        shadow_enabled=config.render.shadow_enabled,
        shadow_offset_x=config.render.shadow_offset_x,
        shadow_offset_y=config.render.shadow_offset_y,
        shadow_blur_radius=config.render.shadow_blur_radius,
        shadow_opacity=config.render.shadow_opacity
    )

    debug_img = renderer.render_debug(pieces, debug_info)
    return np.array(debug_img)


def apply_camera_simulation(
    clean_image: np.ndarray,
    camera_config: CameraConfig
) -> np.ndarray:
    """
    Apply camera simulation effects to image.

    Args:
        clean_image: Clean input image as numpy array
        camera_config: Camera simulation configuration

    Returns:
        Image with camera effects applied as numpy array
    """
    camera = CameraSimulator.from_config(camera_config.get_scaled_params())
    noisy_array = camera.simulate(clean_image)
    return noisy_array


def generate_puzzle_images(
    config: GeneratorConfig,
    cut_types: list[str],
    layout: str = '2x3',
    apply_watermark: bool = True,
    apply_camera: bool = True,
    include_debug: bool = True
) -> PuzzleGenerationResult:
    """
    Generate complete puzzle with all images.

    This is the main API function that orchestrates the entire puzzle generation process.

    Args:
        config: Generator configuration
        cut_types: List of cut type names to use
        layout: Puzzle layout string (e.g., '2x3', '3x2')
        apply_watermark: Whether to apply watermark to images
        apply_camera: Whether to apply camera simulation
        include_debug: Whether to generate debug image

    Returns:
        PuzzleGenerationResult containing all generated images and metadata
    """
    # Generate puzzle geometry
    pieces, seed, generator, debug_info = generate_puzzle_geometry(config, cut_types)

    # Render clean image
    clean_array = render_clean_image(
        pieces, config, debug_info,
        apply_watermark=apply_watermark,
        seed=seed,
        layout=layout,
        cut_types=cut_types
    )

    # Render solution image
    solution_array = render_solution_image(
        generator, config,
        apply_watermark=apply_watermark,
        seed=seed,
        layout=layout,
        cut_types=cut_types
    )

    # Render debug image if requested
    debug_array = None
    if include_debug:
        debug_array = render_debug_image(pieces, config, debug_info)

    # Apply camera simulation if requested
    noisy_array = None
    if apply_camera:
        noisy_array = apply_camera_simulation(clean_array, config.camera)

    # Create result object
    result = PuzzleGenerationResult(
        seed=seed,
        piece_count=config.puzzle.piece_count,
        layout=layout,
        cut_types=cut_types,
        resolution=(config.output.max_width, config.output.max_height),
        clean_image=clean_array,
        solution_image=solution_array,
        debug_image=debug_array,
        noisy_image=noisy_array,
        debug_info=debug_info
    )

    return result
