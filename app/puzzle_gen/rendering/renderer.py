"""Clean image rendering using Pillow."""
import math
from typing import List, Tuple
import random

from PIL import Image, ImageDraw, ImageFilter
import numpy as np

from ..geometry.puzzle import PuzzlePiece


class PuzzleRenderer:
    """Renders puzzle pieces to a clean image."""

    def __init__(
        self,
        width: int,
        height: int,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        piece_color: Tuple[int, int, int] = (0, 0, 0),
        background_noise: float = 0.02,
        shadow_enabled: bool = True,
        shadow_offset_x: int = 15,
        shadow_offset_y: int = 15,
        shadow_blur_radius: int = 20,
        shadow_opacity: float = 0.4
    ):
        """
        Initialize renderer.

        Args:
            width: Canvas width
            height: Canvas height
            background_color: RGB background color
            piece_color: RGB piece color
            background_noise: Background texture noise intensity (0-1)
            shadow_enabled: Enable shadow rendering
            shadow_offset_x: Shadow offset in X direction (pixels)
            shadow_offset_y: Shadow offset in Y direction (pixels)
            shadow_blur_radius: Gaussian blur radius for soft shadows
            shadow_opacity: Shadow opacity (0-1)
        """
        self.width = width
        self.height = height
        self.background_color = background_color
        self.piece_color = piece_color
        self.background_noise = background_noise
        self.shadow_enabled = shadow_enabled
        self.shadow_offset_x = shadow_offset_x
        self.shadow_offset_y = shadow_offset_y
        self.shadow_blur_radius = shadow_blur_radius
        self.shadow_opacity = shadow_opacity

    def _create_background(self) -> Image.Image:
        """
        Create background with subtle texture.

        Returns:
            PIL Image with textured background
        """
        # Create base background
        img = Image.new('RGB', (self.width, self.height), self.background_color)

        if self.background_noise > 0:
            # Add subtle noise texture
            img_array = np.array(img, dtype=np.float32)

            # Generate noise
            noise = np.random.normal(0, 255 * self.background_noise, img_array.shape)
            img_array = img_array + noise

            # Clip to valid range
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_array)

        return img

    def _rotate_point(
        self,
        point: Tuple[float, float],
        center: Tuple[float, float],
        angle_deg: float
    ) -> Tuple[float, float]:
        """
        Rotate a point around a center.

        Args:
            point: Point to rotate (x, y)
            center: Center of rotation (cx, cy)
            angle_deg: Rotation angle in degrees

        Returns:
            Rotated point (x, y)
        """
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Translate to origin
        x = point[0] - center[0]
        y = point[1] - center[1]

        # Rotate
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a

        # Translate back
        return (x_rot + center[0], y_rot + center[1])

    def _transform_piece_points(
        self,
        piece: PuzzlePiece,
        num_points_per_edge: int = 50,
        scale_factor: float = 1.0
    ) -> List[Tuple[float, float]]:
        """
        Get piece outline points with rotation, translation, and scaling applied.

        Args:
            piece: Puzzle piece to transform
            num_points_per_edge: Points per edge
            scale_factor: Scaling factor (1.0 = no scaling)

        Returns:
            List of transformed (x, y) coordinates
        """
        # Get original path points
        points = piece.get_path_points(num_points_per_edge)

        # Calculate original center
        original_center = piece.get_original_center()

        # Apply rotation, translation, and scaling
        transformed_points = []
        for point in points:
            # Rotate around original center
            rotated = self._rotate_point(point, original_center, piece.rotation)

            # Translate to new center
            if piece.center is not None:
                # Translate
                translated_x = rotated[0] - original_center[0] + piece.center[0]
                translated_y = rotated[1] - original_center[1] + piece.center[1]

                # Scale around piece center
                if scale_factor != 1.0:
                    dx = translated_x - piece.center[0]
                    dy = translated_y - piece.center[1]
                    scaled_x = piece.center[0] + dx * scale_factor
                    scaled_y = piece.center[1] + dy * scale_factor
                    transformed_points.append((scaled_x, scaled_y))
                else:
                    transformed_points.append((translated_x, translated_y))
            else:
                transformed_points.append(rotated)

        return transformed_points

    def _render_shadow(
        self,
        piece: PuzzlePiece,
        num_points_per_edge: int,
        scale_factor: float
    ) -> Image.Image:
        """
        Render a soft shadow for a single piece.

        Args:
            piece: Puzzle piece to render shadow for
            num_points_per_edge: Points per edge
            scale_factor: Scaling factor

        Returns:
            PIL Image with shadow (RGBA, transparent background)
        """
        # Create transparent canvas for shadow
        shadow_img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_img)

        # Get transformed outline with offset
        outline = self._transform_piece_points(piece, num_points_per_edge, scale_factor)

        # Apply shadow offset
        offset_outline = [
            (x + self.shadow_offset_x, y + self.shadow_offset_y)
            for x, y in outline
        ]

        # Calculate shadow color with opacity
        shadow_alpha = int(255 * self.shadow_opacity)
        shadow_color = (0, 0, 0, shadow_alpha)

        # Draw shadow polygon
        shadow_draw.polygon(offset_outline, fill=shadow_color)

        # Apply Gaussian blur for soft edges
        if self.shadow_blur_radius > 0:
            shadow_img = shadow_img.filter(
                ImageFilter.GaussianBlur(radius=self.shadow_blur_radius)
            )

        return shadow_img

    def render(
        self,
        pieces: List[PuzzlePiece],
        num_points_per_edge: int = 100,
        scale_factor: float = 1.0
    ) -> Image.Image:
        """
        Render puzzle pieces to image.

        Args:
            pieces: List of puzzle pieces
            num_points_per_edge: Points to generate per edge (higher = smoother)
            scale_factor: Scaling factor for pieces (1.0 = no scaling)

        Returns:
            PIL Image with rendered puzzle
        """
        # Create background
        img = self._create_background()

        # Render shadows first (if enabled)
        if self.shadow_enabled:
            for piece in pieces:
                shadow = self._render_shadow(piece, num_points_per_edge, scale_factor)
                # Composite shadow onto background using alpha blending
                img = Image.alpha_composite(img.convert('RGBA'), shadow).convert('RGB')

        # Create drawing context
        draw = ImageDraw.Draw(img, 'RGBA')

        # Draw each piece on top of shadows
        for piece in pieces:
            # Get transformed outline with scaling
            outline = self._transform_piece_points(piece, num_points_per_edge, scale_factor)

            # Draw filled polygon
            # Pillow expects flat list of coordinates
            flat_coords = [(x, y) for x, y in outline]

            # Draw with anti-aliasing
            draw.polygon(flat_coords, fill=self.piece_color, outline=self.piece_color)

        return img

    def render_to_file(
        self,
        pieces: List[PuzzlePiece],
        output_path: str,
        image_format: str = 'png',
        jpeg_quality: int = 95
    ):
        """
        Render puzzle and save to file.

        Args:
            pieces: List of puzzle pieces
            output_path: Output file path
            image_format: 'png' or 'jpg'
            jpeg_quality: JPEG quality (1-100)
        """
        img = self.render(pieces)

        if image_format.lower() == 'jpg' or image_format.lower() == 'jpeg':
            # Convert to RGB for JPEG (no alpha channel)
            img = img.convert('RGB')
            img.save(output_path, 'JPEG', quality=jpeg_quality, optimize=True)
        else:
            # Convert to palette mode to reduce file size (8 bit instead of 24 bit)
            img = img.convert('P', palette=Image.ADAPTIVE, colors=256)
            img.save(output_path, 'PNG', optimize=True, compress_level=9)

    def get_image_array(self, pieces: List[PuzzlePiece], scale_factor: float = 1.0) -> np.ndarray:
        """
        Render puzzle and return as numpy array (for camera simulation).

        Args:
            pieces: List of puzzle pieces
            scale_factor: Scaling factor for pieces (1.0 = no scaling)

        Returns:
            Numpy array (H, W, 3) with RGB values
        """
        img = self.render(pieces, scale_factor=scale_factor)
        return np.array(img)

    def render_debug(
        self,
        pieces: List[PuzzlePiece],
        grid_info: dict,
        num_points_per_edge: int = 100
    ) -> Image.Image:
        """
        Render puzzle with debug visualization of placement.

        Args:
            pieces: List of puzzle pieces
            grid_info: Dictionary with grid layout info (grid_cols, grid_rows,
                      cell_width, cell_height, grid_start_x, grid_start_y,
                      max_piece_size, safe_radius, scale_factor)
            num_points_per_edge: Points to generate per edge

        Returns:
            PIL Image with debug visualization
        """
        # Extract scale_factor from grid_info
        scale_factor = grid_info.get('scale_factor', 1.0)
        # Create background
        img = self._create_background()
        draw = ImageDraw.Draw(img, 'RGBA')

        # Extract grid info
        grid_cols = grid_info.get('grid_cols', 3)
        grid_rows = grid_info.get('grid_rows', 2)
        cell_width = grid_info.get('cell_width', 0)
        cell_height = grid_info.get('cell_height', 0)
        grid_start_x = grid_info.get('grid_start_x', 0)
        grid_start_y = grid_info.get('grid_start_y', 0)
        max_piece_size = grid_info.get('max_piece_size', 0)
        safe_radius = grid_info.get('safe_radius', 0)

        # Load font for debug text
        from PIL import ImageFont
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            font = ImageFont.load_default()

        # 1. Draw grid cells (light blue rectangles)
        for row in range(grid_rows):
            for col in range(grid_cols):
                x1 = grid_start_x + col * cell_width
                y1 = grid_start_y + row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                draw.rectangle([x1, y1, x2, y2], outline=(100, 150, 255), width=3)

        # Draw each piece with debug info
        for piece in pieces:
            if piece.center is None:
                continue

            # 2. Draw piece center point (red dot)
            cx, cy = piece.center
            r = 10
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 0, 0))

            # 3. Draw safe radius circle (yellow)
            if safe_radius > 0:
                draw.ellipse(
                    [cx-safe_radius, cy-safe_radius, cx+safe_radius, cy+safe_radius],
                    outline=(255, 200, 0),
                    width=2
                )

            # 4. Draw rotated bounding box (green)
            bbox_w, bbox_h = piece.get_rotated_bounding_box(piece.rotation)

            # Get transformed outline to find actual bounding box (with scaling)
            outline = self._transform_piece_points(piece, num_points_per_edge, scale_factor)
            if outline:
                xs = [p[0] for p in outline]
                ys = [p[1] for p in outline]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                # Draw actual bounding box
                draw.rectangle(
                    [min_x, min_y, max_x, max_y],
                    outline=(0, 255, 0),
                    width=2
                )

            # 5. Draw theoretical max piece size circle (cyan)
            half_max = max_piece_size / 2
            draw.ellipse(
                [cx-half_max, cy-half_max, cx+half_max, cy+half_max],
                outline=(0, 255, 255),
                width=2
            )

            # 6. Draw the actual piece (black, semi-transparent)
            flat_coords = [(x, y) for x, y in outline]
            draw.polygon(flat_coords, fill=(0, 0, 0, 200), outline=(0, 0, 0))

            # 7. Draw debug text - show bounding box size
            text = f"{int(bbox_w)}x{int(bbox_h)}"
            draw.text((cx - 40, cy - 50), text, fill=(255, 0, 0), font=font)

        # Draw max_piece_size and safe_radius info at top
        info_text = f"max_piece_size: {int(max_piece_size)}, safe_radius: {int(safe_radius)}, cell: {int(cell_width)}x{int(cell_height)}, scale: {scale_factor:.3f}"
        draw.text((50, 50), info_text, fill=(255, 0, 0), font=font)

        return img
