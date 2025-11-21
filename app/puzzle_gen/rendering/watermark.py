"""Watermark functionality for embedding metadata in images."""
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class Watermark:
    """Adds metadata watermark to puzzle images."""

    def __init__(
        self,
        position: str = 'bottom-right',
        font_size: int = 12,
        text_color: tuple = (128, 128, 128),
        padding: int = 10
    ):
        """
        Initialize watermark.

        Args:
            position: Corner position ('bottom-right', 'bottom-left', 'top-right', 'top-left')
            font_size: Font size in points
            text_color: RGB color for text
            padding: Padding from corner in pixels
        """
        self.position = position
        self.font_size = font_size
        self.text_color = text_color
        self.padding = padding

        # Try to load a default font, fall back to PIL default
        try:
            self.font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            try:
                # Try common Linux font
                self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                # Fall back to PIL default
                self.font = ImageFont.load_default()

    def _create_text_lines(
        self,
        seed: int,
        piece_count: int,
        layout: str,
        cut_types: List[str],
        additional_info: Optional[dict] = None
    ) -> List[str]:
        """
        Create watermark text lines.

        Args:
            seed: Random seed used
            piece_count: Number of puzzle pieces
            layout: Layout string (e.g., "2x3")
            cut_types: List of cut types used
            additional_info: Optional additional parameters

        Returns:
            List of text lines
        """
        lines = [
            f"Seed: {seed}",
            f"Pieces: {piece_count} ({layout})",
            f"Cuts: {', '.join(cut_types)}",
        ]

        if additional_info:
            for key, value in additional_info.items():
                lines.append(f"{key}: {value}")

        return lines

    def _calculate_position(
        self,
        img_width: int,
        img_height: int,
        text_bbox: tuple
    ) -> tuple:
        """
        Calculate watermark position based on corner selection.

        Args:
            img_width: Image width
            img_height: Image height
            text_bbox: Bounding box of text (left, top, right, bottom)

        Returns:
            (x, y) position for text
        """
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        if self.position == 'bottom-right':
            x = img_width - text_width - self.padding
            y = img_height - text_height - self.padding
        elif self.position == 'bottom-left':
            x = self.padding
            y = img_height - text_height - self.padding
        elif self.position == 'top-right':
            x = img_width - text_width - self.padding
            y = self.padding
        elif self.position == 'top-left':
            x = self.padding
            y = self.padding
        else:
            # Default to bottom-right
            x = img_width - text_width - self.padding
            y = img_height - text_height - self.padding

        return (x, y)

    def apply(
        self,
        image: Image.Image,
        seed: int,
        piece_count: int,
        layout: str,
        cut_types: List[str],
        additional_info: Optional[dict] = None
    ) -> Image.Image:
        """
        Apply watermark to image.

        Args:
            image: PIL Image to watermark
            seed: Random seed used
            piece_count: Number of puzzle pieces
            layout: Layout string
            cut_types: Cut types used
            additional_info: Optional additional info

        Returns:
            Watermarked PIL Image
        """
        # Create a copy to avoid modifying original
        img = image.copy()
        draw = ImageDraw.Draw(img)

        # Create text
        lines = self._create_text_lines(seed, piece_count, layout, cut_types, additional_info)
        text = "\n".join(lines)

        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=self.font)

        # Calculate position
        position = self._calculate_position(img.width, img.height, bbox)

        # Draw semi-transparent background rectangle
        bg_padding = 5
        bg_bbox = (
            position[0] - bg_padding,
            position[1] - bg_padding,
            position[0] + (bbox[2] - bbox[0]) + bg_padding,
            position[1] + (bbox[3] - bbox[1]) + bg_padding
        )

        # Create overlay for semi-transparent background
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Draw background rectangle with transparency
        overlay_draw.rectangle(bg_bbox, fill=(255, 255, 255, 200))

        # Composite overlay onto image
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        img = Image.alpha_composite(img, overlay)

        # Draw text
        draw = ImageDraw.Draw(img)
        draw.text(position, text, fill=self.text_color, font=self.font)

        return img

    def apply_to_array(
        self,
        image_array: np.ndarray,
        seed: int,
        piece_count: int,
        layout: str,
        cut_types: List[str],
        additional_info: Optional[dict] = None
    ) -> np.ndarray:
        """
        Apply watermark to numpy array.

        Args:
            image_array: Numpy array (H, W, 3) with RGB values
            seed: Random seed
            piece_count: Number of pieces
            layout: Layout string
            cut_types: Cut types used
            additional_info: Optional additional info

        Returns:
            Watermarked numpy array
        """
        # Convert to PIL Image
        img = Image.fromarray(image_array.astype('uint8'))

        # Apply watermark
        watermarked = self.apply(img, seed, piece_count, layout, cut_types, additional_info)

        # Convert back to RGB if needed
        if watermarked.mode == 'RGBA':
            watermarked = watermarked.convert('RGB')

        # Return as array
        return np.array(watermarked)
