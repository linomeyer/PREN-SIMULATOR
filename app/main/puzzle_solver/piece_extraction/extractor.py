from typing import List, Dict

import cv2
import numpy as np
from cv2 import Mat

class PuzzlePiece:

    def __init__(self, contour, mask, bbox, image):
        self.contour = contour
        self.mask = mask
        self.bbox = bbox
        self.image = image
        self.center = self._calculate_center()

    def _calculate_center(self):
        M = cv2.moments(self.contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
        return 0, 0


class PieceSegmenter:

    # Configurable segmentation parameters
    BLUR_KERNEL_SIZE = 7  # Gaussian blur kernel size (was hardcoded 5)
    MORPH_KERNEL_SIZE = 5  # Morphological operations kernel size (was hardcoded 3)
    MORPH_CLOSE_ITERATIONS = 3  # Closing iterations to fill gaps (was hardcoded 2)
    MORPH_OPEN_ITERATIONS = 1  # Opening iterations to remove noise

    def __init__(self, image_path: str = None, image_array: np.ndarray = None,
                 blur_kernel: int = None,
                 morph_kernel: int = None,
                 morph_close_iter: int = None,
                 morph_open_iter: int = None):
        if image_array is not None:
            # Use provided numpy array
            self.image = image_array
            if len(self.image.shape) == 3 and self.image.shape[2] == 3:
                # Assume BGR format if 3 channels
                self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            else:
                self.image_rgb = self.image
        elif image_path is not None:
            # Load from file path
            self.image = cv2.imread(image_path)
            if self.image is None:
                raise ValueError(f"Could not load image from {image_path}")
            self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Either image_path or image_array must be provided")

        # Store segmentation parameters (use provided or defaults)
        self.blur_kernel = blur_kernel if blur_kernel is not None else self.BLUR_KERNEL_SIZE
        self.morph_kernel = morph_kernel if morph_kernel is not None else self.MORPH_KERNEL_SIZE
        self.morph_close_iter = morph_close_iter if morph_close_iter is not None else self.MORPH_CLOSE_ITERATIONS
        self.morph_open_iter = morph_open_iter if morph_open_iter is not None else self.MORPH_OPEN_ITERATIONS

        self.pieces: List[PuzzlePiece] = []

    def segment_pieces(self, min_area: int = 1000, max_area: int = None) -> List[PuzzlePiece]:
        if max_area is None:
            max_area = (self.image.shape[0] * self.image.shape[1]) // 2

        # remove color
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # noise reduction (configurable kernel size)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        # background separation threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # clean up (configurable kernel and iterations)
        kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=self.morph_close_iter)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=self.morph_open_iter)

        # find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # create PuzzlePiece objects
        self.pieces = []
        for contour in contours:
            puzzle_piece = self._create_puzzle_piece(contour, gray, min_area, max_area)
            if puzzle_piece is not None:
                self.pieces.append(puzzle_piece)

        print(f"Found {len(self.pieces)} puzzle pieces")
        return self.pieces

    def _create_puzzle_piece(self, contour: Mat, gray_scale: Mat, min_area: int, max_area: int) -> PuzzlePiece | None:
        area = cv2.contourArea(contour)

        # filter out unlogically large or small piece
        if area < min_area or area > max_area:
            return None

        mask = np.zeros(gray_scale.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # create a bounding box of the piece
        x, y, w, h = cv2.boundingRect(contour)

        # create a new image of only the piece with transparent background
        piece_mask_roi = mask[y:y + h, x:x + w]
        piece_image_roi = self.image_rgb[y:y + h, x:x + w].copy()
        piece_rgba = cv2.cvtColor(piece_image_roi, cv2.COLOR_RGB2RGBA)
        piece_rgba[:, :, 3] = piece_mask_roi

        return PuzzlePiece(contour, mask, (x, y, w, h), piece_rgba)

    def get_piece_statistics(self) -> Dict:
        if not self.pieces:
            return {}

        areas = [cv2.contourArea(piece.contour) for piece in self.pieces]
        perimeters = [cv2.arcLength(piece.contour, True) for piece in self.pieces]

        stats = {
            'num_pieces': len(self.pieces),
            'avg_area': np.mean(areas),
            'std_area': np.std(areas),
            'min_area': np.min(areas),
            'max_area': np.max(areas),
            'avg_perimeter': np.mean(perimeters),
        }

        return stats
