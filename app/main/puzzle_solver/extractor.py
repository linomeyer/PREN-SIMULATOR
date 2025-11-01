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

    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.pieces: List[PuzzlePiece] = []

    def segment_pieces(self, min_area: int = 1000, max_area: int = None) -> List[PuzzlePiece]:
        if max_area is None:
            max_area = (self.image.shape[0] * self.image.shape[1]) // 2

        # remove color
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # background separation threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

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
