import cv2
import os
from typing import List
from app.main.puzzle_solver.extractor import PieceSegmenter


class PieceVisualizer:

    def __init__(self, output_dir: str = 'app/static/output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def visualize_pieces(self, segmenter: PieceSegmenter, original_filename: str) -> List[str]:
        piece_filenames = []

        for idx, piece in enumerate(segmenter.pieces):
            piece_img = piece.image.copy()

            # Adjust contour coordinates to piece's local coordinate system
            x, y, w, h = piece.bbox
            local_contour = piece.contour - [x, y]

            # Draw contour on the piece using the existing contour
            piece_with_contour = piece_img.copy()
            cv2.drawContours(piece_with_contour, [local_contour], -1, (0, 255, 0, 255), 2)

            piece_filename = f"piece_{idx}_{original_filename}"
            piece_path = os.path.join(self.output_dir, piece_filename)
            cv2.imwrite(piece_path, piece_with_contour)

            piece_filenames.append(piece_filename)

        return piece_filenames