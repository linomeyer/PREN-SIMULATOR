import cv2
import numpy as np
import os
from typing import List
from app.main.puzzle_solver.piece_extraction.extractor import PieceSegmenter
from app.main.puzzle_solver.edge_detection.edge_detector import EdgeDetector


class EdgeVisualizer:
    """Visualizes detected edges of puzzle pieces."""

    def __init__(self, output_dir: str = 'app/static/output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def visualize_piece_edges(self, segmenter: PieceSegmenter,
                              edge_detector: EdgeDetector,
                              original_filename: str) -> List[str]:
        """
        Create visualization of each piece with its edges highlighted.

        Args:
            segmenter: PieceSegmenter with extracted pieces
            edge_detector: EdgeDetector with detected edges
            original_filename: Original filename for naming output

        Returns:
            List of output filenames
        """
        edge_filenames = []

        # Define colors for different edge types
        edge_colors = {
            'top': (255, 0, 0, 255),  # Red
            'right': (0, 255, 0, 255),  # Green
            'bottom': (0, 0, 255, 255),  # Blue
            'left': (255, 255, 0, 255)  # Yellow
        }

        for idx, piece in enumerate(segmenter.pieces):
            # Get piece image
            piece_img = piece.image.copy()

            # Get edges for this piece
            piece_edges = edge_detector.piece_edges.get(idx, {})

            # Adjust coordinates to piece's local coordinate system
            x, y, w, h = piece.bbox

            # Draw each edge
            for edge_type, edge in piece_edges.items():
                color = edge_colors.get(edge_type, (255, 255, 255, 255))

                # Adjust edge points to local coordinates
                local_points = edge.points - [x, y]
                local_points = local_points.astype(np.int32)

                # Draw edge line
                cv2.polylines(piece_img, [local_points], False, color, 3)

                # Draw corner points
                start_local = (edge.start_point[0] - x, edge.start_point[1] - y)
                end_local = (edge.end_point[0] - x, edge.end_point[1] - y)
                cv2.circle(piece_img, start_local, 5, color, -1)
                cv2.circle(piece_img, end_local, 5, color, -1)

                # Add edge label
                mid_point = local_points[len(local_points) // 2][0]
                cv2.putText(piece_img, edge_type, tuple(mid_point),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            # Save image
            edge_filename = f"edges_{idx}_{original_filename}"
            edge_path = os.path.join(self.output_dir, edge_filename)
            cv2.imwrite(edge_path, piece_img)

            edge_filenames.append(edge_filename)

        return edge_filenames