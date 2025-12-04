import os
from typing import List, Optional, Dict, Tuple

import cv2
import numpy as np

from app.main.puzzle_solver.piece_extraction.extractor import PuzzlePiece
from app.main.puzzle_solver.solver.solver import PuzzleSolution, PlacedPiece
from app.main.puzzle_solver.edge_detection.edge_detector import EdgeDetector


class SolutionVisualizer:
    """Visualizes puzzle solutions with pieces properly aligned and touching."""

    def __init__(self, output_dir: str = 'app/static/output', edge_detector: EdgeDetector = None):
        self.output_dir = output_dir
        self.edge_detector = edge_detector
        os.makedirs(output_dir, exist_ok=True)

    def visualize_solution(self, solution: PuzzleSolution,
                           pieces: List[PuzzlePiece],
                           original_filename: str,
                           edge_detector: EdgeDetector = None) -> List[str]:
        """Create visualizations of the puzzle solution."""
        if edge_detector:
            self.edge_detector = edge_detector

        output_files = []

        assembled_img = self._create_assembled_puzzle(solution, pieces)
        if assembled_img is not None:
            filename = f"solution_assembled_{original_filename}"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, assembled_img)
            output_files.append(filename)

        return output_files

    def _create_assembled_puzzle(self, solution: PuzzleSolution,
                                 pieces: List[PuzzlePiece]) -> Optional[np.ndarray]:
        """Create an image showing all pieces assembled in their solved positions."""
        if not solution.placed_pieces:
            return None

        grid_rows = 2
        grid_cols = 3

        piece_data = {}

        for placed in solution.placed_pieces:
            piece = pieces[placed.piece_id]

            # Rotate piece contour according to solver's rotation
            rotated_contour = self._align_piece_to_grid(piece, placed.piece_id, placed.rotation)

            # Get bounding box of rotated contour
            min_x, min_y = rotated_contour.min(axis=0)
            max_x, max_y = rotated_contour.max(axis=0)

            piece_data[(placed.grid_row, placed.grid_col)] = {
                'placed': placed,
                'contour': rotated_contour,
                'min_x': min_x, 'min_y': min_y,
                'max_x': max_x, 'max_y': max_y,
                'width': max_x - min_x,
                'height': max_y - min_y
            }

        # Calculate uniform scale - INCREASED for larger display
        widths = [d['width'] for d in piece_data.values()]
        heights = [d['height'] for d in piece_data.values()]
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)

        # Increased target width for larger visualization
        target_width = 1400
        scale = target_width / (avg_width * grid_cols * 1.15)
        scale = min(scale, 1.2)  # Increased max scale

        margin = 80  # Increased margin
        gap = 25  # Slightly increased gap
        cell_width = avg_width * scale + gap
        cell_height = avg_height * scale + gap

        img_width = int(margin * 2 + cell_width * grid_cols)
        img_height = int(margin * 2 + cell_height * grid_rows + 50)  # More space for text

        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

        # Draw each piece
        for (row, col), data in piece_data.items():
            contour = data['contour'].copy() * scale

            cell_center_x = margin + col * cell_width + cell_width / 2
            cell_center_y = margin + row * cell_height + cell_height / 2

            contour_center_x = (data['min_x'] + data['max_x']) / 2 * scale
            contour_center_y = (data['min_y'] + data['max_y']) / 2 * scale

            offset_x = cell_center_x - contour_center_x
            offset_y = cell_center_y - contour_center_y

            final_contour = contour + np.array([offset_x, offset_y])
            final_contour = final_contour.astype(np.int32).reshape(-1, 1, 2)

            piece_id = data['placed'].piece_id
            colors = [
                (70, 70, 70), (100, 60, 60), (60, 100, 60),
                (60, 60, 100), (90, 90, 60), (60, 90, 90),
            ]
            fill_color = colors[piece_id % len(colors)]

            cv2.drawContours(img, [final_contour], -1, fill_color, -1)
            cv2.drawContours(img, [final_contour], -1, (40, 40, 40), 2)

            # Larger labels
            label = f"P{piece_id}"
            cv2.putText(img, label, (int(cell_center_x - 20), int(cell_center_y + 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Fixed rotation label - use .1f for one decimal place
            rotation_value = data['placed'].rotation
            rot_label = f"{rotation_value:.1f}°"
            cv2.putText(img, rot_label, (int(cell_center_x - 20), int(cell_center_y + 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Larger footer text
        cv2.putText(img, f"Grid: {grid_rows}x{grid_cols}", (10, img_height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(img, f"Confidence: {solution.confidence:.1%}", (160, img_height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

        return img

    def _align_piece_to_grid(self, piece: PuzzlePiece, piece_id: int, solver_rotation: float) -> np.ndarray:
        """
        Align a piece so its edges are in the correct positions for the grid.

        The solver_rotation already contains the full rotation needed (base + discrete),
        so we just apply it directly.
        """
        contour = piece.contour.copy().astype(np.float32).reshape(-1, 2)
        center = np.array(piece.center, dtype=np.float32)
        centered = contour - center

        # The solver_rotation already includes both base rotation and discrete 90° steps
        # So we just apply it directly without calculating base_rotation again
        total_rotation = solver_rotation

        # Apply rotation
        angle_rad = np.radians(total_rotation)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        rotated = centered @ rotation_matrix.T

        return rotated

    def _calculate_base_rotation(self, piece_id: int, center: np.ndarray) -> float:
        """
        Calculate the rotation needed to make the piece's "top" edge point upward.

        Uses the detected edge data to find where the "top" edge currently is,
        then calculates the rotation to make it point to -Y direction.
        """
        if self.edge_detector is None:
            return 0.0

        edges = self.edge_detector.piece_edges.get(piece_id, {})
        if 'top' not in edges:
            return 0.0

        top_edge = edges['top']

        # The "top" edge should go from left to right when the piece is oriented correctly
        # Calculate the midpoint of the top edge and its direction
        start = np.array(top_edge.start_point, dtype=np.float32)
        end = np.array(top_edge.end_point, dtype=np.float32)

        # The midpoint of the top edge
        mid = (start + end) / 2

        # Vector from center to midpoint of top edge - this should point UP (-Y)
        vec_to_top = mid - center

        # Calculate angle of this vector
        # We want it to point UP (angle = -90° or 270°)
        current_angle = np.degrees(np.arctan2(vec_to_top[1], vec_to_top[0]))

        # The target angle for the top edge is -90° (pointing up)
        target_angle = -90.0

        # Rotation needed to align
        rotation = target_angle - current_angle

        # Normalize to [-180, 180]
        while rotation > 180:
            rotation -= 360
        while rotation < -180:
            rotation += 360

        return rotation