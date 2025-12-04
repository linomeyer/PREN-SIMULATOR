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

        grid_img = self._create_grid_diagram(solution)
        if grid_img is not None:
            filename = f"solution_grid_{original_filename}"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, grid_img)
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

        # Calculate uniform scale
        widths = [d['width'] for d in piece_data.values()]
        heights = [d['height'] for d in piece_data.values()]
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)

        target_width = 900
        scale = target_width / (avg_width * grid_cols * 1.15)
        scale = min(scale, 0.8)

        margin = 60
        gap = 20
        cell_width = avg_width * scale + gap
        cell_height = avg_height * scale + gap

        img_width = int(margin * 2 + cell_width * grid_cols)
        img_height = int(margin * 2 + cell_height * grid_rows + 40)

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

            label = f"P{piece_id}"
            cv2.putText(img, label, (int(cell_center_x - 15), int(cell_center_y + 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            rot_label = f"{data['placed'].rotation:.0f}°"
            cv2.putText(img, rot_label, (int(cell_center_x - 15), int(cell_center_y + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.putText(img, f"Grid: {grid_rows}x{grid_cols}", (10, img_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(img, f"Confidence: {solution.confidence:.1%}", (130, img_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

        return img

    def _align_piece_to_grid(self, piece: PuzzlePiece, piece_id: int, solver_rotation: float) -> np.ndarray:
        """
        Align a piece so its edges are in the correct positions for the grid.

        Strategy: Use the detected 'top' edge to find the piece's current orientation,
        then apply the solver's rotation on top of that.
        """
        contour = piece.contour.copy().astype(np.float32).reshape(-1, 2)
        center = np.array(piece.center, dtype=np.float32)
        centered = contour - center

        # Calculate the base rotation needed to make the piece's detected "top" edge
        # actually point upward (negative Y direction)
        base_rotation = self._calculate_base_rotation(piece_id, center)

        # Total rotation = base alignment + solver's rotation
        total_rotation = base_rotation + solver_rotation

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

    def _create_grid_diagram(self, solution: PuzzleSolution) -> np.ndarray:
        """Create a simple grid diagram showing piece IDs."""
        grid_rows = 2
        grid_cols = 3

        cell_size = 70
        margin = 25

        img_width = cell_size * grid_cols + 2 * margin
        img_height = cell_size * grid_rows + 2 * margin + 25

        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

        cv2.putText(img, "LAYOUT", (img_width // 2 - 30, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        grid = solution.get_grid_layout()

        for row in range(grid_rows):
            for col in range(grid_cols):
                x = margin + col * cell_size
                y = 25 + row * cell_size

                cv2.rectangle(img, (x, y), (x + cell_size, y + cell_size), (0, 0, 0), 1)

                piece_id = grid[row][col] if row < len(grid) and col < len(grid[row]) else None

                if piece_id is not None:
                    placed = solution.get_piece_at(row, col)

                    text = f"P{piece_id}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    text_x = x + (cell_size - text_size[0]) // 2
                    text_y = y + cell_size // 2 + 5

                    cv2.putText(img, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    if placed:
                        rot_text = f"{placed.rotation:.0f}°"
                        cv2.putText(img, rot_text, (x + 3, y + cell_size - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (100, 100, 100), 1, cv2.LINE_AA)

        return img