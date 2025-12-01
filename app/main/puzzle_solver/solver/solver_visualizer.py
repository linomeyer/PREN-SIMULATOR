# File: app/main/puzzle_solver/puzzle_solving/puzzle_solver_visualizer.py
"""
Visualizer for puzzle solutions.

Creates visual representations of the solved puzzle showing:
1. How all pieces fit together
2. The grid layout
3. Individual pieces with their rotations applied
"""

import os
from typing import List, Tuple, Optional

import cv2
import numpy as np

from app.main.puzzle_solver.solver.solver import PuzzleSolution, PlacedPiece
from app.main.puzzle_solver.piece_extraction.extractor import PuzzlePiece, PieceSegmenter


class SolutionVisualizer:
    """Visualizes puzzle solutions."""

    def __init__(self, output_dir: str = 'app/static/output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def visualize_solution(self, solution: PuzzleSolution,
                           pieces: List[PuzzlePiece],
                           original_filename: str) -> List[str]:
        """
        Create visualizations of the puzzle solution.

        Args:
            solution: PuzzleSolution object
            pieces: List of PuzzlePiece objects
            original_filename: Original filename for naming output

        Returns:
            List of output filenames
        """
        output_files = []

        # 1. Create assembled puzzle image
        assembled_img = self._create_assembled_puzzle(solution, pieces)
        if assembled_img is not None:
            filename = f"solution_assembled_{original_filename}"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, assembled_img)
            output_files.append(filename)

        # 2. Create grid layout diagram
        grid_img = self._create_grid_diagram(solution)
        if grid_img is not None:
            filename = f"solution_grid_{original_filename}"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, grid_img)
            output_files.append(filename)

        # 3. Create piece rotation guide
        rotation_img = self._create_rotation_guide(solution, pieces)
        if rotation_img is not None:
            filename = f"solution_rotations_{original_filename}"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, rotation_img)
            output_files.append(filename)

        return output_files

    def _create_assembled_puzzle(self, solution: PuzzleSolution,
                                  pieces: List[PuzzlePiece]) -> Optional[np.ndarray]:
        """
        Create an image showing all pieces assembled together.

        Args:
            solution: PuzzleSolution object
            pieces: List of PuzzlePiece objects

        Returns:
            Assembled puzzle image
        """
        if not solution.placed_pieces:
            return None

        # Calculate cell size based on largest piece
        max_width = 0
        max_height = 0
        for placed in solution.placed_pieces:
            piece = pieces[placed.piece_id]
            _, _, w, h = piece.bbox
            # Account for rotation
            if abs(placed.rotation % 180) > 45:
                w, h = h, w
            max_width = max(max_width, w)
            max_height = max(max_height, h)

        # Add padding
        cell_width = int(max_width * 1.1)
        cell_height = int(max_height * 1.1)

        # Create output image
        img_width = cell_width * solution.grid_cols + 100
        img_height = cell_height * solution.grid_rows + 150

        # White background
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

        # Draw header
        cv2.putText(img, "SOLVED PUZZLE", (img_width // 2 - 100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)

        confidence_text = f"Confidence: {solution.confidence:.1%}"
        cv2.putText(img, confidence_text, (img_width // 2 - 80, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1, cv2.LINE_AA)

        # Starting position for puzzle
        start_x = 50
        start_y = 100

        # Draw grid lines
        for row in range(solution.grid_rows + 1):
            y = start_y + row * cell_height
            cv2.line(img, (start_x, y), (start_x + solution.grid_cols * cell_width, y),
                     (200, 200, 200), 1)

        for col in range(solution.grid_cols + 1):
            x = start_x + col * cell_width
            cv2.line(img, (x, start_y), (x, start_y + solution.grid_rows * cell_height),
                     (200, 200, 200), 1)

        # Place each piece
        for placed in solution.placed_pieces:
            piece = pieces[placed.piece_id]

            # Calculate cell center
            cell_x = start_x + placed.grid_col * cell_width + cell_width // 2
            cell_y = start_y + placed.grid_row * cell_height + cell_height // 2

            # Draw the piece (rotated)
            self._draw_rotated_piece(img, piece, (cell_x, cell_y), placed.rotation)

            # Draw piece ID
            label = f"P{placed.piece_id}"
            label_pos = (cell_x - 15, cell_y + 5)
            cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 3, cv2.LINE_AA)  # White outline
            cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Add legend
        legend_y = img_height - 30
        cv2.putText(img, f"Grid: {solution.grid_rows}x{solution.grid_cols}", (20, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"Pieces placed: {len(solution.placed_pieces)}", (200, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return img

    def _draw_rotated_piece(self, img: np.ndarray, piece: PuzzlePiece,
                            center: Tuple[int, int], rotation: float):
        """
        Draw a piece at a specific position with rotation.

        Args:
            img: Image to draw on
            piece: PuzzlePiece to draw
            center: Center position (x, y)
            rotation: Rotation in degrees
        """
        # Get piece contour
        contour = piece.contour.copy()

        # Translate contour to origin
        piece_center = np.array(piece.center, dtype=np.float32)
        contour_float = contour.astype(np.float32).reshape(-1, 2)
        centered_contour = contour_float - piece_center

        # Apply rotation
        angle_rad = np.radians(rotation)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        rotated_contour = centered_contour @ rotation_matrix.T

        # Scale down to fit in cell
        scale = 0.6  # Scale to 60% of original size
        scaled_contour = rotated_contour * scale

        # Translate to target position
        final_contour = scaled_contour + np.array(center)
        final_contour = final_contour.astype(np.int32).reshape(-1, 1, 2)

        # Draw filled contour
        cv2.drawContours(img, [final_contour], -1, (180, 180, 180), -1)  # Fill
        cv2.drawContours(img, [final_contour], -1, (50, 50, 50), 2)  # Border

    def _create_grid_diagram(self, solution: PuzzleSolution) -> np.ndarray:
        """
        Create a simple grid diagram showing piece IDs at each position.

        Args:
            solution: PuzzleSolution object

        Returns:
            Grid diagram image
        """
        cell_size = 100
        margin = 50

        img_width = cell_size * solution.grid_cols + 2 * margin
        img_height = cell_size * solution.grid_rows + 2 * margin + 80

        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

        # Draw title
        cv2.putText(img, "PUZZLE LAYOUT", (img_width // 2 - 80, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

        # Get grid layout
        grid = solution.get_grid_layout()

        # Draw grid
        for row in range(solution.grid_rows):
            for col in range(solution.grid_cols):
                x = margin + col * cell_size
                y = 60 + row * cell_size

                # Draw cell border
                cv2.rectangle(img, (x, y), (x + cell_size, y + cell_size), (0, 0, 0), 2)

                # Get piece at this position
                piece_id = grid[row][col]

                if piece_id is not None:
                    # Find the placed piece for rotation info
                    placed = solution.get_piece_at(row, col)

                    # Draw piece ID
                    text = f"P{piece_id}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    text_x = x + (cell_size - text_size[0]) // 2
                    text_y = y + cell_size // 2

                    cv2.putText(img, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

                    # Draw rotation indicator
                    if placed:
                        rot_text = f"{placed.rotation:.0f}°"
                        cv2.putText(img, rot_text, (x + 5, y + cell_size - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
                else:
                    # Empty cell
                    cv2.putText(img, "?", (x + cell_size // 2 - 10, y + cell_size // 2 + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)

        # Row/column labels
        for row in range(solution.grid_rows):
            y = 60 + row * cell_size + cell_size // 2
            cv2.putText(img, f"R{row}", (10, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

        for col in range(solution.grid_cols):
            x = margin + col * cell_size + cell_size // 2
            cv2.putText(img, f"C{col}", (x - 10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

        return img

    def _create_rotation_guide(self, solution: PuzzleSolution,
                                pieces: List[PuzzlePiece]) -> np.ndarray:
        """
        Create a guide showing each piece with its required rotation.

        Args:
            solution: PuzzleSolution object
            pieces: List of PuzzlePiece objects

        Returns:
            Rotation guide image
        """
        num_pieces = len(solution.placed_pieces)
        if num_pieces == 0:
            return None

        # Calculate layout
        cols = min(4, num_pieces)
        rows = (num_pieces + cols - 1) // cols

        cell_width = 200
        cell_height = 250
        margin = 30

        img_width = cell_width * cols + 2 * margin
        img_height = cell_height * rows + 2 * margin + 60

        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

        # Draw title
        cv2.putText(img, "PIECE ROTATION GUIDE", (img_width // 2 - 130, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw each piece
        for idx, placed in enumerate(solution.placed_pieces):
            row = idx // cols
            col = idx % cols

            x = margin + col * cell_width + cell_width // 2
            y = 80 + row * cell_height + 80  # Leave room for text

            piece = pieces[placed.piece_id]

            # Draw original (left) and rotated (right) view
            # For now, just draw the rotated contour
            self._draw_rotated_piece(img, piece, (x, y), placed.rotation)

            # Draw info
            info_y = 80 + row * cell_height + 20
            cv2.putText(img, f"Piece {placed.piece_id}", (margin + col * cell_width + 10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(img, f"Pos: ({placed.grid_row}, {placed.grid_col})",
                        (margin + col * cell_width + 10, info_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

            cv2.putText(img, f"Rotate: {placed.rotation:.0f}°",
                        (margin + col * cell_width + 10, info_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1, cv2.LINE_AA)

            # Draw arrow showing rotation direction
            arrow_x = x
            arrow_y = y + 60
            arrow_len = 30

            # Draw rotation arrow
            angle_rad = np.radians(placed.rotation)
            end_x = int(arrow_x + arrow_len * np.cos(angle_rad))
            end_y = int(arrow_y + arrow_len * np.sin(angle_rad))

            cv2.arrowedLine(img, (arrow_x, arrow_y), (end_x, end_y),
                            (0, 0, 200), 2, tipLength=0.3)

        return img