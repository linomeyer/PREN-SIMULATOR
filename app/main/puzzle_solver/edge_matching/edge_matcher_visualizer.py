import os
from typing import List, Tuple

import cv2
import numpy as np

from app.main.puzzle_solver.edge_detection.edge_detector import PieceEdge
from app.main.puzzle_solver.edge_matching.edge_matcher import EdgeMatcher, EdgeMatch
from app.main.puzzle_solver.piece_extraction.extractor import PieceSegmenter


class MatchVisualizer:
    """Visualizes edge matches between puzzle pieces."""

    def __init__(self, output_dir: str = 'app/static/output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def visualize_best_matches(self, segmenter: PieceSegmenter,
                               edge_matcher: EdgeMatcher,
                               original_filename: str) -> List[str]:
        """
        Create visualization showing aligned edge comparisons.

        Args:
            segmenter: PieceSegmenter with extracted pieces
            edge_matcher: EdgeMatcher with detected matches
            original_filename: Original filename for naming output

        Returns:
            List of output filenames
        """
        match_filenames = []

        # Get matches
        if not edge_matcher.matches:
            print("No matches to visualize")
            return []

        matches = edge_matcher.matches

        print(f"Creating visualizations for {len(matches)} unique matches")

        # Visualize each match
        for idx, match in enumerate(matches):
            match_img = self._create_aligned_edge_comparison(match)

            # Save image
            match_filename = (f"match_{idx:02d}_p{match.edge1.piece_id}_{match.edge1.edge_type}_"
                              f"p{match.edge2.piece_id}_{match.edge2.edge_type}_{original_filename}")
            match_path = os.path.join(self.output_dir, match_filename)
            cv2.imwrite(match_path, match_img)

            match_filenames.append(match_filename)

        return match_filenames

    def _create_aligned_edge_comparison(self, match: EdgeMatch) -> np.ndarray:
        """
        Create a visualization showing both edges aligned at their matched angle.
        Only shows the edge lines, not the full pieces.

        Args:
            match: EdgeMatch to visualize

        Returns:
            Visualization image
        """
        # Image size
        img_width = 800
        img_height = 600

        # Create white background
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

        # Calculate positions for the two edges - side by side
        edge1_center_x = img_width // 4
        edge2_center_x = 3 * img_width // 4
        center_y = img_height // 2

        # Determine which edge is tab and which is slot
        edge1_classification = match.edge1.get_edge_type_classification()
        edge2_classification = match.edge2.get_edge_type_classification()

        # Base rotations to make edges vertical:
        # Tab should be rotated 270° (or -90°) to be vertical with tab on right
        # Slot should be rotated 90° to be vertical with slot on left
        if edge1_classification == 'tab':
            edge1_base_rotation = -90.0  # 270° = -90°
        else:  # slot
            edge1_base_rotation = 90.0

        if edge2_classification == 'tab':
            edge2_base_rotation = -90.0  # 270° = -90°
        else:  # slot
            edge2_base_rotation = 90.0

        # Draw edge 1 (left side) with just base rotation
        self._draw_edge_at_angle(
            img, match.edge1,
            (edge1_center_x, center_y),
            edge1_base_rotation,
            (0, 0, 255),  # Red
            f"Piece {match.edge1.piece_id} ({match.edge1.edge_type})"
        )

        # Draw edge 2 (right side) with base rotation PLUS alignment
        # The alignment ensures edge2 faces edge1 properly
        alignment_rotation = match.edge1.angle - match.edge2.angle + 180.0
        edge2_total_rotation = edge2_base_rotation + alignment_rotation

        self._draw_edge_at_angle(
            img, match.edge2,
            (edge2_center_x, center_y),
            edge2_total_rotation,
            (255, 0, 0),  # Blue
            f"Piece {match.edge2.piece_id} ({match.edge2.edge_type})"
        )

        # Add header with match information
        header_height = 120
        header = self._create_match_header(match, img_width)

        # Combine header and visualization
        result = np.vstack([header, img])

        return result

    def _draw_edge_at_angle(self, img: np.ndarray, edge: PieceEdge,
                            center: Tuple[int, int], rotation_angle: float,
                            color: Tuple[int, int, int], label: str):
        """
        Draw an edge at a specific position and rotation.

        Args:
            img: Image to draw on
            edge: PieceEdge to draw
            center: Center position (x, y) for the edge
            rotation_angle: Total rotation to apply in degrees
            color: BGR color for the edge
            label: Text label for the edge
        """
        # Get edge points
        points = edge.points.reshape(-1, 2).astype(np.float32)

        # Center the points around origin
        edge_center = np.mean(points, axis=0)
        centered_points = points - edge_center

        # Apply the rotation (remove the original edge angle and apply new rotation)
        total_rotation = np.radians(-edge.angle + rotation_angle)

        # Create rotation matrix
        cos_angle = np.cos(total_rotation)
        sin_angle = np.sin(total_rotation)
        rotation_matrix = np.array([[cos_angle, -sin_angle],
                                    [sin_angle, cos_angle]])

        # Rotate points
        rotated_points = centered_points @ rotation_matrix.T

        # Scale to fit nicely (normalize to around 300 pixels for better visibility)
        scale = 300.0 / max(np.max(np.abs(rotated_points)), 1.0)
        scaled_points = rotated_points * scale

        # Translate to center position
        final_points = scaled_points + np.array(center)
        final_points = final_points.astype(np.int32)

        # Draw the edge line
        cv2.polylines(img, [final_points], False, color, 3, cv2.LINE_AA)

        # Add label above the edge
        label_pos = (center[0] - 80, center[1] - 230)
        cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Add angle information
        angle_text = f"Angle: {edge.angle:.1f}°"
        angle_pos = (center[0] - 60, center[1] - 210)
        cv2.putText(img, angle_text, angle_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (100, 100, 100), 1, cv2.LINE_AA)

        # Add classification
        class_text = f"Type: {edge.get_edge_type_classification()}"
        class_pos = (center[0] - 60, center[1] - 190)
        cv2.putText(img, class_text, class_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (100, 100, 100), 1, cv2.LINE_AA)

        # Add length
        length_text = f"Length: {int(edge.length)}px"
        length_pos = (center[0] - 60, center[1] - 170)
        cv2.putText(img, length_text, length_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (100, 100, 100), 1, cv2.LINE_AA)


    def _create_match_header(self, match: EdgeMatch, width: int) -> np.ndarray:
        """
        Create a header with match information.

        Args:
            match: EdgeMatch object
            width: Width of the header

        Returns:
            Header image
        """
        header_height = 120
        header = np.ones((header_height, width, 3), dtype=np.uint8) * 240

        # Title
        title = "EDGE MATCH COMPARISON - ALIGNED VIEW"
        cv2.putText(header, title, (width // 2 - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

        # Match details - left column
        details_left = [
            f"Edge 1: Piece {match.edge1.piece_id} ({match.edge1.edge_type})",
            f"  Angle: {match.edge1.angle:.1f}°",
            f"  Type: {match.edge1.get_edge_type_classification()}",
            f"  Length: {int(match.edge1.length)}px"
        ]

        y_pos = 55
        for detail in details_left:
            cv2.putText(header, detail, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
            y_pos += 18

        # Match details - right column
        details_right = [
            f"Edge 2: Piece {match.edge2.piece_id} ({match.edge2.edge_type})",
            f"  Angle: {match.edge2.angle:.1f}°",
            f"  Type: {match.edge2.get_edge_type_classification()}",
            f"  Length: {int(match.edge2.length)}px"
        ]

        y_pos = 55
        for detail in details_right:
            cv2.putText(header, detail, (width // 2 + 20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
            y_pos += 18

        # Score in the center
        score_text = f"Match Score: {match.compatibility_score:.3f}"
        cv2.putText(header, score_text, (width // 2 - 120, header_height // 2 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # Color code the score
        score = match.compatibility_score
        if score >= 0.8:
            score_color = (0, 200, 0)  # Green
            score_label = "HIGH"
        elif score >= 0.6:
            score_color = (0, 165, 255)  # Orange
            score_label = "MEDIUM"
        else:
            score_color = (0, 0, 200)  # Red
            score_label = "LOW"

        cv2.putText(header, score_label, (width // 2 - 40, header_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2, cv2.LINE_AA)

        return header