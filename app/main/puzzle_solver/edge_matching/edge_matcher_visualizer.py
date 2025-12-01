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

        if not edge_matcher.matches:
            print("No matches to visualize")
            return []

        matches = edge_matcher.matches

        print(f"Creating visualizations for {len(matches)} unique matches")

        for idx, match in enumerate(matches):
            match_img = self._create_overlay_comparison(match)

            match_filename = (f"match_{idx:02d}_p{match.edge1.piece_id}_{match.edge1.edge_type}_"
                              f"p{match.edge2.piece_id}_{match.edge2.edge_type}_{original_filename}")
            match_path = os.path.join(self.output_dir, match_filename)
            cv2.imwrite(match_path, match_img)

            match_filenames.append(match_filename)

        return match_filenames

    def _create_overlay_comparison(self, match: EdgeMatch) -> np.ndarray:
        """
        Create a visualization showing both edges overlaid after alignment.

        This shows:
        1. Left panel: Edge 1 in red, Edge 2 in blue, both normalized and overlaid
        2. Right panel: Side-by-side view showing how edges fit together

        Args:
            match: EdgeMatch to visualize

        Returns:
            Visualization image
        """
        img_width = 1000
        img_height = 600

        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

        # Panel 1: Overlay view (left side)
        overlay_center = (img_width // 4, img_height // 2)
        self._draw_overlaid_edges(img, match, overlay_center, scale=250)

        # Add panel label
        cv2.putText(img, "OVERLAY VIEW", (overlay_center[0] - 80, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "(Edges should match)", (overlay_center[0] - 90, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

        # Panel 2: Fitting view (right side) - show edges about to connect
        fitting_center = (3 * img_width // 4, img_height // 2)
        self._draw_fitting_view(img, match, fitting_center, scale=200)

        cv2.putText(img, "FITTING VIEW", (fitting_center[0] - 70, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "(How pieces connect)", (fitting_center[0] - 90, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

        # Draw separator
        cv2.line(img, (img_width // 2, 90), (img_width // 2, img_height - 20),
                 (200, 200, 200), 2)

        # Add legend
        legend_y = img_height - 40
        cv2.line(img, (20, legend_y), (50, legend_y), (0, 0, 255), 3)
        cv2.putText(img, f"Edge 1: P{match.edge1.piece_id} {match.edge1.edge_type}",
                    (60, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.line(img, (280, legend_y), (310, legend_y), (255, 0, 0), 3)
        cv2.putText(img, f"Edge 2: P{match.edge2.piece_id} {match.edge2.edge_type}",
                    (320, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Add header
        header = self._create_match_header(match, img_width)
        result = np.vstack([header, img])

        return result

    def _draw_overlaid_edges(self, img: np.ndarray, match: EdgeMatch,
                             center: Tuple[int, int], scale: float):
        """
        Draw both edges overlaid on top of each other after normalizing to same orientation.

        Both edges are rotated to horizontal and centered at the same point.
        Edge 2 is flipped vertically since matching edges are mirror images.
        """
        # Get normalized points for edge 1 (horizontal, centered)
        points1 = self._get_normalized_points(match.edge1, scale, flip=False)

        # Get normalized points for edge 2 (horizontal, centered, flipped)
        points2 = self._get_normalized_points(match.edge2, scale, flip=True)

        if points1 is None or points2 is None:
            return

        # Translate to center
        points1_centered = points1 + np.array(center)
        points2_centered = points2 + np.array(center)

        # Draw edges
        points1_int = points1_centered.astype(np.int32)
        points2_int = points2_centered.astype(np.int32)

        cv2.polylines(img, [points1_int], False, (0, 0, 255), 3, cv2.LINE_AA)  # Red
        cv2.polylines(img, [points2_int], False, (255, 0, 0), 3, cv2.LINE_AA)  # Blue

    def _draw_fitting_view(self, img: np.ndarray, match: EdgeMatch,
                           center: Tuple[int, int], scale: float):
        """
        Draw edges positioned as they would connect - facing each other with a small gap.
        """
        gap = 20  # Pixels between edges

        # Get normalized points
        points1 = self._get_normalized_points(match.edge1, scale, flip=False)
        points2 = self._get_normalized_points(match.edge2, scale, flip=True)

        if points1 is None or points2 is None:
            return

        # Rotate both to be vertical
        rotation_angle = np.radians(90)
        cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        points1_vertical = points1 @ rotation_matrix.T
        points2_vertical = points2 @ rotation_matrix.T

        # Offset: edge1 to the left, edge2 to the right
        offset1 = np.array([center[0] - gap // 2, center[1]])
        offset2 = np.array([center[0] + gap // 2, center[1]])

        points1_final = (points1_vertical + offset1).astype(np.int32)
        points2_final = (points2_vertical + offset2).astype(np.int32)

        cv2.polylines(img, [points1_final], False, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.polylines(img, [points2_final], False, (255, 0, 0), 3, cv2.LINE_AA)

    def _get_normalized_points(self, edge: PieceEdge, scale: float,
                               flip: bool = False) -> np.ndarray:
        """
        Get edge points normalized to horizontal orientation, centered at origin.

        Args:
            edge: PieceEdge to process
            scale: Scale factor for the output points
            flip: If True, flip vertically (for matching edge)

        Returns:
            Normalized points array, or None if edge is invalid
        """
        points = edge.points.reshape(-1, 2).astype(np.float64)

        if len(points) < 3:
            return None

        # Get baseline from start to end
        start = np.array(edge.start_point, dtype=np.float64)
        end = np.array(edge.end_point, dtype=np.float64)

        baseline = end - start
        baseline_length = np.linalg.norm(baseline)

        if baseline_length < 1e-6:
            return None

        # Calculate rotation angle to make baseline horizontal
        angle = np.arctan2(baseline[1], baseline[0])

        # Create rotation matrix to make edge horizontal
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Center points around origin
        edge_center = np.mean(points, axis=0)
        centered_points = points - edge_center

        # Rotate to horizontal
        rotated_points = centered_points @ rotation_matrix.T

        # Scale to desired size
        current_extent = np.max(np.abs(rotated_points[:, 0])) * 2
        if current_extent > 0:
            scale_factor = scale / current_extent
        else:
            scale_factor = 1.0

        scaled_points = rotated_points * scale_factor

        # Flip vertically if requested (for matching edge comparison)
        if flip:
            scaled_points[:, 1] = -scaled_points[:, 1]

        return scaled_points

    def _create_match_header(self, match: EdgeMatch, width: int) -> np.ndarray:
        """Create a header with match information."""
        header_height = 100
        header = np.ones((header_height, width, 3), dtype=np.uint8) * 240

        # Title
        title = "EDGE MATCH VISUALIZATION"
        cv2.putText(header, title, (width // 2 - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

        # Score and details
        score_text = f"Match Score: {match.compatibility_score:.3f}"
        cv2.putText(header, score_text, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        length_text = f"Length Sim: {match.length_similarity:.3f}"
        cv2.putText(header, length_text, (220, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        shape_text = f"Shape Sim: {match.shape_similarity:.3f}"
        cv2.putText(header, shape_text, (420, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Edge info
        e1_text = f"E1: P{match.edge1.piece_id}-{match.edge1.edge_type} ({match.edge1.get_edge_type_classification()})"
        cv2.putText(header, e1_text, (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1, cv2.LINE_AA)

        e2_text = f"E2: P{match.edge2.piece_id}-{match.edge2.edge_type} ({match.edge2.get_edge_type_classification()})"
        cv2.putText(header, e2_text, (320, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 0, 0), 1, cv2.LINE_AA)

        rotation_text = f"Rotation: {match.rotation_angle:.1f}°"
        cv2.putText(header, rotation_text, (620, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Color-coded score indicator
        score = match.compatibility_score
        if score >= 0.8:
            score_color = (0, 180, 0)
            score_label = "EXCELLENT"
        elif score >= 0.65:
            score_color = (0, 200, 0)
            score_label = "GOOD"
        elif score >= 0.5:
            score_color = (0, 165, 255)
            score_label = "MODERATE"
        else:
            score_color = (0, 0, 200)
            score_label = "WEAK"

        cv2.putText(header, score_label, (width - 120, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2, cv2.LINE_AA)

        return header