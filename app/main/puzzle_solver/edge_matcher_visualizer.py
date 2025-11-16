import os
from typing import List, Tuple, Set

import cv2
import numpy as np

from app.main.puzzle_solver.edge_detector import PieceEdge
from app.main.puzzle_solver.edge_matcher import EdgeMatcher, EdgeMatch
from app.main.puzzle_solver.extractor import PieceSegmenter


class MatchVisualizer:
    """Visualizes edge matches between puzzle pieces."""

    def __init__(self, output_dir: str = 'app/static/output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def visualize_best_matches(self, segmenter: PieceSegmenter,
                               edge_matcher: EdgeMatcher,
                               original_filename: str) -> List[str]:
        """
        Create visualization showing the best unique match for each edge.
        Eliminates duplicates by only showing each edge pair once.

        Args:
            segmenter: PieceSegmenter with extracted pieces
            edge_matcher: EdgeMatcher with detected matches
            original_filename: Original filename for naming output

        Returns:
            List of output filenames
        """
        match_filenames = []

        # Get all matches sorted by score
        if not edge_matcher.matches:
            edge_matcher.find_matches()

        # Track which edges have been visualized to avoid duplicates
        # Use the ORIGINAL edge types (before rotation)
        visualized_edges: Set[Tuple[int, str]] = set()
        unique_matches = []

        # Go through matches from best to worst
        for match in edge_matcher.matches:
            # Both edges use their ORIGINAL (unrotated) edge types
            edge1_key = (match.edge1.piece_id, match.edge1.edge_type)
            edge2_key = (match.edge2.piece_id, match.edge2.edge_type)

            # Skip if either edge has already been visualized
            if edge1_key in visualized_edges or edge2_key in visualized_edges:
                continue

            # Skip flat edges
            if (match.edge1.get_edge_type_classification() == 'flat' or
                    match.edge2.get_edge_type_classification() == 'flat'):
                continue

            # Add this match and mark both edges as visualized
            unique_matches.append(match)
            visualized_edges.add(edge1_key)
            visualized_edges.add(edge2_key)

        print(f"Creating visualizations for {len(unique_matches)} unique matches")
        print(f"Visualized edges: {len(visualized_edges)} out of possible edges")

        # Visualize each unique match
        for idx, match in enumerate(unique_matches):
            match_img = self._create_edge_comparison(segmenter, match)

            # Save image
            match_filename = (f"match_{idx:02d}_p{match.edge1.piece_id}_{match.edge1.edge_type}_"
                              f"p{match.edge2.piece_id}_{match.edge2.edge_type}_{original_filename}")
            match_path = os.path.join(self.output_dir, match_filename)
            cv2.imwrite(match_path, match_img)

            match_filenames.append(match_filename)

        return match_filenames

    def _create_edge_comparison(self, segmenter: PieceSegmenter,
                                match: EdgeMatch) -> np.ndarray:
        """
        Create a side-by-side comparison of two matching edges.

        Args:
            segmenter: PieceSegmenter with pieces
            match: EdgeMatch to visualize

        Returns:
            Visualization image showing edges side by side
        """
        # Get the two pieces
        piece1 = segmenter.pieces[match.edge1.piece_id]
        piece2 = segmenter.pieces[match.edge2.piece_id]

        # Extract and visualize each edge
        edge1_img = self._extract_edge_visualization(
            piece1, match.edge1, f"Piece {match.edge1.piece_id}"
        )
        edge2_img = self._extract_edge_visualization(
            piece2, match.edge2, f"Piece {match.edge2.piece_id}"
        )

        # Make both images the same height
        max_height = max(edge1_img.shape[0], edge2_img.shape[0])
        edge1_img = self._pad_to_height(edge1_img, max_height)
        edge2_img = self._pad_to_height(edge2_img, max_height)

        # Create header with match information
        header = self._create_match_header(match, edge1_img.shape[1] + edge2_img.shape[1])

        # Combine edges horizontally
        edges_combined = np.hstack([edge1_img, edge2_img])

        # Add arrow between edges
        arrow_overlay = edges_combined.copy()
        mid_x = edge1_img.shape[1]
        mid_y = edges_combined.shape[0] // 2
        cv2.arrowedLine(arrow_overlay, (mid_x - 50, mid_y), (mid_x + 50, mid_y),
                        (0, 255, 0), 4, tipLength=0.3)
        edges_combined = cv2.addWeighted(edges_combined, 0.7, arrow_overlay, 0.3, 0)

        # Stack header on top
        result = np.vstack([header, edges_combined])

        return result

    def _extract_edge_visualization(self, piece, edge: PieceEdge,
                                    title: str) -> np.ndarray:
        """
        Extract and visualize a single edge with context.

        Args:
            piece: PuzzlePiece object
            edge: PieceEdge to visualize
            title: Title for the visualization

        Returns:
            Image showing the edge
        """
        # Convert piece image to BGR
        if piece.image.shape[2] == 4:
            img = np.ones((piece.image.shape[0], piece.image.shape[1], 3), dtype=np.uint8) * 255
            alpha = piece.image[:, :, 3:4] / 255.0
            piece_bgr = cv2.cvtColor(piece.image[:, :, :3], cv2.COLOR_RGB2BGR)
            img = (img * (1 - alpha) + piece_bgr * alpha).astype(np.uint8)
        else:
            img = piece.image.copy()

        # Get piece bbox to adjust coordinates
        x, y, w, h = piece.bbox

        # Draw the full contour lightly
        local_points = (edge.points - [x, y]).astype(np.int32)

        # Draw all contour in light gray
        piece_contour = (piece.contour - [x, y]).reshape(-1, 2).astype(np.int32)
        cv2.polylines(img, [piece_contour], True, (200, 200, 200), 2)

        # Highlight the specific edge in bright color
        edge_color = (0, 0, 255)  # Red for the matching edge
        cv2.polylines(img, [local_points], False, edge_color, 5)

        # Draw start and end points
        start_local = (edge.start_point[0] - x, edge.start_point[1] - y)
        end_local = (edge.end_point[0] - x, edge.end_point[1] - y)
        cv2.circle(img, start_local, 8, (0, 255, 0), -1)  # Green
        cv2.circle(img, end_local, 8, (255, 0, 0), -1)  # Blue

        # Add edge type label
        label_pos = local_points[len(local_points) // 2][0]
        cv2.putText(img, edge.edge_type.upper(), tuple(label_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, edge_color, 3)

        # Add title and edge info
        title_height = 80
        info_img = np.ones((img.shape[0] + title_height, img.shape[1], 3), dtype=np.uint8) * 250
        info_img[title_height:, :] = img

        # Add title
        cv2.putText(info_img, title, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Add edge details
        edge_class = edge.get_edge_type_classification()
        details = f"{edge.edge_type}: {edge_class} ({int(edge.length)}px)"
        cv2.putText(info_img, details, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

        return info_img

    def _create_match_header(self, match: EdgeMatch, width: int) -> np.ndarray:
        """
        Create a header with match information.

        Args:
            match: EdgeMatch object
            width: Width of the header

        Returns:
            Header image
        """
        header_height = 100
        header = np.ones((header_height, width, 3), dtype=np.uint8) * 240

        # Title
        title = f"EDGE MATCH COMPARISON"
        cv2.putText(header, title, (width // 2 - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        # Match details
        details = [
            f"Compatibility Score: {match.compatibility_score:.3f}",
            f"Length Similarity: {match.length_similarity:.2f}",
            f"Shape Similarity: {match.shape_similarity:.2f}",
            f"Rotation: {match.rotation_offset * 90}°"
        ]

        y_pos = 60
        for detail in details:
            cv2.putText(header, detail, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
            y_pos += 20

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

        cv2.putText(header, score_label, (width - 150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, score_color, 3)

        return header

    def _pad_to_height(self, img: np.ndarray, target_height: int) -> np.ndarray:
        """
        Pad image to target height.

        Args:
            img: Image to pad
            target_height: Desired height

        Returns:
            Padded image
        """
        if img.shape[0] >= target_height:
            return img

        pad_top = (target_height - img.shape[0]) // 2
        pad_bottom = target_height - img.shape[0] - pad_top

        padded = np.ones((target_height, img.shape[1], 3), dtype=np.uint8) * 255
        padded[pad_top:pad_top + img.shape[0], :] = img

        return padded