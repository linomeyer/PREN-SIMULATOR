import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from app.main.puzzle_solver.edge_detector import PieceEdge, EdgeDetector


@dataclass
class EdgeMatch:
    """Represents a potential match between two edges."""
    edge1: PieceEdge
    edge2: PieceEdge
    compatibility_score: float
    length_similarity: float
    shape_similarity: float
    classification_match: bool
    rotation_offset: int  # How many 90-degree rotations needed for edge2 to align with edge1

    def __repr__(self):
        return (f"EdgeMatch(piece_{self.edge1.piece_id}_{self.edge1.edge_type} <-> "
                f"piece_{self.edge2.piece_id}_{self.edge2.edge_type}, "
                f"score={self.compatibility_score:.3f}, rotation={self.rotation_offset * 90}°)")


class EdgeMatcher:
    """Matches puzzle piece edges to find compatible pairs, accounting for rotation."""

    # Mapping of how edges align when pieces are rotated
    # rotation_offset: {original_edge: edge_after_rotation}
    ROTATION_MAP = {
        0: {'top': 'top', 'right': 'right', 'bottom': 'bottom', 'left': 'left'},
        1: {'top': 'right', 'right': 'bottom', 'bottom': 'left', 'left': 'top'},  # 90° CW
        2: {'top': 'bottom', 'right': 'left', 'bottom': 'top', 'left': 'right'},  # 180°
        3: {'top': 'left', 'right': 'top', 'bottom': 'right', 'left': 'bottom'}  # 270° CW
    }

    # Which edges should be adjacent for matching (opposite edges touch)
    # e.g., if piece1's 'right' edge matches, it should connect to piece2's 'left' edge
    ADJACENT_EDGES = {
        'top': 'bottom',
        'bottom': 'top',
        'left': 'right',
        'right': 'left'
    }

    def __init__(self, edge_detector: EdgeDetector):
        """
        Initialize edge matcher with detected edges.

        Args:
            edge_detector: EdgeDetector instance with detected edges
        """
        self.edge_detector = edge_detector
        self.matches: List[EdgeMatch] = []

    def find_matches(self, min_score: float = 0.6, include_flat_edges: bool = False) -> List[EdgeMatch]:
        """
        Find all potential edge matches above a minimum score threshold.
        Considers all possible rotations of pieces.

        Args:
            min_score: Minimum compatibility score (0.0 to 1.0)
            include_flat_edges: If False (default), skip matching flat edges (border pieces)

        Returns:
            List of EdgeMatch objects sorted by score (best first)
        """
        self.matches = []

        # Get all edges organized by piece
        piece_edges = self.edge_detector.piece_edges

        # Compare edges from different pieces
        for piece1_id, edges1 in piece_edges.items():
            for piece2_id, edges2 in piece_edges.items():
                # Don't match edges from the same piece
                if piece1_id == piece2_id:
                    continue

                # Try all possible rotations of piece2
                for rotation in range(4):
                    matches_for_rotation = self._find_matches_with_rotation(
                        piece1_id, edges1, piece2_id, edges2, rotation, min_score, include_flat_edges
                    )
                    self.matches.extend(matches_for_rotation)

        # Sort by compatibility score (best matches first)
        self.matches.sort(key=lambda m: m.compatibility_score, reverse=True)

        print(f"Found {len(self.matches)} potential matches (threshold: {min_score})")
        return self.matches

    def _find_matches_with_rotation(self, piece1_id: int, edges1: Dict[str, PieceEdge],
                                    piece2_id: int, edges2: Dict[str, PieceEdge],
                                    rotation: int, min_score: float,
                                    include_flat_edges: bool) -> List[EdgeMatch]:
        """
        Find matches between two pieces with piece2 rotated by the given amount.

        Args:
            piece1_id: ID of first piece
            edges1: Edges of first piece
            piece2_id: ID of second piece
            edges2: Edges of second piece
            rotation: Number of 90° clockwise rotations (0-3)
            min_score: Minimum compatibility score
            include_flat_edges: If False, skip flat edges

        Returns:
            List of EdgeMatch objects
        """
        matches = []

        # For each edge of piece1
        for edge1_type, edge1 in edges1.items():
            # Skip flat edges if requested (they are border pieces)
            if not include_flat_edges and edge1.get_edge_type_classification() == 'flat':
                continue

            # Determine which edge of piece2 should be adjacent
            adjacent_edge_type = self.ADJACENT_EDGES[edge1_type]

            # Find which edge of the rotated piece2 corresponds to this adjacent edge
            # We need to reverse-map: after rotation, which original edge becomes adjacent_edge_type?
            rotated_edges = self.ROTATION_MAP[rotation]

            # Find which original edge becomes the adjacent edge after rotation
            original_edge_type = None
            for orig, rotated in rotated_edges.items():
                if rotated == adjacent_edge_type:
                    original_edge_type = orig
                    break

            if original_edge_type and original_edge_type in edges2:
                edge2 = edges2[original_edge_type]

                # Skip flat edges if requested
                if not include_flat_edges and edge2.get_edge_type_classification() == 'flat':
                    continue

                # Evaluate match
                match = self._evaluate_match(edge1, edge2, rotation)

                if match.compatibility_score >= min_score:
                    matches.append(match)

        return matches

    def _evaluate_match(self, edge1: PieceEdge, edge2: PieceEdge, rotation: int) -> EdgeMatch:
        """
        Evaluate the compatibility between two edges with a specific rotation.

        Args:
            edge1: First edge
            edge2: Second edge (will be conceptually rotated)
            rotation: Number of 90° rotations applied to piece containing edge2

        Returns:
            EdgeMatch object with compatibility scores
        """
        # 1. Length similarity (edges should have similar lengths)
        length_similarity = self._calculate_length_similarity(edge1, edge2)

        # 2. Shape similarity (compare shape signatures)
        shape_similarity = self._calculate_shape_similarity(edge1, edge2)

        # 3. Classification match (tab should match with slot, flat with flat)
        classification_match = self._check_classification_compatibility(edge1, edge2)
        classification_score = 1.0 if classification_match else 0.0

        # 4. Calculate overall compatibility score (weighted average)
        compatibility_score = (
                0.3 * length_similarity +
                0.5 * shape_similarity +
                0.2 * classification_score
        )

        return EdgeMatch(
            edge1=edge1,
            edge2=edge2,
            compatibility_score=compatibility_score,
            length_similarity=length_similarity,
            shape_similarity=shape_similarity,
            classification_match=classification_match,
            rotation_offset=rotation
        )

    def _calculate_length_similarity(self, edge1: PieceEdge, edge2: PieceEdge) -> float:
        """
        Calculate similarity based on edge lengths.

        Args:
            edge1: First edge
            edge2: Second edge

        Returns:
            Similarity score between 0.0 and 1.0
        """
        len1 = edge1.length
        len2 = edge2.length

        if len1 == 0 or len2 == 0:
            return 0.0

        # Calculate ratio (always <= 1.0)
        ratio = min(len1, len2) / max(len1, len2)

        return ratio

    def _calculate_shape_similarity(self, edge1: PieceEdge, edge2: PieceEdge) -> float:
        """
        Calculate similarity based on shape signatures (curvature).

        For matching edges, one should be approximately the mirror/inverse of the other.

        Args:
            edge1: First edge
            edge2: Second edge

        Returns:
            Similarity score between 0.0 and 1.0
        """
        sig1 = edge1.shape_signature
        sig2 = edge2.shape_signature

        # For puzzle pieces to fit, one edge should be the inverse of the other
        # Try both normal and reversed comparison
        sig2_reversed = sig2[::-1]

        # Calculate correlation (inverted because tab should fit into slot)
        # Use negative correlation since curvatures should be opposite
        correlation_reversed = self._calculate_correlation(-sig1, sig2_reversed)
        correlation_normal = self._calculate_correlation(-sig1, sig2)

        # Take the best correlation
        similarity = max(correlation_reversed, correlation_normal)

        # Normalize to 0-1 range (correlation is -1 to 1)
        similarity = (similarity + 1.0) / 2.0

        return similarity

    def _calculate_correlation(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """
        Calculate Pearson correlation between two signatures.

        Args:
            sig1: First signature
            sig2: Second signature

        Returns:
            Correlation coefficient between -1.0 and 1.0
        """
        if len(sig1) == 0 or len(sig2) == 0:
            return 0.0

        # Normalize signatures
        sig1_normalized = sig1 - np.mean(sig1)
        sig2_normalized = sig2 - np.mean(sig2)

        # Calculate correlation
        numerator = np.sum(sig1_normalized * sig2_normalized)
        denominator = np.sqrt(np.sum(sig1_normalized ** 2) * np.sum(sig2_normalized ** 2))

        if denominator == 0:
            return 0.0

        correlation = numerator / denominator
        return float(correlation)

    def _check_classification_compatibility(self, edge1: PieceEdge, edge2: PieceEdge) -> bool:
        """
        Check if edge classifications are compatible.

        Compatible pairs:
        - flat <-> flat
        - tab <-> slot
        - slot <-> tab

        Args:
            edge1: First edge
            edge2: Second edge

        Returns:
            True if classifications are compatible
        """
        class1 = edge1.get_edge_type_classification()
        class2 = edge2.get_edge_type_classification()

        # Flat edges match with flat edges
        if class1 == 'flat' and class2 == 'flat':
            return True

        # Tab matches with slot
        if (class1 == 'tab' and class2 == 'slot') or (class1 == 'slot' and class2 == 'tab'):
            return True

        return False

    def get_matches_for_piece(self, piece_id: int, min_score: float = 0.6) -> Dict[str, List[EdgeMatch]]:
        """
        Get all matches for a specific piece, organized by edge type.

        Args:
            piece_id: ID of the piece
            min_score: Minimum compatibility score

        Returns:
            Dictionary mapping edge type to list of matches
        """
        if not self.matches:
            self.find_matches(min_score)

        piece_matches = {'top': [], 'right': [], 'bottom': [], 'left': []}

        for match in self.matches:
            if match.edge1.piece_id == piece_id:
                piece_matches[match.edge1.edge_type].append(match)
            elif match.edge2.piece_id == piece_id:
                # Determine which edge type after rotation
                rotated_edge_type = self.ROTATION_MAP[match.rotation_offset][match.edge2.edge_type]
                piece_matches[rotated_edge_type].append(match)

        return piece_matches

    def get_best_match_for_edge(self, piece_id: int, edge_type: str) -> Optional[EdgeMatch]:
        """
        Get the best match for a specific edge.

        Args:
            piece_id: ID of the piece
            edge_type: Type of edge ('top', 'right', 'bottom', 'left')

        Returns:
            Best EdgeMatch or None if no matches found
        """
        if not self.matches:
            self.find_matches()

        best_match = None
        best_score = 0.0

        for match in self.matches:
            if match.edge1.piece_id == piece_id and match.edge1.edge_type == edge_type:
                if match.compatibility_score > best_score:
                    best_match = match
                    best_score = match.compatibility_score
            elif match.edge2.piece_id == piece_id:
                rotated_edge_type = self.ROTATION_MAP[match.rotation_offset][match.edge2.edge_type]
                if rotated_edge_type == edge_type and match.compatibility_score > best_score:
                    best_match = match
                    best_score = match.compatibility_score

        return best_match


    def identify_border_pieces(self) -> Dict[int, Dict[str, any]]:
        """
        Identify border and corner pieces based on flat edges.

        Returns:
            Dictionary mapping piece_id to border information:
            {
                'is_border': bool,
                'is_corner': bool,
                'flat_edges': list of edge types that are flat,
                'num_flat_edges': int
            }
        """
        border_info = {}

        for piece_id, edges in self.edge_detector.piece_edges.items():
            flat_edges = []

            for edge_type, edge in edges.items():
                if edge.get_edge_type_classification() == 'flat':
                    flat_edges.append(edge_type)

            num_flat = len(flat_edges)

            border_info[piece_id] = {
                'is_border': num_flat > 0,
                'is_corner': num_flat >= 2,
                'flat_edges': flat_edges,
                'num_flat_edges': num_flat
            }

        return border_info

    def get_corner_pieces(self) -> List[int]:
        """
        Get IDs of corner pieces (pieces with 2 or more flat edges).

        Returns:
            List of piece IDs that are corner pieces
        """
        border_info = self.identify_border_pieces()
        return [piece_id for piece_id, info in border_info.items() if info['is_corner']]

    def get_border_pieces(self) -> List[int]:
        """
        Get IDs of border pieces (pieces with at least 1 flat edge).

        Returns:
            List of piece IDs that are border pieces
        """
        border_info = self.identify_border_pieces()
        return [piece_id for piece_id, info in border_info.items() if info['is_border']]

    def get_interior_pieces(self) -> List[int]:
        """
        Get IDs of interior pieces (pieces with no flat edges).

        Returns:
            List of piece IDs that are interior pieces
        """
        border_info = self.identify_border_pieces()
        return [piece_id for piece_id, info in border_info.items() if not info['is_border']]



    def get_match_statistics(self) -> Dict:
        """
        Get statistics about the matches.

        Returns:
            Dictionary with match statistics
        """
        if not self.matches:
            return {}

        scores = [m.compatibility_score for m in self.matches]

        # Count matches by rotation
        rotation_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for match in self.matches:
            rotation_counts[match.rotation_offset] += 1

        # Count matches by classification type
        flat_flat = sum(1 for m in self.matches if
                        m.edge1.get_edge_type_classification() == 'flat' and
                        m.edge2.get_edge_type_classification() == 'flat')
        tab_slot = sum(1 for m in self.matches if m.classification_match and
                       'flat' not in [m.edge1.get_edge_type_classification(),
                                      m.edge2.get_edge_type_classification()])

        # Get border piece information
        border_info = self.identify_border_pieces()
        corner_pieces = [pid for pid, info in border_info.items() if info['is_corner']]
        border_pieces = [pid for pid, info in border_info.items() if info['is_border']]
        interior_pieces = [pid for pid, info in border_info.items() if not info['is_border']]

        stats = {
            'total_matches': len(self.matches),
            'avg_score': float(np.mean(scores)) if scores else 0.0,
            'max_score': float(np.max(scores)) if scores else 0.0,
            'min_score': float(np.min(scores)) if scores else 0.0,
            'flat_flat_matches': flat_flat,
            'tab_slot_matches': tab_slot,
            'high_confidence_matches': sum(1 for s in scores if s >= 0.8),
            'medium_confidence_matches': sum(1 for s in scores if 0.6 <= s < 0.8),
            'matches_by_rotation': {
                '0°': rotation_counts[0],
                '90°': rotation_counts[1],
                '180°': rotation_counts[2],
                '270°': rotation_counts[3]
            },
            'piece_classification': {
                'corner_pieces': len(corner_pieces),
                'border_pieces': len(border_pieces),
                'interior_pieces': len(interior_pieces),
                'corner_piece_ids': corner_pieces,
                'border_piece_ids': border_pieces,
                'interior_piece_ids': interior_pieces
            }
        }

        return stats