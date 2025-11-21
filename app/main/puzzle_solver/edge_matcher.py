import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
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
    rotation_offset: int  # Discrete 90° rotations for backward compatibility
    rotation_angle: float  # NEW: Actual rotation angle in degrees

    def __repr__(self):
        return (f"EdgeMatch(piece_{self.edge1.piece_id}_{self.edge1.edge_type} <-> "
                f"piece_{self.edge2.piece_id}_{self.edge2.edge_type}, "
                f"score={self.compatibility_score:.3f}, rotation={self.rotation_angle:.1f}°)")


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

    def find_unique_best_matches(self, min_score: float = 0.0) -> List[EdgeMatch]:
        """
        Find unique best matches for all non-flat edges.
        Each edge can only be used once. Eliminates duplicate matches.
        Considers actual edge angles, not just 90° rotations.

        Args:
            min_score: Minimum compatibility score (0.0 to 1.0)

        Returns:
            List of unique EdgeMatch objects (no duplicates, one match per edge pair)
        """
        all_potential_matches = []
        seen_edge_pairs: Set[Tuple[Tuple[int, str], Tuple[int, str]]] = set()

        # Get all edges organized by piece
        piece_edges = self.edge_detector.piece_edges

        # Compare edges from different pieces
        for piece1_id, edges1 in piece_edges.items():
            for piece2_id, edges2 in piece_edges.items():
                # Only process each piece pair once (avoid A->B and B->A)
                if piece1_id >= piece2_id:
                    continue

                # Compare all edges with angle-based alignment
                matches_for_pair = self._find_matches_with_angles(
                    piece1_id, edges1, piece2_id, edges2, min_score
                )

                # Filter out duplicate edge pairs
                for match in matches_for_pair:
                    edge1_key = (match.edge1.piece_id, match.edge1.edge_type)
                    edge2_key = (match.edge2.piece_id, match.edge2.edge_type)

                    # Create normalized pair (smaller piece_id first)
                    edge_pair = tuple(sorted([edge1_key, edge2_key]))

                    if edge_pair not in seen_edge_pairs:
                        seen_edge_pairs.add(edge_pair)
                        all_potential_matches.append(match)

        # Sort by compatibility score (best matches first)
        all_potential_matches.sort(key=lambda m: m.compatibility_score, reverse=True)

        print(f"Found {len(all_potential_matches)} unique potential matches (threshold: {min_score})")

        # Now select the best match for each edge, ensuring no edge is used twice
        used_edges: Set[Tuple[int, str]] = set()
        final_matches: List[EdgeMatch] = []

        for match in all_potential_matches:
            edge1_key = (match.edge1.piece_id, match.edge1.edge_type)
            edge2_key = (match.edge2.piece_id, match.edge2.edge_type)

            # Skip if either edge has already been matched
            if edge1_key in used_edges or edge2_key in used_edges:
                continue

            # This is the best available match for these edges
            final_matches.append(match)
            used_edges.add(edge1_key)
            used_edges.add(edge2_key)

        print(f"Selected {len(final_matches)} best unique matches")
        print(f"Matched {len(used_edges)} edges total")

        # Update self.matches
        self.matches = final_matches

        return final_matches

    def _find_matches_with_angles(self, piece1_id: int, edges1: Dict[str, PieceEdge],
                                  piece2_id: int, edges2: Dict[str, PieceEdge],
                                  min_score: float) -> List[EdgeMatch]:
        """
        Find matches between two pieces considering actual edge angles.

        For two edges to match, they should be approximately opposite in direction
        (180° apart) when placed adjacent to each other.

        Args:
            piece1_id: ID of first piece
            edges1: Edges of first piece
            piece2_id: ID of second piece
            edges2: Edges of second piece
            min_score: Minimum compatibility score

        Returns:
            List of EdgeMatch objects
        """
        matches = []
        angle_tolerance = 10.0  # Tightened to 10° deviation for matching

        # For each edge of piece1
        for edge1_type, edge1 in edges1.items():
            edge1_classification = edge1.get_edge_type_classification()

            # Skip flat edges
            if edge1_classification == 'flat':
                continue

            # For each edge of piece2
            for edge2_type, edge2 in edges2.items():
                edge2_classification = edge2.get_edge_type_classification()

                # Skip flat edges
                if edge2_classification == 'flat':
                    continue

                # Only compare complementary edges (tab with slot)
                if not self._are_complementary_edges(edge1_classification, edge2_classification):
                    continue

                # Calculate required rotation for alignment
                # Two edges should be opposite (180° apart) to fit together
                angle_diff = self._calculate_angle_difference(edge1.angle, edge2.angle)

                # Check if edges are approximately opposite (around 180°)
                is_opposite = abs(angle_diff - 180.0) < angle_tolerance

                if is_opposite:
                    # Calculate the exact rotation needed
                    rotation_angle = self._normalize_angle(edge2.angle - edge1.angle + 180.0)

                    # Evaluate match with actual rotation angle
                    match = self._evaluate_match_with_angle(edge1, edge2, rotation_angle)

                    if match.compatibility_score >= min_score:
                        matches.append(match)

        return matches

    def _evaluate_match_with_angle(self, edge1: PieceEdge, edge2: PieceEdge,
                                   rotation_angle: float) -> EdgeMatch:
        """
        Evaluate the compatibility between two edges with a specific rotation angle.

        Args:
            edge1: First edge
            edge2: Second edge
            rotation_angle: Actual rotation angle in degrees

        Returns:
            EdgeMatch object with compatibility scores
        """
        # 1. Length similarity (edges should have similar lengths)
        length_similarity = self._calculate_length_similarity(edge1, edge2)

        # 2. Shape similarity (compare shape signatures)
        shape_similarity = self._calculate_shape_similarity(edge1, edge2)

        # 3. Classification match (tab should match with slot)
        classification_match = self._check_classification_compatibility(edge1, edge2)
        classification_score = 1.0 if classification_match else 0.0

        # 4. Angle alignment bonus - reward closer angle matches
        angle_diff = abs(self._calculate_angle_difference(edge1.angle, edge2.angle) - 180.0)
        angle_similarity = max(0.0, 1.0 - (angle_diff / 10.0))  # Linear falloff over 10°

        # 5. Calculate overall compatibility score (weighted average)
        compatibility_score = (
                0.25 * length_similarity +
                0.45 * shape_similarity +
                0.15 * classification_score +
                0.15 * angle_similarity  # NEW: Angle alignment component
        )

        # Convert rotation angle to nearest 90° step for backward compatibility
        rotation_offset = int(round(rotation_angle / 90.0)) % 4

        return EdgeMatch(
            edge1=edge1,
            edge2=edge2,
            compatibility_score=compatibility_score,
            length_similarity=length_similarity,
            shape_similarity=shape_similarity,
            classification_match=classification_match,
            rotation_offset=rotation_offset,
            rotation_angle=rotation_angle  # Store actual angle
        )

    def _calculate_angle_difference(self, angle1: float, angle2: float) -> float:
        """
        Calculate the absolute difference between two angles (0-360°).

        Args:
            angle1: First angle in degrees
            angle2: Second angle in degrees

        Returns:
            Absolute angle difference (0-180°)
        """
        diff = abs(angle1 - angle2)

        # Normalize to 0-180 range (take shorter angle)
        if diff > 180:
            diff = 360 - diff

        return diff

    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to 0-360 range.

        Args:
            angle: Angle in degrees

        Returns:
            Normalized angle (0-360°)
        """
        angle = angle % 360
        if angle < 0:
            angle += 360
        return angle


    def _are_complementary_edges(self, class1: str, class2: str) -> bool:
        """
        Check if two edge classifications are complementary (can fit together).

        Args:
            class1: Classification of first edge ('flat', 'tab', or 'slot')
            class2: Classification of second edge ('flat', 'tab', or 'slot')

        Returns:
            True if edges can fit together, False otherwise
        """
        # Tab fits with slot, slot fits with tab
        if (class1 == 'tab' and class2 == 'slot') or (class1 == 'slot' and class2 == 'tab'):
            return True

        # Flat edges can match with flat edges (border pieces)
        if class1 == 'flat' and class2 == 'flat':
            return True

        # All other combinations cannot fit
        # (tab-tab, slot-slot, tab-flat, slot-flat, etc.)
        return False


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
        - tab <-> slot
        - slot <-> tab

        Flat edges should NOT match with anything (they are borders).

        Args:
            edge1: First edge
            edge2: Second edge

        Returns:
            True if classifications are compatible
        """
        class1 = edge1.get_edge_type_classification()
        class2 = edge2.get_edge_type_classification()

        # Flat edges should never match - they are border pieces
        if class1 == 'flat' or class2 == 'flat':
            return False

        # Tab matches with slot
        if (class1 == 'tab' and class2 == 'slot') or (class1 == 'slot' and class2 == 'tab'):
            return True

        return False


    def identify_border_pieces(self) -> Dict[int, Dict[str, any]]:
        """
        Identify border and corner pieces based on flat edges.

        Returns:
            Dictionary mapping piece_id to border information
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

    def get_match_statistics(self) -> Dict:
        """Get statistics about the matches."""
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