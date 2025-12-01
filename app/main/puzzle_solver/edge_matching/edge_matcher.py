import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from app.main.puzzle_solver.edge_detection.edge_detector import PieceEdge, EdgeDetector


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
    rotation_angle: float  # Actual rotation angle in degrees

    def __repr__(self):
        return (f"EdgeMatch(piece_{self.edge1.piece_id}_{self.edge1.edge_type} <-> "
                f"piece_{self.edge2.piece_id}_{self.edge2.edge_type}, "
                f"score={self.compatibility_score:.3f}, rotation={self.rotation_angle:.1f}°)")


class EdgeMatcher:
    """Matches puzzle piece edges to find compatible pairs, accounting for rotation."""

    # Mapping of how edges align when pieces are rotated
    ROTATION_MAP = {
        0: {'top': 'top', 'right': 'right', 'bottom': 'bottom', 'left': 'left'},
        1: {'top': 'right', 'right': 'bottom', 'bottom': 'left', 'left': 'top'},  # 90° CW
        2: {'top': 'bottom', 'right': 'left', 'bottom': 'top', 'left': 'right'},  # 180°
        3: {'top': 'left', 'right': 'top', 'bottom': 'right', 'left': 'bottom'}  # 270° CW
    }

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

        Args:
            min_score: Minimum compatibility score (0.0 to 1.0)

        Returns:
            List of unique EdgeMatch objects (no duplicates, one match per edge pair)
        """
        all_potential_matches = []
        seen_edge_pairs: Set[Tuple[Tuple[int, str], Tuple[int, str]]] = set()

        piece_edges = self.edge_detector.piece_edges

        # Compare edges from different pieces
        for piece1_id, edges1 in piece_edges.items():
            for piece2_id, edges2 in piece_edges.items():
                if piece1_id >= piece2_id:
                    continue

                # Find all potential matches between these two pieces
                matches_for_pair = self._find_matches_between_pieces(
                    piece1_id, edges1, piece2_id, edges2, min_score
                )

                for match in matches_for_pair:
                    edge1_key = (match.edge1.piece_id, match.edge1.edge_type)
                    edge2_key = (match.edge2.piece_id, match.edge2.edge_type)
                    edge_pair = tuple(sorted([edge1_key, edge2_key]))

                    if edge_pair not in seen_edge_pairs:
                        seen_edge_pairs.add(edge_pair)
                        all_potential_matches.append(match)

        # Sort by compatibility score (best matches first)
        all_potential_matches.sort(key=lambda m: m.compatibility_score, reverse=True)

        print(f"Found {len(all_potential_matches)} unique potential matches (threshold: {min_score})")

        # Select best match for each edge, ensuring no edge is used twice
        used_edges: Set[Tuple[int, str]] = set()
        final_matches: List[EdgeMatch] = []

        for match in all_potential_matches:
            edge1_key = (match.edge1.piece_id, match.edge1.edge_type)
            edge2_key = (match.edge2.piece_id, match.edge2.edge_type)

            if edge1_key in used_edges or edge2_key in used_edges:
                continue

            final_matches.append(match)
            used_edges.add(edge1_key)
            used_edges.add(edge2_key)

        print(f"Selected {len(final_matches)} best unique matches")
        print(f"Matched {len(used_edges)} edges total")

        self.matches = final_matches
        return final_matches

    def _find_matches_between_pieces(self, piece1_id: int, edges1: Dict[str, PieceEdge],
                                     piece2_id: int, edges2: Dict[str, PieceEdge],
                                     min_score: float) -> List[EdgeMatch]:
        """
        Find matches between two pieces by comparing all edge combinations.

        The key insight: we don't filter by angle here. Instead, we compare
        the SHAPE of edges after normalizing them to a common orientation.
        Two edges match if their shapes are complementary (one is the "inverse" of the other).

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

        for edge1_type, edge1 in edges1.items():
            edge1_classification = edge1.get_edge_type_classification()

            # Skip flat edges (border edges)
            if edge1_classification == 'flat':
                continue

            for edge2_type, edge2 in edges2.items():
                edge2_classification = edge2.get_edge_type_classification()

                # Skip flat edges
                if edge2_classification == 'flat':
                    continue

                # Only compare complementary edges (tab with slot)
                if not self._are_complementary_edges(edge1_classification, edge2_classification):
                    continue

                # Evaluate match by comparing normalized edge shapes
                match = self._evaluate_edge_match(edge1, edge2)

                if match.compatibility_score >= min_score:
                    matches.append(match)

        return matches

    def _evaluate_edge_match(self, edge1: PieceEdge, edge2: PieceEdge) -> EdgeMatch:
        """
        Evaluate the compatibility between two edges.

        This normalizes both edges to a common reference frame and compares:
        1. Length similarity
        2. Shape similarity (curvature profile match)
        3. Classification compatibility

        Args:
            edge1: First edge
            edge2: Second edge

        Returns:
            EdgeMatch object with compatibility scores
        """
        # 1. Length similarity
        length_similarity = self._calculate_length_similarity(edge1, edge2)

        # 2. Shape similarity - compare normalized edge profiles
        shape_similarity = self._calculate_normalized_shape_similarity(edge1, edge2)

        # 3. Classification match
        classification_match = self._check_classification_compatibility(edge1, edge2)
        classification_score = 1.0 if classification_match else 0.0

        # Calculate overall compatibility score
        compatibility_score = (
                0.20 * length_similarity +
                0.60 * shape_similarity +
                0.20 * classification_score
        )

        # Calculate the rotation angle needed to align edge2 to edge1
        # When edges match, they should be oriented 180° apart (facing each other)
        rotation_angle = self._normalize_angle(edge1.angle - edge2.angle + 180.0)
        rotation_offset = int(round(rotation_angle / 90.0)) % 4

        return EdgeMatch(
            edge1=edge1,
            edge2=edge2,
            compatibility_score=compatibility_score,
            length_similarity=length_similarity,
            shape_similarity=shape_similarity,
            classification_match=classification_match,
            rotation_offset=rotation_offset,
            rotation_angle=rotation_angle
        )

    def _calculate_normalized_shape_similarity(self, edge1: PieceEdge, edge2: PieceEdge) -> float:
        """
        Calculate shape similarity by normalizing both edges to a common frame.

        Steps:
        1. Extract edge points
        2. Normalize both edges: translate to origin, rotate to horizontal, scale to unit length
        3. For edge2, also flip vertically (since matching edges are mirror images)
        4. Compare the resulting profiles

        Args:
            edge1: First edge
            edge2: Second edge

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Get normalized deviation profiles for both edges
        profile1 = self._get_normalized_edge_profile(edge1, flip=False)
        profile2 = self._get_normalized_edge_profile(edge2, flip=True)  # Flip for matching

        if profile1 is None or profile2 is None:
            return 0.0

        # Compare profiles - try both forward and reversed
        similarity_forward = self._compare_profiles(profile1, profile2)
        similarity_reversed = self._compare_profiles(profile1, profile2[::-1])

        return max(similarity_forward, similarity_reversed)

    def _get_normalized_edge_profile(self, edge: PieceEdge, flip: bool = False,
                                     num_samples: int = 100) -> np.ndarray:
        """
        Get a normalized profile of edge deviations from the baseline.

        This creates a rotation-invariant representation of the edge shape by:
        1. Rotating the edge to be horizontal
        2. Measuring perpendicular distances from the baseline
        3. Normalizing distances by edge length

        Args:
            edge: PieceEdge to process
            flip: If True, negate the deviations (for matching edge comparison)
            num_samples: Number of samples along the edge

        Returns:
            Array of normalized deviation values, or None if edge is invalid
        """
        points = edge.points.reshape(-1, 2).astype(np.float64)

        if len(points) < 3:
            return None

        # Get start and end points
        start = np.array(edge.start_point, dtype=np.float64)
        end = np.array(edge.end_point, dtype=np.float64)

        # Calculate baseline vector and length
        baseline = end - start
        baseline_length = np.linalg.norm(baseline)

        if baseline_length < 1e-6:
            return None

        # Normalize baseline direction
        baseline_dir = baseline / baseline_length

        # Calculate perpendicular direction (90° counter-clockwise)
        perp_dir = np.array([-baseline_dir[1], baseline_dir[0]])

        # For each point, calculate:
        # - t: position along baseline (0 to 1)
        # - d: perpendicular distance from baseline (normalized by length)
        deviations = []

        for point in points:
            relative = point - start
            t = np.dot(relative, baseline_dir) / baseline_length
            d = np.dot(relative, perp_dir) / baseline_length
            deviations.append((t, d))

        # Sort by position along baseline
        deviations.sort(key=lambda x: x[0])

        # Resample to fixed number of points
        t_values = np.array([d[0] for d in deviations])
        d_values = np.array([d[1] for d in deviations])

        # Interpolate to uniform samples
        t_uniform = np.linspace(0, 1, num_samples)
        profile = np.interp(t_uniform, t_values, d_values)

        if flip:
            profile = -profile

        return profile

    def _compare_profiles(self, profile1: np.ndarray, profile2: np.ndarray) -> float:
        """
        Compare two edge profiles using correlation and distance metrics.

        Args:
            profile1: First normalized profile
            profile2: Second normalized profile

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if len(profile1) != len(profile2):
            # Resample to match lengths
            profile2 = np.interp(
                np.linspace(0, 1, len(profile1)),
                np.linspace(0, 1, len(profile2)),
                profile2
            )

        # Method 1: Correlation
        correlation = self._calculate_correlation(profile1, profile2)
        correlation_score = (correlation + 1.0) / 2.0  # Map [-1, 1] to [0, 1]

        # Method 2: Mean absolute difference
        diff = np.abs(profile1 - profile2)
        mean_diff = np.mean(diff)
        # Typical deviations are 0-0.3 of edge length, so scale accordingly
        diff_score = max(0.0, 1.0 - mean_diff / 0.15)

        # Combine scores
        similarity = 0.6 * correlation_score + 0.4 * diff_score

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

        sig1_normalized = sig1 - np.mean(sig1)
        sig2_normalized = sig2 - np.mean(sig2)

        numerator = np.sum(sig1_normalized * sig2_normalized)
        denominator = np.sqrt(np.sum(sig1_normalized ** 2) * np.sum(sig2_normalized ** 2))

        if denominator < 1e-10:
            return 0.0

        return float(numerator / denominator)

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to 0-360 range."""
        angle = angle % 360
        if angle < 0:
            angle += 360
        return angle

    def _are_complementary_edges(self, class1: str, class2: str) -> bool:
        """Check if two edge classifications are complementary."""
        if (class1 == 'tab' and class2 == 'slot') or (class1 == 'slot' and class2 == 'tab'):
            return True
        return False

    def _calculate_length_similarity(self, edge1: PieceEdge, edge2: PieceEdge) -> float:
        """Calculate similarity based on edge lengths."""
        len1 = edge1.length
        len2 = edge2.length

        if len1 == 0 or len2 == 0:
            return 0.0

        ratio = min(len1, len2) / max(len1, len2)
        return ratio

    def _check_classification_compatibility(self, edge1: PieceEdge, edge2: PieceEdge) -> bool:
        """Check if edge classifications are compatible (tab matches slot)."""
        class1 = edge1.get_edge_type_classification()
        class2 = edge2.get_edge_type_classification()

        if class1 == 'flat' or class2 == 'flat':
            return False

        if (class1 == 'tab' and class2 == 'slot') or (class1 == 'slot' and class2 == 'tab'):
            return True

        return False

    def identify_border_pieces(self) -> Dict[int, Dict[str, any]]:
        """Identify border and corner pieces based on flat edges."""
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

        rotation_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for match in self.matches:
            rotation_counts[match.rotation_offset] += 1

        flat_flat = sum(1 for m in self.matches if
                        m.edge1.get_edge_type_classification() == 'flat' and
                        m.edge2.get_edge_type_classification() == 'flat')
        tab_slot = sum(1 for m in self.matches if m.classification_match and
                       'flat' not in [m.edge1.get_edge_type_classification(),
                                      m.edge2.get_edge_type_classification()])

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