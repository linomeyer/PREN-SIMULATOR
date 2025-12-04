import numpy as np
from typing import List, Tuple, Dict, Set, Optional
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
    rotation_angle: float

    def __repr__(self):
        return (f"EdgeMatch(piece_{self.edge1.piece_id}_{self.edge1.edge_type} <-> "
                f"piece_{self.edge2.piece_id}_{self.edge2.edge_type}, "
                f"score={self.compatibility_score:.3f}, rotation={self.rotation_angle:.1f}°)")


class EdgeMatcher:
    """Matches puzzle piece edges to find compatible pairs."""

    ADJACENT_EDGES = {
        'top': 'bottom',
        'bottom': 'top',
        'left': 'right',
        'right': 'left'
    }

    def __init__(self, edge_detector: EdgeDetector):
        self.edge_detector = edge_detector
        self.matches: List[EdgeMatch] = []
        self.border_info = self._identify_border_pieces()

    def _identify_border_pieces(self) -> Dict[int, Dict[str, any]]:
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
                'is_middle_border': num_flat == 1,
                'flat_edges': flat_edges,
                'num_flat_edges': num_flat
            }

        return border_info

    def find_unique_best_matches(self, min_score: float = 0.0) -> List[EdgeMatch]:
        """
        Find unique best matches for all non-flat edges.

        For a 2x3 puzzle, we need exactly 7 matches:
        - 4 horizontal adjacencies (corners to middles)
        - 2 vertical corner-to-corner adjacencies
        - 1 vertical middle-to-middle adjacency

        Args:
            min_score: Minimum compatibility score (0.0 to 1.0)

        Returns:
            List of unique EdgeMatch objects (should be 7 for a 2x3 puzzle)
        """
        # First pass: strict matching (tab <-> slot only)
        final_matches, used_edges = self._find_matches_with_classification_filter(
            min_score, strict_classification=True
        )

        print(f"Selected {len(final_matches)} best unique matches (strict)")
        print(f"Matched {len(used_edges)} edges total")

        # Debug: print what edges were matched
        for m in final_matches:
            print(f"  Match: P{m.edge1.piece_id}.{m.edge1.edge_type} <-> P{m.edge2.piece_id}.{m.edge2.edge_type} "
                  f"(score={m.compatibility_score:.3f})")

        # Check if we have 7 matches - if not, try fallback matching
        if len(final_matches) != 7:
            print(f"WARNING: Expected 7 matches for 2x3 puzzle, got {len(final_matches)}")
            print("Attempting fallback matching with relaxed classification...")

            # Second pass: find remaining matches with relaxed classification
            fallback_matches = self._find_fallback_matches(min_score, used_edges)

            if fallback_matches:
                print(f"Found {len(fallback_matches)} fallback matches:")
                for m in fallback_matches:
                    e1_class = m.edge1.get_edge_type_classification()
                    e2_class = m.edge2.get_edge_type_classification()
                    print(f"  FALLBACK Match: P{m.edge1.piece_id}.{m.edge1.edge_type} ({e1_class}) <-> "
                          f"P{m.edge2.piece_id}.{m.edge2.edge_type} ({e2_class}) "
                          f"(score={m.compatibility_score:.3f})")
                    final_matches.append(m)
                    used_edges.add((m.edge1.piece_id, m.edge1.edge_type))
                    used_edges.add((m.edge2.piece_id, m.edge2.edge_type))

        # Final check
        if len(final_matches) != 7:
            print(f"FINAL WARNING: Still have {len(final_matches)} matches instead of 7")

        self.matches = final_matches
        return final_matches

    def _find_matches_with_classification_filter(self, min_score: float,
                                                 strict_classification: bool) -> Tuple[List[EdgeMatch], Set[Tuple[int, str]]]:
        """
        Find matches with optional strict classification filtering.

        Args:
            min_score: Minimum compatibility score
            strict_classification: If True, only match tab<->slot. If False, allow any non-flat combination.

        Returns:
            Tuple of (matches list, used edges set)
        """
        all_potential_matches = []
        seen_edge_pairs: Set[Tuple[Tuple[int, str], Tuple[int, str]]] = set()

        piece_edges = self.edge_detector.piece_edges

        for piece1_id, edges1 in piece_edges.items():
            for piece2_id, edges2 in piece_edges.items():
                if piece1_id >= piece2_id:
                    continue

                matches_for_pair = self._find_matches_between_pieces_ex(
                    piece1_id, edges1, piece2_id, edges2, min_score, strict_classification
                )

                for match in matches_for_pair:
                    edge1_key = (match.edge1.piece_id, match.edge1.edge_type)
                    edge2_key = (match.edge2.piece_id, match.edge2.edge_type)
                    edge_pair = tuple(sorted([edge1_key, edge2_key]))

                    if edge_pair not in seen_edge_pairs:
                        seen_edge_pairs.add(edge_pair)
                        all_potential_matches.append(match)

        all_potential_matches.sort(key=lambda m: m.compatibility_score, reverse=True)

        print(f"Found {len(all_potential_matches)} potential matches (threshold: {min_score}, strict: {strict_classification})")

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

        return final_matches, used_edges

    def _find_fallback_matches(self, min_score: float,
                               used_edges: Set[Tuple[int, str]]) -> List[EdgeMatch]:
        """
        Find matches for remaining unmatched edges with relaxed classification.
        Allows slot<->slot or tab<->tab matches if classification was wrong.

        Args:
            min_score: Minimum compatibility score
            used_edges: Set of edges already matched

        Returns:
            List of fallback matches
        """
        piece_edges = self.edge_detector.piece_edges

        # Collect all unmatched non-flat edges
        unmatched_edges: List[Tuple[int, str, PieceEdge]] = []
        for piece_id, edges in piece_edges.items():
            for edge_type, edge in edges.items():
                if edge.get_edge_type_classification() != 'flat':
                    edge_key = (piece_id, edge_type)
                    if edge_key not in used_edges:
                        unmatched_edges.append((piece_id, edge_type, edge))

        if len(unmatched_edges) < 2:
            return []

        # Find best matches among unmatched edges (ignoring classification)
        fallback_matches = []

        for i, (p1_id, e1_type, edge1) in enumerate(unmatched_edges):
            for j, (p2_id, e2_type, edge2) in enumerate(unmatched_edges):
                if i >= j:
                    continue
                if p1_id == p2_id:
                    continue  # Don't match edges from same piece

                # Evaluate match without classification requirement
                match = self._evaluate_edge_match(edge1, edge2)

                if match.compatibility_score >= min_score:
                    fallback_matches.append(match)

        # Sort by score and select non-overlapping matches
        fallback_matches.sort(key=lambda m: m.compatibility_score, reverse=True)

        selected_fallbacks = []
        newly_used: Set[Tuple[int, str]] = set()

        for match in fallback_matches:
            edge1_key = (match.edge1.piece_id, match.edge1.edge_type)
            edge2_key = (match.edge2.piece_id, match.edge2.edge_type)

            if edge1_key in newly_used or edge2_key in newly_used:
                continue

            selected_fallbacks.append(match)
            newly_used.add(edge1_key)
            newly_used.add(edge2_key)

        return selected_fallbacks

    def _find_matches_between_pieces_ex(self, piece1_id: int, edges1: Dict[str, PieceEdge],
                                        piece2_id: int, edges2: Dict[str, PieceEdge],
                                        min_score: float, strict_classification: bool) -> List[EdgeMatch]:
        """Find matches between two pieces with optional classification strictness."""
        matches = []

        for edge1_type, edge1 in edges1.items():
            edge1_classification = edge1.get_edge_type_classification()

            if edge1_classification == 'flat':
                continue

            for edge2_type, edge2 in edges2.items():
                edge2_classification = edge2.get_edge_type_classification()

                if edge2_classification == 'flat':
                    continue

                # Check classification compatibility
                if strict_classification:
                    if not self._are_complementary_edges(edge1_classification, edge2_classification):
                        continue

                match = self._evaluate_edge_match(edge1, edge2)

                if match.compatibility_score >= min_score:
                    matches.append(match)

        return matches

    def _find_matches_between_pieces(self, piece1_id: int, edges1: Dict[str, PieceEdge],
                                     piece2_id: int, edges2: Dict[str, PieceEdge],
                                     min_score: float) -> List[EdgeMatch]:
        """Find matches between two pieces by comparing all edge combinations."""
        return self._find_matches_between_pieces_ex(piece1_id, edges1, piece2_id, edges2,
                                                    min_score, strict_classification=True)

    def _evaluate_edge_match(self, edge1: PieceEdge, edge2: PieceEdge) -> EdgeMatch:
        """Evaluate the compatibility between two edges."""
        length_similarity = self._calculate_length_similarity(edge1, edge2)
        shape_similarity = self._calculate_normalized_shape_similarity(edge1, edge2)
        classification_match = self._check_classification_compatibility(edge1, edge2)
        classification_score = 1.0 if classification_match else 0.0

        compatibility_score = (
                0.20 * length_similarity +
                0.60 * shape_similarity +
                0.20 * classification_score
        )

        rotation_angle = self._normalize_angle(edge1.angle - edge2.angle + 180.0)

        return EdgeMatch(
            edge1=edge1,
            edge2=edge2,
            compatibility_score=compatibility_score,
            length_similarity=length_similarity,
            shape_similarity=shape_similarity,
            classification_match=classification_match,
            rotation_angle=rotation_angle
        )

    def _calculate_normalized_shape_similarity(self, edge1: PieceEdge, edge2: PieceEdge) -> float:
        """Calculate shape similarity by normalizing both edges to a common frame."""
        profile1 = self._get_normalized_edge_profile(edge1, flip=False)
        profile2 = self._get_normalized_edge_profile(edge2, flip=True)

        if profile1 is None or profile2 is None:
            return 0.0

        similarity_forward = self._compare_profiles(profile1, profile2)
        similarity_reversed = self._compare_profiles(profile1, profile2[::-1])

        return max(similarity_forward, similarity_reversed)

    def _get_normalized_edge_profile(self, edge: PieceEdge, flip: bool = False,
                                     num_samples: int = 100) -> Optional[np.ndarray]:
        """Get a normalized profile of edge deviations from the baseline."""
        points = edge.points.reshape(-1, 2).astype(np.float64)

        if len(points) < 3:
            return None

        start = np.array(edge.start_point, dtype=np.float64)
        end = np.array(edge.end_point, dtype=np.float64)

        baseline = end - start
        baseline_length = np.linalg.norm(baseline)

        if baseline_length < 1e-6:
            return None

        baseline_dir = baseline / baseline_length
        perp_dir = np.array([-baseline_dir[1], baseline_dir[0]])

        deviations = []
        for point in points:
            relative = point - start
            t = np.dot(relative, baseline_dir) / baseline_length
            d = np.dot(relative, perp_dir) / baseline_length
            deviations.append((t, d))

        deviations.sort(key=lambda x: x[0])

        t_values = np.array([d[0] for d in deviations])
        d_values = np.array([d[1] for d in deviations])

        t_uniform = np.linspace(0, 1, num_samples)
        profile = np.interp(t_uniform, t_values, d_values)

        if flip:
            profile = -profile

        return profile

    def _compare_profiles(self, profile1: np.ndarray, profile2: np.ndarray) -> float:
        """Compare two edge profiles using correlation and distance metrics."""
        if len(profile1) != len(profile2):
            profile2 = np.interp(
                np.linspace(0, 1, len(profile1)),
                np.linspace(0, 1, len(profile2)),
                profile2
            )

        correlation = self._calculate_correlation(profile1, profile2)
        correlation_score = (correlation + 1.0) / 2.0

        diff = np.abs(profile1 - profile2)
        mean_diff = np.mean(diff)
        diff_score = max(0.0, 1.0 - mean_diff / 0.15)

        similarity = 0.6 * correlation_score + 0.4 * diff_score

        return similarity

    def _calculate_correlation(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Calculate Pearson correlation between two signatures."""
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
        return (class1 == 'tab' and class2 == 'slot') or (class1 == 'slot' and class2 == 'tab')

    def _calculate_length_similarity(self, edge1: PieceEdge, edge2: PieceEdge) -> float:
        """Calculate similarity based on edge lengths."""
        len1 = edge1.length
        len2 = edge2.length

        if len1 == 0 or len2 == 0:
            return 0.0

        return min(len1, len2) / max(len1, len2)

    def _check_classification_compatibility(self, edge1: PieceEdge, edge2: PieceEdge) -> bool:
        """Check if edge classifications are compatible (tab matches slot)."""
        class1 = edge1.get_edge_type_classification()
        class2 = edge2.get_edge_type_classification()

        if class1 == 'flat' or class2 == 'flat':
            return False

        return (class1 == 'tab' and class2 == 'slot') or (class1 == 'slot' and class2 == 'tab')

    def identify_border_pieces(self) -> Dict[int, Dict[str, any]]:
        """Return cached border info."""
        return self.border_info

    def get_match_statistics(self) -> Dict:
        """Get statistics about the matches."""
        if not self.matches:
            return {}

        scores = [m.compatibility_score for m in self.matches]


        flat_flat = sum(1 for m in self.matches if
                        m.edge1.get_edge_type_classification() == 'flat' and
                        m.edge2.get_edge_type_classification() == 'flat')
        tab_slot = sum(1 for m in self.matches if m.classification_match and
                       'flat' not in [m.edge1.get_edge_type_classification(),
                                      m.edge2.get_edge_type_classification()])

        corner_pieces = [pid for pid, info in self.border_info.items() if info['is_corner']]
        border_pieces = [pid for pid, info in self.border_info.items() if info['is_border']]
        interior_pieces = [pid for pid, info in self.border_info.items() if not info['is_border']]

        stats = {
            'total_matches': len(self.matches),
            'avg_score': float(np.mean(scores)) if scores else 0.0,
            'max_score': float(np.max(scores)) if scores else 0.0,
            'min_score': float(np.min(scores)) if scores else 0.0,
            'flat_flat_matches': flat_flat,
            'tab_slot_matches': tab_slot,
            'high_confidence_matches': sum(1 for s in scores if s >= 0.8),
            'medium_confidence_matches': sum(1 for s in scores if 0.6 <= s < 0.8),
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