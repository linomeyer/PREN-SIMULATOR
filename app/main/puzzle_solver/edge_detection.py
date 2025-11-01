import cv2
import numpy as np
from typing import List, Tuple, Dict
from app.main.puzzle_solver.extractor import PuzzlePiece


class PieceEdge:
    """Represents one edge of a puzzle piece."""

    def __init__(self, piece_id: int, edge_type: str, points: np.ndarray, start_point: Tuple[int, int], end_point: Tuple[int, int]):
        """
        Initialize a puzzle piece edge.

        Args:
            piece_id: ID of the piece this edge belongs to
            edge_type: Type of edge ('top', 'right', 'bottom', 'left')
            points: Array of points forming the edge contour
            start_point: Starting corner point of the edge
            end_point: Ending corner point of the edge
        """
        self.piece_id = piece_id
        self.edge_type = edge_type
        self.points = points
        self.start_point = start_point
        self.end_point = end_point
        self.length = self._calculate_length()
        self.shape_signature = self._compute_shape_signature()

    def _calculate_length(self) -> float:
        """Calculate the length of the edge."""
        return cv2.arcLength(self.points, False)

    def _compute_shape_signature(self, num_samples: int = 50) -> np.ndarray:
        """
        Compute a shape signature for the edge based on curvature.

        Args:
            num_samples: Number of points to sample along the edge

        Returns:
            Array representing the shape signature
        """
        if len(self.points) < 3:
            return np.zeros(num_samples)

        # Sample points uniformly along the edge
        points_2d = self.points.reshape(-1, 2)
        indices = np.linspace(0, len(points_2d) - 1, num_samples, dtype=int)
        sampled_points = points_2d[indices]

        # Calculate curvature at each sampled point
        curvatures = []
        for i in range(1, len(sampled_points) - 1):
            p1 = sampled_points[i - 1]
            p2 = sampled_points[i]
            p3 = sampled_points[i + 1]

            # Vectors
            v1 = p2 - p1
            v2 = p3 - p2

            # Calculate angle between vectors
            angle1 = np.arctan2(v1[1], v1[0])
            angle2 = np.arctan2(v2[1], v2[0])

            # Compute angle difference (curvature)
            curvature = angle2 - angle1
            # Normalize to [-pi, pi]
            curvature = np.arctan2(np.sin(curvature), np.cos(curvature))
            curvatures.append(curvature)

        # Pad to match num_samples
        while len(curvatures) < num_samples:
            curvatures.append(0)

        return np.array(curvatures[:num_samples])

    def get_edge_type_classification(self) -> str:
        """
        Classify the edge as flat, tab (out), or slot (in).

        Returns:
            'flat', 'tab', or 'slot'
        """
        # Calculate deviation from straight line
        start = np.array(self.start_point)
        end = np.array(self.end_point)

        # Create straight line
        line_length = np.linalg.norm(end - start)
        if line_length == 0:
            return 'flat'

        # Calculate maximum perpendicular distance from line
        points_2d = self.points.reshape(-1, 2)
        max_dist = 0
        max_side = 0

        for point in points_2d:
            # Calculate perpendicular distance from point to line
            # Using cross product method
            v1 = end - start
            v2 = point - start
            cross = np.cross(v1, v2)
            dist = abs(cross) / line_length

            if dist > abs(max_dist):
                # Determine which side of the line the point is on
                side = np.sign(cross)
                max_dist = dist
                max_side = side

        # Threshold for classification (adjust based on your puzzle pieces)
        threshold = line_length * 0.1  # 10% of edge length

        if max_dist < threshold:
            return 'flat'
        elif max_side > 0:
            return 'tab'  # Protrusion
        else:
            return 'slot'  # Indentation


class EdgeDetector:
    """Detects and analyzes edges of puzzle pieces."""

    def __init__(self, pieces: List[PuzzlePiece]):
        """
        Initialize edge detector with extracted puzzle pieces.

        Args:
            pieces: List of PuzzlePiece objects from the extractor
        """
        self.pieces = pieces
        self.edges: List[PieceEdge] = []
        self.piece_edges: Dict[int, Dict[str, PieceEdge]] = {}  # piece_id -> {edge_type -> edge}

    def detect_edges(self) -> List[PieceEdge]:
        """
        Detect all edges for all puzzle pieces.

        Returns:
            List of PieceEdge objects
        """
        self.edges = []
        self.piece_edges = {}

        for idx, piece in enumerate(self.pieces):
            piece_edges = self._detect_piece_edges(piece, idx)
            self.piece_edges[idx] = piece_edges
            self.edges.extend(piece_edges.values())

        print(f"Detected {len(self.edges)} edges from {len(self.pieces)} pieces")
        return self.edges

    def _detect_piece_edges(self, piece: PuzzlePiece, piece_id: int) -> Dict[str, PieceEdge]:
        """
        Detect the four edges of a single puzzle piece.

        Args:
            piece: PuzzlePiece object
            piece_id: Unique identifier for the piece

        Returns:
            Dictionary mapping edge type to PieceEdge object
        """
        # Find corner points of the piece
        corners = self._find_corners(piece.contour)

        if len(corners) < 4:
            # If we can't find 4 corners, use bounding box corners
            x, y, w, h = piece.bbox
            corners = [
                (x, y),  # top-left
                (x + w, y),  # top-right
                (x + w, y + h),  # bottom-right
                (x, y + h)  # bottom-left
            ]

        # Sort corners to get consistent ordering (top-left, top-right, bottom-right, bottom-left)
        corners = self._sort_corners(corners)

        # Extract edge segments between corners
        edges = {}
        edge_types = ['top', 'right', 'bottom', 'left']

        for i, edge_type in enumerate(edge_types):
            start_corner = corners[i]
            end_corner = corners[(i + 1) % 4]

            # Extract contour points between these corners
            edge_points = self._extract_edge_points(piece.contour, start_corner, end_corner)

            # Create edge object
            edge = PieceEdge(piece_id, edge_type, edge_points, start_corner, end_corner)
            edges[edge_type] = edge

        return edges

    def _find_corners(self, contour: np.ndarray, num_corners: int = 4) -> List[Tuple[int, int]]:
        """
        Find corner points in the contour using corner detection.

        Args:
            contour: Contour of the puzzle piece
            num_corners: Expected number of corners

        Returns:
            List of corner coordinates
        """
        # Approximate the contour to reduce points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If approximation gives us roughly the right number of corners, use those
        if len(approx) >= num_corners and len(approx) <= num_corners + 2:
            return [tuple(point[0]) for point in approx[:num_corners]]

        # Otherwise, find corners using curvature analysis
        contour_2d = contour.reshape(-1, 2)
        angles = []

        window_size = max(5, len(contour_2d) // 50)

        for i in range(len(contour_2d)):
            p1_idx = (i - window_size) % len(contour_2d)
            p2_idx = i
            p3_idx = (i + window_size) % len(contour_2d)

            p1 = contour_2d[p1_idx]
            p2 = contour_2d[p2_idx]
            p3 = contour_2d[p3_idx]

            v1 = p2 - p1
            v2 = p3 - p2

            # Calculate angle
            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angle = abs(np.arctan2(np.sin(angle), np.cos(angle)))
            angles.append((i, angle))

        # Find peaks (corners) in angles
        angles.sort(key=lambda x: x[1], reverse=True)
        corner_indices = [angles[i][0] for i in range(min(num_corners, len(angles)))]
        corner_indices.sort()

        corners = [tuple(contour_2d[idx]) for idx in corner_indices]
        return corners

    def _sort_corners(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Sort corners in clockwise order starting from top-left.

        Args:
            corners: List of corner coordinates

        Returns:
            Sorted list of corners
        """
        center = np.mean(corners, axis=0)

        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])

        sorted_corners = sorted(corners, key=angle_from_center)

        # Rotate list so top-left corner is first
        # Top-left has smallest x+y sum
        min_idx = min(range(len(sorted_corners)),
                      key=lambda i: sorted_corners[i][0] + sorted_corners[i][1])
        sorted_corners = sorted_corners[min_idx:] + sorted_corners[:min_idx]

        return sorted_corners

    def _extract_edge_points(self, contour: np.ndarray,
                             start_corner: Tuple[int, int],
                             end_corner: Tuple[int, int]) -> np.ndarray:
        """
        Extract contour points between two corner points.

        Args:
            contour: Full contour of the piece
            start_corner: Starting corner point
            end_corner: Ending corner point

        Returns:
            Array of points forming the edge
        """
        contour_2d = contour.reshape(-1, 2)

        # Find indices of corners in contour
        start_idx = self._find_nearest_point_index(contour_2d, start_corner)
        end_idx = self._find_nearest_point_index(contour_2d, end_corner)

        # Extract points between corners
        if end_idx > start_idx:
            edge_points = contour_2d[start_idx:end_idx + 1]
        else:
            # Wrap around
            edge_points = np.vstack([contour_2d[start_idx:], contour_2d[:end_idx + 1]])

        return edge_points.reshape(-1, 1, 2)

    @staticmethod
    def find_nearest_point_index(points: np.ndarray, target: Tuple[int, int]) -> int:
        distances = np.linalg.norm(points - np.array(target), axis=1)
        return int(np.argmin(distances))

    def get_edge_statistics(self) -> Dict:
        if not self.edges:
            return {}

        lengths = [edge.length for edge in self.edges]
        classifications = [edge.get_edge_type_classification() for edge in self.edges]

        stats = {
            'total_edges': len(self.edges),
            'avg_edge_length': np.mean(lengths),
            'min_edge_length': np.min(lengths),
            'max_edge_length': np.max(lengths),
            'flat_edges': classifications.count('flat'),
            'tab_edges': classifications.count('tab'),
            'slot_edges': classifications.count('slot')
        }

        return stats

    def get_piece_edge_info(self, piece_id: int) -> Dict:
        if piece_id not in self.piece_edges:
            return {}

        edges = self.piece_edges[piece_id]
        info = {}

        for edge_type, edge in edges.items():
            info[edge_type] = {
                'length': edge.length,
                'classification': edge.get_edge_type_classification(),
                'start_point': edge.start_point,
                'end_point': edge.end_point,
                'num_points': len(edge.points)
            }

        return info