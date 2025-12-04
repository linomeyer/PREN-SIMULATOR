"""
Puzzle solver that assembles puzzle pieces using edge matches.

For a 2x3 grid:
- 4 corner pieces (2 flat edges) at (0,0), (0,2), (1,0), (1,2)
- 2 middle border pieces (1 flat edge) at (0,1), (1,1)

The solver uses edge matches to determine:
1. Which specific corner goes to which corner position
2. Which middle border piece goes top vs bottom
3. The exact rotation for each piece
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from itertools import permutations

from app.main.puzzle_solver.edge_matching.edge_matcher import EdgeMatcher, EdgeMatch
from app.main.puzzle_solver.edge_detection.edge_detector import EdgeDetector
from app.main.puzzle_solver.piece_extraction.extractor import PuzzlePiece


@dataclass
class PlacedPiece:
    """Represents a piece placed in the solved puzzle."""
    piece_id: int
    grid_row: int
    grid_col: int
    rotation: float  # Rotation in degrees
    piece: PuzzlePiece

    def __repr__(self):
        return f"PlacedPiece(id={self.piece_id}, pos=({self.grid_row}, {self.grid_col}), rot={self.rotation:.1f}°)"


@dataclass
class PuzzleSolution:
    """Represents the complete puzzle solution."""
    placed_pieces: List[PlacedPiece]
    grid_rows: int
    grid_cols: int
    matches_used: List[EdgeMatch]
    confidence: float

    def get_piece_at(self, row: int, col: int) -> Optional[PlacedPiece]:
        """Get the piece at a specific grid position."""
        for piece in self.placed_pieces:
            if piece.grid_row == row and piece.grid_col == col:
                return piece
        return None

    def get_grid_layout(self) -> List[List[Optional[int]]]:
        """Get a 2D grid showing piece IDs at each position."""
        grid = [[None for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        for piece in self.placed_pieces:
            if 0 <= piece.grid_row < self.grid_rows and 0 <= piece.grid_col < self.grid_cols:
                grid[piece.grid_row][piece.grid_col] = piece.piece_id
        return grid


class PuzzleSolver:
    """
    Solves 2x3 puzzles using edge matches to determine piece positions.

    Strategy:
    1. Build an adjacency graph from edge matches
    2. Place pieces by following the match connections
    3. Determine rotations based on which edges are matched
    """

    GRID_ROWS = 2
    GRID_COLS = 3

    # Required flat edges for each position
    POSITION_FLAT_EDGES = {
        (0, 0): {'top', 'left'},
        (0, 1): {'top'},
        (0, 2): {'top', 'right'},
        (1, 0): {'bottom', 'left'},
        (1, 1): {'bottom'},
        (1, 2): {'bottom', 'right'},
    }

    def __init__(self, edge_matcher: EdgeMatcher, pieces: List[PuzzlePiece]):
        self.edge_matcher = edge_matcher
        self.pieces = pieces
        self.edge_detector = edge_matcher.edge_detector
        self.border_info = edge_matcher.identify_border_pieces()

        # Build adjacency info from matches
        self.adjacency = self._build_adjacency_from_matches()

    def _build_adjacency_from_matches(self) -> Dict[Tuple[int, str], Tuple[int, str, EdgeMatch]]:
        """
        Build adjacency lookup from edge matches.

        Returns dict mapping (piece_id, edge_type) -> (other_piece_id, other_edge_type, match)
        """
        adjacency = {}

        for match in self.edge_matcher.matches:
            p1_id = match.edge1.piece_id
            p1_edge = match.edge1.edge_type
            p2_id = match.edge2.piece_id
            p2_edge = match.edge2.edge_type

            adjacency[(p1_id, p1_edge)] = (p2_id, p2_edge, match)
            adjacency[(p2_id, p2_edge)] = (p1_id, p1_edge, match)

        return adjacency

    def solve(self) -> Optional[PuzzleSolution]:
        """Solve the puzzle using edge matches to build the grid."""
        print(f"Solving puzzle: {len(self.pieces)} pieces, grid: {self.GRID_ROWS}x{self.GRID_COLS}")
        print(f"Available matches: {len(self.edge_matcher.matches)}")

        # Print adjacency info
        print("Adjacency from matches:")
        for (pid, edge), (other_pid, other_edge, match) in self.adjacency.items():
            print(f"  P{pid}.{edge} <-> P{other_pid}.{other_edge}")

        # Identify corners and middles
        corners = [pid for pid, info in self.border_info.items() if info['is_corner']]
        middles = [pid for pid, info in self.border_info.items() if info['is_middle_border']]

        if len(corners) != 4:
            print(f"Error: Expected 4 corner pieces, found {len(corners)}: {corners}")
            return None
        if len(middles) != 2:
            print(f"Error: Expected 2 middle pieces, found {len(middles)}: {middles}")
            return None

        print(f"Corner pieces: {corners}")
        print(f"Middle pieces: {middles}")

        # Try building from each corner as top-left
        best_solution = None
        best_score = -1

        for start_corner in corners:
            solution = self._build_from_corner(start_corner, corners, middles)
            if solution and solution.confidence > best_score:
                best_score = solution.confidence
                best_solution = solution

        if best_solution:
            print(f"Solution found! Placed {len(best_solution.placed_pieces)} pieces")
            print(f"Solution confidence: {best_solution.confidence:.2%}")
        else:
            print("Could not find a valid solution")

        return best_solution

    def _build_from_corner(self, start_corner: int, corners: List[int],
                           middles: List[int]) -> Optional[PuzzleSolution]:
        """
        Try to build the puzzle starting from a specific corner as top-left.

        Grid layout:
        (0,0) - (0,1) - (0,2)
          |       |       |
        (1,0) - (1,1) - (1,2)
        """
        # Calculate discrete rotation for start corner at (0,0)
        discrete_rotation = self._calculate_rotation_for_position(start_corner, (0, 0))
        if discrete_rotation is None:
            return None  # Can't place this corner at top-left

        # Add base rotation from edge detector to get accurate total rotation
        base_rotation = self._get_base_rotation_from_detector(start_corner)
        start_rotation = base_rotation + discrete_rotation

        placements = {}
        matches_used = []
        total_score = 0.0

        # Place starting corner at (0,0)
        placements[(0, 0)] = {
            'piece_id': start_corner,
            'rotation': start_rotation
        }

        # Get flat edges for start corner (after rotation)
        start_flat = self.border_info[start_corner]['flat_edges']

        # Find non-flat edges of start corner (these connect to neighbors)
        start_edges = self.edge_detector.piece_edges[start_corner]

        # Build grid by following matches
        # From (0,0), go right to find (0,1), then continue to (0,2)
        # From (0,0), go down to find (1,0)
        # etc.

        # Step 1: Find piece to the right of (0,0) -> position (0,1)
        # The right edge of piece at (0,0) matches with left edge of piece at (0,1)
        right_edge_after_rotation = self._get_edge_after_rotation('right', discrete_rotation)

        neighbor_info = self.adjacency.get((start_corner, right_edge_after_rotation))
        if not neighbor_info:
            # Try finding which original edge becomes 'right' after rotation
            original_right = self._get_original_edge('right', discrete_rotation)
            neighbor_info = self.adjacency.get((start_corner, original_right))

        if neighbor_info:
            neighbor_id, neighbor_edge, match = neighbor_info
            # Calculate rotation for neighbor: its 'neighbor_edge' should become 'left'
            neighbor_rotation = self._calculate_rotation_to_align_edge(neighbor_id, neighbor_edge, 'left')
            if neighbor_rotation is not None:
                placements[(0, 1)] = {
                    'piece_id': neighbor_id,
                    'rotation': neighbor_rotation
                }
                matches_used.append(match)
                total_score += match.compatibility_score

        # Continue building the grid...
        # This is complex - let me use a more systematic approach

        return self._systematic_build(start_corner, start_rotation)

    def _systematic_build(self, start_piece: int, start_rotation: float) -> Optional[PuzzleSolution]:
        """
        Systematically build the grid starting from top-left corner.
        """
        grid = {}  # (row, col) -> {'piece_id': int, 'rotation': float}
        used_pieces = set()
        matches_used = []
        total_score = 0.0

        # Place starting piece at (0, 0)
        grid[(0, 0)] = {'piece_id': start_piece, 'rotation': start_rotation}
        used_pieces.add(start_piece)

        # Process positions in order: row by row, left to right
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        for pos in positions:
            row, col = pos

            # Find which neighbor we can use to determine this position
            # Prefer left neighbor for horizontal connection, or top neighbor for vertical

            placed = False

            # Try to place based on left neighbor
            if col > 0 and (row, col - 1) in grid:
                left_info = grid[(row, col - 1)]
                left_piece = left_info['piece_id']
                left_rotation = left_info['rotation']

                # Find which edge of left_piece is its "right" edge after rotation
                original_right_edge = self._get_original_edge('right', left_rotation)

                # Look up what this edge matches to
                match_info = self.adjacency.get((left_piece, original_right_edge))

                if match_info:
                    neighbor_id, neighbor_edge, match = match_info
                    if neighbor_id not in used_pieces:
                        # Calculate rotation: neighbor_edge should become 'left' after rotation
                        discrete_rotation = self._calculate_rotation_to_align_edge(
                            neighbor_id, neighbor_edge, 'left'
                        )

                        # Also verify flat edges match position requirements
                        if discrete_rotation is not None:
                            expected_flat = self.POSITION_FLAT_EDGES[pos]
                            actual_flat = self._get_flat_edges_after_rotation(neighbor_id, discrete_rotation)

                            if expected_flat == actual_flat:
                                # Calculate the accurate base rotation from edge detector
                                base_rotation = self._get_base_rotation_from_detector(neighbor_id)
                                # Combine base rotation with discrete 90° steps
                                accurate_rotation = base_rotation + discrete_rotation

                                grid[pos] = {'piece_id': neighbor_id, 'rotation': accurate_rotation}
                                used_pieces.add(neighbor_id)
                                matches_used.append(match)
                                total_score += match.compatibility_score
                                placed = True
                                print(f"  Placed P{neighbor_id} at {pos} (from left neighbor P{left_piece})")

            # Try to place based on top neighbor if not placed yet
            if not placed and row > 0 and (row - 1, col) in grid:
                top_info = grid[(row - 1, col)]
                top_piece = top_info['piece_id']
                top_rotation = top_info['rotation']

                # Find which edge of top_piece is its "bottom" edge after rotation
                original_bottom_edge = self._get_original_edge('bottom', top_rotation)

                # Look up what this edge matches to
                match_info = self.adjacency.get((top_piece, original_bottom_edge))

                if match_info:
                    neighbor_id, neighbor_edge, match = match_info
                    if neighbor_id not in used_pieces:
                        # Calculate rotation: neighbor_edge should become 'top' after rotation
                        discrete_rotation = self._calculate_rotation_to_align_edge(
                            neighbor_id, neighbor_edge, 'top'
                        )

                        # Also verify flat edges match position requirements
                        if discrete_rotation is not None:
                            expected_flat = self.POSITION_FLAT_EDGES[pos]
                            actual_flat = self._get_flat_edges_after_rotation(neighbor_id, discrete_rotation)

                            if expected_flat == actual_flat:
                                # Calculate the accurate base rotation from edge detector
                                base_rotation = self._get_base_rotation_from_detector(neighbor_id)
                                # Combine base rotation with discrete 90° steps
                                accurate_rotation = base_rotation + discrete_rotation

                                grid[pos] = {'piece_id': neighbor_id, 'rotation': accurate_rotation}
                                used_pieces.add(neighbor_id)
                                matches_used.append(match)
                                total_score += match.compatibility_score
                                placed = True
                                print(f"  Placed P{neighbor_id} at {pos} (from top neighbor P{top_piece})")

            if not placed:
                print(f"  Could not place piece at {pos}")
                return None

        # Build solution
        placements = []
        for pos, info in grid.items():
            placements.append(PlacedPiece(
                piece_id=info['piece_id'],
                grid_row=pos[0],
                grid_col=pos[1],
                rotation=info['rotation'],
                piece=self.pieces[info['piece_id']]
            ))

        confidence = total_score / len(matches_used) if matches_used else 0.0

        return PuzzleSolution(
            placed_pieces=placements,
            grid_rows=self.GRID_ROWS,
            grid_cols=self.GRID_COLS,
            matches_used=matches_used,
            confidence=confidence
        )

    def _get_base_rotation_from_detector(self, piece_id: int) -> float:
        """
        Get the base rotation for a piece from the edge detector.
        This is the rotation needed to align the piece's detected orientation.
        """
        if not hasattr(self.edge_detector, 'piece_edges'):
            return 0.0

        edges = self.edge_detector.piece_edges.get(piece_id, {})
        if 'top' not in edges:
            return 0.0

        top_edge = edges['top']
        piece = self.pieces[piece_id]
        center = np.array(piece.center, dtype=np.float32)

        # Calculate the midpoint of the top edge
        start = np.array(top_edge.start_point, dtype=np.float32)
        end = np.array(top_edge.end_point, dtype=np.float32)
        mid = (start + end) / 2

        # Vector from center to midpoint of top edge - this should point UP (-Y)
        vec_to_top = mid - center

        # Calculate angle of this vector
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

    def _get_original_edge(self, target_edge: str, rotation: float) -> str:
        """
        Given a target edge position and rotation, find which original edge
        ends up at that position.

        E.g., if rotation=90°, what was originally 'left' becomes 'top'.
        So if target='top' and rotation=90°, original='left'.

        Rotation is clockwise.
        """
        edges = ['top', 'right', 'bottom', 'left']
        target_idx = edges.index(target_edge)
        steps = int(round(rotation / 90.0)) % 4
        # Reverse the rotation to find original
        original_idx = (target_idx - steps) % 4
        return edges[original_idx]

    def _get_edge_after_rotation(self, original_edge: str, rotation: float) -> str:
        """
        Given an original edge and rotation, find where it ends up.

        Rotation is clockwise.
        """
        edges = ['top', 'right', 'bottom', 'left']
        original_idx = edges.index(original_edge)
        steps = int(round(rotation / 90.0)) % 4
        new_idx = (original_idx + steps) % 4
        return edges[new_idx]

    def _calculate_rotation_to_align_edge(self, piece_id: int, edge: str, target_position: str) -> Optional[float]:
        """
        Calculate rotation needed so that 'edge' ends up at 'target_position'.

        E.g., if edge='bottom' and target='left', we need to rotate so bottom becomes left.
        """
        edges = ['top', 'right', 'bottom', 'left']
        edge_idx = edges.index(edge)
        target_idx = edges.index(target_position)

        steps = (target_idx - edge_idx) % 4
        return steps * 90.0

    def _get_flat_edges_after_rotation(self, piece_id: int, rotation: float) -> Set[str]:
        """Get the set of flat edges after applying rotation."""
        original_flat = set(self.border_info[piece_id]['flat_edges'])
        return self._rotate_edge_set(original_flat, rotation)

    def _calculate_rotation_for_position(self, piece_id: int, position: Tuple[int, int]) -> Optional[float]:
        """
        Calculate the rotation needed for a piece at a given position.

        The piece must be rotated so its flat edges align with the grid boundaries.
        """
        flat_edges = set(self.border_info[piece_id]['flat_edges'])
        required_flat = self.POSITION_FLAT_EDGES[position]

        # Try all 4 rotations
        for rotation_steps in range(4):
            rotation = rotation_steps * 90.0
            rotated_flat = self._rotate_edge_set(flat_edges, rotation)

            if rotated_flat == required_flat:
                return rotation

        return None  # No valid rotation found

    def _rotate_edge_set(self, edges: Set[str], rotation: float) -> Set[str]:
        """Rotate a set of edge types by the given rotation."""
        edge_list = ['top', 'right', 'bottom', 'left']
        steps = int(round(rotation / 90.0)) % 4

        rotated = set()
        for edge in edges:
            idx = edge_list.index(edge)
            new_idx = (idx + steps) % 4
            rotated.add(edge_list[new_idx])

        return rotated