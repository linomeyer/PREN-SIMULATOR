"""
Puzzle solver that assembles puzzle pieces using edge matches.

This module takes the edge matches from EdgeMatcher and determines:
1. The relative positions of all pieces
2. The rotation needed for each piece
3. The final solved puzzle layout
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

from app.main.puzzle_solver.edge_matching.edge_matcher import EdgeMatcher, EdgeMatch
from app.main.puzzle_solver.edge_detection.edge_detector import EdgeDetector
from app.main.puzzle_solver.piece_extraction.extractor import PuzzlePiece


@dataclass
class PlacedPiece:
    """Represents a piece placed in the solved puzzle."""
    piece_id: int
    grid_row: int
    grid_col: int
    rotation: float  # Rotation in degrees to apply to the piece
    piece: PuzzlePiece  # Reference to the original piece

    def __repr__(self):
        return f"PlacedPiece(id={self.piece_id}, pos=({self.grid_row}, {self.grid_col}), rot={self.rotation:.1f}°)"


@dataclass
class PuzzleSolution:
    """Represents the complete puzzle solution."""
    placed_pieces: List[PlacedPiece]
    grid_rows: int
    grid_cols: int
    matches_used: List[EdgeMatch]
    confidence: float  # Overall solution confidence (0-1)

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
    Solves puzzles by assembling pieces based on edge matches.

    Strategy:
    1. Identify corner pieces (2 flat edges) - these anchor the puzzle
    2. Identify border pieces (1 flat edge) - these form the frame
    3. Start from a corner and build outward
    4. Use edge matches to determine which pieces connect
    5. Calculate rotations needed to align pieces
    """

    # Edge direction mappings
    OPPOSITE_EDGE = {
        'top': 'bottom',
        'bottom': 'top',
        'left': 'right',
        'right': 'left'
    }

    # Grid offsets for each edge direction
    EDGE_OFFSETS = {
        'top': (-1, 0),    # piece above
        'bottom': (1, 0),  # piece below
        'left': (0, -1),   # piece to the left
        'right': (0, 1)    # piece to the right
    }

    def __init__(self, edge_matcher: EdgeMatcher, pieces: List[PuzzlePiece]):
        """
        Initialize the puzzle solver.

        Args:
            edge_matcher: EdgeMatcher with computed matches
            pieces: List of extracted puzzle pieces
        """
        self.edge_matcher = edge_matcher
        self.pieces = pieces
        self.edge_detector = edge_matcher.edge_detector

        # Build match lookup for quick access
        self.match_lookup = self._build_match_lookup()

        # Identify piece types
        self.border_info = edge_matcher.identify_border_pieces()

    def _build_match_lookup(self) -> Dict[Tuple[int, str], EdgeMatch]:
        """
        Build a lookup dictionary for finding matches by piece_id and edge_type.

        Returns:
            Dict mapping (piece_id, edge_type) to EdgeMatch
        """
        lookup = {}
        for match in self.edge_matcher.matches:
            key1 = (match.edge1.piece_id, match.edge1.edge_type)
            key2 = (match.edge2.piece_id, match.edge2.edge_type)
            lookup[key1] = match
            lookup[key2] = match
        return lookup

    def solve(self) -> Optional[PuzzleSolution]:
        """
        Solve the puzzle by placing all pieces in their correct positions.

        Returns:
            PuzzleSolution if successful, None if puzzle cannot be solved
        """
        num_pieces = len(self.pieces)

        # Determine grid dimensions
        grid_rows, grid_cols = self._estimate_grid_size(num_pieces)

        print(f"Solving puzzle: {num_pieces} pieces, estimated grid: {grid_rows}x{grid_cols}")

        # Find corner pieces (should have 2 flat edges)
        corners = self._find_corner_pieces()
        if not corners:
            print("Error: No corner pieces found!")
            return None

        print(f"Found {len(corners)} corner pieces: {corners}")

        # Start from the first corner and build the puzzle
        solution = self._build_solution(corners[0], grid_rows, grid_cols)

        if solution:
            print(f"Solution found! Placed {len(solution.placed_pieces)} pieces")
            print(f"Solution confidence: {solution.confidence:.2%}")
        else:
            print("Could not find a complete solution")

        return solution

    def _estimate_grid_size(self, num_pieces: int) -> Tuple[int, int]:
        """
        Estimate the grid dimensions based on number of pieces.

        For 6 pieces: 2x3 or 3x2
        For 4 pieces: 2x2
        For 9 pieces: 3x3
        etc.
        """
        # Common puzzle dimensions
        common_grids = {
            4: (2, 2),
            6: (2, 3),
            8: (2, 4),
            9: (3, 3),
            12: (3, 4),
            16: (4, 4),
            20: (4, 5),
            24: (4, 6),
            25: (5, 5)
        }

        if num_pieces in common_grids:
            return common_grids[num_pieces]

        # For other counts, find factors closest to square
        sqrt_n = int(np.sqrt(num_pieces))
        for rows in range(sqrt_n, 0, -1):
            if num_pieces % rows == 0:
                cols = num_pieces // rows
                return (rows, cols)

        # Fallback
        return (1, num_pieces)

    def _find_corner_pieces(self) -> List[int]:
        """Find pieces that are corners (have 2 flat edges)."""
        corners = []
        for piece_id, info in self.border_info.items():
            if info['is_corner'] and info['num_flat_edges'] >= 2:
                corners.append(piece_id)
        return corners

    def _find_border_pieces(self) -> List[int]:
        """Find pieces that are on the border but not corners (have 1 flat edge)."""
        border = []
        for piece_id, info in self.border_info.items():
            if info['is_border'] and info['num_flat_edges'] == 1:
                border.append(piece_id)
        return border

    def _find_interior_pieces(self) -> List[int]:
        """Find pieces that are interior (have 0 flat edges)."""
        interior = []
        for piece_id, info in self.border_info.items():
            if not info['is_border']:
                interior.append(piece_id)
        return interior

    def _build_solution(self, start_corner: int, grid_rows: int, grid_cols: int) -> Optional[PuzzleSolution]:
        """
        Build the puzzle solution starting from a corner piece.

        Args:
            start_corner: Piece ID of the starting corner
            grid_rows: Number of rows in the grid
            grid_cols: Number of columns in the grid

        Returns:
            PuzzleSolution if successful, None otherwise
        """
        placed: Dict[int, PlacedPiece] = {}  # piece_id -> PlacedPiece
        grid: Dict[Tuple[int, int], int] = {}  # (row, col) -> piece_id
        matches_used: List[EdgeMatch] = []

        # Determine which corner position this piece should be at
        # based on which edges are flat
        flat_edges = self.border_info[start_corner]['flat_edges']

        # Calculate the rotation needed to put flat edges at top-left corner
        rotation, corner_pos = self._get_corner_placement(flat_edges, grid_rows, grid_cols)

        # Place the starting corner
        start_placed = PlacedPiece(
            piece_id=start_corner,
            grid_row=corner_pos[0],
            grid_col=corner_pos[1],
            rotation=rotation,
            piece=self.pieces[start_corner]
        )

        placed[start_corner] = start_placed
        grid[corner_pos] = start_corner

        print(f"Placed starting corner piece {start_corner} at {corner_pos} with rotation {rotation}°")

        # Use BFS to place remaining pieces
        to_process = [start_corner]
        processed = {start_corner}

        while to_process:
            current_id = to_process.pop(0)
            current_placed = placed[current_id]

            # Try to find matches for each non-flat edge
            edges = self.edge_detector.piece_edges.get(current_id, {})

            for edge_type, edge in edges.items():
                # Skip flat edges
                if edge.get_edge_type_classification() == 'flat':
                    continue

                # Find match for this edge
                match = self.match_lookup.get((current_id, edge_type))
                if not match:
                    continue

                # Get the other piece in the match
                if match.edge1.piece_id == current_id:
                    other_piece_id = match.edge2.piece_id
                    other_edge_type = match.edge2.edge_type
                else:
                    other_piece_id = match.edge1.piece_id
                    other_edge_type = match.edge1.edge_type

                # Skip if already placed
                if other_piece_id in processed:
                    continue

                # Calculate position and rotation for the new piece
                new_pos, new_rotation = self._calculate_neighbor_placement(
                    current_placed, edge_type, other_edge_type, match
                )

                # Check if position is valid and not occupied
                if new_pos in grid:
                    continue  # Position already taken

                if not (0 <= new_pos[0] < grid_rows and 0 <= new_pos[1] < grid_cols):
                    continue  # Out of bounds

                # Place the new piece
                new_placed = PlacedPiece(
                    piece_id=other_piece_id,
                    grid_row=new_pos[0],
                    grid_col=new_pos[1],
                    rotation=new_rotation,
                    piece=self.pieces[other_piece_id]
                )

                placed[other_piece_id] = new_placed
                grid[new_pos] = other_piece_id
                matches_used.append(match)
                processed.add(other_piece_id)
                to_process.append(other_piece_id)

                print(f"Placed piece {other_piece_id} at {new_pos} with rotation {new_rotation:.1f}°")

        # Calculate confidence based on how many pieces were placed
        confidence = len(placed) / len(self.pieces)

        return PuzzleSolution(
            placed_pieces=list(placed.values()),
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            matches_used=matches_used,
            confidence=confidence
        )

    def _get_corner_placement(self, flat_edges: List[str], grid_rows: int, grid_cols: int) -> Tuple[float, Tuple[int, int]]:
        """
        Determine the rotation and position for a corner piece.

        Args:
            flat_edges: List of edge types that are flat
            grid_rows: Number of rows
            grid_cols: Number of columns

        Returns:
            Tuple of (rotation_degrees, (row, col))
        """
        flat_set = set(flat_edges)

        # Top-left corner: flat on top and left
        if 'top' in flat_set and 'left' in flat_set:
            return (0.0, (0, 0))

        # Top-right corner: flat on top and right
        if 'top' in flat_set and 'right' in flat_set:
            return (0.0, (0, grid_cols - 1))

        # Bottom-left corner: flat on bottom and left
        if 'bottom' in flat_set and 'left' in flat_set:
            return (0.0, (grid_rows - 1, 0))

        # Bottom-right corner: flat on bottom and right
        if 'bottom' in flat_set and 'right' in flat_set:
            return (0.0, (grid_rows - 1, grid_cols - 1))

        # If we can't determine, default to top-left with rotation
        # This means we need to rotate the piece
        if 'top' in flat_set:
            if 'bottom' in flat_set:
                # This is a border piece, not a corner
                return (0.0, (0, 0))
            # Rotate to get another edge to be left
            return (90.0, (0, 0))

        # Default: place at top-left, will need rotation
        return (0.0, (0, 0))

    def _calculate_neighbor_placement(self, current: PlacedPiece, current_edge: str,
                                      neighbor_edge: str, match: EdgeMatch) -> Tuple[Tuple[int, int], float]:
        """
        Calculate the position and rotation for a neighboring piece.

        Args:
            current: The already-placed piece
            current_edge: The edge of the current piece that connects
            neighbor_edge: The edge of the neighbor piece that connects
            match: The EdgeMatch between these pieces

        Returns:
            Tuple of ((row, col), rotation_degrees)
        """
        # Calculate the actual edge direction after current piece's rotation
        rotated_edge = self._rotate_edge_type(current_edge, current.rotation)

        # Get grid offset for this edge
        offset = self.EDGE_OFFSETS[rotated_edge]
        new_row = current.grid_row + offset[0]
        new_col = current.grid_col + offset[1]

        # Calculate rotation for the neighbor piece
        # The neighbor's connecting edge should face the opposite direction
        required_neighbor_edge = self.OPPOSITE_EDGE[rotated_edge]

        # Calculate how much to rotate the neighbor so its edge aligns
        rotation = self._calculate_rotation_for_alignment(neighbor_edge, required_neighbor_edge)

        return ((new_row, new_col), rotation)

    def _rotate_edge_type(self, edge_type: str, rotation: float) -> str:
        """
        Get the edge type after rotation.

        Args:
            edge_type: Original edge type ('top', 'right', 'bottom', 'left')
            rotation: Rotation in degrees (clockwise)

        Returns:
            New edge type after rotation
        """
        edges = ['top', 'right', 'bottom', 'left']
        idx = edges.index(edge_type)

        # Each 90° clockwise rotation shifts the edge index
        steps = int(round(rotation / 90.0)) % 4
        new_idx = (idx + steps) % 4

        return edges[new_idx]

    def _calculate_rotation_for_alignment(self, current_edge: str, required_edge: str) -> float:
        """
        Calculate the rotation needed to move current_edge to required_edge position.

        Args:
            current_edge: Current edge position ('top', 'right', 'bottom', 'left')
            required_edge: Required edge position

        Returns:
            Rotation in degrees (clockwise)
        """
        edges = ['top', 'right', 'bottom', 'left']
        current_idx = edges.index(current_edge)
        required_idx = edges.index(required_edge)

        steps = (required_idx - current_idx) % 4
        return steps * 90.0