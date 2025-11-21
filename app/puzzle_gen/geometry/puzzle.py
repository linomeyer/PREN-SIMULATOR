"""Puzzle geometry generation and piece placement."""
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .cuts import Cut, StraightCut, create_cut, ReversedCut


@dataclass
class PieceEdge:
    """Represents one edge of a puzzle piece."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    cut: Cut
    is_border: bool = False  # True if this is an outer edge


@dataclass
class PuzzlePiece:
    """Represents a single puzzle piece."""
    row: int
    col: int
    edges: List[PieceEdge]  # Top, Right, Bottom, Left (clockwise)
    rotation: float = 0.0  # Rotation in degrees
    center: Optional[Tuple[float, float]] = None  # Randomized center position

    def get_path_points(self, num_points_per_edge: int = 50) -> List[Tuple[float, float]]:
        """
        Generate complete path around the piece.

        Args:
            num_points_per_edge: Points to generate per edge

        Returns:
            List of (x, y) coordinates forming closed path
        """
        all_points = []

        for edge in self.edges:
            points = edge.cut.generate_points(edge.start, edge.end, num_points_per_edge)
            # Skip last point to avoid duplicates at corners
            all_points.extend(points[:-1])

        return all_points

    def get_original_center(self) -> Tuple[float, float]:
        """Calculate the geometric center of the piece in its original position."""
        # Use the bounding box of the original edges
        xs = [edge.start[0] for edge in self.edges] + [edge.end[0] for edge in self.edges]
        ys = [edge.start[1] for edge in self.edges] + [edge.end[1] for edge in self.edges]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def get_bounding_box_size(self) -> Tuple[float, float]:
        """
        Calculate the bounding box size of the piece including cuts.
        Returns (width, height) of the axis-aligned bounding box.
        """
        # Get all points of the piece outline
        points = self.get_path_points(num_points_per_edge=100)

        if not points:
            return (0, 0)

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        # Account for rotation - when rotated, the bounding box can be larger
        # Use diagonal as worst-case size
        diagonal = math.sqrt(width**2 + height**2)

        return (diagonal, diagonal)

    def get_rotated_bounding_box(self, rotation_degrees: float) -> Tuple[float, float]:
        """
        Calculate the axis-aligned bounding box size after rotation.
        This gives the actual extent of the piece when rotated.

        Args:
            rotation_degrees: Rotation angle in degrees

        Returns:
            (width, height) of the rotated piece's bounding box
        """
        # Get all points of the piece outline
        points = self.get_path_points(num_points_per_edge=100)

        if not points:
            return (0, 0)

        # Get the original center point
        original_center = self.get_original_center()

        # Convert rotation to radians
        rotation_rad = math.radians(rotation_degrees)
        cos_angle = math.cos(rotation_rad)
        sin_angle = math.sin(rotation_rad)

        # Rotate all points around the original center
        rotated_points = []
        for x, y in points:
            # Translate to origin
            dx = x - original_center[0]
            dy = y - original_center[1]

            # Rotate
            rotated_x = dx * cos_angle - dy * sin_angle
            rotated_y = dx * sin_angle + dy * cos_angle

            # Translate back
            rotated_x += original_center[0]
            rotated_y += original_center[1]

            rotated_points.append((rotated_x, rotated_y))

        # Calculate bounding box of rotated points
        xs = [p[0] for p in rotated_points]
        ys = [p[1] for p in rotated_points]

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        return (width, height)


class PuzzleGenerator:
    """Generates puzzle geometry with randomized placement."""

    def __init__(
        self,
        width_px: int,
        height_px: int,
        rows: int,
        cols: int,
        cut_depth_ratio: float = 0.2,
        seed: Optional[int] = None
    ):
        """
        Initialize puzzle generator.

        Args:
            width_px: Base rectangle width in pixels
            height_px: Base rectangle height in pixels
            rows: Number of rows in grid
            cols: Number of columns in grid
            cut_depth_ratio: Cut depth as ratio of edge length
            seed: Random seed for reproducibility
        """
        self.width_px = width_px
        self.height_px = height_px
        self.rows = rows
        self.cols = cols
        self.cut_depth_ratio = cut_depth_ratio

        if seed is not None:
            random.seed(seed)

        self.piece_width = width_px / cols
        self.piece_height = height_px / rows

        # Store edge cut assignments (shared between adjacent pieces)
        # Key: (row, col, direction) where direction is 'h' (horizontal) or 'v' (vertical)
        self.edge_cuts: dict = {}

    def _assign_edge_cuts(self, cut_types: List[str], wave_frequency: int = 2):
        """
        Assign cut types to all internal edges, ensuring male/female pairing.

        Args:
            cut_types: List of available cut types
            wave_frequency: Frequency parameter for wavy cuts
        """
        # Horizontal internal edges (between rows)
        for row in range(self.rows - 1):
            for col in range(self.cols):
                cut_type = random.choice(cut_types)
                is_male = random.choice([True, False])

                # Create the cut
                cut = create_cut(
                    cut_type,
                    is_male,
                    self.cut_depth_ratio,
                    frequency=wave_frequency
                )

                # Store for top piece (bottom edge)
                self.edge_cuts[(row, col, 'bottom')] = cut

                # Create complementary cut for bottom piece (top edge)
                # Bottom edge goes right-to-left, top edge goes left-to-right
                # So we need to reverse the cut
                self.edge_cuts[(row + 1, col, 'top')] = ReversedCut(cut)

        # Vertical internal edges (between columns)
        for row in range(self.rows):
            for col in range(self.cols - 1):
                cut_type = random.choice(cut_types)
                is_male = random.choice([True, False])

                cut = create_cut(
                    cut_type,
                    is_male,
                    self.cut_depth_ratio,
                    frequency=wave_frequency
                )

                # Store for left piece (right edge)
                self.edge_cuts[(row, col, 'right')] = cut

                # Create complementary cut for right piece (left edge)
                # Right edge goes top-to-bottom, left edge goes bottom-to-top
                # So we need to reverse the cut
                self.edge_cuts[(row, col + 1, 'left')] = ReversedCut(cut)

        # Border edges (straight)
        for row in range(self.rows):
            for col in range(self.cols):
                # Top border
                if row == 0:
                    self.edge_cuts[(row, col, 'top')] = StraightCut(True, 0)
                # Bottom border
                if row == self.rows - 1:
                    self.edge_cuts[(row, col, 'bottom')] = StraightCut(True, 0)
                # Left border
                if col == 0:
                    self.edge_cuts[(row, col, 'left')] = StraightCut(True, 0)
                # Right border
                if col == self.cols - 1:
                    self.edge_cuts[(row, col, 'right')] = StraightCut(True, 0)

    def _create_piece(self, row: int, col: int) -> PuzzlePiece:
        """
        Create a single puzzle piece with its edges.

        Args:
            row: Row index
            col: Column index

        Returns:
            PuzzlePiece instance
        """
        # Calculate corner positions in original grid
        x0 = col * self.piece_width
        y0 = row * self.piece_height
        x1 = x0 + self.piece_width
        y1 = y0 + self.piece_height

        # Create edges: Top, Right, Bottom, Left (clockwise)
        edges = [
            # Top edge (left to right)
            PieceEdge(
                start=(x0, y0),
                end=(x1, y0),
                cut=self.edge_cuts[(row, col, 'top')],
                is_border=(row == 0)
            ),
            # Right edge (top to bottom)
            PieceEdge(
                start=(x1, y0),
                end=(x1, y1),
                cut=self.edge_cuts[(row, col, 'right')],
                is_border=(col == self.cols - 1)
            ),
            # Bottom edge (right to left)
            PieceEdge(
                start=(x1, y1),
                end=(x0, y1),
                cut=self.edge_cuts[(row, col, 'bottom')],
                is_border=(row == self.rows - 1)
            ),
            # Left edge (bottom to top)
            PieceEdge(
                start=(x0, y1),
                end=(x0, y0),
                cut=self.edge_cuts[(row, col, 'left')],
                is_border=(col == 0)
            ),
        ]

        return PuzzlePiece(row=row, col=col, edges=edges)

    def _randomize_placement(
        self,
        pieces: List[PuzzlePiece],
        margin_px: int,
        spacing_px: int,
        canvas_width: int,
        canvas_height: int
    ) -> dict:
        """
        Randomize rotation and position of pieces with guaranteed spacing.

        Args:
            pieces: List of puzzle pieces
            margin_px: Margin from canvas edges
            spacing_px: Minimum spacing between pieces
            canvas_width: Canvas width
            canvas_height: Canvas height
        """
        # Calculate grid layout for pieces to ensure good distribution
        num_pieces = len(pieces)

        # Determine grid dimensions based on piece count
        if num_pieces == 6:
            grid_cols = 3
            grid_rows = 2
        elif num_pieces == 4:
            grid_cols = 2
            grid_rows = 2
        else:
            # Approximate square grid
            grid_cols = int(math.ceil(math.sqrt(num_pieces)))
            grid_rows = int(math.ceil(num_pieces / grid_cols))

        # IMPORTANT: Assign rotations FIRST, before calculating bounding boxes
        # This ensures we calculate the correct size for the rotated pieces
        for piece in pieces:
            piece.rotation = random.uniform(0, 360)

        # Calculate maximum piece size including cuts AFTER rotation
        # Check all pieces to find the largest rotated bounding box
        max_piece_size = 0
        if pieces:
            for piece in pieces:
                # Use the rotated bounding box for accurate size
                bbox_w, bbox_h = piece.get_rotated_bounding_box(piece.rotation)
                piece_size = max(bbox_w, bbox_h)
                max_piece_size = max(max_piece_size, piece_size)
        else:
            max_piece_size = max(self.piece_width, self.piece_height) * 1.3

        # Add safety margin to account for cut variations
        # Minimal margin since bounding boxes are calculated after rotation
        max_piece_size = max_piece_size * 1.05

        # Calculate cell size with spacing between cells
        # Each cell must fit: piece + spacing on ALL sides + additional gap between cells
        # Additional gap ensures green bounding boxes don't touch
        inter_cell_gap = spacing_px * 0.1  # 10% additional gap between cells
        min_cell_width = max_piece_size + 2 * spacing_px + inter_cell_gap
        min_cell_height = max_piece_size + 2 * spacing_px + inter_cell_gap

        # Total grid size needed
        total_grid_width = min_cell_width * grid_cols
        total_grid_height = min_cell_height * grid_rows

        # Calculate required canvas with margins
        required_width = total_grid_width + 2 * margin_px
        required_height = total_grid_height + 2 * margin_px

        # If canvas is smaller than required, adjust cell sizes proportionally
        # Store scale factor to also scale the pieces during rendering
        if required_width > canvas_width or required_height > canvas_height:
            scale_x = (canvas_width - 2 * margin_px) / total_grid_width
            scale_y = (canvas_height - 2 * margin_px) / total_grid_height
            scale_factor = min(scale_x, scale_y)

            cell_width = min_cell_width * scale_factor
            cell_height = min_cell_height * scale_factor
            total_grid_width = cell_width * grid_cols
            total_grid_height = cell_height * grid_rows
        else:
            scale_factor = 1.0  # No scaling needed
            cell_width = min_cell_width
            cell_height = min_cell_height

        # Center the grid on canvas with proper margin
        grid_start_x = (canvas_width - total_grid_width) / 2
        grid_start_y = (canvas_height - total_grid_height) / 2

        # Ensure minimum margin from edge
        grid_start_x = max(grid_start_x, margin_px)
        grid_start_y = max(grid_start_y, margin_px)

        # Shuffle pieces to randomize grid assignment
        shuffled_pieces = pieces.copy()
        random.shuffle(shuffled_pieces)

        # Place each piece in a grid cell with random offset
        for idx, piece in enumerate(shuffled_pieces):
            grid_row = idx // grid_cols
            grid_col = idx % grid_cols

            # Note: Rotation already assigned above before bounding box calculation

            # Calculate cell center
            cell_center_x = grid_start_x + (grid_col + 0.5) * cell_width
            cell_center_y = grid_start_y + (grid_row + 0.5) * cell_height

            # Calculate safe area within cell (keep pieces away from cell edges)
            # This ensures spacing_px is maintained between pieces
            # Conservative calculation: ensure piece stays well within cell boundaries
            safe_radius = (min(cell_width, cell_height) - max_piece_size) / 2
            # Add extra safety margin to guarantee no overlap
            # Use 0.4 to allow more variation while preventing overlap
            safe_radius = safe_radius * 0.4
            safe_radius = max(0, safe_radius)

            # Add random offset within safe area
            if safe_radius > 0:
                offset_x = random.uniform(-safe_radius, safe_radius)
                offset_y = random.uniform(-safe_radius, safe_radius)
            else:
                offset_x = 0
                offset_y = 0

            x = cell_center_x + offset_x
            y = cell_center_y + offset_y

            # Set position
            piece.center = (x, y)

        # Return debug info including scale factor
        return {
            'grid_cols': grid_cols,
            'grid_rows': grid_rows,
            'cell_width': cell_width,
            'cell_height': cell_height,
            'grid_start_x': grid_start_x,
            'grid_start_y': grid_start_y,
            'max_piece_size': max_piece_size,
            'safe_radius': safe_radius,
            'scale_factor': scale_factor
        }

    def _create_solution_placement(
        self,
        pieces: List[PuzzlePiece],
        gap_px: int,
        canvas_width: int,
        canvas_height: int
    ):
        """
        Place pieces in their correct solution positions with small gaps.

        Args:
            pieces: List of puzzle pieces
            gap_px: Gap between pieces in pixels
            canvas_width: Canvas width
            canvas_height: Canvas height
        """
        # Calculate dimensions with gaps between pieces
        total_width = self.width_px + (self.cols - 1) * gap_px
        total_height = self.height_px + (self.rows - 1) * gap_px

        # Center the solution on canvas
        offset_x = (canvas_width - total_width) / 2
        offset_y = (canvas_height - total_height) / 2

        for piece in pieces:
            # No rotation for solution
            piece.rotation = 0

            # Calculate position with gaps between pieces
            piece_center_x = offset_x + piece.col * (self.piece_width + gap_px) + self.piece_width / 2
            piece_center_y = offset_y + piece.row * (self.piece_height + gap_px) + self.piece_height / 2

            # Set position
            piece.center = (piece_center_x, piece_center_y)

    def generate(
        self,
        cut_types: List[str],
        canvas_width: int,
        canvas_height: int,
        margin_px: int = 200,
        spacing_px: int = 100,
        wave_frequency: int = 2
    ) -> Tuple[List[PuzzlePiece], dict]:
        """
        Generate complete puzzle with randomized pieces.

        Args:
            cut_types: List of cut type names to use
            canvas_width: Final canvas width
            canvas_height: Final canvas height
            margin_px: Margin around canvas
            spacing_px: Minimum spacing between pieces
            wave_frequency: Wave frequency for wavy cuts

        Returns:
            Tuple of (pieces, debug_info) where debug_info contains grid layout data
        """
        # Assign cuts to all edges
        self._assign_edge_cuts(cut_types, wave_frequency)

        # Create all pieces
        pieces = []
        for row in range(self.rows):
            for col in range(self.cols):
                piece = self._create_piece(row, col)
                pieces.append(piece)

        # Randomize placement and get debug info
        debug_info = self._randomize_placement(pieces, margin_px, spacing_px, canvas_width, canvas_height)

        return pieces, debug_info

    def generate_solution(
        self,
        canvas_width: int,
        canvas_height: int,
        gap_px: int = 20
    ) -> List[PuzzlePiece]:
        """
        Generate puzzle in solution configuration (correct positions with gaps).
        Uses the same cuts as previously generated puzzle.

        Args:
            canvas_width: Final canvas width
            canvas_height: Final canvas height
            gap_px: Gap between pieces in pixels

        Returns:
            List of PuzzlePiece instances in solution positions
        """
        # Check if cuts have been assigned
        if not self.edge_cuts:
            raise RuntimeError("Must call generate() before generate_solution()")

        # Create all pieces (using existing edge_cuts)
        pieces = []
        for row in range(self.rows):
            for col in range(self.cols):
                piece = self._create_piece(row, col)
                pieces.append(piece)

        # Place in solution configuration
        self._create_solution_placement(pieces, gap_px, canvas_width, canvas_height)

        return pieces
