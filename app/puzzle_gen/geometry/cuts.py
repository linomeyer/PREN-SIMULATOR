"""Parametric cut functions for puzzle edges."""
import math
import random
from abc import ABC, abstractmethod
from typing import List, Tuple


class Cut(ABC):
    """Base class for parametric cut functions."""

    def __init__(self, is_male: bool, depth_ratio: float = 0.2):
        """
        Initialize a cut.

        Args:
            is_male: True for outward protrusion, False for inward
            depth_ratio: Depth of cut as ratio of edge length
        """
        self.is_male = is_male
        self.depth_ratio = depth_ratio

    @abstractmethod
    def generate_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 50
    ) -> List[Tuple[float, float]]:
        """
        Generate points along the cut from start to end.

        Args:
            start: Starting point (x, y)
            end: Ending point (x, y)
            num_points: Number of points to generate

        Returns:
            List of (x, y) coordinates defining the cut
        """
        pass

    def _perpendicular_offset(
        self,
        t: float,
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Calculate perpendicular offset direction.

        Args:
            t: Parameter along line (0 to 1)
            start: Start point
            end: End point

        Returns:
            Unit perpendicular vector (dx, dy)
        """
        # Vector along the edge
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)

        # Perpendicular vector (rotate 90 degrees)
        # For male cuts, offset outward (to the right when moving start->end)
        perp_x = -dy / length
        perp_y = dx / length

        # Flip for female cuts
        if not self.is_male:
            perp_x = -perp_x
            perp_y = -perp_y

        return perp_x, perp_y


class StraightCut(Cut):
    """Straight edge (for puzzle borders)."""

    def generate_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 50
    ) -> List[Tuple[float, float]]:
        """Generate a straight line."""
        return [
            (
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1])
            )
            for t in [i / (num_points - 1) for i in range(num_points)]
        ]


class WavyCut(Cut):
    """Sinusoidal wavy cut."""

    def __init__(self, is_male: bool, depth_ratio: float = 0.2, frequency: int = 2):
        """
        Initialize wavy cut.

        Args:
            is_male: True for outward, False for inward
            depth_ratio: Maximum wave amplitude as ratio of edge length
            frequency: Number of complete waves along the edge
        """
        super().__init__(is_male, depth_ratio)
        self.frequency = frequency

    def generate_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 50
    ) -> List[Tuple[float, float]]:
        """Generate wavy edge using sine wave."""
        edge_length = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        max_amplitude = edge_length * self.depth_ratio

        points = []
        for i in range(num_points):
            t = i / (num_points - 1)

            # Base position along the line
            base_x = start[0] + t * (end[0] - start[0])
            base_y = start[1] + t * (end[1] - start[1])

            # Sine wave amplitude
            amplitude = max_amplitude * math.sin(2 * math.pi * self.frequency * t)

            # Perpendicular offset
            perp_x, perp_y = self._perpendicular_offset(t, start, end)

            # Apply offset
            x = base_x + amplitude * perp_x
            y = base_y + amplitude * perp_y

            points.append((x, y))

        return points


class KnobSquareCut(Cut):
    """Rectangular knob protrusion."""

    def generate_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 50
    ) -> List[Tuple[float, float]]:
        """Generate rectangular knob in the middle of the edge."""
        edge_length = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        knob_depth = edge_length * self.depth_ratio
        knob_width = edge_length * 0.3  # Knob width is 30% of edge

        points = []

        # Parameters for knob position
        knob_start_t = 0.35
        knob_end_t = 0.65

        for i in range(num_points):
            t = i / (num_points - 1)

            base_x = start[0] + t * (end[0] - start[0])
            base_y = start[1] + t * (end[1] - start[1])

            perp_x, perp_y = self._perpendicular_offset(t, start, end)

            # Determine offset based on position
            if knob_start_t <= t <= knob_end_t:
                # Inside knob region
                offset = knob_depth
            else:
                # Outside knob region
                offset = 0

            x = base_x + offset * perp_x
            y = base_y + offset * perp_y

            points.append((x, y))

        return points


class KnobTriangularCut(Cut):
    """Triangular knob protrusion."""

    def generate_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 50
    ) -> List[Tuple[float, float]]:
        """Generate triangular knob in the middle of the edge."""
        edge_length = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        knob_depth = edge_length * self.depth_ratio

        points = []

        # Knob spans from 35% to 65% of the edge
        knob_center_t = 0.5
        knob_half_width_t = 0.15

        for i in range(num_points):
            t = i / (num_points - 1)

            base_x = start[0] + t * (end[0] - start[0])
            base_y = start[1] + t * (end[1] - start[1])

            perp_x, perp_y = self._perpendicular_offset(t, start, end)

            # Calculate triangular offset
            if abs(t - knob_center_t) <= knob_half_width_t:
                # Inside knob region - linear ramp to center
                relative_t = (knob_half_width_t - abs(t - knob_center_t)) / knob_half_width_t
                offset = knob_depth * relative_t
            else:
                offset = 0

            x = base_x + offset * perp_x
            y = base_y + offset * perp_y

            points.append((x, y))

        return points


class KnobWavyCut(Cut):
    """Rounded/wavy knob protrusion."""

    def generate_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 50
    ) -> List[Tuple[float, float]]:
        """Generate rounded knob using half-sine wave."""
        edge_length = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        knob_depth = edge_length * self.depth_ratio

        points = []

        # Knob spans from 35% to 65% of the edge
        knob_start_t = 0.35
        knob_end_t = 0.65

        for i in range(num_points):
            t = i / (num_points - 1)

            base_x = start[0] + t * (end[0] - start[0])
            base_y = start[1] + t * (end[1] - start[1])

            perp_x, perp_y = self._perpendicular_offset(t, start, end)

            # Calculate smooth knob offset using sine
            if knob_start_t <= t <= knob_end_t:
                # Map t from [knob_start_t, knob_end_t] to [0, pi]
                knob_t = (t - knob_start_t) / (knob_end_t - knob_start_t)
                offset = knob_depth * math.sin(math.pi * knob_t)
            else:
                offset = 0

            x = base_x + offset * perp_x
            y = base_y + offset * perp_y

            points.append((x, y))

        return points


class ChevronCut(Cut):
    """Single chevron/angle cut in the middle of the edge."""

    def __init__(self, is_male: bool, depth_ratio: float = 0.2, apex_position: float = 0.5):
        """
        Initialize chevron cut.

        Args:
            is_male: True for outward, False for inward
            depth_ratio: Maximum offset at the apex as ratio of edge length
            apex_position: Position of the apex along the edge (0-1), default 0.5
        """
        super().__init__(is_male, depth_ratio)
        self.apex_position = apex_position

    def generate_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 50
    ) -> List[Tuple[float, float]]:
        """Generate chevron/angle edge with a single apex."""
        edge_length = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        max_offset = edge_length * self.depth_ratio

        points = []

        for i in range(num_points):
            t = i / (num_points - 1)

            # Base position along the line
            base_x = start[0] + t * (end[0] - start[0])
            base_y = start[1] + t * (end[1] - start[1])

            # Perpendicular offset
            perp_x, perp_y = self._perpendicular_offset(t, start, end)

            # Calculate offset: increases to apex, then decreases
            if t <= self.apex_position:
                # Before apex: linear increase
                offset = max_offset * (t / self.apex_position)
            else:
                # After apex: linear decrease
                offset = max_offset * ((1 - t) / (1 - self.apex_position))

            # Apply offset
            x = base_x + offset * perp_x
            y = base_y + offset * perp_y

            points.append((x, y))

        return points


class SingleWaveCut(Cut):
    """Single smooth S-curve wave along the entire edge."""

    def generate_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 50
    ) -> List[Tuple[float, float]]:
        """Generate single wave (one complete sine period)."""
        edge_length = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        max_amplitude = edge_length * self.depth_ratio

        points = []
        for i in range(num_points):
            t = i / (num_points - 1)

            # Base position along the line
            base_x = start[0] + t * (end[0] - start[0])
            base_y = start[1] + t * (end[1] - start[1])

            # Single sine wave (one complete period from 0 to 2π)
            amplitude = max_amplitude * math.sin(2 * math.pi * t)

            # Perpendicular offset
            perp_x, perp_y = self._perpendicular_offset(t, start, end)

            # Apply offset
            x = base_x + amplitude * perp_x
            y = base_y + amplitude * perp_y

            points.append((x, y))

        return points


class PartialWaveCut(Cut):
    """Hybrid cut: straight beginning, then waves, then straight end."""

    def __init__(self, is_male: bool, depth_ratio: float = 0.2,
                 straight_ratio: float = 0.3, wave_count: int = 1):
        """
        Initialize partial wave cut.

        Args:
            is_male: True for outward, False for inward
            depth_ratio: Maximum wave amplitude as ratio of edge length
            straight_ratio: Portion of edge that remains straight (0.1-0.5)
            wave_count: Number of waves in the wavy section (1-3)
        """
        super().__init__(is_male, depth_ratio)
        self.straight_ratio = max(0.1, min(0.5, straight_ratio))
        self.wave_count = max(1, min(3, wave_count))

    def generate_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 50
    ) -> List[Tuple[float, float]]:
        """Generate straight + wave hybrid edge."""
        edge_length = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        max_amplitude = edge_length * self.depth_ratio

        points = []

        # Calculate sections
        wave_section_length = 1 - self.straight_ratio

        for i in range(num_points):
            t = i / (num_points - 1)

            # Base position along the line
            base_x = start[0] + t * (end[0] - start[0])
            base_y = start[1] + t * (end[1] - start[1])

            # Perpendicular offset
            perp_x, perp_y = self._perpendicular_offset(t, start, end)

            # Calculate amplitude based on position
            if t < self.straight_ratio:
                # Straight beginning
                amplitude = 0
            else:
                # Wavy section
                # Map t from [straight_ratio, 1] to [0, wave_count * 2π]
                wave_t = (t - self.straight_ratio) / wave_section_length
                amplitude = max_amplitude * math.sin(2 * math.pi * self.wave_count * wave_t)

            # Apply offset
            x = base_x + amplitude * perp_x
            y = base_y + amplitude * perp_y

            points.append((x, y))

        return points


class DoubleChevronCut(Cut):
    """Double chevron cut with two peaks (Berg/Tal symmetrie)."""

    def generate_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 50
    ) -> List[Tuple[float, float]]:
        """Generate double chevron edge with two peaks."""
        edge_length = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        max_offset = edge_length * self.depth_ratio

        points = []

        # First peak at 25%, second peak at 75%
        peak1_t = 0.25
        peak2_t = 0.75

        for i in range(num_points):
            t = i / (num_points - 1)

            # Base position along the line
            base_x = start[0] + t * (end[0] - start[0])
            base_y = start[1] + t * (end[1] - start[1])

            # Perpendicular offset
            perp_x, perp_y = self._perpendicular_offset(t, start, end)

            # Calculate offset: two triangular peaks
            if t <= peak1_t:
                # Rising to first peak
                offset = max_offset * (t / peak1_t)
            elif t <= 0.5:
                # Falling from first peak to valley
                offset = max_offset * ((0.5 - t) / (0.5 - peak1_t))
            elif t <= peak2_t:
                # Rising to second peak
                offset = max_offset * ((t - 0.5) / (peak2_t - 0.5))
            else:
                # Falling from second peak
                offset = max_offset * ((1 - t) / (1 - peak2_t))

            # Apply offset
            x = base_x + offset * perp_x
            y = base_y + offset * perp_y

            points.append((x, y))

        return points


class KnobSquareNewCut(Cut):
    """New rectangular knob variant (Klassisch eckig)."""

    def generate_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 50
    ) -> List[Tuple[float, float]]:
        """Generate rectangular knob in the middle of the edge."""
        edge_length = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        knob_depth = edge_length * self.depth_ratio
        knob_width = edge_length * 0.3  # Knob width is 30% of edge

        points = []

        # Parameters for knob position
        knob_start_t = 0.35
        knob_end_t = 0.65

        for i in range(num_points):
            t = i / (num_points - 1)

            base_x = start[0] + t * (end[0] - start[0])
            base_y = start[1] + t * (end[1] - start[1])

            perp_x, perp_y = self._perpendicular_offset(t, start, end)

            # Determine offset based on position
            if knob_start_t <= t <= knob_end_t:
                # Inside knob region
                offset = knob_depth
            else:
                # Outside knob region
                offset = 0

            x = base_x + offset * perp_x
            y = base_y + offset * perp_y

            points.append((x, y))

        return points


class KnobTrapezoidCut(Cut):
    """Trapezoidal knob protrusion (Klassisch trapez)."""

    def generate_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 50
    ) -> List[Tuple[float, float]]:
        """Generate trapezoidal knob in the middle of the edge."""
        edge_length = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        knob_depth = edge_length * self.depth_ratio

        points = []

        # Trapezoid parameters: centered, narrow end is 50% of wide end
        knob_center = 0.5
        knob_half_width_bottom = 0.15  # 30% total width at base
        knob_half_width_top = 0.075    # 15% total width at top (50% of base)

        for i in range(num_points):
            t = i / (num_points - 1)

            base_x = start[0] + t * (end[0] - start[0])
            base_y = start[1] + t * (end[1] - start[1])

            perp_x, perp_y = self._perpendicular_offset(t, start, end)

            # Calculate trapezoidal offset
            dist_from_center = abs(t - knob_center)

            if dist_from_center <= knob_half_width_bottom:
                if dist_from_center <= knob_half_width_top:
                    # At full depth (top of trapezoid)
                    offset = knob_depth
                else:
                    # Sloped sides
                    slope_t = (dist_from_center - knob_half_width_top) / (knob_half_width_bottom - knob_half_width_top)
                    offset = knob_depth * (1 - slope_t)
            else:
                offset = 0

            x = base_x + offset * perp_x
            y = base_y + offset * perp_y

            points.append((x, y))

        return points


class KnobRoundCut(Cut):
    """Round/circular knob protrusion (Klassisch rund)."""

    def generate_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 50
    ) -> List[Tuple[float, float]]:
        """Generate circular knob using semicircle."""
        edge_length = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        knob_radius = edge_length * self.depth_ratio

        points = []

        # Knob centered at 50%, spanning 35% to 65%
        knob_center = 0.5
        knob_half_width = 0.15

        for i in range(num_points):
            t = i / (num_points - 1)

            base_x = start[0] + t * (end[0] - start[0])
            base_y = start[1] + t * (end[1] - start[1])

            perp_x, perp_y = self._perpendicular_offset(t, start, end)

            # Calculate circular offset
            dist_from_center = abs(t - knob_center)

            if dist_from_center <= knob_half_width:
                # Inside knob region - use semicircle formula
                # Normalize to [-1, 1] range
                normalized_pos = dist_from_center / knob_half_width
                # Semicircle: sqrt(1 - x^2)
                offset = knob_radius * math.sqrt(1 - normalized_pos ** 2)
            else:
                offset = 0

            x = base_x + offset * perp_x
            y = base_y + offset * perp_y

            points.append((x, y))

        return points


class VariableAmplitudeIncreasingCut(Cut):
    """Sine wave with linearly increasing amplitude (Var Amp Stiigend)."""

    def __init__(self, is_male: bool, depth_ratio: float = 0.2, wave_count: int = 1):
        """
        Initialize increasing amplitude wave cut.

        Args:
            is_male: True for outward, False for inward
            depth_ratio: Maximum amplitude at end as ratio of edge length
            wave_count: Number of complete waves (1-3)
        """
        super().__init__(is_male, depth_ratio)
        self.wave_count = max(1, min(3, wave_count))

    def generate_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 50
    ) -> List[Tuple[float, float]]:
        """Generate sine wave with amplitude increasing from 0 to max."""
        edge_length = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        max_amplitude = edge_length * self.depth_ratio

        points = []
        for i in range(num_points):
            t = i / (num_points - 1)

            # Base position along the line
            base_x = start[0] + t * (end[0] - start[0])
            base_y = start[1] + t * (end[1] - start[1])

            # Amplitude increases linearly from 0 to max
            current_amplitude = max_amplitude * t

            # Sine wave
            wave_value = math.sin(2 * math.pi * self.wave_count * t)
            amplitude = current_amplitude * wave_value

            # Perpendicular offset
            perp_x, perp_y = self._perpendicular_offset(t, start, end)

            # Apply offset
            x = base_x + amplitude * perp_x
            y = base_y + amplitude * perp_y

            points.append((x, y))

        return points


class VariableAmplitudeChaoticCut(Cut):
    """Sine wave with random varying amplitude per wave (Var Amp Wirr)."""

    def __init__(self, is_male: bool, depth_ratio: float = 0.2, wave_count: int = 4):
        """
        Initialize chaotic amplitude wave cut.

        Args:
            is_male: True for outward, False for inward
            depth_ratio: Maximum possible amplitude as ratio of edge length
            wave_count: Number of complete waves (3-6)
        """
        super().__init__(is_male, depth_ratio)
        self.wave_count = max(3, min(6, wave_count))
        # Generate random amplitudes for each wave (between 25% and 100% of max)
        self.wave_amplitudes = [random.uniform(0.25, 1.0) for _ in range(self.wave_count)]

    def generate_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 50
    ) -> List[Tuple[float, float]]:
        """Generate sine wave with chaotic amplitude variation."""
        edge_length = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        max_amplitude = edge_length * self.depth_ratio

        points = []
        for i in range(num_points):
            t = i / (num_points - 1)

            # Base position along the line
            base_x = start[0] + t * (end[0] - start[0])
            base_y = start[1] + t * (end[1] - start[1])

            # Determine which wave we're in
            wave_position = t * self.wave_count
            wave_index = int(wave_position) % self.wave_count

            # Get amplitude for current wave
            current_max_amplitude = max_amplitude * self.wave_amplitudes[wave_index]

            # Sine wave
            wave_value = math.sin(2 * math.pi * self.wave_count * t)
            amplitude = current_max_amplitude * wave_value

            # Perpendicular offset
            perp_x, perp_y = self._perpendicular_offset(t, start, end)

            # Apply offset
            x = base_x + amplitude * perp_x
            y = base_y + amplitude * perp_y

            points.append((x, y))

        return points


class ReversedCut(Cut):
    """Wraps a cut and reverses its points for opposite edge directions."""

    def __init__(self, base_cut: Cut):
        """Initialize with a base cut to reverse."""
        super().__init__(base_cut.is_male, base_cut.depth_ratio)
        self.base_cut = base_cut
        # Flip male/female for reversed direction
        self.is_male = not base_cut.is_male

    def generate_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 50
    ) -> List[Tuple[float, float]]:
        """Generate points by reversing the base cut."""
        # Generate base cut points with swapped start/end
        points = self.base_cut.generate_points(end, start, num_points)
        # Reverse the order of points
        return list(reversed(points))


def create_cut(cut_type: str, is_male: bool, depth_ratio: float = 0.2, **kwargs) -> Cut:
    """
    Factory function to create cuts by type name.

    Args:
        cut_type: One of 'wavy', 'knob_square', 'knob_triangular', 'knob_wavy',
                  'chevron', 'single_wave', 'partial_wave', 'straight',
                  'double_chevron', 'knob_square_new', 'knob_trapezoid', 'knob_round',
                  'var_amp_increasing', 'var_amp_chaotic'
        is_male: True for outward protrusion
        depth_ratio: Cut depth ratio
        **kwargs: Additional parameters (e.g., frequency for WavyCut)

    Returns:
        Cut instance
    """
    cut_map = {
        'straight': StraightCut,
        'wavy': WavyCut,
        'knob_square': KnobSquareCut,
        'knob_triangular': KnobTriangularCut,
        'knob_wavy': KnobWavyCut,
        'chevron': ChevronCut,
        'single_wave': SingleWaveCut,
        'partial_wave': PartialWaveCut,
        'double_chevron': DoubleChevronCut,
        'knob_square_new': KnobSquareNewCut,
        'knob_trapezoid': KnobTrapezoidCut,
        'knob_round': KnobRoundCut,
        'var_amp_increasing': VariableAmplitudeIncreasingCut,
        'var_amp_chaotic': VariableAmplitudeChaoticCut,
    }

    if cut_type not in cut_map:
        raise ValueError(f"Unknown cut type: {cut_type}")

    cut_class = cut_map[cut_type]

    # Handle special parameters
    if cut_type == 'wavy' and 'frequency' in kwargs:
        return cut_class(is_male, depth_ratio, frequency=kwargs['frequency'])
    elif cut_type == 'chevron':
        # Randomize apex position between 40-60% of edge
        apex_position = kwargs.get('apex_position', random.uniform(0.4, 0.6))
        return cut_class(is_male, depth_ratio, apex_position=apex_position)
    elif cut_type == 'partial_wave':
        # Randomize straight ratio (10-50%) and wave count (1-3)
        straight_ratio = kwargs.get('straight_ratio', random.uniform(0.1, 0.5))
        wave_count = kwargs.get('wave_count', random.randint(1, 3))
        return cut_class(is_male, depth_ratio, straight_ratio=straight_ratio, wave_count=wave_count)
    elif cut_type == 'var_amp_increasing':
        # Randomize wave count (1-3)
        wave_count = kwargs.get('wave_count', random.randint(1, 3))
        return cut_class(is_male, depth_ratio, wave_count=wave_count)
    elif cut_type == 'var_amp_chaotic':
        # Randomize wave count (3-6)
        wave_count = kwargs.get('wave_count', random.randint(3, 6))
        return cut_class(is_male, depth_ratio, wave_count=wave_count)
    else:
        return cut_class(is_male, depth_ratio)
