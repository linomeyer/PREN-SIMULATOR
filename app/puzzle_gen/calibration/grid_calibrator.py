"""
Grid-based calibration for barrel distortion.

This module detects calibration grids in distorted images and calculates
barrel distortion coefficients (k1, k2, k3) by analyzing how straight lines
become curved.
"""

import cv2
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class GridCalibrator:
    """
    Detects calibration grids and calculates barrel distortion parameters.

    Workflow:
    1. detect_grid_lines() - Find horizontal/vertical lines using Hough transform
    2. measure_line_curvature() - Extract points along lines and fit polynomials
    3. calculate_distortion_coefficients() - Optimize k1, k2, k3 to minimize curvature
    """

    def __init__(self, expected_spacing_px: float = 236.0):
        """
        Initialize grid calibrator.

        Args:
            expected_spacing_px: Expected grid spacing in pixels (undistorted)
        """
        self.expected_spacing_px = expected_spacing_px
        self.detected_lines = []
        self.line_points = []

    def calibrate_from_image(self, image: np.ndarray) -> Dict:
        """
        Perform full calibration from a distorted image with grid.

        Args:
            image: Distorted image with calibration grid (BGR or grayscale)

        Returns:
            Dictionary with calibration parameters:
            {
                'k1': float,
                'k2': float,
                'k3': float,
                'fx': float,
                'fy': float,
                'cx': float,
                'cy': float,
                'image_size': [width, height],
                'num_lines_detected': int,
                'rms_error': float
            }
        """
        # 1. Detect grid lines
        h_lines, v_lines = self.detect_grid_lines(image)

        if len(h_lines) < 3 or len(v_lines) < 3:
            raise ValueError(
                f"Insufficient grid lines detected. "
                f"Found {len(h_lines)} horizontal and {len(v_lines)} vertical lines. "
                f"Need at least 3 of each."
            )

        # 2. Extract points along lines
        all_line_points = self._extract_line_points(image, h_lines + v_lines)

        # 3. Calculate distortion coefficients
        h, w = image.shape[:2]
        distortion_params = self.calculate_distortion_coefficients(
            all_line_points,
            image_size=(w, h)
        )

        distortion_params['num_lines_detected'] = len(h_lines) + len(v_lines)

        return distortion_params

    def detect_grid_lines(
        self,
        image: np.ndarray,
        canny_threshold1: int = 50,
        canny_threshold2: int = 150,
        hough_threshold: int = 100
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Detect horizontal and vertical grid lines using Hough transform.

        Args:
            image: Input image (BGR or grayscale)
            canny_threshold1: Lower threshold for Canny edge detection
            canny_threshold2: Upper threshold for Canny edge detection
            hough_threshold: Threshold for Hough line detection

        Returns:
            Tuple of (horizontal_lines, vertical_lines)
            Each line is represented as [x1, y1, x2, y2]
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

        # Hough Line Transform (probabilistic)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=hough_threshold,
            minLineLength=100,
            maxLineGap=50
        )

        if lines is None:
            return [], []

        # Classify lines as horizontal or vertical
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1))

            # Horizontal: angle close to 0 or π
            if angle < np.pi/8 or angle > 7*np.pi/8:
                horizontal_lines.append(line[0])
            # Vertical: angle close to π/2
            elif np.pi/8 < angle < 3*np.pi/8 or 5*np.pi/8 < angle < 7*np.pi/8:
                vertical_lines.append(line[0])

        # Merge nearby lines
        horizontal_lines = self._merge_similar_lines(horizontal_lines, orientation='horizontal')
        vertical_lines = self._merge_similar_lines(vertical_lines, orientation='vertical')

        self.detected_lines = horizontal_lines + vertical_lines

        return horizontal_lines, vertical_lines

    def _merge_similar_lines(
        self,
        lines: List[np.ndarray],
        orientation: str = 'horizontal',
        distance_threshold: float = 20.0
    ) -> List[np.ndarray]:
        """
        Merge lines that are close to each other.

        Args:
            lines: List of lines [x1, y1, x2, y2]
            orientation: 'horizontal' or 'vertical'
            distance_threshold: Maximum distance to merge lines

        Returns:
            List of merged lines
        """
        if not lines:
            return []

        # Group lines by position
        if orientation == 'horizontal':
            # Group by y-coordinate
            lines_sorted = sorted(lines, key=lambda l: (l[1] + l[3]) / 2)
        else:
            # Group by x-coordinate
            lines_sorted = sorted(lines, key=lambda l: (l[0] + l[2]) / 2)

        merged = []
        current_group = [lines_sorted[0]]

        for i in range(1, len(lines_sorted)):
            if orientation == 'horizontal':
                prev_y = (current_group[0][1] + current_group[0][3]) / 2
                curr_y = (lines_sorted[i][1] + lines_sorted[i][3]) / 2
                distance = abs(curr_y - prev_y)
            else:
                prev_x = (current_group[0][0] + current_group[0][2]) / 2
                curr_x = (lines_sorted[i][0] + lines_sorted[i][2]) / 2
                distance = abs(curr_x - prev_x)

            if distance < distance_threshold:
                current_group.append(lines_sorted[i])
            else:
                # Average the group
                merged.append(self._average_lines(current_group))
                current_group = [lines_sorted[i]]

        # Add last group
        merged.append(self._average_lines(current_group))

        return merged

    def _average_lines(self, lines: List[np.ndarray]) -> np.ndarray:
        """Average multiple lines into one."""
        if len(lines) == 1:
            return lines[0]

        x1_avg = int(np.mean([l[0] for l in lines]))
        y1_avg = int(np.mean([l[1] for l in lines]))
        x2_avg = int(np.mean([l[2] for l in lines]))
        y2_avg = int(np.mean([l[3] for l in lines]))

        return np.array([x1_avg, y1_avg, x2_avg, y2_avg])

    def _extract_line_points(
        self,
        image: np.ndarray,
        lines: List[np.ndarray],
        num_points: int = 50
    ) -> List[List[Tuple[float, float]]]:
        """
        Extract points along detected lines.

        Args:
            image: Input image
            lines: List of lines [x1, y1, x2, y2]
            num_points: Number of points to sample per line

        Returns:
            List of point lists, one per line
        """
        all_points = []

        for line in lines:
            x1, y1, x2, y2 = line

            # Sample points along the line
            t = np.linspace(0, 1, num_points)
            x_points = x1 + t * (x2 - x1)
            y_points = y1 + t * (y2 - y1)

            points = [(x, y) for x, y in zip(x_points, y_points)]
            all_points.append(points)

        self.line_points = all_points
        return all_points

    def calculate_distortion_coefficients(
        self,
        line_points: List[List[Tuple[float, float]]],
        image_size: Tuple[int, int]
    ) -> Dict:
        """
        Calculate barrel distortion coefficients from line points.

        Uses optimization to find k1, k2, k3 that best straighten the lines.

        Args:
            line_points: List of point lists, one per detected line
            image_size: Image dimensions (width, height)

        Returns:
            Dictionary with distortion parameters
        """
        w, h = image_size

        # Camera matrix (approximate)
        fx = fy = max(h, w)
        cx, cy = w / 2, h / 2

        # Initial guess for distortion coefficients
        k1_init = -0.3
        k2_init = 0.05
        k3_init = 0.0

        # Objective function: minimize line curvature
        def objective(params):
            k1, k2, k3 = params

            total_error = 0.0

            for points in line_points:
                # Apply undistortion to points
                undistorted_points = self._undistort_points(
                    points, k1, k2, k3, fx, fy, cx, cy
                )

                # Measure how straight the line is
                # Fit a line and measure residuals
                if len(undistorted_points) < 3:
                    continue

                x_coords = np.array([p[0] for p in undistorted_points])
                y_coords = np.array([p[1] for p in undistorted_points])

                # Fit line y = mx + b or x = my + b (whichever has better fit)
                # Try y = mx + b
                if np.std(x_coords) > np.std(y_coords):
                    # Fit y as function of x
                    coeffs = np.polyfit(x_coords, y_coords, 1)
                    y_fit = np.polyval(coeffs, x_coords)
                    residuals = y_coords - y_fit
                else:
                    # Fit x as function of y
                    coeffs = np.polyfit(y_coords, x_coords, 1)
                    x_fit = np.polyval(coeffs, y_coords)
                    residuals = x_coords - x_fit

                # Sum squared residuals
                total_error += np.sum(residuals ** 2)

            return total_error

        # Optimize distortion coefficients
        result = minimize(
            objective,
            x0=[k1_init, k2_init, k3_init],
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-6}
        )

        k1, k2, k3 = result.x
        rms_error = np.sqrt(result.fun / sum(len(pts) for pts in line_points))

        return {
            'k1': float(k1),
            'k2': float(k2),
            'k3': float(k3),
            'fx': float(fx),
            'fy': float(fy),
            'cx': float(cx),
            'cy': float(cy),
            'image_size': [w, h],
            'rms_error': float(rms_error)
        }

    def _undistort_points(
        self,
        points: List[Tuple[float, float]],
        k1: float, k2: float, k3: float,
        fx: float, fy: float, cx: float, cy: float
    ) -> List[Tuple[float, float]]:
        """
        Apply undistortion to a list of points.

        Args:
            points: List of (x, y) points
            k1, k2, k3: Radial distortion coefficients
            fx, fy: Focal lengths
            cx, cy: Principal point

        Returns:
            List of undistorted (x, y) points
        """
        undistorted = []

        for x, y in points:
            # Normalize coordinates
            x_norm = (x - cx) / fx
            y_norm = (y - cy) / fy

            # Calculate radius squared
            r2 = x_norm**2 + y_norm**2
            r4 = r2**2
            r6 = r2**3

            # Radial distortion
            radial = 1 + k1 * r2 + k2 * r4 + k3 * r6

            # Apply undistortion
            x_undist = x_norm / radial
            y_undist = y_norm / radial

            # Denormalize
            x_final = x_undist * fx + cx
            y_final = y_undist * fy + cy

            undistorted.append((x_final, y_final))

        return undistorted

    def visualize_detection(
        self,
        image: np.ndarray,
        horizontal_lines: List[np.ndarray],
        vertical_lines: List[np.ndarray]
    ) -> np.ndarray:
        """
        Visualize detected grid lines on image.

        Args:
            image: Input image
            horizontal_lines: Detected horizontal lines
            vertical_lines: Detected vertical lines

        Returns:
            Image with lines drawn
        """
        vis = image.copy()

        # Draw horizontal lines in red
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw vertical lines in blue
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return vis
