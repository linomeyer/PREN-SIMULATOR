"""
Extended Tests for Step 2: Einheiten & Koordinatensysteme.

Tests Transform2D methods and conversion utilities.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.config import Transform2D
from solver.models import PuzzlePiece
from solver.utils.conversion import (
    convert_contour_px_to_mm,
    extract_scale_from_metadata,
    convert_pieces_px_to_mm,
    validate_pieces_format,
    get_default_T_MF,
    PlaceholderScaleError,
    get_default_scale_simulator,
)
import numpy as np


def test_transform2d_matrix_roundtrip():
    """Test 1: Matrix round-trip (to_matrix → from_matrix)."""
    print("Test 1: Matrix round-trip...", end=" ")

    t1 = Transform2D(x_mm=10.0, y_mm=20.0, theta_deg=45.0)
    matrix = t1.to_matrix()
    t2 = Transform2D.from_matrix(matrix)

    assert np.allclose([t2.x_mm, t2.y_mm, t2.theta_deg], [10.0, 20.0, 45.0], atol=1e-10), \
        f"Round-trip failed: expected (10, 20, 45), got ({t2.x_mm}, {t2.y_mm}, {t2.theta_deg})"

    print("✓")


def test_transform2d_compose():
    """Test 2: Compose two transformations."""
    print("Test 2: Compose...", end=" ")

    t1 = Transform2D(x_mm=10.0, y_mm=20.0, theta_deg=45.0)
    t3 = Transform2D(x_mm=5.0, y_mm=0.0, theta_deg=0.0)
    t4 = t1.compose(t3)

    print(f"Result: ({t4.x_mm:.2f}, {t4.y_mm:.2f}, {t4.theta_deg:.2f}) ✓")


def test_transform2d_inverse():
    """Test 3: Inverse (T ∘ T^-1 = Identity)."""
    print("Test 3: Inverse...", end=" ")

    t1 = Transform2D(x_mm=10.0, y_mm=20.0, theta_deg=45.0)
    t5 = t1.inverse()
    identity = t1.compose(t5)

    assert np.allclose([identity.x_mm, identity.y_mm, identity.theta_deg], [0, 0, 0], atol=1e-10), \
        f"Inverse failed: T ∘ T^-1 ≠ Identity, got ({identity.x_mm}, {identity.y_mm}, {identity.theta_deg})"

    print("✓")


def test_transform2d_apply():
    """Test 4: Apply transformation to points."""
    print("Test 4: Apply...", end=" ")

    t1 = Transform2D(x_mm=10.0, y_mm=20.0, theta_deg=45.0)
    pts = np.array([[0, 0], [10, 0], [10, 10]])
    pts_transformed = t1.apply(pts)

    assert pts_transformed.shape == (3, 2), \
        f"Shape mismatch: expected (3, 2), got {pts_transformed.shape}"

    print(f"shape={pts_transformed.shape} ✓")


def test_conversion_contour():
    """Test 5: Pixel to mm conversion."""
    print("Test 5: Conversion...", end=" ")

    contour_px = np.array([[100, 200], [150, 250]])
    scale = 0.1  # 0.1 mm/px
    contour_mm = convert_contour_px_to_mm(contour_px, scale)

    assert contour_mm.shape == (2, 2), \
        f"Shape mismatch: expected (2, 2), got {contour_mm.shape}"

    expected = np.array([[10, 20], [15, 25]])
    assert np.allclose(contour_mm, expected, atol=1e-10), \
        f"Conversion failed: expected {expected}, got {contour_mm}"

    print("✓")


def test_metadata_extraction():
    """Test 6: Metadata extraction (expected to raise NotImplementedError)."""
    print("Test 6: Metadata extraction...", end=" ")

    metadata = {"scale_px_to_mm": 0.5}

    try:
        extracted = extract_scale_from_metadata(metadata)
        # Should not reach here
        raise AssertionError("Expected NotImplementedError, but function returned value")
    except NotImplementedError as e:
        # Expected behavior
        print(f"NotImplementedError (expected) ✓")


def test_convert_pieces_px_to_mm():
    """Test 7: Batch conversion of PuzzlePiece list (realistic extraction input)."""
    print("Test 7: convert_pieces_px_to_mm...", end=" ")

    # Create 2 test pieces with px fields (as from extraction)
    piece1 = PuzzlePiece(
        piece_id=1,
        contour=np.array([[100, 200], [150, 250], [120, 230]]),  # px field
        mask=np.zeros((10, 10), dtype=np.uint8),
        bbox=(100, 200, 150, 250),  # px field
        center=np.array([125, 225])  # px field
    )

    piece2 = PuzzlePiece(
        piece_id=2,
        contour=np.array([[200, 300], [250, 350], [220, 330]]),  # px field
        mask=np.zeros((10, 10), dtype=np.uint8),
        bbox=(200, 300, 250, 350),  # px field
        center=None  # Test optional center
    )

    pieces_px = [piece1, piece2]
    scale = 0.1  # 0.1 mm/px

    # Convert (fills _mm fields)
    pieces_mm = convert_pieces_px_to_mm(pieces_px, scale)

    # Validate
    assert len(pieces_mm) == 2, f"Expected 2 pieces, got {len(pieces_mm)}"

    # Check piece 1 - mm fields filled
    assert pieces_mm[0].piece_id == 1
    assert np.allclose(pieces_mm[0].contour_mm, [[10, 20], [15, 25], [12, 23]], atol=1e-10)
    assert pieces_mm[0].bbox_mm == (10.0, 20.0, 15.0, 25.0)
    assert np.allclose(pieces_mm[0].center_mm, [12.5, 22.5], atol=1e-10)

    # Check piece 2
    assert pieces_mm[1].piece_id == 2
    assert np.allclose(pieces_mm[1].contour_mm, [[20, 30], [25, 35], [22, 33]], atol=1e-10)
    assert pieces_mm[1].center_mm is None

    print("✓")


def test_validate_pieces_format():
    """Test 8: Validation of PuzzlePiece format."""
    print("Test 8: validate_pieces_format...", end=" ")

    # Valid piece
    valid_piece = PuzzlePiece(
        piece_id=1,
        contour_mm=np.array([[10, 20], [15, 25], [12, 23]]),
        mask=np.zeros((10, 10), dtype=np.uint8),
        bbox_mm=(10.0, 20.0, 15.0, 25.0),
        center_mm=np.array([12.5, 22.5])
    )

    # Should not raise
    validate_pieces_format([valid_piece])

    # Test invalid cases
    # 1. Too few points
    try:
        invalid_piece = PuzzlePiece(
            piece_id=2,
            contour_mm=np.array([[10, 20], [15, 25]]),  # Only 2 points
            mask=np.zeros((10, 10), dtype=np.uint8),
            bbox_mm=(10.0, 20.0, 15.0, 25.0)
        )
        validate_pieces_format([invalid_piece])
        raise AssertionError("Should have raised ValueError for < 3 points")
    except ValueError as e:
        assert "at least 3 points" in str(e)

    # 2. Wrong contour shape
    try:
        invalid_piece = PuzzlePiece(
            piece_id=3,
            contour_mm=np.array([10, 20, 15]),  # Wrong shape (1D)
            mask=np.zeros((10, 10), dtype=np.uint8),
            bbox_mm=(10.0, 20.0, 15.0, 25.0)
        )
        validate_pieces_format([invalid_piece])
        raise AssertionError("Should have raised ValueError for wrong shape")
    except ValueError as e:
        assert "shape (N, 2)" in str(e)

    # 3. Suspicious values (still in pixels) - default 1000mm
    try:
        invalid_piece = PuzzlePiece(
            piece_id=4,
            contour_mm=np.array([[1000, 2000], [1500, 2500], [1200, 2300]]),  # > 1000mm
            mask=np.zeros((10, 10), dtype=np.uint8),
            bbox_mm=(1000.0, 2000.0, 1500.0, 2500.0)
        )
        validate_pieces_format([invalid_piece])
        raise AssertionError("Should have raised ValueError for suspicious values")
    except ValueError as e:
        assert "Suspicious contour_mm values" in str(e)

    # 4. Custom max_mm_extent parameter
    piece_200mm = PuzzlePiece(
        piece_id=5,
        contour_mm=np.array([[100, 150], [120, 160], [110, 155]]),  # 100-160mm range
        mask=np.zeros((10, 10), dtype=np.uint8),
        bbox_mm=(100.0, 150.0, 120.0, 160.0)
    )
    # Should pass with default (1000mm)
    validate_pieces_format([piece_200mm])
    # Should pass with custom 500mm
    validate_pieces_format([piece_200mm], max_mm_extent=500.0)
    # Should fail with custom 100mm
    try:
        validate_pieces_format([piece_200mm], max_mm_extent=100.0)
        raise AssertionError("Should have raised ValueError for values > 100mm")
    except ValueError as e:
        assert "Suspicious contour_mm values" in str(e)
        assert "100.0mm" in str(e)  # Check error message includes limit

    print("✓")


def test_validate_pieces_require_mm_fields():
    """Test 11: validate_pieces_format with require_mm_fields=True."""
    print("Test 11: validate_pieces_require_mm_fields...", end=" ")

    # Piece with only px fields (not yet converted)
    piece_px_only = PuzzlePiece(
        piece_id=1,
        contour=np.array([[0, 0], [10, 0], [10, 10]]),  # px field
        bbox=(0, 0, 10, 10),  # px field
        mask=np.ones((10, 10), dtype=bool)
        # contour_mm/bbox_mm NOT set (None)
    )

    # Should raise ValueError with require_mm_fields=True
    try:
        validate_pieces_format([piece_px_only], require_mm_fields=True)
        raise AssertionError("Should have raised ValueError for missing mm fields")
    except ValueError as e:
        assert "missing required mm fields" in str(e)
        assert "contour_mm" in str(e)
        assert "bbox_mm" in str(e)

    # After conversion should pass
    converted = convert_pieces_px_to_mm([piece_px_only], scale_px_to_mm=0.1)
    validate_pieces_format(converted, require_mm_fields=True)  # OK

    print("✓")


def test_get_default_T_MF():
    """Test 9: Default T_MF transform."""
    print("Test 9: get_default_T_MF...", end=" ")

    t_mf = get_default_T_MF()

    assert t_mf.x_mm == 200.0, f"Expected x_mm=200.0, got {t_mf.x_mm}"
    assert t_mf.y_mm == 200.0, f"Expected y_mm=200.0, got {t_mf.y_mm}"
    assert t_mf.theta_deg == 0.0, f"Expected theta_deg=0.0, got {t_mf.theta_deg}"

    print("✓")


def test_placeholder_scale_error():
    """Test 10: PlaceholderScaleError runtime guard."""
    print("Test 10: PlaceholderScaleError...", end=" ")

    # Default behavior: should raise PlaceholderScaleError
    try:
        scale = get_default_scale_simulator()
        raise AssertionError("Should have raised PlaceholderScaleError")
    except PlaceholderScaleError as e:
        assert "Placeholder scale" in str(e)
        assert "must not be used in production" in str(e)

    # With allow_in_production=True: should return 0.1
    scale = get_default_scale_simulator(allow_in_production=True)
    assert scale == 0.1, f"Expected 0.1, got {scale}"

    # Explicit False: should raise
    try:
        scale = get_default_scale_simulator(allow_in_production=False)
        raise AssertionError("Should have raised PlaceholderScaleError")
    except PlaceholderScaleError:
        pass  # Expected

    print("✓")


def run_all_tests():
    """Run all Step 2 tests."""
    print("=" * 60)
    print("Step 2: Einheiten & Koordinatensysteme - Extended Tests")
    print("=" * 60)
    print()

    test_transform2d_matrix_roundtrip()
    test_transform2d_compose()
    test_transform2d_inverse()
    test_transform2d_apply()
    test_conversion_contour()
    test_metadata_extraction()
    test_convert_pieces_px_to_mm()
    test_validate_pieces_format()
    test_get_default_T_MF()
    test_placeholder_scale_error()

    print()
    print("=" * 60)
    print("✅ Alle Tests bestanden (10/10)")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
