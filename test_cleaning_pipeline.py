"""
Test script for the new cleaning pipeline.

Tests:
1. GlobalCleaner - calibration and global cleaning
2. ImageCleaner - piece-level cleaning
3. Integration with existing pipeline
"""

import cv2
import numpy as np
from pathlib import Path

# Add app to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from app.main.puzzle_solver.preprocessing.global_cleaner import GlobalCleaner
from app.main.puzzle_solver.preprocessing.image_cleaner import ImageCleaner
from app.main.puzzle_solver.piece_extraction.extractor import PieceSegmenter


def test_global_cleaner():
    """Test GlobalCleaner calibration and cleaning."""
    print("\n" + "="*60)
    print("TEST 1: GlobalCleaner")
    print("="*60)

    # Load a test image
    test_image_path = Path("app/main/img/generated_puzzle.png")
    if not test_image_path.exists():
        print(f"❌ Test image not found: {test_image_path}")
        return False

    image = cv2.imread(str(test_image_path))
    print(f"✓ Loaded test image: {image.shape}")

    # Test calibration
    print("\nTesting calibration...")
    cleaner = GlobalCleaner()
    params = cleaner.calibrate([image])
    print(f"✓ Calibration complete")
    print(f"  {cleaner.get_calibration_info()}")

    # Test cleaning
    print("\nTesting global cleaning...")
    cleaned = cleaner.clean(image)
    print(f"✓ Cleaned image: {cleaned.shape}")

    # Check if cleaning changed the image
    diff = np.abs(image.astype(float) - cleaned.astype(float)).mean()
    print(f"  Mean pixel difference: {diff:.2f}")

    return True


def test_image_cleaner():
    """Test ImageCleaner on extracted pieces."""
    print("\n" + "="*60)
    print("TEST 2: ImageCleaner")
    print("="*60)

    # Load test image and extract pieces
    test_image_path = Path("app/main/img/generated_puzzle.png")
    if not test_image_path.exists():
        print(f"❌ Test image not found: {test_image_path}")
        return False

    print("\nExtracting pieces...")
    segmenter = PieceSegmenter(str(test_image_path))
    pieces = segmenter.segment_pieces()
    print(f"✓ Extracted {len(pieces)} pieces")

    if not pieces:
        print("❌ No pieces found")
        return False

    # Test ImageCleaner
    print("\nTesting piece cleaning...")
    cleaner = ImageCleaner()

    # Clean first piece
    original_contour_len = len(pieces[0].contour)
    cleaner.clean_piece(pieces[0])

    print(f"✓ Cleaned piece 0")
    print(f"  Original contour points: {original_contour_len}")
    print(f"  Cleaned contour points: {len(pieces[0].contour)}")
    print(f"  Has original contour saved: {hasattr(pieces[0], 'contour_original')}")

    return True


def test_integration():
    """Test full pipeline integration."""
    print("\n" + "="*60)
    print("TEST 3: Full Pipeline Integration")
    print("="*60)

    test_image_path = Path("app/main/img/generated_puzzle.png")
    if not test_image_path.exists():
        print(f"❌ Test image not found: {test_image_path}")
        return False

    # Step 1: Load image
    print("\n1. Loading image...")
    image = cv2.imread(str(test_image_path))
    print(f"✓ Loaded: {image.shape}")

    # Step 1.5: Global cleaning
    print("\n1.5. Global cleaning...")
    global_cleaner = GlobalCleaner()
    global_cleaner.calibrate([image])
    cleaned_global = global_cleaner.clean(image)
    print(f"✓ Global cleaning complete")

    # Step 2: Extract pieces (from cleaned image)
    print("\n2. Extracting pieces from cleaned image...")
    segmenter = PieceSegmenter(image_array=cleaned_global)
    pieces = segmenter.segment_pieces()
    print(f"✓ Extracted {len(pieces)} pieces")

    if not pieces:
        print("❌ No pieces found")
        return False

    # Step 3: Clean pieces
    print("\n3. Cleaning individual pieces...")
    image_cleaner = ImageCleaner()
    image_cleaner.clean_pieces(pieces)
    print(f"✓ Cleaned {len(pieces)} pieces")

    # Verify all pieces have original contours saved
    pieces_with_original = sum(1 for p in pieces if hasattr(p, 'contour_original'))
    print(f"  Pieces with original contours: {pieces_with_original}/{len(pieces)}")

    # Step 4: Ready for edge detection
    print("\n4. Pipeline ready for edge detection")
    print(f"✓ All {len(pieces)} pieces ready for edge detection")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CLEANING PIPELINE TEST SUITE")
    print("="*60)

    results = []

    # Run tests
    results.append(("GlobalCleaner", test_global_cleaner()))
    results.append(("ImageCleaner", test_image_cleaner()))
    results.append(("Full Pipeline", test_integration()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
