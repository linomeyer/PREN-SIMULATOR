from flask import render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import cv2
import numpy as np
from app.main import main_bp
from app.main.puzzle_solver.edge_matching.edge_matcher import EdgeMatcher
from app.main.puzzle_solver.edge_matching.edge_matcher_visualizer import MatchVisualizer
from app.main.puzzle_solver.piece_extraction.extractor import PieceSegmenter
from app.main.puzzle_solver.piece_extraction.extractor_visualizer import PieceVisualizer
from app.main.puzzle_solver.edge_detection.edge_detector import EdgeDetector
from app.main.puzzle_solver.edge_detection.edge_detector_visualizer import EdgeVisualizer
from app.main.puzzle_solver.preprocessing.global_cleaner import GlobalCleaner
from app.main.puzzle_solver.preprocessing.image_cleaner import ImageCleaner
from app.main.puzzle_solver.solver.solver import PuzzleSolver
from app.main.puzzle_solver.solver.solver_visualizer import SolutionVisualizer

# Get absolute base path (app/ directory)
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / 'main' / 'img'
OUTPUT_FOLDER = BASE_DIR / 'static' / 'output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

segmenters_cache = {}
cleaned_images_cache = {}  # Cache for globally cleaned images

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@main_bp.route('/')
def index():
    return render_template("index.html")


@main_bp.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))

        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'File uploaded successfully'
        })

    return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400


@main_bp.route('/calibrate', methods=['POST'])
def calibrate():
    """Calibrate global cleaning parameters from test images."""
    if 'files' not in request.files:
        return jsonify({'error': 'No test images provided'}), 400

    files = request.files.getlist('files')

    if not files or files[0].filename == '':
        return jsonify({'error': 'No test images selected'}), 400

    try:
        # Load test images
        test_images = []
        for file in files:
            if file and allowed_file(file.filename):
                # Read image from file
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is not None:
                    test_images.append(img)

        if not test_images:
            return jsonify({'error': 'No valid test images found'}), 400

        # Initialize GlobalCleaner and calibrate
        global_cleaner = GlobalCleaner()
        params = global_cleaner.calibrate(test_images)

        # Get calibration info
        info = global_cleaner.get_calibration_info()

        return jsonify({
            'success': True,
            'message': f'Calibration complete with {len(test_images)} test image(s)',
            'calibration_info': info,
            'parameters': {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                          for k, v in params.items()}
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main_bp.route('/clean-global/<filename>')
def clean_global(filename):
    """Apply global cleaning to full puzzle image (Step 1.5)."""
    filepath = UPLOAD_FOLDER / filename

    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404

    try:
        # Load image
        image = cv2.imread(str(filepath))
        if image is None:
            return jsonify({'error': 'Failed to load image'}), 500

        # Initialize GlobalCleaner and load calibration
        global_cleaner = GlobalCleaner()

        if not global_cleaner.is_calibrated:
            return jsonify({
                'warning': 'No calibration found. Using default corrections.',
                'message': 'Please calibrate first using POST /calibrate for optimal results.'
            }), 200

        # Apply global cleaning
        cleaned_image = global_cleaner.clean(image)

        # Cache cleaned image for extraction step
        cleaned_images_cache[filename] = cleaned_image

        # Save visualization (before/after comparison)
        before_after = np.hstack([image, cleaned_image])
        vis_filename = f"global_clean_{filename}"
        vis_path = OUTPUT_FOLDER / vis_filename
        cv2.imwrite(str(vis_path), before_after)

        # Get calibration info
        calib_info = global_cleaner.get_calibration_info()

        return jsonify({
            'success': True,
            'filename': filename,
            'calibration_info': calib_info,
            'images': {
                'before_after': vis_filename
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main_bp.route('/clean-pieces/<filename>')
def clean_pieces(filename):
    """Apply local cleaning to extracted pieces (Step 3)."""

    # Check if we have extracted pieces
    if filename not in segmenters_cache:
        return jsonify({'error': 'Please extract pieces first'}), 400

    try:
        segmenter = segmenters_cache[filename]

        # Initialize ImageCleaner
        image_cleaner = ImageCleaner()

        # Clean all pieces
        cleaned_pieces = image_cleaner.clean_pieces(segmenter.pieces)

        # Create visualizations
        visualizer = PieceVisualizer(output_dir=str(OUTPUT_FOLDER))

        # Visualize cleaned pieces
        cleaned_images = visualizer.visualize_pieces(segmenter, f"cleaned_{filename}")

        # Create contour comparison visualization
        filepath = UPLOAD_FOLDER / filename
        original_image = cv2.imread(str(filepath))

        # Check if cleaned image exists in cache
        if filename in cleaned_images_cache:
            original_image = cleaned_images_cache[filename]

        # Draw both contours (red = original, green = cleaned)
        comparison = original_image.copy()
        for piece in segmenter.pieces:
            if hasattr(piece, 'contour_original'):
                cv2.drawContours(comparison, [piece.contour_original], -1, (0, 0, 255), 2)  # Red
            cv2.drawContours(comparison, [piece.contour], -1, (0, 255, 0), 2)  # Green

        # Save comparison
        comparison_filename = f"contour_comparison_{filename}"
        comparison_path = OUTPUT_FOLDER / comparison_filename
        cv2.imwrite(str(comparison_path), comparison)

        # Get piece statistics after cleaning
        stats = segmenter.get_piece_statistics()

        return jsonify({
            'success': True,
            'num_pieces': len(cleaned_pieces),
            'statistics': {k: float(v) if isinstance(v, (np.float64, np.float32)) else v
                          for k, v in stats.items()},
            'images': {
                'cleaned_pieces': cleaned_images,
                'contour_comparison': comparison_filename
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main_bp.route('/extract/<filename>')
def extract_pieces(filename):
    filepath = UPLOAD_FOLDER / filename

    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404

    try:
        # Get optional segmentation parameters from query string
        blur_kernel = request.args.get('blur_kernel', type=int)
        morph_kernel = request.args.get('morph_kernel', type=int)
        morph_close_iter = request.args.get('morph_close_iter', type=int)
        morph_open_iter = request.args.get('morph_open_iter', type=int)

        # Check if we have a globally cleaned version
        if filename in cleaned_images_cache:
            # Use cleaned image for segmentation
            image = cleaned_images_cache[filename]
            segmenter = PieceSegmenter(
                image_array=image,
                blur_kernel=blur_kernel,
                morph_kernel=morph_kernel,
                morph_close_iter=morph_close_iter,
                morph_open_iter=morph_open_iter
            )
        else:
            # Use original uploaded image
            segmenter = PieceSegmenter(
                str(filepath),
                blur_kernel=blur_kernel,
                morph_kernel=morph_kernel,
                morph_close_iter=morph_close_iter,
                morph_open_iter=morph_open_iter
            )

        visualizer = PieceVisualizer(output_dir=str(OUTPUT_FOLDER))

        # Segment pieces
        pieces = segmenter.segment_pieces()

        if not pieces:
            return jsonify({'error': 'No puzzle pieces detected. Try adjusting the parameters.'}), 400

        # Store segmenter in cache for edge detection step
        segmenters_cache[filename] = segmenter

        # Get statistics
        stats = segmenter.get_piece_statistics()

        # Visualize pieces
        piece_images = visualizer.visualize_pieces(segmenter, filename)

        # Prepare piece info
        piece_info = []
        for idx, piece in enumerate(pieces):
            x, y, w, h = piece.bbox
            piece_info.append({
                'id': idx,
                'center': piece.center,
                'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                'area': float(cv2.contourArea(piece.contour)),
                'perimeter': float(cv2.arcLength(piece.contour, True))
            })

        return jsonify({
            'success': True,
            'num_pieces': len(pieces),
            'statistics': {k: float(v) if isinstance(v, (np.float64, np.float32)) else v
                           for k, v in stats.items()},
            'pieces': piece_info,
            'images': {
                'pieces': piece_images
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main_bp.route('/detect-edges/<filename>')
def detect_edges(filename):
    """Detect edges for extracted puzzle pieces."""

    # Check if we have a cached segmenter
    if filename not in segmenters_cache:
        return jsonify({'error': 'Please extract pieces first'}), 400

    try:
        segmenter = segmenters_cache[filename]

        # Initialize edge detector
        edge_detector = EdgeDetector(segmenter.pieces)

        # Detect edges
        edges = edge_detector.detect_edges()

        if not edges:
            return jsonify({'error': 'No edges detected'}), 400

        # Get edge statistics
        edge_stats = edge_detector.get_edge_statistics()

        # Visualize edges
        edge_visualizer = EdgeVisualizer(output_dir=str(OUTPUT_FOLDER))
        edge_images = edge_visualizer.visualize_piece_edges(segmenter, edge_detector, filename)

        # Prepare edge info for each piece
        edge_info = []
        for piece_id in range(len(segmenter.pieces)):
            piece_edge_info = edge_detector.get_piece_edge_info(piece_id)
            edge_info.append({
                'piece_id': piece_id,
                'edges': piece_edge_info
            })

        return jsonify({
            'success': True,
            'num_edges': len(edges),
            'statistics': {k: float(v) if isinstance(v, (np.float64, np.float32)) else v
                          for k, v in edge_stats.items()},
            'edge_info': edge_info,
            'images': {
                'edge_pieces': edge_images,
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main_bp.route('/output/<path:filename>')
def get_output_image(filename):
    """Serve output images."""
    filepath = OUTPUT_FOLDER / filename

    if filepath.exists():
        return send_file(str(filepath))

    return jsonify({'error': 'Image not found'}), 404


@main_bp.route('/match-edges/<filename>')
def match_edges(filename):
    """Find matching edges between puzzle pieces."""

    # Check if we have a cached segmenter
    if filename not in segmenters_cache:
        return jsonify({'error': 'Please extract pieces first'}), 400

    try:
        segmenter = segmenters_cache[filename]

        # Initialize edge detector if not already done
        edge_detector = EdgeDetector(segmenter.pieces)
        edge_detector.detect_edges()

        # Initialize edge matcher
        edge_matcher = EdgeMatcher(edge_detector)

        # Find unique best matches - one match per edge, no duplicates
        min_score = float(request.args.get('min_score', 0.0))
        matches = edge_matcher.find_unique_best_matches(min_score=min_score)

        # Get match statistics (includes border piece info)
        match_stats = edge_matcher.get_match_statistics()

        # Visualize matches
        match_visualizer = MatchVisualizer(output_dir=str(OUTPUT_FOLDER))
        match_images = match_visualizer.visualize_best_matches(segmenter, edge_matcher, filename)

        # Prepare match info
        match_info = []
        for match in matches[:50]:  # Limit to top 50 matches for response size
            match_info.append({
                'edge1': {
                    'piece_id': match.edge1.piece_id,
                    'edge_type': match.edge1.edge_type,
                    'classification': match.edge1.get_edge_type_classification(),
                    'length': float(match.edge1.length)
                },
                'edge2': {
                    'piece_id': match.edge2.piece_id,
                    'edge_type': match.edge2.edge_type,
                    'classification': match.edge2.get_edge_type_classification(),
                    'length': float(match.edge2.length)
                },
                'scores': {
                    'compatibility': float(match.compatibility_score),
                    'length_similarity': float(match.length_similarity),
                    'shape_similarity': float(match.shape_similarity),
                    'classification_match': match.classification_match
                },
                'rotation': {
                    'degrees': match.rotation_angle,
                }
            })

        return jsonify({
            'success': True,
            'num_matches': len(matches),
            'statistics': match_stats,
            'matches': match_info,
            'images': {
                'match_visualizations': match_images
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_bp.route('/solve-puzzle/<filename>')
def solve_puzzle(filename):
    """Solve the puzzle by assembling pieces based on edge matches."""

    # Check if we have a cached segmenter
    if filename not in segmenters_cache:
        return jsonify({'error': 'Please extract pieces first'}), 400

    try:
        segmenter = segmenters_cache[filename]

        # Initialize edge detector
        edge_detector = EdgeDetector(segmenter.pieces)
        edge_detector.detect_edges()

        # Initialize edge matcher and find matches
        edge_matcher = EdgeMatcher(edge_detector)
        min_score = float(request.args.get('min_score', 0.0))
        edge_matcher.find_unique_best_matches(min_score=min_score)

        # Initialize puzzle solver
        puzzle_solver = PuzzleSolver(edge_matcher, segmenter.pieces)

        # Solve the puzzle
        solution = puzzle_solver.solve()

        if solution is None:
            return jsonify({
                'success': False,
                'error': 'Could not solve puzzle - no valid solution found'
            }), 400

        # Visualize solution - pass edge_detector for proper orientation
        solution_visualizer = SolutionVisualizer(output_dir=str(OUTPUT_FOLDER))
        solution_images = solution_visualizer.visualize_solution(
            solution, segmenter.pieces, filename, edge_detector=edge_detector
        )

        # Prepare solution info
        placed_pieces = []
        for placed in solution.placed_pieces:
            placed_pieces.append({
                'piece_id': placed.piece_id,
                'grid_position': {
                    'row': placed.grid_row,
                    'col': placed.grid_col
                },
                'rotation': float(placed.rotation)  # Convert numpy float to Python float
            })

        # Get grid layout
        grid_layout = solution.get_grid_layout()

        return jsonify({
            'success': True,
            'solution': {
                'grid_rows': solution.grid_rows,
                'grid_cols': solution.grid_cols,
                'confidence': float(solution.confidence),  # Convert to Python float
                'pieces_placed': len(solution.placed_pieces),
                'total_pieces': len(segmenter.pieces),
                'placed_pieces': placed_pieces,
                'grid_layout': grid_layout,
                'matches_used': len(solution.matches_used)
            },
            'images': {
                'solution_visualizations': solution_images
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500