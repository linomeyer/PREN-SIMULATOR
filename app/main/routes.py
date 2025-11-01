from flask import render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from app.main import main_bp
from app.main.puzzle_solver.extractor import PieceSegmenter
from app.main.puzzle_solver.extractor_visualizer import PieceVisualizer
from app.main.puzzle_solver.edge_detector import EdgeDetector
from app.main.puzzle_solver.edge_detector_visualizer import EdgeVisualizer

UPLOAD_FOLDER = 'app/main/img'
OUTPUT_FOLDER = '/static/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

segmenters_cache = {}

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
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'File uploaded successfully'
        })

    return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400


@main_bp.route('/extract/<filename>')
def extract_pieces(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    try:
        # Initialize segmenter and visualizer
        segmenter = PieceSegmenter(filepath)
        visualizer = PieceVisualizer(output_dir=OUTPUT_FOLDER)

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
        edge_visualizer = EdgeVisualizer(output_dir=OUTPUT_FOLDER)
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
    filepath = os.path.join(OUTPUT_FOLDER, filename)

    if os.path.exists(filepath):
        return send_file(filepath)

    return jsonify({'error': 'Image not found'}), 404