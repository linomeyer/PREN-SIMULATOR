from flask import jsonify, request
from app.puzzle_gen import puzzle_gen_bp
from app.puzzle_gen.api import generate_puzzle_images
from app.puzzle_gen.config import (
    GeneratorConfig, PuzzleConfig, RenderConfig,
    CameraConfig, OutputConfig, Layout
)
from app.puzzle_gen.performance import print_performance_report
from app.puzzle_gen.calibration import GridCalibrator
from PIL import Image
import io
import base64
import random
import cv2
import numpy as np
import json
from pathlib import Path


# All available cut types (13 variants)
AVAILABLE_CUT_TYPES = [
    #'wavy',
    'knob_square',
    'knob_triangular',
    'knob_wavy',
    'knob_square_new',
    'knob_trapezoid',
    'knob_round',
    'chevron',
    'double_chevron',
    'single_wave',
    'partial_wave',
    #'var_amp_increasing',
    #'var_amp_chaotic'
]


@puzzle_gen_bp.route('/generate', methods=['POST'])
def generate():
    """
    Generate a new puzzle with both noisy and solution images.

    Request body (JSON, all optional):
    {
        "layout": "2x3" or "3x2" (default: "2x3"),
        "seed": int or null (default: null = random),
        "cut_types": ["wavy", "knob_square", ...] (default: ["wavy", "knob_square"]),
        "apply_watermark": bool (default: true),
        "camera_intensity": float 0.0-1.0 (default: 1.0)
    }

    Returns:
    {
        "success": true,
        "noisy_image": "base64...",
        "solution_image": "base64...",
        "metadata": {
            "seed": 12345,
            "layout": "2x3",
            "cut_types": ["wavy", "knob_square"],
            "piece_count": 6,
            "resolution": [4608, 2592]
        }
    }
    """
    try:
        # Parse request parameters
        data = request.get_json() or {}

        layout_str = data.get('layout', '2x3')
        seed = data.get('seed', None)

        # Generate seed early if not provided (for reproducible cut_types)
        if seed is None:
            seed = random.randint(0, 999999)

        # If no cut_types provided, randomly select 3-5 from all available types
        # IMPORTANT: If seed is provided, use it to make cut_types reproducible
        if 'cut_types' not in data:
            # Seed the RNG for cut_types selection (seed is always set now)
            random.seed(seed)

            num_types = random.randint(3, 5)
            cut_types = random.sample(AVAILABLE_CUT_TYPES, num_types)

            # Note: The seed will be set again in PuzzleGenerator.__init__
            # This ensures cut_types are reproducible with the same seed
        else:
            cut_types = data.get('cut_types')

        apply_watermark = data.get('apply_watermark', True)
        camera_intensity = data.get('camera_intensity', 1.0)
        enable_perf_logging = data.get('enable_performance_logging', False)

        # Validate layout
        if layout_str not in ['2x3', '3x2']:
            return jsonify({'error': 'Layout must be "2x3" or "3x2"'}), 400

        # Determine Layout enum
        layout_enum = Layout.LAYOUT_2X3 if layout_str == '2x3' else Layout.LAYOUT_3X2

        # Create configuration
        config = GeneratorConfig(
            puzzle=PuzzleConfig(
                layout=layout_enum,
                seed=seed
            ),
            render=RenderConfig(
                margin_px=200,
                piece_spacing_px=100
            ),
            camera=CameraConfig(
                intensity=camera_intensity
            ),
            output=OutputConfig(
                max_width=4608,
                max_height=2592
            )
        )

        # Set performance logging flag
        GeneratorConfig.enable_performance_logging = enable_perf_logging

        # Generate puzzle
        result = generate_puzzle_images(
            config=config,
            cut_types=cut_types,
            layout=layout_str,
            apply_watermark=apply_watermark,
            apply_camera=True,
            include_debug=False
        )

        # Print performance report if enabled
        print_performance_report()

        # Convert images to base64
        def array_to_base64(img_array):
            img = Image.fromarray(img_array)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')

        noisy_b64 = array_to_base64(result.noisy_image)
        solution_b64 = array_to_base64(result.solution_image)

        # Prepare response
        response = {
            'success': True,
            'noisy_image': noisy_b64,
            'solution_image': solution_b64,
            'metadata': {
                'seed': result.seed,
                'layout': result.layout,
                'cut_types': result.cut_types,
                'piece_count': result.piece_count,
                'resolution': list(result.resolution)
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@puzzle_gen_bp.route('/calibrate-from-grid', methods=['POST'])
def calibrate_from_grid():
    """
    Calibrate barrel distortion from a puzzle image with calibration grid.

    Accepts either:
    1. A base64-encoded image in JSON body: {"image": "base64..."}
    2. A file upload (multipart/form-data)

    Returns:
    {
        "success": true,
        "calibration": {
            "k1": -0.3,
            "k2": 0.05,
            "k3": 0.0,
            "fx": 4608,
            "fy": 4608,
            "cx": 2304.0,
            "cy": 1296.0,
            "image_size": [4608, 2592],
            "num_lines_detected": 35,
            "rms_error": 1.2
        },
        "visualization": "base64..."  # Optional: image with detected lines
    }
    """
    try:
        # Determine input type
        if request.is_json:
            # JSON with base64 image
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': 'Missing "image" in request body'}), 400

            # Decode base64 image
            image_b64 = data['image']
            image_bytes = base64.b64decode(image_b64)
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        elif 'file' in request.files:
            # File upload
            file = request.files['file']
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        else:
            return jsonify({'error': 'No image provided. Send JSON with "image" or upload file.'}), 400

        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Initialize calibrator with expected grid spacing
        # Default: 20mm at 300 DPI = 236.22 pixels
        expected_spacing = request.args.get('grid_spacing', 236.0, type=float)
        calibrator = GridCalibrator(expected_spacing_px=expected_spacing)

        # Perform calibration
        calibration_params = calibrator.calibrate_from_image(image)

        # Save calibration to file (same location as main puzzle solver)
        calibration_file = Path(__file__).parent.parent / 'main' / 'calibration_params.json'
        calibration_data = {
            'calibration_date': __import__('datetime').datetime.now().isoformat(),
            'image_shape': calibration_params['image_size'][::-1],  # [height, width]
            'barrel_distortion': {
                'k1': calibration_params['k1'],
                'k2': calibration_params['k2'],
                'k3': calibration_params['k3'],
                'fx': calibration_params['fx'],
                'fy': calibration_params['fy'],
                'cx': calibration_params['cx'],
                'cy': calibration_params['cy'],
                'image_size': calibration_params['image_size']
            },
            'perspective': {'enabled': False, 'transform_matrix': None},
            'vignette': {'strength': 0.2, 'enabled': True, 'center': [calibration_params['cx'], calibration_params['cy']], 'radius': max(calibration_params['image_size']) / 2},
            'chromatic_aberration': {'enabled': False, 'r_offset': [0, 0], 'b_offset': [0, 0]}
        }

        calibration_file.parent.mkdir(parents=True, exist_ok=True)
        with open(calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        # Create visualization (optional)
        create_vis = request.args.get('visualize', 'true').lower() == 'true'
        vis_b64 = None

        if create_vis:
            # Re-detect lines for visualization
            h_lines, v_lines = calibrator.detect_grid_lines(image)
            vis_image = calibrator.visualize_detection(image, h_lines, v_lines)

            # Convert to base64
            _, buffer = cv2.imencode('.png', vis_image)
            vis_b64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare response
        response = {
            'success': True,
            'calibration': calibration_params,
            'calibration_saved': str(calibration_file)
        }

        if vis_b64:
            response['visualization'] = vis_b64

        return jsonify(response)

    except ValueError as e:
        # Insufficient lines detected
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

    except Exception as e:
        # Other errors
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@puzzle_gen_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'puzzle-generator'
    })
