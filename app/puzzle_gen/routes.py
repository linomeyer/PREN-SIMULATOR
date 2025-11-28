from flask import jsonify, request
from app.puzzle_gen import puzzle_gen_bp
from app.puzzle_gen.api import generate_puzzle_images
from app.puzzle_gen.config import (
    GeneratorConfig, PuzzleConfig, RenderConfig,
    CameraConfig, OutputConfig, Layout
)
from app.puzzle_gen.performance import print_performance_report
from PIL import Image
import io
import base64
import random


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

        # If no cut_types provided, randomly select 3-5 from all available types
        if 'cut_types' not in data:
            num_types = random.randint(3, 5)
            cut_types = random.sample(AVAILABLE_CUT_TYPES, num_types)
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


@puzzle_gen_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'puzzle-generator'
    })
