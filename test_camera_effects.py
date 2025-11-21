"""Test script for new camera simulation effects."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.puzzle_gen.config import GeneratorConfig, CutType
from app.puzzle_gen.api import generate_puzzle_images

def test_camera_effects():
    """Generate a puzzle with all new camera effects enabled."""
    print("🎬 Testing new camera simulation effects...")
    print("-" * 60)

    # Create configuration with default values
    config = GeneratorConfig()

    # Print camera settings
    print("\n📸 Camera Configuration:")
    print(f"  Barrel Distortion: {config.camera.fisheye_strength:.2f}")
    print(f"  Perspective Distortion: {config.camera.perspective_strength:.2f}")
    print(f"  Lens Softness: {config.camera.lens_softness:.2f}")
    print(f"  Chromatic Aberration: {config.camera.color_aberration:.2f}")
    print(f"  Color Noise: {config.camera.color_noise:.2f}")
    print(f"  Purple Fringing: {config.camera.purple_fringing_intensity:.2f}")
    print(f"  Oversharpening: {config.camera.oversharpening_amount:.2f}")
    print(f"  Noise Reduction: {config.camera.noise_reduction_strength:.2f}")

    # Print shadow settings
    print("\n🌑 Shadow Configuration:")
    print(f"  Shadow Enabled: {config.render.shadow_enabled}")
    print(f"  Shadow Offset: ({config.render.shadow_offset_x}, {config.render.shadow_offset_y})")
    print(f"  Shadow Blur: {config.render.shadow_blur_radius}")
    print(f"  Shadow Opacity: {config.render.shadow_opacity:.2f}")

    # Use a variety of cut types
    cut_types = [
        CutType.WAVY.value,
        CutType.KNOB_ROUND.value,
        CutType.CHEVRON.value,
        CutType.SINGLE_WAVE.value,
        CutType.KNOB_TRIANGULAR.value,
        CutType.PARTIAL_WAVE.value
    ]

    print(f"\n🧩 Generating puzzle with {len(cut_types)} pieces...")
    print(f"  Cut types: {', '.join(cut_types)}")

    # Generate puzzle
    result = generate_puzzle_images(
        config=config,
        cut_types=cut_types,
        layout='2x3',
        apply_watermark=True,
        apply_camera=True,
        include_debug=True
    )

    # Save images
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    print(f"\n💾 Saving images to {output_dir}/...")
    saved_files = result.save_all(
        output_dir=output_dir,
        image_format='png',
        save_debug=True
    )

    print("\n✅ Successfully generated puzzle images:")
    for img_type, filepath in saved_files.items():
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  {img_type:12s}: {filepath.name:40s} ({file_size_mb:.2f} MB)")

    print(f"\n🎯 Seed: {result.seed}")
    print(f"📐 Resolution: {result.resolution[0]} x {result.resolution[1]}")
    print("-" * 60)
    print("✨ Test complete! Check the 'output' directory for generated images.")
    print("\n💡 Compare 'clean' vs 'noisy' images to see the camera effects.")

if __name__ == '__main__':
    test_camera_effects()
