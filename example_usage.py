#!/usr/bin/env python3
"""
Example usage of the DINO-SAM2 pipeline for automatic object segmentation

This script demonstrates how to use the pipeline to automatically detect and segment objects
"""

import cv2
import numpy as np
from pathlib import Path
from src.dino_sam_pipeline import DinoSamPipeline


def example_single_image():
    """Example: Process a single image"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Process Single Image")
    print("=" * 70)

    # Initialize pipeline
    pipeline = DinoSamPipeline(
        dino_model="dinov2_vitb14",  # Options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
    )

    # Process image
    results = pipeline.process_image(
        image_path="path/to/your/image.jpg",
        num_objects=5,              # Max number of objects to detect
        prompt_type="box",           # 'box' or 'point'
        attention_threshold=0.6,     # Lower = more regions detected
        save_dir="output/example1"
    )

    print(f"\nDetected {len(results['dino_regions'])} regions")
    print(f"Segmented {len(results['segmentations'])} objects")


def example_batch_processing():
    """Example: Process multiple images"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Batch Processing")
    print("=" * 70)

    # Get all images from a directory
    image_dir = Path("path/to/your/images")
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    # Initialize pipeline
    pipeline = DinoSamPipeline()

    # Process all images
    results = pipeline.process_batch(
        image_paths=[str(p) for p in image_paths],
        num_objects=3,
        save_dir="output/batch"
    )

    print(f"\nProcessed {len(results)} images")


def example_with_custom_parameters():
    """Example: Fine-tune detection parameters"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Custom Parameters")
    print("=" * 70)

    pipeline = DinoSamPipeline(
        dino_model="dinov2_vitl14",  # Larger model for better features
    )

    results = pipeline.process_image(
        image_path="path/to/your/image.jpg",
        num_objects=10,              # Detect more objects
        prompt_type="point",         # Use point prompts instead of boxes
        min_area=50,                 # Smaller minimum area
        attention_threshold=0.5,     # Lower threshold = more sensitive
        save_dir="output/custom"
    )

    # Access individual results
    for idx, seg in enumerate(results['segmentations']):
        print(f"\nObject {idx + 1}:")
        print(f"  SAM2 Score: {seg['score']:.3f}")
        print(f"  DINO Confidence: {seg['confidence']:.3f}")
        print(f"  Mask Area: {seg['mask'].sum()} pixels")


def example_programmatic_usage():
    """Example: Use results programmatically"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Programmatic Usage")
    print("=" * 70)

    pipeline = DinoSamPipeline()

    # Process without saving
    results = pipeline.process_image(
        image_path="path/to/your/image.jpg",
        num_objects=5,
        visualize=True,
        save_dir=None  # Don't save automatically
    )

    # Work with the results
    image = results['image']
    regions = results['dino_regions']
    segmentations = results['segmentations']

    # Filter high-confidence detections
    high_conf_segs = [
        seg for seg in segmentations
        if seg.get('confidence', 0) > 0.7
    ]

    print(f"\nFound {len(high_conf_segs)} high-confidence objects")

    # Extract individual object crops
    for idx, seg in enumerate(high_conf_segs):
        mask = seg['mask']
        bbox = seg['bbox'].astype(int)

        # Crop object
        x1, y1, x2, y2 = bbox
        cropped = image[y1:y2, x1:x2].copy()

        # Apply mask
        mask_crop = mask[y1:y2, x1:x2]
        cropped[~mask_crop] = 0  # Black background

        # Save individual object
        output_path = f"output/object_{idx:03d}.png"
        cv2.imwrite(output_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(high_conf_segs)} object crops")


def create_test_image():
    """Create a simple test image with geometric shapes"""
    print("\n" + "=" * 70)
    print("Creating test image...")
    print("=" * 70)

    # Create white canvas
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # Draw some colorful shapes
    # Red circle
    cv2.circle(img, (150, 150), 80, (255, 0, 0), -1)

    # Green rectangle
    cv2.rectangle(img, (300, 80), (500, 220), (0, 255, 0), -1)

    # Blue triangle
    pts = np.array([[650, 100], [750, 100], [700, 200]], np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 255))

    # Yellow ellipse
    cv2.ellipse(img, (250, 400), (120, 80), 0, 0, 360, (0, 255, 255), -1)

    # Purple polygon
    pts = np.array([[550, 350], [650, 350], [680, 450], [520, 450]], np.int32)
    cv2.fillPoly(img, [pts], (200, 0, 200))

    # Save test image
    Path("test_images").mkdir(exist_ok=True)
    test_path = "test_images/test_shapes.jpg"
    cv2.imwrite(test_path, img)

    print(f"Created test image: {test_path}")

    return test_path


def run_test_demo():
    """Run a complete demo with test image"""
    print("\n" + "=" * 70)
    print("RUNNING DEMO WITH TEST IMAGE")
    print("=" * 70)

    # Create test image
    test_image = create_test_image()

    # Initialize pipeline
    pipeline = DinoSamPipeline()

    # Process test image
    print("\nProcessing test image...")
    results = pipeline.process_image(
        image_path=test_image,
        num_objects=5,
        prompt_type="box",
        attention_threshold=0.5,
        save_dir="output/demo"
    )

    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: output/demo/")
    print(f"  - Visualization: output/demo/test_shapes_visualization.jpg")
    print(f"  - Individual masks: output/demo/test_shapes_masks/")
    print(f"  - Summary: output/demo/test_shapes_summary.txt")
    print("\nDetection Summary:")
    print(f"  Total regions detected: {len(results['dino_regions'])}")
    print(f"  Total objects segmented: {len(results['segmentations'])}")

    for idx, seg in enumerate(results['segmentations']):
        print(f"\n  Object {idx + 1}:")
        print(f"    SAM2 Score: {seg['score']:.3f}")
        print(f"    DINO Confidence: {seg.get('confidence', 0):.3f}")


if __name__ == "__main__":
    import sys

    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║             DINO-SAM2 Automatic Segmentation Examples            ║
    ║                                                                   ║
    ║  Combining DINOv2 object detection with SAM2 segmentation        ║
    ║  for fully automatic object segmentation                          ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    if len(sys.argv) > 1:
        # Run with command line arguments
        from src.dino_sam_pipeline import main
        main()
    else:
        # Run demo
        print("\nNo arguments provided. Running demo with test image...\n")
        run_test_demo()

        print("\n" + "=" * 70)
        print("To run with your own images:")
        print("=" * 70)
        print("\n  python example_usage.py path/to/image1.jpg path/to/image2.jpg\n")
        print("Options:")
        print("  --num-objects N      Max objects to detect (default: 5)")
        print("  --prompt-type TYPE   'box' or 'point' (default: box)")
        print("  --threshold T        Detection threshold 0-1 (default: 0.6)")
        print("  --save-dir DIR       Output directory (default: output)")
        print("  --dino-model MODEL   Model size: vits14/vitb14/vitl14/vitg14")
        print()
