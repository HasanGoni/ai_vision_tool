"""
Example: Custom Pin Detection

This example demonstrates how to use the custom grounding pipeline to detect
your specific "Pin" objects that are not regular pins.

Workflow:
1. Load reference image of your custom Pin (with optional mask)
2. Extract visual features using DINOv3
3. Define custom text description for Grounding DINO
4. Process target image to find and segment all similar Pins
5. Visualize results

Use Case:
    You have a custom Pin design that's not a standard object.
    You want to:
    - Find all instances of this Pin in images
    - Teach the model what YOUR specific Pin looks like (visually + text description)
    - Get precise segmentation masks
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from PIL import Image
import argparse

from pipelines.custom_grounding import CustomGroundingPipeline
from utils.preprocessing import load_image, load_mask, crop_to_mask
from utils.visualization import save_results, create_comparison_view


def main():
    parser = argparse.ArgumentParser(description='Custom Pin Detection Example')
    parser.add_argument('--reference', type=str, required=True,
                       help='Path to reference Pin image')
    parser.add_argument('--reference-mask', type=str, default=None,
                       help='Optional mask for reference image (focuses on Pin region)')
    parser.add_argument('--target', type=str, required=True,
                       help='Path to target image to search')
    parser.add_argument('--text-prompt', type=str,
                       default='my custom Pin with unique design',
                       help='Text description of your Pin')
    parser.add_argument('--output', type=str, default='output/pin_detection_result.jpg',
                       help='Output path for visualization')
    parser.add_argument('--similarity-threshold', type=float, default=0.5,
                       help='Visual similarity threshold (0-1)')
    parser.add_argument('--detection-threshold', type=float, default=0.3,
                       help='Text detection threshold (0-1)')
    parser.add_argument('--fusion-strategy', type=str, default='multiply',
                       choices=['multiply', 'max', 'weighted'],
                       help='How to fuse visual and text signals')
    parser.add_argument('--use-visual', action='store_true', default=True,
                       help='Use visual matching (DINOv3)')
    parser.add_argument('--use-text', action='store_true', default=True,
                       help='Use text grounding (Grounding DINO)')
    parser.add_argument('--no-visual', action='store_false', dest='use_visual',
                       help='Disable visual matching')
    parser.add_argument('--no-text', action='store_false', dest='use_text',
                       help='Disable text grounding')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("CUSTOM PIN DETECTION PIPELINE")
    print("="*70)

    # Load images
    print("\n[1/6] Loading images...")
    reference_image = load_image(args.reference)
    target_image = load_image(args.target)

    print(f"  Reference image: {reference_image.shape}")
    print(f"  Target image: {target_image.shape}")

    # Load mask if provided
    reference_mask = None
    if args.reference_mask is not None:
        print(f"  Loading reference mask: {args.reference_mask}")
        reference_mask = load_mask(args.reference_mask)

        # Optionally crop reference to mask region for better feature extraction
        reference_image_cropped, reference_mask_cropped = crop_to_mask(
            reference_image, reference_mask, padding=20
        )
        print(f"  Cropped reference to mask region: {reference_image_cropped.shape}")
    else:
        reference_image_cropped = reference_image

    # Initialize pipeline
    print("\n[2/6] Initializing Custom Grounding Pipeline...")
    pipeline = CustomGroundingPipeline(
        dinov3_model="dinov2_vitb14",
        sam2_model="vit_b",
        use_visual_matching=args.use_visual,
        use_text_grounding=args.use_text
    )

    # Process reference image
    print("\n[3/6] Extracting reference features...")
    print(f"  Text prompt: '{args.text_prompt}'")

    reference_features = None
    if args.use_visual:
        reference_features = pipeline.process_reference_image(
            reference_image=Image.fromarray(reference_image_cropped),
            reference_mask=None  # Already cropped
        )
        print("  ✓ Reference features extracted")

    # Detect and segment in target image
    print("\n[4/6] Detecting custom Pins in target image...")
    print(f"  Visual matching: {'ON' if args.use_visual else 'OFF'}")
    print(f"  Text grounding: {'ON' if args.use_text else 'OFF'}")
    print(f"  Fusion strategy: {args.fusion_strategy}")

    results = pipeline.detect_and_segment(
        target_image=Image.fromarray(target_image),
        reference_features=reference_features,
        text_prompt=args.text_prompt if args.use_text else None,
        fusion_strategy=args.fusion_strategy,
        similarity_threshold=args.similarity_threshold,
        detection_threshold=args.detection_threshold,
        min_region_area=100
    )

    # Display results
    print("\n[5/6] Results:")
    print(f"  Detected {len(results['masks'])} custom Pin instances")

    for i, (score, source) in enumerate(zip(results['scores'], results['sources'])):
        phrase = results['phrases'][i] if results['phrases'][i] else 'N/A'
        print(f"    Pin {i+1}: score={score:.3f}, source={source}, phrase='{phrase}'")

    # Visualize
    print("\n[6/6] Creating visualization...")
    vis = pipeline.visualize_results(
        image=target_image,
        results=results,
        show_boxes=True,
        show_masks=True,
        show_labels=True
    )

    # Save results
    save_results(
        output_path=args.output,
        image=target_image,
        results=results,
        save_individual_masks=True
    )

    print(f"\n✓ Results saved to: {args.output}")
    print("\n" + "="*70)
    print("DETECTION COMPLETE!")
    print("="*70 + "\n")


def example_usage_without_cli():
    """
    Example usage without command line arguments
    For use in notebooks or scripts
    """
    print("Example: Custom Pin Detection (Programmatic Usage)")
    print("="*70)

    # Configuration
    reference_path = "data/reference/my_custom_pin.jpg"
    reference_mask_path = "data/reference/my_custom_pin_mask.png"  # Optional
    target_path = "data/target/street_scene.jpg"
    text_description = "my custom Pin with red top and blue bottom and unique pattern"

    # Load images
    print("\n1. Loading images...")
    try:
        reference_image = load_image(reference_path)
        target_image = load_image(target_path)
        print("  ✓ Images loaded")
    except Exception as e:
        print(f"  ✗ Error loading images: {e}")
        print("\n  Please update the paths to point to your actual images!")
        return

    # Initialize pipeline
    print("\n2. Initializing pipeline...")
    pipeline = CustomGroundingPipeline(
        use_visual_matching=True,
        use_text_grounding=True
    )

    # Extract reference features
    print("\n3. Processing reference image...")
    reference_features = pipeline.process_reference_image(
        reference_image=Image.fromarray(reference_image)
    )

    # Detect and segment
    print(f"\n4. Detecting: '{text_description}'...")
    results = pipeline.detect_and_segment(
        target_image=Image.fromarray(target_image),
        reference_features=reference_features,
        text_prompt=text_description,
        fusion_strategy='multiply',
        similarity_threshold=0.5,
        detection_threshold=0.3
    )

    # Results
    print(f"\n5. Found {len(results['masks'])} instances")

    # Visualize
    vis = pipeline.visualize_results(target_image, results)

    print("\n✓ Done!")

    return results, vis


if __name__ == "__main__":
    # Check if running with CLI arguments
    if len(sys.argv) > 1:
        main()
    else:
        print("\nNo CLI arguments provided.")
        print("\nUsage:")
        print("  python custom_pin_detection.py \\")
        print("    --reference data/reference/pin.jpg \\")
        print("    --target data/target/scene.jpg \\")
        print("    --text-prompt 'my custom Pin' \\")
        print("    --output results/detection.jpg")
        print("\n" + "="*70)
        print("Running example usage instead...")
        print("="*70)
        example_usage_without_cli()
