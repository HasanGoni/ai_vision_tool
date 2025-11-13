"""
Simple Example: Quick Start Guide

This is a minimal example to get started with the custom grounding pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.custom_grounding import CustomGroundingPipeline
from PIL import Image
import numpy as np


def simple_reference_matching():
    """
    Example 1: Reference-based matching only
    Use case: Find all red cars similar to a reference red car image
    """
    print("\n" + "="*60)
    print("Example 1: Reference-Based Visual Matching")
    print("="*60)

    # Initialize pipeline (visual matching only)
    pipeline = CustomGroundingPipeline(
        use_visual_matching=True,
        use_text_grounding=False
    )

    # Load your images
    # reference_image = Image.open("reference_red_car.jpg")
    # target_image = Image.open("street_scene.jpg")

    # For demo purposes, create dummy images
    print("\nNote: Using dummy images for demo.")
    print("Replace with your actual images!")

    reference_image = Image.new('RGB', (224, 224), color='red')
    target_image = Image.new('RGB', (800, 600), color='blue')

    # Extract reference features
    print("\n1. Extracting reference features...")
    ref_features = pipeline.process_reference_image(reference_image)

    # Find similar regions in target
    print("2. Finding similar regions...")
    results = pipeline.detect_and_segment(
        target_image=target_image,
        reference_features=ref_features,
        text_prompt=None,  # No text grounding
        similarity_threshold=0.5
    )

    print(f"\n✓ Found {len(results['masks'])} similar objects")

    return results


def simple_text_grounding():
    """
    Example 2: Text-based grounding only
    Use case: Detect objects using custom text description
    """
    print("\n" + "="*60)
    print("Example 2: Text-Based Object Detection")
    print("="*60)

    # Initialize pipeline (text grounding only)
    pipeline = CustomGroundingPipeline(
        use_visual_matching=False,
        use_text_grounding=True
    )

    # Load your image
    # target_image = Image.open("scene.jpg")

    # For demo
    print("\nNote: Using dummy image for demo.")
    print("Replace with your actual image!")

    target_image = Image.new('RGB', (800, 600), color='green')

    # Detect using text
    print("\n1. Detecting with text prompt: 'my custom Pin'")
    results = pipeline.detect_and_segment(
        target_image=target_image,
        reference_features=None,  # No visual matching
        text_prompt="my custom Pin with red top",
        detection_threshold=0.3
    )

    print(f"\n✓ Found {len(results['masks'])} objects matching the description")

    return results


def simple_hybrid_approach():
    """
    Example 3: Combined visual + text grounding (RECOMMENDED)
    Use case: Most accurate detection using both visual similarity and text
    """
    print("\n" + "="*60)
    print("Example 3: Hybrid (Visual + Text) Detection")
    print("="*60)

    # Initialize full pipeline
    pipeline = CustomGroundingPipeline(
        use_visual_matching=True,
        use_text_grounding=True
    )

    # Load images
    print("\nNote: Using dummy images for demo.")
    print("Replace with your actual images!")

    reference_image = Image.new('RGB', (224, 224), color='red')
    target_image = Image.new('RGB', (800, 600), color='white')

    # Process reference
    print("\n1. Processing reference image...")
    ref_features = pipeline.process_reference_image(reference_image)

    # Detect with both visual + text
    print("2. Detecting with hybrid approach...")
    results = pipeline.detect_and_segment(
        target_image=target_image,
        reference_features=ref_features,
        text_prompt="my custom Pin",
        fusion_strategy='multiply',  # Requires both visual AND text match
        similarity_threshold=0.5,
        detection_threshold=0.3
    )

    print(f"\n✓ Found {len(results['masks'])} objects")
    print("\nDetection sources:")
    for i, source in enumerate(results['sources']):
        print(f"  Object {i+1}: {source}")

    return results


def minimal_code_example():
    """
    Minimal code for quick copy-paste
    """
    print("\n" + "="*60)
    print("Minimal Code Example (Copy-Paste Ready)")
    print("="*60)

    code = '''
from pipelines.custom_grounding import CustomGroundingPipeline
from PIL import Image

# 1. Initialize
pipeline = CustomGroundingPipeline()

# 2. Load images
ref_img = Image.open("my_custom_pin_reference.jpg")
target_img = Image.open("scene_to_search.jpg")

# 3. Extract reference features
ref_features = pipeline.process_reference_image(ref_img)

# 4. Detect and segment
results = pipeline.detect_and_segment(
    target_image=target_img,
    reference_features=ref_features,
    text_prompt="my custom Pin",
    similarity_threshold=0.5,
    detection_threshold=0.3
)

# 5. Get results
print(f"Found {len(results['masks'])} objects")

# 6. Visualize
vis = pipeline.visualize_results(target_img, results)
    '''

    print(code)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SIMPLE EXAMPLES FOR CUSTOM GROUNDING PIPELINE")
    print("="*70)

    # Run all examples
    print("\n\nRunning all examples...\n")

    # Example 1: Visual only
    simple_reference_matching()

    # Example 2: Text only
    simple_text_grounding()

    # Example 3: Hybrid (recommended)
    simple_hybrid_approach()

    # Minimal code
    minimal_code_example()

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE!")
    print("="*70)

    print("\n\nNext Steps:")
    print("  1. Replace dummy images with your actual images")
    print("  2. Adjust thresholds based on your use case")
    print("  3. Try different fusion strategies: 'multiply', 'max', 'weighted'")
    print("  4. See custom_pin_detection.py for full CLI example")
    print()
