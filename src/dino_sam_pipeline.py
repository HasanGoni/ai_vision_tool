"""DINO + SAM2 Integration Pipeline

This module combines DINOv2 detection with SAM2 segmentation for automatic object segmentation
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .dino_detector import DINOv2Detector
from .sam2_segmenter import SAM2Segmenter


class DinoSamPipeline:
    """
    Automatic object segmentation pipeline combining DINOv2 and SAM2

    Workflow:
    1. DINOv2 detects salient regions/objects automatically
    2. Generate prompts (boxes/points) from DINO detections
    3. SAM2 performs precise segmentation using those prompts
    4. Save and visualize results
    """

    def __init__(
        self,
        dino_model: str = "dinov2_vitb14",
        sam2_config: str = "sam2_hiera_l.yaml",
        sam2_checkpoint: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the pipeline

        Args:
            dino_model: DINOv2 model variant
            sam2_config: SAM2 configuration
            sam2_checkpoint: Path to SAM2 checkpoint
            device: Device to run on
        """
        print("=" * 60)
        print("Initializing DINO-SAM2 Pipeline")
        print("=" * 60)

        # Initialize DINOv2 detector
        self.dino = DINOv2Detector(model_name=dino_model, device=device)

        # Initialize SAM2 segmenter
        self.sam2 = SAM2Segmenter(
            model_cfg=sam2_config,
            checkpoint=sam2_checkpoint,
            device=device
        )

        print("Pipeline ready!")
        print("=" * 60)

    def process_image(
        self,
        image_path: str,
        num_objects: int = 5,
        prompt_type: str = "box",
        min_area: int = 100,
        attention_threshold: float = 0.6,
        visualize: bool = True,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Process a single image: detect and segment objects automatically

        Args:
            image_path: Path to input image
            num_objects: Maximum number of objects to detect
            prompt_type: Type of prompt for SAM2 ('box' or 'point')
            min_area: Minimum area for detected regions
            attention_threshold: Threshold for DINO attention
            visualize: Whether to create visualizations
            save_dir: Directory to save results (if None, don't save)

        Returns:
            Dictionary with results:
                - 'image': Original image
                - 'dino_regions': Detected regions from DINO
                - 'segmentations': SAM2 segmentation results
                - 'visualization': Visualization image (if visualize=True)
        """
        print(f"\nProcessing: {image_path}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 1: Detect regions with DINOv2
        print(f"  [1/3] Detecting salient regions with DINOv2...")
        regions = self.dino.detect_salient_regions(
            image_rgb,
            num_regions=num_objects,
            min_area=min_area,
            attention_threshold=attention_threshold
        )
        print(f"        Found {len(regions)} regions")

        if len(regions) == 0:
            print("  Warning: No regions detected. Try lowering attention_threshold.")
            return {
                'image': image_rgb,
                'dino_regions': [],
                'segmentations': [],
                'visualization': image_rgb
            }

        # Step 2: Segment with SAM2
        print(f"  [2/3] Segmenting with SAM2 using '{prompt_type}' prompts...")
        self.sam2.set_image(image_rgb)
        segmentations = self.sam2.segment_from_regions(regions, prompt_type=prompt_type)
        print(f"        Segmented {len(segmentations)} objects")

        # Step 3: Visualize and save
        results = {
            'image': image_rgb,
            'dino_regions': regions,
            'segmentations': segmentations
        }

        if visualize:
            print(f"  [3/3] Creating visualizations...")
            vis_image = self._create_visualization(image_rgb, regions, segmentations)
            results['visualization'] = vis_image

        if save_dir:
            self._save_results(image_path, results, save_dir)

        return results

    def process_batch(
        self,
        image_paths: List[str],
        **kwargs
    ) -> List[Dict]:
        """
        Process multiple images

        Args:
            image_paths: List of image paths
            **kwargs: Arguments passed to process_image

        Returns:
            List of results for each image
        """
        results = []

        print(f"\nProcessing batch of {len(image_paths)} images...")
        print("=" * 60)

        for idx, img_path in enumerate(image_paths, 1):
            print(f"\nImage {idx}/{len(image_paths)}")
            result = self.process_image(img_path, **kwargs)
            results.append(result)

        print("\n" + "=" * 60)
        print(f"Batch processing complete! Processed {len(results)} images")

        return results

    def _create_visualization(
        self,
        image: np.ndarray,
        regions: List[Dict],
        segmentations: List[Dict]
    ) -> np.ndarray:
        """Create comprehensive visualization"""

        # Create multi-panel visualization
        # Panel 1: Original image
        panel1 = image.copy()

        # Panel 2: DINO detections
        features = self.dino.extract_features(image)
        h, w = image.shape[:2]
        attention_map = self.dino._compute_attention_map(features, h, w)
        panel2 = self.dino.visualize_detections(image, regions, attention_map)

        # Panel 3: SAM2 segmentations
        panel3 = self.sam2.visualize_masks(image, segmentations)

        # Combine panels
        vis = self._combine_panels([panel1, panel2, panel3],
                                   labels=["Original", "DINO Detections", "SAM2 Segments"])

        return vis

    def _combine_panels(
        self,
        panels: List[np.ndarray],
        labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """Combine multiple panels into one visualization"""

        if labels is None:
            labels = [f"Panel {i+1}" for i in range(len(panels))]

        # Add labels to panels
        labeled_panels = []
        for panel, label in zip(panels, labels):
            panel_copy = panel.copy()
            cv2.putText(
                panel_copy, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                panel_copy, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA
            )
            labeled_panels.append(panel_copy)

        # Concatenate horizontally
        combined = np.hstack(labeled_panels)

        return combined

    def _save_results(
        self,
        image_path: str,
        results: Dict,
        save_dir: str
    ):
        """Save results to disk"""

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Get image name without extension
        img_name = Path(image_path).stem

        # Save visualization
        if 'visualization' in results:
            vis_path = save_path / f"{img_name}_visualization.jpg"
            vis_bgr = cv2.cvtColor(results['visualization'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(vis_path), vis_bgr)
            print(f"        Saved visualization: {vis_path}")

        # Save individual masks
        masks_dir = save_path / f"{img_name}_masks"
        self.sam2.save_masks(
            results['segmentations'],
            str(masks_dir),
            prefix="mask"
        )

        # Save summary
        summary_path = save_path / f"{img_name}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Image: {image_path}\n")
            f.write(f"Detected regions: {len(results['dino_regions'])}\n")
            f.write(f"Segmented objects: {len(results['segmentations'])}\n\n")

            for idx, seg in enumerate(results['segmentations']):
                f.write(f"Object {idx + 1}:\n")
                f.write(f"  SAM2 Score: {seg.get('score', 0.0):.4f}\n")
                f.write(f"  DINO Confidence: {seg.get('confidence', 0.0):.4f}\n")
                if 'bbox' in seg:
                    bbox = seg['bbox']
                    f.write(f"  BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]\n")
                mask_area = seg['mask'].sum()
                f.write(f"  Mask Area: {mask_area} pixels\n\n")

        print(f"        Saved summary: {summary_path}")


def main():
    """Example usage of the pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="DINO + SAM2 Automatic Segmentation")
    parser.add_argument("images", nargs="+", help="Input image path(s)")
    parser.add_argument("--num-objects", type=int, default=5,
                       help="Maximum number of objects to detect")
    parser.add_argument("--prompt-type", choices=["box", "point"], default="box",
                       help="Type of prompt for SAM2")
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="Attention threshold for DINO detection")
    parser.add_argument("--save-dir", type=str, default="output",
                       help="Directory to save results")
    parser.add_argument("--dino-model", type=str, default="dinov2_vitb14",
                       choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                       help="DINOv2 model variant")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = DinoSamPipeline(dino_model=args.dino_model)

    # Process images
    pipeline.process_batch(
        args.images,
        num_objects=args.num_objects,
        prompt_type=args.prompt_type,
        attention_threshold=args.threshold,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
