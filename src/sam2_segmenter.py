"""SAM2 Segmenter Module

This module provides SAM2 segmentation functionality with prompts
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2
from PIL import Image


class SAM2Segmenter:
    """SAM2-based segmenter that takes prompts and generates precise masks"""

    def __init__(
        self,
        model_cfg: str = "sam2_hiera_l.yaml",
        checkpoint: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize SAM2 segmenter

        Args:
            model_cfg: SAM2 model configuration
            checkpoint: Path to SAM2 checkpoint
            device: Device to run on (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SAM2 on {self.device}")

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Build SAM2 model
            if checkpoint:
                sam2_model = build_sam2(model_cfg, checkpoint, device=self.device)
            else:
                # Try to load from default location
                sam2_model = build_sam2(model_cfg, device=self.device)

            self.predictor = SAM2ImagePredictor(sam2_model)
            self.model_loaded = True

        except Exception as e:
            print(f"Warning: Could not load SAM2: {e}")
            print("SAM2 will use placeholder mode. Install SAM2 for full functionality.")
            self.predictor = None
            self.model_loaded = False

    def set_image(self, image: np.ndarray):
        """
        Set the image for segmentation

        Args:
            image: Input image (H, W, 3) in RGB format
        """
        if self.model_loaded:
            self.predictor.set_image(image)
        self.current_image = image

    def segment_from_boxes(
        self,
        boxes: np.ndarray,
        multimask_output: bool = False
    ) -> List[Dict[str, np.ndarray]]:
        """
        Segment objects from bounding box prompts

        Args:
            boxes: Array of bounding boxes (N, 4) in format [x1, y1, x2, y2]
            multimask_output: Whether to return multiple masks per prompt

        Returns:
            List of segmentation results with keys:
                - 'mask': Binary segmentation mask
                - 'score': Confidence score
                - 'bbox': Original bounding box
        """
        if not self.model_loaded:
            return self._placeholder_segment_from_boxes(boxes)

        results = []

        for box in boxes:
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=multimask_output
            )

            # Take the best mask
            best_idx = np.argmax(scores)

            results.append({
                'mask': masks[best_idx],
                'score': float(scores[best_idx]),
                'bbox': box
            })

        return results

    def segment_from_points(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        multimask_output: bool = True
    ) -> List[Dict[str, np.ndarray]]:
        """
        Segment objects from point prompts

        Args:
            points: Array of points (N, 2) in format [x, y]
            labels: Point labels (1 = foreground, 0 = background)
            multimask_output: Whether to return multiple masks

        Returns:
            List of segmentation results
        """
        if not self.model_loaded:
            return self._placeholder_segment_from_points(points)

        if labels is None:
            labels = np.ones(len(points), dtype=np.int32)

        results = []

        for point, label in zip(points, labels):
            masks, scores, _ = self.predictor.predict(
                point_coords=point[None, :],
                point_labels=np.array([label]),
                box=None,
                multimask_output=multimask_output
            )

            # Take the best mask
            best_idx = np.argmax(scores)

            results.append({
                'mask': masks[best_idx],
                'score': float(scores[best_idx]),
                'point': point,
                'label': label
            })

        return results

    def segment_from_regions(
        self,
        regions: List[Dict],
        prompt_type: str = "box"
    ) -> List[Dict[str, np.ndarray]]:
        """
        Segment from DINO-detected regions

        Args:
            regions: List of regions from DINOv2Detector
            prompt_type: Type of prompt to use ('box' or 'point')

        Returns:
            List of segmentation results
        """
        if prompt_type == "box":
            boxes = np.array([r['bbox'] for r in regions])
            results = self.segment_from_boxes(boxes)

            # Add region metadata
            for result, region in zip(results, regions):
                result['confidence'] = region['confidence']
                result['dino_attention'] = region.get('attention_map')

        elif prompt_type == "point":
            points = np.array([r['center'] for r in regions])
            results = self.segment_from_points(points)

            # Add region metadata
            for result, region in zip(results, regions):
                result['confidence'] = region['confidence']
                result['dino_attention'] = region.get('attention_map')
                result['bbox'] = region['bbox']

        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        return results

    def _placeholder_segment_from_boxes(self, boxes: np.ndarray) -> List[Dict]:
        """Placeholder segmentation using simple morphology (when SAM2 not available)"""
        results = []
        h, w = self.current_image.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True

            results.append({
                'mask': mask,
                'score': 0.5,
                'bbox': box
            })

        return results

    def _placeholder_segment_from_points(self, points: np.ndarray) -> List[Dict]:
        """Placeholder segmentation from points"""
        results = []
        h, w = self.current_image.shape[:2]

        for point in points:
            x, y = point.astype(int)
            mask = np.zeros((h, w), dtype=bool)
            # Create circular mask
            cv2.circle(mask.astype(np.uint8), (x, y), 50, 1, -1)

            results.append({
                'mask': mask.astype(bool),
                'score': 0.5,
                'point': point,
                'label': 1
            })

        return results

    def visualize_masks(
        self,
        image: np.ndarray,
        results: List[Dict],
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Visualize segmentation masks on image

        Args:
            image: Original image (H, W, 3)
            results: List of segmentation results
            alpha: Transparency of masks

        Returns:
            Image with overlaid masks
        """
        vis_image = image.copy()

        # Color palette for different instances
        colors = self._generate_colors(len(results))

        for idx, result in enumerate(results):
            mask = result['mask']
            color = colors[idx]

            # Create colored mask
            colored_mask = np.zeros_like(image)
            colored_mask[mask] = color

            # Blend with original image
            vis_image = cv2.addWeighted(vis_image, 1.0, colored_mask, alpha, 0)

            # Draw contours
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis_image, contours, -1, color, 2)

            # Add label
            if 'bbox' in result:
                bbox = result['bbox'].astype(int)
                score = result.get('score', 0.0)
                label = f"Obj {idx+1}: {score:.2f}"
                cv2.putText(
                    vis_image, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        return vis_image

    @staticmethod
    def _generate_colors(n: int) -> List[Tuple[int, int, int]]:
        """Generate N distinct colors"""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]),
                cv2.COLOR_HSV2RGB
            )[0, 0]
            colors.append(tuple(map(int, color)))
        return colors

    def save_masks(
        self,
        results: List[Dict],
        output_dir: str,
        prefix: str = "mask"
    ):
        """
        Save segmentation masks to disk

        Args:
            results: List of segmentation results
            output_dir: Output directory
            prefix: Filename prefix
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        for idx, result in enumerate(results):
            mask = result['mask']

            # Save as PNG
            mask_path = os.path.join(output_dir, f"{prefix}_{idx:03d}.png")
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

            # Save metadata
            meta_path = os.path.join(output_dir, f"{prefix}_{idx:03d}.txt")
            with open(meta_path, 'w') as f:
                f.write(f"Score: {result.get('score', 0.0):.4f}\n")
                if 'bbox' in result:
                    bbox = result['bbox']
                    f.write(f"BBox: {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}\n")
                if 'confidence' in result:
                    f.write(f"DINO Confidence: {result['confidence']:.4f}\n")

        print(f"Saved {len(results)} masks to {output_dir}")
