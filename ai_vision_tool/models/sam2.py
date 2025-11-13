"""
SAM2 (Segment Anything Model 2) Wrapper
Provides zero-shot segmentation for any object given prompts
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from PIL import Image
import cv2


class SAM2Model:
    """
    Wrapper for Segment Anything Model 2 (SAM2)
    Performs instance segmentation from various prompts (points, boxes, masks)
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize SAM2 model

        Args:
            model_type: Model size (vit_h, vit_l, vit_b, vit_tiny)
            checkpoint_path: Path to model checkpoint (None = download from hub)
            device: Device to run on (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type

        # Import SAM2 (requires sam2 package installation)
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Install with: "
                "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )

        print(f"Loading SAM2 model: {model_type} on {self.device}")

        # Build model
        if checkpoint_path is None:
            # Download from torch hub
            checkpoint_path = self._download_checkpoint(model_type)

        self.sam2_model = build_sam2(model_type, checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

        print(f"SAM2 initialized successfully")

    def _download_checkpoint(self, model_type: str) -> str:
        """Download SAM2 checkpoint from hub"""
        checkpoint_urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_large.pt",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_large.pt",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_base_plus.pt",
            "vit_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_tiny.pt"
        }

        url = checkpoint_urls.get(model_type)
        if url is None:
            raise ValueError(f"Unknown model type: {model_type}")

        # Use torch hub to download
        checkpoint_path = torch.hub.load_state_dict_from_url(url)
        return checkpoint_path

    def set_image(self, image: Union[np.ndarray, Image.Image]):
        """
        Set the image to be segmented

        Args:
            image: Input image (numpy array or PIL Image)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert RGB to BGR if needed (SAM2 expects RGB)
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        self.predictor.set_image(image)
        self.current_image_shape = image.shape[:2]

    def segment_from_points(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        multimask_output: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Segment object from point prompts

        Args:
            point_coords: Point coordinates [N, 2] in (x, y) format
            point_labels: Point labels [N] (1 = foreground, 0 = background)
            multimask_output: Whether to return multiple masks

        Returns:
            Dictionary containing:
                - 'masks': Segmentation masks [num_masks, H, W]
                - 'scores': Confidence scores [num_masks]
                - 'logits': Low-res logits [num_masks, 256, 256]
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output
        )

        return {
            'masks': masks,
            'scores': scores,
            'logits': logits
        }

    def segment_from_box(
        self,
        box: np.ndarray,
        multimask_output: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Segment object from bounding box

        Args:
            box: Bounding box [x1, y1, x2, y2]
            multimask_output: Whether to return multiple masks

        Returns:
            Dictionary with masks, scores, logits
        """
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=multimask_output
        )

        return {
            'masks': masks,
            'scores': scores,
            'logits': logits
        }

    def segment_from_boxes(
        self,
        boxes: List[np.ndarray],
        multimask_output: bool = False
    ) -> List[Dict[str, np.ndarray]]:
        """
        Segment multiple objects from multiple bounding boxes

        Args:
            boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            multimask_output: Whether to return multiple masks per box

        Returns:
            List of dictionaries, each containing masks, scores, logits
        """
        results = []
        for box in boxes:
            result = self.segment_from_box(box, multimask_output)
            results.append(result)

        return results

    def segment_from_mask(
        self,
        mask_input: np.ndarray,
        multimask_output: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Refine segmentation from a rough mask

        Args:
            mask_input: Low-res mask logits [1, 256, 256]
            multimask_output: Whether to return multiple masks

        Returns:
            Dictionary with refined masks, scores, logits
        """
        masks, scores, logits = self.predictor.predict(
            mask_input=mask_input,
            multimask_output=multimask_output
        )

        return {
            'masks': masks,
            'scores': scores,
            'logits': logits
        }

    def segment_everything(
        self,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        crop_n_layers: int = 1,
        min_mask_region_area: int = 100
    ) -> List[Dict[str, np.ndarray]]:
        """
        Automatic mask generation (segment everything in the image)

        Args:
            points_per_side: Number of points per side for grid
            pred_iou_thresh: IoU threshold for filtering predictions
            stability_score_thresh: Stability score threshold
            crop_n_layers: Number of crop layers
            min_mask_region_area: Minimum mask area

        Returns:
            List of dictionaries with segmentation, bbox, area, etc.
        """
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            min_mask_region_area=min_mask_region_area
        )

        # Need to pass the image directly (not through set_image)
        masks = mask_generator.generate(self.predictor.original_image)

        return masks

    def get_best_mask(
        self,
        prediction_result: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Get the mask with highest confidence score

        Args:
            prediction_result: Result from segment_from_* methods

        Returns:
            Best mask [H, W]
        """
        masks = prediction_result['masks']
        scores = prediction_result['scores']

        best_idx = np.argmax(scores)
        return masks[best_idx]

    def combine_masks(
        self,
        masks: List[np.ndarray],
        method: str = 'union'
    ) -> np.ndarray:
        """
        Combine multiple masks

        Args:
            masks: List of binary masks
            method: 'union' or 'intersection'

        Returns:
            Combined mask
        """
        if not masks:
            return None

        combined = masks[0].astype(bool)

        for mask in masks[1:]:
            if method == 'union':
                combined = combined | mask.astype(bool)
            elif method == 'intersection':
                combined = combined & mask.astype(bool)

        return combined.astype(np.uint8)

    def postprocess_mask(
        self,
        mask: np.ndarray,
        remove_small_regions: bool = True,
        min_area: int = 100
    ) -> np.ndarray:
        """
        Postprocess mask to remove noise

        Args:
            mask: Binary mask [H, W]
            remove_small_regions: Remove small disconnected regions
            min_area: Minimum area for regions

        Returns:
            Cleaned mask
        """
        if not remove_small_regions:
            return mask

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )

        # Filter small regions
        cleaned_mask = np.zeros_like(mask)
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned_mask[labels == i] = 1

        return cleaned_mask

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_type': self.model_type,
            'device': self.device,
            'image_shape': getattr(self, 'current_image_shape', None)
        }


if __name__ == "__main__":
    # Example usage
    print("SAM2 Model Wrapper")
    print("=" * 50)

    # Initialize model
    model = SAM2Model(model_type="vit_b")

    # Print model info
    info = model.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")
