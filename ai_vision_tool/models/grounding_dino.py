"""
Grounding DINO Model Wrapper
Open-vocabulary object detection with custom text prompts
Allows teaching custom concepts like "my custom Pin"
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from PIL import Image
import torchvision.transforms as T


class GroundingDINOModel:
    """
    Wrapper for Grounding DINO - Open-vocabulary object detection
    Detects objects based on text prompts, enabling custom grounding
    """

    def __init__(
        self,
        model_config: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ):
        """
        Initialize Grounding DINO model

        Args:
            model_config: Path to model config file
            checkpoint_path: Path to checkpoint (None = download)
            device: Device to run on (cuda/cpu)
            box_threshold: Confidence threshold for box detection
            text_threshold: Threshold for text-image similarity
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Import Grounding DINO
        try:
            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict
        except ImportError:
            raise ImportError(
                "Grounding DINO not installed. Install with: "
                "pip install groundingdino-py"
            )

        print(f"Loading Grounding DINO on {self.device}")

        # Load config
        args = SLConfig.fromfile(model_config)
        args.device = self.device

        # Build model
        self.model = build_model(args)

        # Load checkpoint
        if checkpoint_path is None:
            checkpoint_path = self._download_checkpoint()

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = T.Compose([
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print(f"Grounding DINO initialized successfully")

    def _download_checkpoint(self) -> str:
        """Download Grounding DINO checkpoint"""
        checkpoint_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        checkpoint_path = torch.hub.load_state_dict_from_url(checkpoint_url)
        return checkpoint_path

    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for Grounding DINO

        Args:
            image: Input image

        Returns:
            - Preprocessed tensor
            - Original image size (H, W)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        original_size = image.size  # (W, H)

        # Transform
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        return image_tensor, (image.height, image.width)

    def preprocess_caption(self, caption: str) -> str:
        """
        Preprocess text caption for Grounding DINO

        Args:
            caption: Text prompt (e.g., "red car . blue truck . custom Pin")

        Returns:
            Formatted caption
        """
        # Grounding DINO expects phrases separated by " . "
        # Lowercase and clean
        caption = caption.lower().strip()

        # Ensure proper separator format
        if not caption.endswith('.'):
            caption = caption + '.'

        return caption

    @torch.no_grad()
    def detect(
        self,
        image: Union[np.ndarray, Image.Image],
        text_prompt: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None
    ) -> Dict[str, Union[np.ndarray, List[str]]]:
        """
        Detect objects based on text prompt

        Args:
            image: Input image
            text_prompt: Text description (e.g., "red car . my custom Pin . person")
            box_threshold: Override default box threshold
            text_threshold: Override default text threshold

        Returns:
            Dictionary containing:
                - 'boxes': Bounding boxes [N, 4] in xyxy format (normalized 0-1)
                - 'scores': Confidence scores [N]
                - 'labels': Predicted labels [N]
                - 'phrases': Detected phrases [N]
        """
        box_threshold = box_threshold or self.box_threshold
        text_threshold = text_threshold or self.text_threshold

        # Preprocess
        image_tensor, (orig_h, orig_w) = self.preprocess_image(image)
        caption = self.preprocess_caption(text_prompt)

        # Predict
        from groundingdino.util.inference import predict

        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )

        # Convert boxes to absolute coordinates
        boxes_abs = self._normalize_to_absolute(boxes, orig_h, orig_w)

        return {
            'boxes': boxes_abs,  # [N, 4] in absolute xyxy
            'boxes_normalized': boxes,  # [N, 4] in normalized xyxy
            'scores': logits,  # [N]
            'phrases': phrases,  # List[str]
            'labels': [self._phrase_to_label(p, caption) for p in phrases]  # List[int]
        }

    def _normalize_to_absolute(
        self,
        boxes: torch.Tensor,
        height: int,
        width: int
    ) -> np.ndarray:
        """
        Convert normalized boxes [0, 1] to absolute pixel coordinates

        Args:
            boxes: Normalized boxes [N, 4] in (cx, cy, w, h) format
            height: Image height
            width: Image width

        Returns:
            Absolute boxes [N, 4] in (x1, y1, x2, y2) format
        """
        if len(boxes) == 0:
            return np.array([])

        boxes = boxes.cpu().numpy()

        # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

        # Scale to absolute coordinates
        boxes_xyxy[:, [0, 2]] *= width
        boxes_xyxy[:, [1, 3]] *= height

        return boxes_xyxy

    def _phrase_to_label(self, phrase: str, caption: str) -> int:
        """Map detected phrase to label index"""
        phrases = caption.split('.')
        phrases = [p.strip() for p in phrases if p.strip()]

        try:
            return phrases.index(phrase.strip())
        except ValueError:
            return -1

    def filter_by_phrase(
        self,
        detection_result: Dict,
        target_phrase: str
    ) -> Dict:
        """
        Filter detections to only include specific phrase

        Args:
            detection_result: Result from detect()
            target_phrase: Phrase to keep (e.g., "my custom Pin")

        Returns:
            Filtered detection result
        """
        phrases = detection_result['phrases']
        target_phrase = target_phrase.lower().strip()

        # Find matching indices
        matching_indices = [
            i for i, phrase in enumerate(phrases)
            if target_phrase in phrase.lower()
        ]

        if not matching_indices:
            return {
                'boxes': np.array([]),
                'boxes_normalized': torch.tensor([]),
                'scores': torch.tensor([]),
                'phrases': [],
                'labels': []
            }

        # Filter results
        filtered = {
            'boxes': detection_result['boxes'][matching_indices],
            'boxes_normalized': detection_result['boxes_normalized'][matching_indices],
            'scores': detection_result['scores'][matching_indices],
            'phrases': [detection_result['phrases'][i] for i in matching_indices],
            'labels': [detection_result['labels'][i] for i in matching_indices]
        }

        return filtered

    def detect_custom_object(
        self,
        image: Union[np.ndarray, Image.Image],
        custom_description: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25
    ) -> Dict:
        """
        Detect custom object with specific description
        Perfect for teaching new concepts like "my custom Pin"

        Args:
            image: Input image
            custom_description: Custom object description (e.g., "my special Pin that looks like X")
            box_threshold: Detection threshold
            text_threshold: Text similarity threshold

        Returns:
            Detection results filtered to custom object
        """
        # Detect with custom prompt
        results = self.detect(
            image=image,
            text_prompt=custom_description,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        return results

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'device': self.device,
            'box_threshold': self.box_threshold,
            'text_threshold': self.text_threshold
        }


if __name__ == "__main__":
    # Example usage
    print("Grounding DINO Model Wrapper")
    print("=" * 50)

    # Initialize model
    model = GroundingDINOModel()

    # Print model info
    info = model.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    # Example: Detect custom Pin
    print("\nExample: Detecting custom 'Pin' object")
    print("Text prompt: 'my custom Pin that has a red top and blue bottom'")
