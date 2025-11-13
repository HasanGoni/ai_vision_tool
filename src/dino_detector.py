"""DINOv2 Feature Extractor and Object Detector

This module uses DINOv2 to:
1. Extract visual features from images
2. Detect salient objects/regions
3. Generate prompts (bounding boxes, points) for SAM2
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
import cv2


class DINOv2Detector:
    """DINOv2-based object detector that generates prompts for SAM2"""

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: Optional[str] = None
    ):
        """
        Initialize DINOv2 detector

        Args:
            model_name: DINOv2 model variant (dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)
            device: Device to run model on (cuda/cpu). Auto-detects if None
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading DINOv2 model: {model_name} on {self.device}")

        # Load DINOv2 from torch hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Model configuration
        self.patch_size = 14
        self.model_name = model_name

    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        """
        Extract dense features from image using DINOv2

        Args:
            image: Input image as numpy array (H, W, 3) in RGB

        Returns:
            Feature tensor of shape (H//14, W//14, feature_dim)
        """
        # Prepare image
        img_tensor = self._preprocess_image(image)

        # Extract features
        with torch.no_grad():
            features = self.model.forward_features(img_tensor)
            # Get patch tokens (excluding CLS token)
            patch_features = features['x_norm_patchtokens']

        return patch_features

    def detect_salient_regions(
        self,
        image: np.ndarray,
        num_regions: int = 5,
        min_area: int = 100,
        attention_threshold: float = 0.6
    ) -> List[Dict[str, np.ndarray]]:
        """
        Detect salient regions in the image using DINO features

        Args:
            image: Input image (H, W, 3) in RGB
            num_regions: Maximum number of regions to detect
            min_area: Minimum area for detected regions (in pixels)
            attention_threshold: Threshold for attention map (0-1)

        Returns:
            List of dictionaries with keys:
                - 'bbox': Bounding box [x1, y1, x2, y2]
                - 'center': Center point [x, y]
                - 'confidence': Detection confidence score
                - 'attention_map': Attention map for this region
        """
        h, w = image.shape[:2]

        # Extract features
        features = self.extract_features(image)

        # Compute self-attention map
        attention_map = self._compute_attention_map(features, h, w)

        # Find salient regions using attention map
        regions = self._find_regions_from_attention(
            attention_map,
            num_regions=num_regions,
            min_area=min_area,
            threshold=attention_threshold
        )

        return regions

    def _compute_attention_map(
        self,
        features: torch.Tensor,
        target_h: int,
        target_w: int
    ) -> np.ndarray:
        """
        Compute attention map from DINO features

        Args:
            features: Feature tensor from DINOv2
            target_h: Target height for upsampling
            target_w: Target width for upsampling

        Returns:
            Attention map as numpy array (target_h, target_w)
        """
        # features shape: (1, num_patches, feature_dim)
        batch_size, num_patches, feature_dim = features.shape

        # Calculate grid size
        grid_size = int(np.sqrt(num_patches))

        # Reshape to spatial grid
        features_spatial = features.reshape(batch_size, grid_size, grid_size, feature_dim)
        features_spatial = features_spatial.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Compute self-similarity (attention)
        # Use L2 norm of features as saliency indicator
        attention = torch.norm(features_spatial, dim=1, keepdim=True)

        # Normalize attention
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

        # Upsample to original image size
        attention_upsampled = F.interpolate(
            attention,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )

        attention_map = attention_upsampled[0, 0].cpu().numpy()

        return attention_map

    def _find_regions_from_attention(
        self,
        attention_map: np.ndarray,
        num_regions: int = 5,
        min_area: int = 100,
        threshold: float = 0.6
    ) -> List[Dict[str, np.ndarray]]:
        """
        Find distinct regions from attention map using connected components

        Args:
            attention_map: Attention map (H, W)
            num_regions: Maximum number of regions to return
            min_area: Minimum area for valid regions
            threshold: Threshold for binarization

        Returns:
            List of region dictionaries
        """
        # Threshold attention map
        binary_map = (attention_map > threshold).astype(np.uint8)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_map, connectivity=8
        )

        regions = []

        # Process each component (skip background label 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if area < min_area:
                continue

            # Extract bounding box
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            bbox = np.array([x, y, x + w, y + h])
            center = centroids[i].astype(int)

            # Calculate confidence based on mean attention in region
            mask = (labels == i)
            confidence = attention_map[mask].mean()

            regions.append({
                'bbox': bbox,
                'center': center,
                'confidence': float(confidence),
                'attention_map': mask.astype(np.uint8),
                'area': area
            })

        # Sort by confidence and return top N
        regions = sorted(regions, key=lambda x: x['confidence'], reverse=True)

        return regions[:num_regions]

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for DINOv2

        Args:
            image: Input image (H, W, 3) in RGB format

        Returns:
            Preprocessed tensor
        """
        # Convert to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # DINOv2 uses ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # Resize to multiple of patch_size
        w, h = image.size
        new_w = (w // self.patch_size) * self.patch_size
        new_h = (h // self.patch_size) * self.patch_size

        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), Image.BILINEAR)

        # Convert to tensor and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = (img_array - mean) / std
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

        return img_tensor.to(self.device)

    def visualize_detections(
        self,
        image: np.ndarray,
        regions: List[Dict],
        attention_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Visualize detected regions on image

        Args:
            image: Original image (H, W, 3)
            regions: List of detected regions
            attention_map: Optional attention map to overlay

        Returns:
            Visualized image
        """
        vis_image = image.copy()

        # Overlay attention map if provided
        if attention_map is not None:
            heatmap = cv2.applyColorMap(
                (attention_map * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            vis_image = cv2.addWeighted(vis_image, 0.6, heatmap, 0.4, 0)

        # Draw bounding boxes and centers
        for idx, region in enumerate(regions):
            bbox = region['bbox'].astype(int)
            center = region['center'].astype(int)
            confidence = region['confidence']

            # Draw bounding box
            color = (0, 255, 0)
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Draw center point
            cv2.circle(vis_image, tuple(center), 5, (255, 0, 0), -1)

            # Add label
            label = f"Region {idx+1}: {confidence:.2f}"
            cv2.putText(
                vis_image, label, (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        return vis_image
