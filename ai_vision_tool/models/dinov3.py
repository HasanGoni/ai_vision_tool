"""
DINOv3 Model Wrapper for Reference Image Feature Extraction
Extracts dense visual features for image matching and similarity detection
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class DINOv3Model:
    """
    Wrapper for DINOv3 (Vision Transformer with self-distillation)
    Used for extracting dense visual features from images for matching
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: Optional[str] = None,
        use_registers: bool = True
    ):
        """
        Initialize DINOv3 model

        Args:
            model_name: Model variant (dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)
            device: Device to run model on (cuda/cpu)
            use_registers: Whether to use register tokens for better feature quality
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.use_registers = use_registers

        # Load model
        print(f"Loading DINOv3 model: {model_name} on {self.device}")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get patch size and embedding dimension
        self.patch_size = self.model.patch_size
        self.embed_dim = self.model.embed_dim

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),  # DINOv2 optimal size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"DINOv3 initialized: patch_size={self.patch_size}, embed_dim={self.embed_dim}")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for DINOv3

        Args:
            image: PIL Image

        Returns:
            Preprocessed tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        img_tensor = self.transform(image).unsqueeze(0)
        return img_tensor.to(self.device)

    @torch.no_grad()
    def extract_features(
        self,
        image: Image.Image,
        return_spatial: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Extract dense features from image

        Args:
            image: Input image (PIL Image or numpy array)
            return_spatial: Whether to return spatial feature maps

        Returns:
            Dictionary containing:
                - 'cls_token': Global image representation [1, embed_dim]
                - 'patch_tokens': Spatial patch features [1, num_patches, embed_dim]
                - 'features': Reshaped spatial features [1, embed_dim, H, W]
        """
        # Preprocess
        img_tensor = self.preprocess_image(image)

        # Get original image size for reshaping
        h, w = img_tensor.shape[2:]
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size

        # Extract features
        output = self.model.forward_features(img_tensor)

        # Split into CLS token and patch tokens
        cls_token = output['x_norm_clstoken']  # [1, embed_dim]
        patch_tokens = output['x_norm_patchtokens']  # [1, num_patches, embed_dim]

        result = {
            'cls_token': cls_token,
            'patch_tokens': patch_tokens,
        }

        if return_spatial:
            # Reshape patch tokens to spatial grid
            spatial_features = patch_tokens.reshape(
                1, num_patches_h, num_patches_w, self.embed_dim
            ).permute(0, 3, 1, 2)  # [1, embed_dim, H, W]
            result['features'] = spatial_features

        return result

    def compute_similarity(
        self,
        ref_features: Dict[str, torch.Tensor],
        target_features: Dict[str, torch.Tensor],
        similarity_type: str = 'cosine',
        use_spatial: bool = True
    ) -> torch.Tensor:
        """
        Compute similarity between reference and target features

        Args:
            ref_features: Reference image features from extract_features()
            target_features: Target image features from extract_features()
            similarity_type: 'cosine' or 'l2'
            use_spatial: Use spatial features (True) or global CLS token (False)

        Returns:
            Similarity map [1, H, W] if use_spatial, else scalar similarity
        """
        if use_spatial:
            ref_feat = ref_features['features']  # [1, C, H, W]
            target_feat = target_features['features']  # [1, C, H', W']

            # Normalize features
            ref_feat_norm = F.normalize(ref_feat, p=2, dim=1)
            target_feat_norm = F.normalize(target_feat, p=2, dim=1)

            # Global average pooling on reference to get prototype
            ref_prototype = ref_feat_norm.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

            # Compute cosine similarity at each spatial location
            similarity_map = (target_feat_norm * ref_prototype).sum(dim=1, keepdim=True)  # [1, 1, H', W']

            return similarity_map
        else:
            # Use global CLS tokens
            ref_cls = F.normalize(ref_features['cls_token'], p=2, dim=1)
            target_cls = F.normalize(target_features['cls_token'], p=2, dim=1)

            if similarity_type == 'cosine':
                similarity = (ref_cls * target_cls).sum(dim=1)
            else:  # L2 distance
                similarity = -torch.norm(ref_cls - target_cls, p=2, dim=1)

            return similarity

    def find_similar_regions(
        self,
        ref_image: Image.Image,
        target_image: Image.Image,
        threshold: float = 0.5,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Find regions in target image similar to reference image

        Args:
            ref_image: Reference image (e.g., red car with mask)
            target_image: Target image to search
            threshold: Similarity threshold (0-1)
            top_k: Return top-k most similar regions

        Returns:
            - similarity_map: Spatial similarity scores [H, W]
            - coordinates: Top-k region coordinates [(x, y, score), ...]
        """
        # Extract features
        ref_features = self.extract_features(ref_image, return_spatial=True)
        target_features = self.extract_features(target_image, return_spatial=True)

        # Compute similarity map
        similarity_map = self.compute_similarity(
            ref_features,
            target_features,
            use_spatial=True
        )  # [1, 1, H, W]

        similarity_map = similarity_map.squeeze().cpu()  # [H, W]

        # Find top-k peaks above threshold
        flat_sim = similarity_map.flatten()
        above_threshold = (flat_sim >= threshold).nonzero(as_tuple=True)[0]

        if len(above_threshold) == 0:
            return similarity_map, np.array([])

        # Get top-k
        top_k = min(top_k, len(above_threshold))
        top_values, top_indices = torch.topk(flat_sim[above_threshold], top_k)
        top_indices = above_threshold[top_indices]

        # Convert to 2D coordinates
        h, w = similarity_map.shape
        y_coords = (top_indices // w).cpu().numpy()
        x_coords = (top_indices % w).cpu().numpy()
        scores = top_values.cpu().numpy()

        coordinates = np.stack([x_coords, y_coords, scores], axis=1)

        return similarity_map, coordinates

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'device': self.device,
            'use_registers': self.use_registers
        }


if __name__ == "__main__":
    # Example usage
    print("DINOv3 Model Wrapper")
    print("=" * 50)

    # Initialize model
    model = DINOv3Model(model_name="dinov2_vitb14")

    # Print model info
    info = model.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")
