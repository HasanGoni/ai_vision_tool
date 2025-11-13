"""
Custom Reference-Based Grounding Pipeline

Combines three powerful models:
1. DINOv3: Find regions similar to reference image (visual similarity)
2. Grounding DINO: Detect objects using custom text descriptions (semantic grounding)
3. SAM2: Segment the matched regions precisely

Use Case Example:
    Reference: [Image of your custom Pin + mask]
    Target: [Street scene with many objects]
    Text: "my custom Pin with red top"

    Pipeline:
    1. DINOv3 finds visually similar regions to reference Pin
    2. Grounding DINO detects objects matching "my custom Pin with red top"
    3. Combine both signals (visual + semantic)
    4. SAM2 segments only the matched custom Pins
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from PIL import Image
import cv2

from ..models.dinov3 import DINOv3Model
from ..models.sam2 import SAM2Model
from ..models.grounding_dino import GroundingDINOModel


class CustomGroundingPipeline:
    """
    End-to-end pipeline for custom object grounding and segmentation

    Workflow:
        1. Visual Matching (DINOv3): Find regions similar to reference
        2. Text Grounding (Grounding DINO): Detect with custom description
        3. Fusion: Combine visual + semantic signals
        4. Segmentation (SAM2): Precise instance masks
    """

    def __init__(
        self,
        dinov3_model: str = "dinov2_vitb14",
        sam2_model: str = "vit_b",
        grounding_dino_config: Optional[str] = None,
        device: Optional[str] = None,
        use_visual_matching: bool = True,
        use_text_grounding: bool = True
    ):
        """
        Initialize custom grounding pipeline

        Args:
            dinov3_model: DINOv3 model variant
            sam2_model: SAM2 model variant
            grounding_dino_config: Grounding DINO config path
            device: Device to run on
            use_visual_matching: Enable DINOv3 visual matching
            use_text_grounding: Enable Grounding DINO text grounding
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_visual_matching = use_visual_matching
        self.use_text_grounding = use_text_grounding

        print("=" * 60)
        print("Initializing Custom Grounding Pipeline")
        print("=" * 60)

        # Initialize models
        if use_visual_matching:
            print("\n[1/3] Loading DINOv3 for visual matching...")
            self.dinov3 = DINOv3Model(model_name=dinov3_model, device=self.device)
        else:
            self.dinov3 = None

        if use_text_grounding:
            print("\n[2/3] Loading Grounding DINO for text grounding...")
            self.grounding_dino = GroundingDINOModel(
                model_config=grounding_dino_config,
                device=self.device
            )
        else:
            self.grounding_dino = None

        print("\n[3/3] Loading SAM2 for segmentation...")
        self.sam2 = SAM2Model(model_type=sam2_model, device=self.device)

        print("\n" + "=" * 60)
        print("Pipeline Ready!")
        print("=" * 60)

    def process_reference_image(
        self,
        reference_image: Union[np.ndarray, Image.Image],
        reference_mask: Optional[np.ndarray] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from reference image
        Optionally use mask to focus on specific region

        Args:
            reference_image: Reference image (e.g., your custom Pin)
            reference_mask: Optional mask to isolate object of interest

        Returns:
            Reference features for matching
        """
        if not self.use_visual_matching:
            return None

        # Apply mask if provided
        if reference_mask is not None:
            if isinstance(reference_image, Image.Image):
                reference_image = np.array(reference_image)

            # Crop to masked region
            reference_image = self._apply_mask(reference_image, reference_mask)
            reference_image = Image.fromarray(reference_image)

        # Extract features
        ref_features = self.dinov3.extract_features(reference_image, return_spatial=True)

        return ref_features

    def detect_and_segment(
        self,
        target_image: Union[np.ndarray, Image.Image],
        reference_features: Optional[Dict[str, torch.Tensor]] = None,
        text_prompt: Optional[str] = None,
        fusion_strategy: str = 'multiply',
        similarity_threshold: float = 0.5,
        detection_threshold: float = 0.3,
        min_region_area: int = 100
    ) -> Dict[str, Union[np.ndarray, List]]:
        """
        Main pipeline: Detect and segment custom objects

        Args:
            target_image: Image to search
            reference_features: Features from reference image (from process_reference_image)
            text_prompt: Custom text description (e.g., "my custom Pin")
            fusion_strategy: How to combine signals ('multiply', 'max', 'weighted')
            similarity_threshold: Visual similarity threshold (0-1)
            detection_threshold: Text detection threshold (0-1)
            min_region_area: Minimum mask area in pixels

        Returns:
            Dictionary containing:
                - 'masks': List of segmentation masks
                - 'boxes': List of bounding boxes
                - 'scores': List of confidence scores
                - 'sources': List of detection sources ('visual', 'text', 'both')
                - 'visualization': Optional visualization
        """
        if isinstance(target_image, Image.Image):
            target_image_np = np.array(target_image)
        else:
            target_image_np = target_image.copy()

        # Step 1: Visual Matching (DINOv3)
        visual_candidates = []
        if self.use_visual_matching and reference_features is not None:
            print("\n[Step 1/4] Visual matching with DINOv3...")
            visual_candidates = self._visual_matching(
                target_image,
                reference_features,
                similarity_threshold
            )
            print(f"  Found {len(visual_candidates)} visual matches")

        # Step 2: Text Grounding (Grounding DINO)
        text_candidates = []
        if self.use_text_grounding and text_prompt is not None:
            print("\n[Step 2/4] Text-based detection with Grounding DINO...")
            text_candidates = self._text_grounding(
                target_image,
                text_prompt,
                detection_threshold
            )
            print(f"  Found {len(text_candidates)} text matches")

        # Step 3: Fusion
        print("\n[Step 3/4] Fusing visual and semantic signals...")
        fused_candidates = self._fuse_candidates(
            visual_candidates,
            text_candidates,
            fusion_strategy
        )
        print(f"  Total candidates after fusion: {len(fused_candidates)}")

        # Step 4: Segmentation (SAM2)
        print("\n[Step 4/4] Segmenting with SAM2...")
        results = self._segment_candidates(
            target_image_np,
            fused_candidates,
            min_region_area
        )
        print(f"  Generated {len(results['masks'])} final masks")

        return results

    def _visual_matching(
        self,
        target_image: Union[np.ndarray, Image.Image],
        reference_features: Dict[str, torch.Tensor],
        threshold: float
    ) -> List[Dict]:
        """Find visually similar regions using DINOv3"""
        similarity_map, coordinates = self.dinov3.find_similar_regions(
            ref_image=None,  # Features already extracted
            target_image=target_image,
            threshold=threshold,
            top_k=10
        )

        # Need to recompute with features
        target_features = self.dinov3.extract_features(target_image, return_spatial=True)
        similarity_map = self.dinov3.compute_similarity(
            reference_features,
            target_features,
            use_spatial=True
        )

        # Convert similarity map to candidates
        similarity_map_np = similarity_map.squeeze().cpu().numpy()

        # Upsample similarity map to image size
        if isinstance(target_image, Image.Image):
            h, w = target_image.size[1], target_image.size[0]
        else:
            h, w = target_image.shape[:2]

        similarity_map_resized = cv2.resize(similarity_map_np, (w, h))

        # Find peaks
        threshold_mask = similarity_map_resized > threshold
        if not threshold_mask.any():
            return []

        # Connected components to find regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            threshold_mask.astype(np.uint8), connectivity=8
        )

        candidates = []
        for i in range(1, num_labels):  # Skip background
            x, y, w_box, h_box, area = stats[i]
            cx, cy = centroids[i]

            # Get average similarity in region
            region_mask = (labels == i)
            avg_similarity = similarity_map_resized[region_mask].mean()

            candidates.append({
                'box': np.array([x, y, x + w_box, y + h_box]),
                'score': float(avg_similarity),
                'source': 'visual',
                'center': np.array([cx, cy])
            })

        return candidates

    def _text_grounding(
        self,
        target_image: Union[np.ndarray, Image.Image],
        text_prompt: str,
        threshold: float
    ) -> List[Dict]:
        """Detect objects using text description with Grounding DINO"""
        detection_result = self.grounding_dino.detect(
            image=target_image,
            text_prompt=text_prompt,
            box_threshold=threshold
        )

        candidates = []
        boxes = detection_result['boxes']
        scores = detection_result['scores']
        phrases = detection_result['phrases']

        for i in range(len(boxes)):
            box = boxes[i]
            candidates.append({
                'box': box,
                'score': float(scores[i]),
                'source': 'text',
                'phrase': phrases[i],
                'center': np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])
            })

        return candidates

    def _fuse_candidates(
        self,
        visual_candidates: List[Dict],
        text_candidates: List[Dict],
        strategy: str = 'multiply'
    ) -> List[Dict]:
        """
        Fuse visual and text candidates

        Strategy:
            - 'multiply': score_visual * score_text (both must be confident)
            - 'max': max(score_visual, score_text) (either can be confident)
            - 'weighted': 0.6 * score_visual + 0.4 * score_text
        """
        if not visual_candidates and not text_candidates:
            return []

        if not visual_candidates:
            return text_candidates

        if not text_candidates:
            return visual_candidates

        # Match candidates based on spatial proximity
        fused = []
        matched_text = set()

        for v_cand in visual_candidates:
            v_center = v_cand['center']
            v_score = v_cand['score']

            # Find closest text candidate
            best_match = None
            best_distance = float('inf')
            best_idx = -1

            for idx, t_cand in enumerate(text_candidates):
                if idx in matched_text:
                    continue

                t_center = t_cand['center']
                distance = np.linalg.norm(v_center - t_center)

                if distance < best_distance:
                    best_distance = distance
                    best_match = t_cand
                    best_idx = idx

            # If close enough, fuse
            if best_match is not None and best_distance < 100:  # pixels
                t_score = best_match['score']

                if strategy == 'multiply':
                    fused_score = v_score * t_score
                elif strategy == 'max':
                    fused_score = max(v_score, t_score)
                elif strategy == 'weighted':
                    fused_score = 0.6 * v_score + 0.4 * t_score
                else:
                    fused_score = (v_score + t_score) / 2

                # Average the boxes
                fused_box = (v_cand['box'] + best_match['box']) / 2

                fused.append({
                    'box': fused_box,
                    'score': fused_score,
                    'source': 'both',
                    'phrase': best_match.get('phrase', '')
                })

                matched_text.add(best_idx)
            else:
                # Keep visual-only candidate
                fused.append(v_cand)

        # Add unmatched text candidates
        for idx, t_cand in enumerate(text_candidates):
            if idx not in matched_text:
                fused.append(t_cand)

        return fused

    def _segment_candidates(
        self,
        image: np.ndarray,
        candidates: List[Dict],
        min_area: int
    ) -> Dict:
        """Segment each candidate using SAM2"""
        if not candidates:
            return {
                'masks': [],
                'boxes': [],
                'scores': [],
                'sources': [],
                'phrases': []
            }

        # Set image for SAM2
        self.sam2.set_image(image)

        results = {
            'masks': [],
            'boxes': [],
            'scores': [],
            'sources': [],
            'phrases': []
        }

        for cand in candidates:
            box = cand['box'].astype(np.float32)

            # Segment with SAM2
            seg_result = self.sam2.segment_from_box(box, multimask_output=False)

            if len(seg_result['masks']) == 0:
                continue

            mask = seg_result['masks'][0]

            # Filter small masks
            if mask.sum() < min_area:
                continue

            # Postprocess
            mask = self.sam2.postprocess_mask(mask, remove_small_regions=True, min_area=min_area)

            results['masks'].append(mask)
            results['boxes'].append(box)
            results['scores'].append(cand['score'])
            results['sources'].append(cand['source'])
            results['phrases'].append(cand.get('phrase', ''))

        return results

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to image and crop to bounding box"""
        # Find bounding box
        coords = cv2.findNonZero(mask.astype(np.uint8))
        if coords is None:
            return image

        x, y, w, h = cv2.boundingRect(coords)

        # Crop and mask
        cropped = image[y:y+h, x:x+w].copy()
        mask_cropped = mask[y:y+h, x:x+w]

        # Apply mask (set background to white)
        if len(cropped.shape) == 3:
            cropped[mask_cropped == 0] = [255, 255, 255]
        else:
            cropped[mask_cropped == 0] = 255

        return cropped

    def visualize_results(
        self,
        image: Union[np.ndarray, Image.Image],
        results: Dict,
        show_boxes: bool = True,
        show_masks: bool = True,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Visualize detection and segmentation results

        Args:
            image: Original image
            results: Results from detect_and_segment()
            show_boxes: Draw bounding boxes
            show_masks: Overlay masks
            show_labels: Show text labels

        Returns:
            Visualization image
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        vis = image.copy()

        # Draw masks
        if show_masks:
            for i, mask in enumerate(results['masks']):
                color = self._get_color(i)
                colored_mask = np.zeros_like(vis)
                colored_mask[mask > 0] = color
                vis = cv2.addWeighted(vis, 1.0, colored_mask, 0.4, 0)

        # Draw boxes
        if show_boxes:
            for i, box in enumerate(results['boxes']):
                color = self._get_color(i)
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                # Add label
                if show_labels:
                    source = results['sources'][i]
                    score = results['scores'][i]
                    phrase = results['phrases'][i] if results['phrases'][i] else source

                    label = f"{phrase} ({score:.2f})"
                    cv2.putText(vis, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis

    def _get_color(self, index: int) -> Tuple[int, int, int]:
        """Get distinct color for visualization"""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        return colors[index % len(colors)]


if __name__ == "__main__":
    print("Custom Reference-Based Grounding Pipeline")
    print("=" * 60)
    print("\nThis pipeline combines:")
    print("  1. DINOv3: Visual similarity matching")
    print("  2. Grounding DINO: Custom text-based detection")
    print("  3. SAM2: Precise segmentation")
