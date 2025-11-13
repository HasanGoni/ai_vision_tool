"""
Visualization utilities for displaying results
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Union
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_boxes(
    image: np.ndarray,
    boxes: List[np.ndarray],
    labels: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on image

    Args:
        image: Input image
        boxes: List of boxes [x1, y1, x2, y2]
        labels: Optional text labels
        scores: Optional confidence scores
        color: Box color (B, G, R)
        thickness: Line thickness

    Returns:
        Image with boxes drawn
    """
    vis = image.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # Add label
        if labels is not None or scores is not None:
            label_parts = []
            if labels is not None and i < len(labels):
                label_parts.append(labels[i])
            if scores is not None and i < len(scores):
                label_parts.append(f"{scores[i]:.2f}")

            label = " - ".join(label_parts)

            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(vis, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

            # Draw label text
            cv2.putText(
                vis, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

    return vis


def draw_masks(
    image: np.ndarray,
    masks: List[np.ndarray],
    colors: Optional[List[Tuple[int, int, int]]] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay segmentation masks on image

    Args:
        image: Input image
        masks: List of binary masks
        colors: List of colors for each mask (B, G, R)
        alpha: Transparency (0-1)

    Returns:
        Image with masks overlaid
    """
    vis = image.copy()

    if colors is None:
        colors = [get_distinct_color(i) for i in range(len(masks))]

    for mask, color in zip(masks, colors):
        colored_mask = np.zeros_like(vis)
        colored_mask[mask > 0] = color
        vis = cv2.addWeighted(vis, 1.0, colored_mask, alpha, 0)

    return vis


def draw_points(
    image: np.ndarray,
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    radius: int = 5
) -> np.ndarray:
    """
    Draw points on image

    Args:
        image: Input image
        points: Point coordinates [N, 2] (x, y)
        labels: Point labels [N] (1=foreground, 0=background)
        radius: Point radius

    Returns:
        Image with points drawn
    """
    vis = image.copy()

    for i, (x, y) in enumerate(points):
        # Color based on label
        if labels is not None:
            color = (0, 255, 0) if labels[i] == 1 else (255, 0, 0)
        else:
            color = (0, 255, 0)

        cv2.circle(vis, (int(x), int(y)), radius, color, -1)
        cv2.circle(vis, (int(x), int(y)), radius + 2, (255, 255, 255), 2)

    return vis


def draw_heatmap(
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    normalize: bool = True
) -> np.ndarray:
    """
    Convert heatmap to color visualization

    Args:
        heatmap: 2D heatmap
        colormap: OpenCV colormap
        normalize: Normalize to 0-255 range

    Returns:
        Colored heatmap
    """
    if normalize:
        heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)

    colored = cv2.applyColorMap(heatmap, colormap)
    return colored


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on image

    Args:
        image: Base image
        heatmap: 2D heatmap (will be resized to image size)
        alpha: Heatmap transparency
        colormap: OpenCV colormap

    Returns:
        Image with heatmap overlay
    """
    # Resize heatmap to image size
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to colored heatmap
    colored_heatmap = draw_heatmap(heatmap_resized, colormap)

    # Overlay
    vis = cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)
    return vis


def get_distinct_color(index: int) -> Tuple[int, int, int]:
    """
    Get distinct color for given index

    Args:
        index: Color index

    Returns:
        BGR color tuple
    """
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (128, 0, 0),      # Maroon
        (0, 128, 0),      # Dark Green
        (0, 0, 128),      # Navy
        (128, 128, 0),    # Olive
        (128, 0, 128),    # Purple
        (0, 128, 128),    # Teal
        (255, 128, 0),    # Orange
        (255, 0, 128),    # Pink
        (0, 255, 128),    # Spring Green
        (128, 255, 0),    # Chartreuse
    ]
    return colors[index % len(colors)]


def create_comparison_view(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> np.ndarray:
    """
    Create side-by-side comparison view

    Args:
        images: List of images to compare
        titles: Optional titles for each image
        figsize: Figure size

    Returns:
        Comparison visualization
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for i, (ax, img) in enumerate(zip(axes, images)):
        # Convert BGR to RGB for matplotlib
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img

        ax.imshow(img_rgb)
        ax.axis('off')

        if titles is not None and i < len(titles):
            ax.set_title(titles[i])

    plt.tight_layout()

    # Convert to numpy array
    fig.canvas.draw()
    vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return vis


def draw_similarity_map(
    image: np.ndarray,
    similarity_map: np.ndarray,
    threshold: Optional[float] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Visualize similarity map on image

    Args:
        image: Base image
        similarity_map: 2D similarity scores
        threshold: Optional threshold to highlight
        alpha: Overlay transparency

    Returns:
        Visualization
    """
    vis = overlay_heatmap(image, similarity_map, alpha, cv2.COLORMAP_JET)

    # Draw threshold contour if provided
    if threshold is not None:
        h, w = image.shape[:2]
        similarity_resized = cv2.resize(similarity_map, (w, h))
        threshold_mask = (similarity_resized > threshold).astype(np.uint8)

        contours, _ = cv2.findContours(
            threshold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    return vis


def save_results(
    output_path: str,
    image: np.ndarray,
    results: dict,
    save_individual_masks: bool = False
):
    """
    Save visualization results

    Args:
        output_path: Base output path
        image: Original image
        results: Detection/segmentation results
        save_individual_masks: Save each mask separately
    """
    import os
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save combined visualization
    vis = draw_boxes(image, results['boxes'], scores=results['scores'])
    vis = draw_masks(vis, results['masks'])

    cv2.imwrite(str(output_path), vis)
    print(f"Saved visualization to: {output_path}")

    # Save individual masks
    if save_individual_masks:
        for i, mask in enumerate(results['masks']):
            mask_path = output_path.parent / f"{output_path.stem}_mask_{i}.png"
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)

        print(f"Saved {len(results['masks'])} individual masks")


if __name__ == "__main__":
    print("Visualization utilities for AI Vision Tool")
