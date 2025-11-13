"""
Preprocessing utilities for images and masks
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Union
from PIL import Image


def load_image(
    image_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    convert_rgb: bool = True
) -> np.ndarray:
    """
    Load image from file

    Args:
        image_path: Path to image
        target_size: Optional (width, height) to resize to
        convert_rgb: Convert to RGB

    Returns:
        Image as numpy array
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    if convert_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if target_size is not None:
        image = cv2.resize(image, target_size)

    return image


def load_mask(
    mask_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    threshold: int = 127
) -> np.ndarray:
    """
    Load binary mask from file

    Args:
        mask_path: Path to mask image
        target_size: Optional (width, height) to resize to
        threshold: Binarization threshold

    Returns:
        Binary mask
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Failed to load mask: {mask_path}")

    if target_size is not None:
        mask = cv2.resize(mask, target_size)

    # Binarize
    mask = (mask > threshold).astype(np.uint8)

    return mask


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect: bool = True,
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> Tuple[np.ndarray, dict]:
    """
    Resize image with optional aspect ratio preservation

    Args:
        image: Input image
        target_size: Target (width, height)
        keep_aspect: Preserve aspect ratio and pad
        pad_color: Padding color

    Returns:
        - Resized image
        - Resize info dict (scale, padding)
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    if not keep_aspect:
        resized = cv2.resize(image, target_size)
        return resized, {'scale': (target_w / w, target_h / h), 'padding': (0, 0, 0, 0)}

    # Calculate scale to fit
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h))

    # Pad to target size
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    if len(image.shape) == 3:
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=pad_color
        )
    else:
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=pad_color[0]
        )

    resize_info = {
        'scale': scale,
        'padding': (pad_left, pad_top, pad_right, pad_bottom)
    }

    return padded, resize_info


def crop_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop image and mask to bounding box of mask

    Args:
        image: Input image
        mask: Binary mask
        padding: Additional padding around bbox

    Returns:
        - Cropped image
        - Cropped mask
    """
    # Find bounding box
    coords = cv2.findNonZero(mask.astype(np.uint8))

    if coords is None:
        return image, mask

    x, y, w, h = cv2.boundingRect(coords)

    # Add padding
    h_img, w_img = image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)

    # Crop
    image_cropped = image[y1:y2, x1:x2].copy()
    mask_cropped = mask[y1:y2, x1:x2].copy()

    return image_cropped, mask_cropped


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Normalize image (ImageNet normalization by default)

    Args:
        image: Input image [0, 255]
        mean: Mean values
        std: Std values

    Returns:
        Normalized image
    """
    image = image.astype(np.float32) / 255.0

    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)

    normalized = (image - mean) / std

    return normalized


def denormalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Denormalize image back to [0, 255]

    Args:
        image: Normalized image
        mean: Mean values used for normalization
        std: Std values used for normalization

    Returns:
        Denormalized image [0, 255]
    """
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)

    denormalized = (image * std + mean) * 255.0
    denormalized = np.clip(denormalized, 0, 255).astype(np.uint8)

    return denormalized


def create_gaussian_mask(
    center: Tuple[int, int],
    image_shape: Tuple[int, int],
    sigma: float = 50.0
) -> np.ndarray:
    """
    Create Gaussian attention mask around a center point

    Args:
        center: (x, y) center point
        image_shape: (height, width)
        sigma: Gaussian sigma

    Returns:
        Gaussian mask [0, 1]
    """
    h, w = image_shape
    cx, cy = center

    # Create coordinate grids
    y, x = np.ogrid[:h, :w]

    # Gaussian formula
    mask = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

    return mask


def apply_mask_to_image(
    image: np.ndarray,
    mask: np.ndarray,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Apply binary mask to image

    Args:
        image: Input image
        mask: Binary mask
        background_color: Color for masked-out regions

    Returns:
        Masked image
    """
    result = image.copy()

    if len(image.shape) == 3:
        for c in range(image.shape[2]):
            result[:, :, c][mask == 0] = background_color[c]
    else:
        result[mask == 0] = background_color[0]

    return result


def extract_patches(
    image: np.ndarray,
    patch_size: int = 224,
    stride: int = 112
) -> Tuple[np.ndarray, list]:
    """
    Extract overlapping patches from image

    Args:
        image: Input image
        patch_size: Size of square patches
        stride: Stride between patches

    Returns:
        - Patches array [N, patch_size, patch_size, C]
        - Patch coordinates [(x, y), ...]
    """
    h, w = image.shape[:2]
    patches = []
    coordinates = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coordinates.append((x, y))

    return np.array(patches), coordinates


def augment_image(
    image: np.ndarray,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    rotate_angle: float = 0.0,
    brightness_factor: float = 1.0
) -> np.ndarray:
    """
    Simple image augmentation

    Args:
        image: Input image
        flip_horizontal: Horizontal flip
        flip_vertical: Vertical flip
        rotate_angle: Rotation angle in degrees
        brightness_factor: Brightness adjustment (1.0 = no change)

    Returns:
        Augmented image
    """
    result = image.copy()

    # Flip
    if flip_horizontal:
        result = cv2.flip(result, 1)

    if flip_vertical:
        result = cv2.flip(result, 0)

    # Rotate
    if rotate_angle != 0:
        h, w = result.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
        result = cv2.warpAffine(result, matrix, (w, h))

    # Brightness
    if brightness_factor != 1.0:
        result = np.clip(result * brightness_factor, 0, 255).astype(np.uint8)

    return result


if __name__ == "__main__":
    print("Preprocessing utilities for AI Vision Tool")
