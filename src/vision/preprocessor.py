"""
Vision Preprocessing Module.
Crops facial regions, applies normalization, and formats tensors for inference.
"""

from typing import Optional, Tuple
import numpy as np
from PIL import Image
from src.config import IMAGE_INPUT_SIZE
from src.utils.logger import get_logger

logger = get_logger("vision.preprocessor")


def preprocess_face_crop(
    image: Image.Image,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    target_size: Tuple[int, int] = IMAGE_INPUT_SIZE,
    margin_ratio: float = 0.15
) -> np.ndarray:
    """
    Extract facial crop with margin, resize to target dimensions, and normalize.
    Returns:
        np.ndarray with shape (1, target_height, target_width, 3) normalized to [0, 1].
    """
    img = image.convert("RGB")
    width, height = img.size

    if bbox:
        x, y, w, h = bbox
        # Apply padding margin around face box
        mx = int(w * margin_ratio)
        my = int(h * margin_ratio)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(width, x + w + mx)
        y2 = min(height, y + h + my)

        cropped = img.crop((x1, y1, x2, y2))
    else:
        # No specific bounding box provided: use full image
        cropped = img

    resized = cropped.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0

    # Ensure shape is (1, H, W, 3)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)

    return arr
