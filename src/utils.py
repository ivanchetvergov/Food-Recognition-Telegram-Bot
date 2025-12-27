"""Shared utility functions."""
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def get_image_data(img: np.ndarray) -> Tuple[float, float, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray) / 255.0)
    h, w = img.shape[:2]
    aspect_ratio = w / h if h > 0 else 1.0

    return blur_score, brightness, aspect_ratio

def get_image_quality(image_path: str | Path) -> Tuple[float, float, float]:
    """Calculate image quality metrics."""
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Could not load image: {image_path}")
        return 0.0, 0.0, 1.0

    return get_image_data(img)

def get_image_quality_from_array(image_np: np.ndarray) -> Tuple[float, float, float]:
    """Calculate image quality metrics from numpy array."""
    if image_np is None or image_np.size == 0:
        return 0.0, 0.0, 1.0
    
    return get_image_data(image_np)

def ensure_dir(path: Path | str) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_image_path(path: str | Path) -> Optional[Path]:
    """Validate that image path exists and has valid extension."""
    path = Path(path)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    if not path.exists():
        logger.error(f"Image not found: {path}")
        return None
    
    if path.suffix.lower() not in valid_extensions:
        logger.error(f"Invalid image extension: {path.suffix}")
        return None
    
    return path


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=level
    )
