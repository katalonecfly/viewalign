from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def discover_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    paths = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(paths, key=lambda p: p.name.lower())


def load_oriented_bgr(path: Path) -> np.ndarray:
    """Load an image and normalize EXIF orientation."""
    try:
        with Image.open(path) as pil_image:
            corrected = ImageOps.exif_transpose(pil_image)
            rgb = corrected.convert("RGB")
            arr = np.asarray(rgb)
    except Exception as exc:  # Pillow exceptions vary by file/format.
        raise ValueError(f"Could not read image {path}: {exc}") from exc

    # Pillow gives RGB; OpenCV operations below use BGR.
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def resize_for_working(image: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float]:
    """Resize while preserving aspect ratio and return (image, scale)."""
    if max_dimension <= 0:
        raise ValueError("working size must be greater than zero")

    height, width = image.shape[:2]
    current_max = max(height, width)
    if current_max <= max_dimension:
        return image.copy(), 1.0

    scale = max_dimension / current_max
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # Exact scale can differ very slightly from the requested value due to rounding.
    return resized, new_width / width


def image_timestamp(path: Path) -> tuple[int, str] | None:
    """Return EXIF DateTimeOriginal/DateTime when available, normalized for sorting."""
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            # 36867 = DateTimeOriginal, 306 = DateTime.
            value = exif.get(36867) or exif.get(306)
    except Exception:
        return None

    if not value:
        return None
    return (0, str(value))
