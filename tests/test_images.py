from pathlib import Path

import numpy as np
from PIL import Image

from viewalign.images import image_timestamp, load_oriented_bgr


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    # Stored as 10x20, EXIF orientation 6 means rotate 90° clockwise when displayed.
    image = Image.new("RGB", (10, 20), "white")
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "portrait.jpg"
    image.save(path, exif=exif)

    loaded = load_oriented_bgr(path)
    assert loaded.shape[:2] == (10, 20)


def test_timestamp_reads_datetime_original(tmp_path: Path) -> None:
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), "RGB")
    exif = image.getexif()
    exif[36867] = "2025:04:05 12:34:56"
    path = tmp_path / "photo.jpg"
    image.save(path, exif=exif)

    assert image_timestamp(path) == (0, "2025:04:05 12:34:56")
