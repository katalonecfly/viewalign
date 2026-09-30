from pathlib import Path

import cv2
import numpy as np

from viewalign.derived import RenderedFrame
from viewalign.ordering import order_images, order_rendered_frames


def test_filename_ordering() -> None:
    paths = [Path("b.jpg"), Path("A.jpg"), Path("c.png")]
    ordered, missing = order_images(paths, "filename")
    assert [p.name for p in ordered] == ["A.jpg", "b.jpg", "c.png"]
    assert missing == []


def _frame(
    tmp_path: Path,
    name: str,
    value: int,
    transform: np.ndarray | None = None,
) -> RenderedFrame:
    image = np.full((10, 10, 3), value, dtype=np.uint8)
    path = tmp_path / name
    assert cv2.imwrite(str(path), image)
    if transform is None:
        transform = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    return RenderedFrame(path, path, transform, 10, 10)


def test_video_brightness_order(tmp_path: Path) -> None:
    dark = _frame(tmp_path, "dark.jpg", 20)
    light = _frame(tmp_path, "light.jpg", 220)
    middle = _frame(tmp_path, "middle.jpg", 120)

    ordered = order_rendered_frames(
        [dark, light, middle],
        "brightness",
        reference_size=(10, 10),
        crop="none",
        crop_rect=None,
        fit_origin=(0.0, 0.0),
        output_size=(10, 10),
    )
    assert [frame.source_path.name for frame in ordered] == ["light.jpg", "middle.jpg", "dark.jpg"]


def test_video_brightness_ignores_geometric_black_borders(tmp_path: Path) -> None:
    bright_with_border = _frame(
        tmp_path,
        "bright.jpg",
        100,
        np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0]], dtype=np.float64),
    )
    slightly_darker = _frame(tmp_path, "darker.jpg", 90)

    ordered = order_rendered_frames(
        [slightly_darker, bright_with_border],
        "brightness",
        reference_size=(10, 10),
        crop="none",
        crop_rect=None,
        fit_origin=(0.0, 0.0),
        output_size=(15, 10),
    )
    assert [frame.source_path.name for frame in ordered] == ["bright.jpg", "darker.jpg"]


def test_video_center_order_left_to_right(tmp_path: Path) -> None:
    right = _frame(
        tmp_path,
        "right.jpg",
        100,
        np.array([[1.0, 0.0, 8.0], [0.0, 1.0, 0.0]], dtype=np.float64),
    )
    left = _frame(
        tmp_path,
        "left.jpg",
        100,
        np.array([[1.0, 0.0, -8.0], [0.0, 1.0, 0.0]], dtype=np.float64),
    )
    center = _frame(tmp_path, "center.jpg", 100)

    ordered = order_rendered_frames(
        [right, center, left],
        "center",
        reference_size=(10, 10),
        crop="none",
        crop_rect=None,
        fit_origin=(0.0, 0.0),
        output_size=(10, 10),
    )
    assert [frame.source_path.name for frame in ordered] == ["left.jpg", "center.jpg", "right.jpg"]


def test_video_random_order_keeps_all_frames(tmp_path: Path, monkeypatch) -> None:
    frames = [_frame(tmp_path, f"{index}.jpg", 100 + index) for index in range(4)]

    def reverse_shuffle(items: list[RenderedFrame]) -> None:
        items.reverse()

    monkeypatch.setattr("viewalign.ordering.random.shuffle", reverse_shuffle)
    ordered = order_rendered_frames(
        frames,
        "random",
        reference_size=(10, 10),
        crop="none",
        crop_rect=None,
        fit_origin=(0.0, 0.0),
        output_size=(10, 10),
    )
    assert [frame.source_path.name for frame in ordered] == ["3.jpg", "2.jpg", "1.jpg", "0.jpg"]
