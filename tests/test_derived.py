import cv2
import numpy as np

from viewalign.derived import RenderedFrame, create_average, create_mosaic


def _frame(tmp_path, name: str, image: np.ndarray, transform=None) -> RenderedFrame:
    path = tmp_path / name
    assert cv2.imwrite(str(path), image)
    if transform is None:
        transform = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    return RenderedFrame(path, path, transform, image.shape[1], image.shape[0])


def test_average_ignores_geometric_black_borders(tmp_path) -> None:
    image_a = np.full((4, 6, 3), 100, dtype=np.uint8)
    image_b = np.full((4, 6, 3), 200, dtype=np.uint8)
    shift = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    frame_a = _frame(tmp_path, "a.jpg", image_a)
    frame_b = _frame(tmp_path, "b.jpg", image_b, shift)

    average, valid = create_average(
        [frame_a, frame_b],
        reference_size=(6, 4),
        crop="none",
        crop_rect=None,
        fit_origin=(0.0, 0.0),
        output_size=(6, 4),
    )

    assert np.all(average[:, :2] == 100)
    assert np.all(average[:, 2:] == 150)
    assert np.all(valid == 255)


def test_average_keeps_real_black_content(tmp_path) -> None:
    image_a = np.zeros((4, 6, 3), dtype=np.uint8)
    image_b = np.full((4, 6, 3), 100, dtype=np.uint8)
    frame_a = _frame(tmp_path, "a.jpg", image_a)
    frame_b = _frame(tmp_path, "b.jpg", image_b)

    average, valid = create_average(
        [frame_a, frame_b],
        reference_size=(6, 4),
        crop="none",
        crop_rect=None,
        fit_origin=(0.0, 0.0),
        output_size=(6, 4),
    )

    assert np.all(average == 50)
    assert np.all(valid == 255)


def test_mosaic_does_not_copy_invalid_border_pixels(tmp_path) -> None:
    image_a = np.full((10, 10, 3), 100, dtype=np.uint8)
    image_b = np.full((10, 10, 3), 200, dtype=np.uint8)
    shift = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    frame_a = _frame(tmp_path, "a.jpg", image_a)
    frame_b = _frame(tmp_path, "b.jpg", image_b, shift)
    mosaic, placed = create_mosaic(
        [frame_a, frame_b],
        reference_size=(10, 10),
        crop="none",
        crop_rect=None,
        fit_origin=(0.0, 0.0),
        output_size=(10, 10),
        polygon_count=5,
        seed=123,
    )

    assert 0 <= placed <= 5
    # The invalid left four columns of image_b may never become input.
    # Uncovered pixels are black because the mosaic is an independent composition.
    assert set(np.unique(mosaic)).issubset({0, 100, 200})


def test_mosaic_does_not_use_average_as_background(tmp_path) -> None:
    image = np.full((10, 10, 3), 100, dtype=np.uint8)
    frame = _frame(tmp_path, "a.jpg", image)

    mosaic, placed = create_mosaic(
        [frame],
        reference_size=(10, 10),
        crop="none",
        crop_rect=None,
        fit_origin=(0.0, 0.0),
        output_size=(10, 10),
        polygon_count=0,
        seed=123,
    )

    assert placed == 0
    assert np.all(mosaic == 0)
