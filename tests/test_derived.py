import random

import cv2
import numpy as np

from viewalign.derived import (
    RenderedFrame,
    compute_coverage_crop,
    create_average,
    create_mosaic,
)


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
    mosaic, placed, _, _ = create_mosaic(
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

    mosaic, placed, _, _ = create_mosaic(
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
    assert np.all(mosaic == 100)


def test_mosaic_guarantees_coverable_area_is_filled(tmp_path) -> None:
    image_a = np.zeros((4, 8, 3), dtype=np.uint8)
    image_a[:, :6] = 100
    image_b = np.zeros((4, 8, 3), dtype=np.uint8)
    image_b[:, 2:8] = 200
    shift = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    path_a = tmp_path / "a.jpg"
    path_b = tmp_path / "b.jpg"
    assert cv2.imwrite(str(path_a), image_a)
    assert cv2.imwrite(str(path_b), image_b)
    frame_a = RenderedFrame(path_a, path_a, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 6, 4)
    frame_b = RenderedFrame(path_b, path_b, shift, 6, 4)

    mosaic, placed, coverable, fill_pixels = create_mosaic(
        [frame_a, frame_b],
        reference_size=(8, 4),
        crop="none",
        crop_rect=None,
        fit_origin=(0.0, 0.0),
        output_size=(8, 4),
        polygon_count=1,
        seed=123,
        polygon_size=0.5,
    )

    assert 0 <= placed <= 1
    assert coverable == 8 * 4
    assert fill_pixels > 0
    assert np.all(mosaic > 0)


def test_polygon_size_scales_polygon(tmp_path) -> None:
    from viewalign.derived import _make_random_polygon

    center = (100, 100)
    small = _make_random_polygon(center, (400, 400), random.Random(7), size_scale=0.5)
    large = _make_random_polygon(center, (400, 400), random.Random(7), size_scale=2.0)

    small_span = float(np.ptp(small, axis=0).max())
    large_span = float(np.ptp(large, axis=0).max())
    assert large_span > small_span


def test_random_valid_center_accepts_2d_find_nonzero_shape(monkeypatch):
    import viewalign.derived as derived

    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[3, 5] = 255

    monkeypatch.setattr(derived.cv2, "findNonZero", lambda _mask: np.array([[5, 3]], dtype=np.int32))
    center = derived._random_valid_center(mask, random.Random(1))

    assert center == (5, 3)


def test_coverage_crop_requires_every_pixel_to_meet_threshold() -> None:
    counts = np.array(
        [
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
            [4, 4, 2, 4, 4],
            [4, 4, 4, 4, 4],
        ],
        dtype=np.uint16,
    )

    crop, required, active_pixels = compute_coverage_crop(
        counts, image_count=4, min_coverage=75.0,
    )

    assert required == 3
    assert active_pixels == 19
    assert crop == (0, 0, 5, 2)
    x, y, w, h = crop
    assert np.all(counts[y : y + h, x : x + w] >= required)


def test_coverage_crop_zero_threshold_keeps_full_canvas() -> None:
    counts = np.zeros((3, 5), dtype=np.uint16)
    crop, required, active_pixels = compute_coverage_crop(
        counts, image_count=4, min_coverage=0.0,
    )

    assert crop == (0, 0, 5, 3)
    assert required == 0
    assert active_pixels == 15


def test_coverage_crop_can_have_no_valid_rectangle() -> None:
    counts = np.array([[1, 1], [1, 1]], dtype=np.uint16)
    crop, required, active_pixels = compute_coverage_crop(
        counts, image_count=2, min_coverage=100.0,
    )

    # One image has disappeared from the whole frame, so no pixel satisfies 100%.
    assert crop is None
    assert required == 2
    assert active_pixels == 0
