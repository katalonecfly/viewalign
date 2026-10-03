from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class RenderedFrame:
    """Metadata needed to reconstruct an aligned frame's valid-pixel mask."""

    source_path: Path
    output_path: Path
    transform_full: np.ndarray
    source_width: int
    source_height: int


def _identity_mask(height: int, width: int) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


def valid_mask_for_frame(
    frame: RenderedFrame,
    *,
    reference_size: tuple[int, int],
    crop: str,
    crop_rect: tuple[int, int, int, int] | None,
    fit_origin: tuple[float, float],
    output_size: tuple[int, int],
) -> np.ndarray:
    """Return a mask of pixels in an aligned output that came from the source image.

    The mask is geometric rather than based on pixel values, so legitimately black
    content (for example a night sky) remains part of the calculations.
    """
    ref_width, ref_height = reference_size
    output_width, output_height = output_size
    source_mask = _identity_mask(frame.source_height, frame.source_width)

    warped = cv2.warpAffine(
        source_mask,
        frame.transform_full,
        (ref_width, ref_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if crop == "fit":
        offset = np.array(
            [[1.0, 0.0, -fit_origin[0]], [0.0, 1.0, -fit_origin[1]]],
            dtype=np.float64,
        )
        transform = offset @ np.vstack([frame.transform_full, [0.0, 0.0, 1.0]])
        return cv2.warpAffine(
            source_mask,
            transform[:2, :],
            (output_width, output_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    if crop == "intersection":
        if crop_rect is None:
            raise ValueError("intersection crop requires a crop rectangle")
        x, y, w, h = crop_rect
        return warped[y : y + h, x : x + w]

    return warped



def compute_valid_coverage(
    frames: Sequence[RenderedFrame],
    *,
    reference_size: tuple[int, int],
    crop: str,
    crop_rect: tuple[int, int, int, int] | None,
    fit_origin: tuple[float, float],
    output_size: tuple[int, int],
) -> np.ndarray:
    """Return the number of aligned frames that geometrically cover each output pixel."""
    if not frames:
        raise ValueError("Cannot compute coverage for an empty frame set")

    output_width, output_height = output_size
    counts = np.zeros((output_height, output_width), dtype=np.uint16)
    for frame in frames:
        mask = valid_mask_for_frame(
            frame,
            reference_size=reference_size,
            crop=crop,
            crop_rect=crop_rect,
            fit_origin=fit_origin,
            output_size=output_size,
        )
        counts += (mask > 0).astype(np.uint16)
    return counts


def _largest_active_rectangle(active: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the largest axis-aligned rectangle containing only active pixels."""
    if active.ndim != 2:
        raise ValueError("active coverage mask must be two-dimensional")
    height, width = active.shape
    if height == 0 or width == 0 or not np.any(active):
        return None

    heights = np.zeros(width, dtype=np.int32)
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    best_short_side = -1

    for y, row in enumerate(active):
        heights[row] += 1
        heights[~row] = 0

        stack: list[int] = []
        for x in range(width + 1):
            current_height = int(heights[x]) if x < width else 0
            while stack and int(heights[stack[-1]]) > current_height:
                index = stack.pop()
                rect_height = int(heights[index])
                left = stack[-1] + 1 if stack else 0
                rect_width = x - left
                area = rect_width * rect_height
                if area <= 0:
                    continue
                short_side = min(rect_width, rect_height)
                if area > best_area or (area == best_area and short_side > best_short_side):
                    best_area = area
                    best_short_side = short_side
                    best = (left, y - rect_height + 1, rect_width, rect_height)
            stack.append(x)

    return best


def compute_coverage_crop(
    coverage_counts: np.ndarray,
    *,
    image_count: int,
    min_coverage: float,
) -> tuple[tuple[int, int, int, int] | None, int, int]:
    """Find a rectangular crop whose every pixel is covered by at least ``min_coverage`` of images.

    Returns ``(crop_rect, required_image_count, active_pixel_count)``. With a zero
    threshold the full image is active and no crop is necessary.
    """
    if image_count <= 0:
        raise ValueError("image_count must be greater than zero")
    if not 0.0 <= min_coverage <= 100.0 or not np.isfinite(min_coverage):
        raise ValueError("min_coverage must be between 0 and 100")
    if coverage_counts.ndim != 2:
        raise ValueError("coverage_counts must be two-dimensional")
    if coverage_counts.size == 0:
        return None, 0, 0
    if min_coverage == 0.0:
        height, width = coverage_counts.shape
        return (0, 0, width, height), 0, int(coverage_counts.size)

    required_images = max(1, int(np.ceil((min_coverage / 100.0) * image_count - 1e-12)))
    active = coverage_counts >= required_images
    active_pixel_count = int(np.count_nonzero(active))
    return _largest_active_rectangle(active), required_images, active_pixel_count


def crop_image(image: np.ndarray, crop_rect: tuple[int, int, int, int]) -> np.ndarray:
    """Return an exact copy of ``image`` cropped to ``(x, y, width, height)``."""
    x, y, width, height = (int(value) for value in crop_rect)
    image_height, image_width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("crop rectangle must have positive width and height")
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError("crop rectangle is outside the image")
    return image[y : y + height, x : x + width].copy()


def create_average(
    frames: Sequence[RenderedFrame],
    *,
    reference_size: tuple[int, int],
    crop: str,
    crop_rect: tuple[int, int, int, int] | None,
    fit_origin: tuple[float, float],
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Average aligned images while ignoring geometrically invalid border pixels.

    Returns (average_bgr, valid_union_mask).
    """
    if not frames:
        raise ValueError("Cannot average an empty frame set")

    output_width, output_height = output_size
    sums = np.zeros((output_height, output_width, 3), dtype=np.float64)
    counts = np.zeros((output_height, output_width), dtype=np.uint16)

    for frame in frames:
        image = cv2.imread(str(frame.output_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read aligned image: {frame.output_path}")
        if image.shape[1] != output_width or image.shape[0] != output_height:
            raise ValueError(
                f"Aligned image has unexpected size: {frame.output_path} "
                f"({image.shape[1]}x{image.shape[0]} vs {output_width}x{output_height})"
            )

        mask = valid_mask_for_frame(
            frame,
            reference_size=reference_size,
            crop=crop,
            crop_rect=crop_rect,
            fit_origin=fit_origin,
            output_size=output_size,
        )
        valid = mask > 0
        sums[valid] += image[valid]
        counts[valid] += 1

    average = np.zeros_like(sums, dtype=np.uint8)
    valid_union = counts > 0
    if np.any(valid_union):
        average[valid_union] = np.clip(
            sums[valid_union] / counts[valid_union, None], 0, 255
        ).astype(np.uint8)

    return average, (valid_union.astype(np.uint8) * 255)


def _polygon_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, points.astype(np.int32), 255)
    return mask


def _random_valid_center(mask: np.ndarray, rng: random.Random) -> tuple[int, int] | None:
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    for _ in range(100):
        px = rng.randint(x, max(x, x + w - 1))
        py = rng.randint(y, max(y, y + h - 1))
        if mask[py, px]:
            return px, py
    # Deterministic fallback for unusual masks with a thin valid region.
    # OpenCV may return either (N, 1, 2) or (N, 2); normalize both forms.
    points = np.asarray(coords).reshape(-1, 2)
    point = points[rng.randrange(len(points))]
    return int(point[0]), int(point[1])


def _make_random_polygon(
    center: tuple[int, int],
    canvas_shape: tuple[int, int],
    rng: random.Random,
    *,
    size_scale: float = 1.0,
) -> np.ndarray:
    height, width = canvas_shape
    min_dimension = min(width, height)
    base_radius = rng.uniform(0.045, 0.12) * min_dimension * size_scale
    vertices = rng.randint(5, 8)
    start_angle = rng.uniform(0.0, 2.0 * np.pi)

    points: list[tuple[int, int]] = []
    for index in range(vertices):
        angle = start_angle + (2.0 * np.pi * index / vertices) + rng.uniform(-0.18, 0.18)
        radius = base_radius * rng.uniform(0.72, 1.28)
        px = round(center[0] + np.cos(angle) * radius)
        py = round(center[1] + np.sin(angle) * radius)
        points.append((px, py))

    return np.asarray(points, dtype=np.float32)


def _load_frame_data(
    frame: RenderedFrame,
    *,
    reference_size: tuple[int, int],
    crop: str,
    crop_rect: tuple[int, int, int, int] | None,
    fit_origin: tuple[float, float],
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(frame.output_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read aligned image: {frame.output_path}")
    if image.shape[1] != output_size[0] or image.shape[0] != output_size[1]:
        raise ValueError(
            f"Aligned image has unexpected size: {frame.output_path} "
            f"({image.shape[1]}x{image.shape[0]} vs {output_size[0]}x{output_size[1]})"
        )
    mask = valid_mask_for_frame(
        frame,
        reference_size=reference_size,
        crop=crop,
        crop_rect=crop_rect,
        fit_origin=fit_origin,
        output_size=output_size,
    )
    return image, mask


def create_mosaic(
    frames: Sequence[RenderedFrame],
    *,
    reference_size: tuple[int, int],
    crop: str,
    crop_rect: tuple[int, int, int, int] | None,
    fit_origin: tuple[float, float],
    output_size: tuple[int, int],
    polygon_count: int = 50,
    seed: int | None = None,
    polygon_size: float = 1.0,
    min_valid_coverage: float = 0.20,
) -> tuple[np.ndarray, int, int, int]:
    """Build an independent random-polygon mosaic with guaranteed geometric coverage.

    The random phase places clipped polygon patches preferentially in currently
    uncovered regions. A deterministic final fill pass then assigns every remaining
    geometrically coverable pixel from one of the aligned frames. Pixels outside the
    union of all valid source regions remain black.

    Returns ``(mosaic, polygons_placed, coverable_pixels, fill_pixels)``.
    """
    if not frames:
        raise ValueError("Cannot create a mosaic from an empty frame set")
    if polygon_count < 0:
        raise ValueError("polygon_count must be non-negative")
    if polygon_size <= 0 or not np.isfinite(polygon_size):
        raise ValueError("polygon_size must be greater than zero")

    mosaic = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    covered = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
    rng = random.Random(seed)
    cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    cache_order: list[Path] = []
    max_cache = min(8, len(frames))

    def get_frame_data(frame: RenderedFrame) -> tuple[np.ndarray, np.ndarray]:
        if frame.output_path not in cache:
            cache[frame.output_path] = _load_frame_data(
                frame,
                reference_size=reference_size,
                crop=crop,
                crop_rect=crop_rect,
                fit_origin=fit_origin,
                output_size=output_size,
            )
            cache_order.append(frame.output_path)
            if len(cache_order) > max_cache:
                oldest = cache_order.pop(0)
                cache.pop(oldest, None)
        return cache[frame.output_path]

    # The union mask is the hard upper bound for what the mosaic can cover.
    coverable_mask = np.zeros_like(covered)
    for frame in frames:
        _, valid_mask = get_frame_data(frame)
        coverable_mask = cv2.bitwise_or(coverable_mask, valid_mask)

    coverable = coverable_mask > 0
    coverable_pixels = int(np.count_nonzero(coverable))
    placed = 0

    # The random phase is coverage-aware: choose a frame with some currently
    # uncovered area, then place a polygon around an uncovered pixel from it.
    max_attempts = max(1, polygon_count * 12)
    attempts = 0
    stagnant_attempts = 0
    while placed < polygon_count and attempts < max_attempts:
        attempts += 1
        frame = frames[rng.randrange(len(frames))]
        image, valid_mask = get_frame_data(frame)
        target_mask = cv2.bitwise_and(valid_mask, cv2.bitwise_not(covered))
        center = _random_valid_center(target_mask, rng)
        if center is None:
            stagnant_attempts += 1
            continue

        points = _make_random_polygon(
            center,
            valid_mask.shape,
            rng,
            size_scale=polygon_size,
        )
        polygon_mask = _polygon_mask(valid_mask.shape, points)
        polygon_area = int(np.count_nonzero(polygon_mask))
        if polygon_area == 0:
            stagnant_attempts += 1
            continue

        # Clip polygons to the geometrically valid region. This allows a polygon
        # to contribute useful area even when it crosses a warped-image boundary.
        clipped = cv2.bitwise_and(polygon_mask, valid_mask)
        newly_covered = cv2.bitwise_and(clipped, cv2.bitwise_not(covered))
        new_pixels = int(np.count_nonzero(newly_covered))
        coverage = new_pixels / polygon_area
        if coverage < min_valid_coverage or new_pixels == 0:
            stagnant_attempts += 1
            continue

        mosaic[newly_covered > 0] = image[newly_covered > 0]
        covered[newly_covered > 0] = 255
        placed += 1
        stagnant_attempts = 0

        if stagnant_attempts > max(50, len(frames) * 3):
            break

    # Guaranteed fill: every remaining geometrically coverable pixel is assigned
    # from an aligned image that actually contains it. Randomize source priority
    # so the fallback is not always biased toward the same input image.
    remaining = cv2.bitwise_and(coverable_mask, cv2.bitwise_not(covered))
    fill_pixels = 0
    fill_order = list(frames)
    rng.shuffle(fill_order)
    for frame in fill_order:
        if not np.any(remaining):
            break
        image, valid_mask = get_frame_data(frame)
        to_fill = cv2.bitwise_and(valid_mask, remaining)
        count = int(np.count_nonzero(to_fill))
        if count == 0:
            continue
        mosaic[to_fill > 0] = image[to_fill > 0]
        covered[to_fill > 0] = 255
        remaining[to_fill > 0] = 0
        fill_pixels += count

    uncovered = int(np.count_nonzero(cv2.bitwise_and(coverable_mask, cv2.bitwise_not(covered))))
    if uncovered:
        raise RuntimeError(
            f"Mosaic coverage fill failed: {uncovered} geometrically coverable pixels remain uncovered"
        )

    return mosaic, placed, coverable_pixels, fill_pixels
