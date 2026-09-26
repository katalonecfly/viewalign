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
    point = coords[rng.randrange(len(coords))][0]
    return int(point[0]), int(point[1])


def _make_random_polygon(
    center: tuple[int, int],
    canvas_shape: tuple[int, int],
    rng: random.Random,
) -> np.ndarray:
    height, width = canvas_shape
    min_dimension = min(width, height)
    base_radius = rng.uniform(0.045, 0.12) * min_dimension
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
    min_valid_coverage: float = 0.98,
) -> tuple[np.ndarray, int]:
    """Build a collage by replacing random polygons with random aligned frames.

    Polygons are accepted only when almost all of their area is valid for the
    selected source frame, preventing black border pixels from entering the mosaic.
    The background is typically the black-aware average image.
    """
    if not frames:
        raise ValueError("Cannot create a mosaic from an empty frame set")
    if polygon_count < 0:
        raise ValueError("polygon_count must be non-negative")

    mosaic = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    rng = random.Random(seed)
    cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    cache_order: list[Path] = []
    max_cache = min(8, len(frames))
    placed = 0

    for _ in range(polygon_count * 6):
        if placed >= polygon_count:
            break
        frame = frames[rng.randrange(len(frames))]

        if frame.output_path not in cache:
            image = cv2.imread(str(frame.output_path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"Could not read aligned image: {frame.output_path}")
            mask = valid_mask_for_frame(
                frame,
                reference_size=reference_size,
                crop=crop,
                crop_rect=crop_rect,
                fit_origin=fit_origin,
                output_size=output_size,
            )
            cache[frame.output_path] = (image, mask)
            cache_order.append(frame.output_path)
            if len(cache_order) > max_cache:
                oldest = cache_order.pop(0)
                cache.pop(oldest, None)

        image, valid_mask = cache[frame.output_path]
        center = _random_valid_center(valid_mask, rng)
        if center is None:
            continue
        points = _make_random_polygon(center, valid_mask.shape, rng)
        polygon_mask = _polygon_mask(valid_mask.shape, points)
        polygon_area = int(np.count_nonzero(polygon_mask))
        if polygon_area == 0:
            continue

        covered = cv2.bitwise_and(polygon_mask, valid_mask)
        coverage = np.count_nonzero(covered) / polygon_area
        if coverage < min_valid_coverage:
            continue

        mosaic[covered > 0] = image[covered > 0]
        placed += 1

    return mosaic, placed
