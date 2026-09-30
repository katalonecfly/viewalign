from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .derived import RenderedFrame, valid_mask_for_frame
from .images import image_timestamp


def order_images(paths: list[Path], mode: str) -> tuple[list[Path], list[Path]]:
    """Return (ordered paths, files missing a timestamp)."""
    if mode == "filename":
        return sorted(paths, key=lambda p: p.name.lower()), []

    missing: list[Path] = []
    keyed: list[tuple[str, str, Path]] = []
    for path in paths:
        timestamp = image_timestamp(path)
        if timestamp is None:
            missing.append(path)
            # Keep missing files deterministic and after timestamped files.
            keyed.append(("~", path.name.lower(), path))
        else:
            keyed.append((timestamp[1], path.name.lower(), path))

    keyed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in keyed], missing


def _frame_center_x(
    frame: RenderedFrame,
    *,
    crop: str,
    crop_rect: tuple[int, int, int, int] | None,
    fit_origin: tuple[float, float],
) -> float:
    """Return the source image center's x coordinate in output-image pixels."""
    center = np.array(
        [[[frame.source_width / 2.0, frame.source_height / 2.0]]],
        dtype=np.float32,
    )
    transformed = cv2.transform(center, frame.transform_full).reshape(2)
    x = float(transformed[0])

    if crop == "fit":
        x -= fit_origin[0]
    elif crop == "intersection":
        if crop_rect is None:
            raise ValueError("intersection crop requires a crop rectangle")
        x -= crop_rect[0]

    return x


def _frame_mean_luminance(
    frame: RenderedFrame,
    *,
    reference_size: tuple[int, int],
    crop: str,
    crop_rect: tuple[int, int, int, int] | None,
    fit_origin: tuple[float, float],
    output_size: tuple[int, int],
) -> float:
    """Return mean grayscale luminance over geometrically valid output pixels."""
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
    valid = mask > 0
    if not np.any(valid):
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray[valid]))


def order_rendered_frames(
    frames: Sequence[RenderedFrame],
    mode: str,
    *,
    reference_size: tuple[int, int],
    crop: str,
    crop_rect: tuple[int, int, int, int] | None,
    fit_origin: tuple[float, float],
    output_size: tuple[int, int],
) -> list[RenderedFrame]:
    """Return aligned frames in the requested video order."""
    if mode == "random":
        ordered = list(frames)
        random.shuffle(ordered)
        return ordered

    if mode == "brightness":
        keyed = [
            (
                _frame_mean_luminance(
                    frame,
                    reference_size=reference_size,
                    crop=crop,
                    crop_rect=crop_rect,
                    fit_origin=fit_origin,
                    output_size=output_size,
                ),
                frame.source_path.name.lower(),
                frame,
            )
            for frame in frames
        ]
        keyed.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in keyed]

    if mode == "center":
        keyed = [
            (
                _frame_center_x(
                    frame,
                    crop=crop,
                    crop_rect=crop_rect,
                    fit_origin=fit_origin,
                ),
                frame.source_path.name.lower(),
                frame,
            )
            for frame in frames
        ]
        keyed.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in keyed]

    raise ValueError(f"Unknown video order: {mode}")
