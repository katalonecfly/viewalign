from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class UprightEstimate:
    """Estimated single global roll correction for a registered image set."""

    rotation_degrees: float
    confidence: str
    supporting_lines: int
    supporting_images: int
    spread_degrees: float
    applied: bool
    reason: str


def _fold_angle_to_axis(angle: float) -> float:
    """Fold an image-line angle into [-pi/4, pi/4], modulo 90 degrees."""
    return float((angle + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)


def estimate_rotation_from_angles(
    angles_radians: Sequence[float],
    weights: Sequence[float] | None = None,
    *,
    min_axis_support_degrees: float = 12.0,
) -> tuple[float | None, float, int]:
    """Estimate the common roll from line angles already expressed in global coordinates.

    Returns (rotation_radians, weighted_spread_radians, accepted_count). The returned
    rotation is the correction parameter used by ViewAlign's similarity transform convention:
    applying it subtracts that angle from image-line orientations.
    """
    if not angles_radians:
        return None, float("inf"), 0

    if weights is None:
        weights_array = np.ones(len(angles_radians), dtype=np.float64)
    else:
        if len(weights) != len(angles_radians):
            raise ValueError("weights must have the same length as angles_radians")
        weights_array = np.asarray(weights, dtype=np.float64)

    accepted_angles: list[float] = []
    accepted_weights: list[float] = []
    limit = np.deg2rad(min_axis_support_degrees)

    for angle, weight in zip(angles_radians, weights_array):
        if not np.isfinite(angle) or not np.isfinite(weight) or weight <= 0.0:
            continue
        folded = _fold_angle_to_axis(float(angle))
        if abs(folded) <= limit:
            accepted_angles.append(folded)
            accepted_weights.append(float(weight))

    if not accepted_angles:
        return None, float("inf"), 0

    values = np.asarray(accepted_angles, dtype=np.float64)
    weights_used = np.asarray(accepted_weights, dtype=np.float64)
    total_weight = float(weights_used.sum())
    if total_weight <= 0.0:
        return None, float("inf"), 0

    estimate = float(np.sum(values * weights_used) / total_weight)
    residuals = values - estimate
    spread = float(np.sqrt(np.sum(weights_used * residuals * residuals) / total_weight))
    return estimate, spread, len(values)


def _line_segments(image: np.ndarray, max_lines: int, min_length: float) -> list[tuple[np.ndarray, np.ndarray, float]]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(gray)[0]
    if detected is None:
        return []

    # OpenCV's LineSegmentDetector has returned both (N, 1, 4) and (N, 4)
    # shaped arrays across versions/builds. Normalize both forms before
    # iterating so a perfectly valid detection cannot crash the pipeline.
    detected_array = np.asarray(detected, dtype=np.float64)
    if detected_array.size == 0:
        return []
    if detected_array.size % 4 != 0:
        return []
    lines = detected_array.reshape(-1, 4)

    segments: list[tuple[np.ndarray, np.ndarray, float]] = []
    for line in lines:
        x1, y1, x2, y2 = map(float, line)
        length = float(np.hypot(x2 - x1, y2 - y1))
        if not np.isfinite(length) or length < min_length:
            continue
        segments.append(
            (
                np.array([x1, y1], dtype=np.float64),
                np.array([x2, y2], dtype=np.float64),
                length,
            )
        )

    segments.sort(key=lambda item: item[2], reverse=True)
    return segments[:max_lines]


def estimate_upright(
    images: Mapping[Path, np.ndarray],
    transforms: Mapping[Path, np.ndarray],
    scales: Mapping[Path, float],
    *,
    max_lines_per_image: int = 60,
    min_line_length_ratio: float = 0.06,
    max_axis_support_degrees: float = 12.0,
    min_supporting_images: int = 3,
    min_supporting_lines: int = 8,
    apply_min_confidence: str = "medium",
) -> UprightEstimate:
    """Estimate one global upright correction from long horizontal/vertical scene lines.

    Lines are detected in each working-resolution image, transformed into the global anchor
    coordinate system, and aggregated across images. Each image contributes at most unit total
    weight, preventing one image with many detected lines from dominating the result.
    """
    angles: list[float] = []
    weights: list[float] = []
    supporting_images = 0

    for path, image in images.items():
        transform = transforms.get(path)
        scale = scales.get(path)
        if transform is None or scale is None or scale <= 0.0:
            continue

        height, width = image.shape[:2]
        min_length = max(40.0, min(height, width) * min_line_length_ratio)
        segments = _line_segments(image, max_lines_per_image, min_length)
        if not segments:
            continue

        linear = np.asarray(transform, dtype=np.float64)[:, :2]
        if linear.shape != (2, 2) or not np.all(np.isfinite(linear)):
            continue

        local_angles: list[float] = []
        local_weights: list[float] = []
        for first, second, _ in segments:
            # Convert working-resolution coordinates back to source pixels before applying
            # the final global transform.
            vector_source = (second - first) / float(scale)
            vector_global = linear @ vector_source
            length_global = float(np.linalg.norm(vector_global))
            if length_global <= 1e-6 or not np.isfinite(length_global):
                continue
            local_angles.append(float(np.arctan2(vector_global[1], vector_global[0])))
            # Long lines are more useful, but the per-image normalization below stops one
            # image from dominating the entire estimate.
            local_weights.append(min(length_global, 2000.0))

        if not local_angles:
            continue

        total = float(np.sum(local_weights))
        if total <= 0.0:
            continue
        normalized = np.asarray(local_weights, dtype=np.float64) / total
        angles.extend(local_angles)
        weights.extend(normalized.tolist())

        # Count an image only when it contributes at least one line that actually supports
        # an axis-like orientation. This prevents a single image with many lines from making
        # the confidence appear stronger than it is.
        accepted_here = any(
            abs(_fold_angle_to_axis(angle)) <= np.deg2rad(max_axis_support_degrees)
            for angle in local_angles
        )
        if accepted_here:
            supporting_images += 1

    estimate, spread, accepted_lines = estimate_rotation_from_angles(
        angles,
        weights,
        min_axis_support_degrees=max_axis_support_degrees,
    )

    if estimate is None:
        return UprightEstimate(
            rotation_degrees=0.0,
            confidence="low",
            supporting_lines=0,
            supporting_images=0,
            spread_degrees=float("inf"),
            applied=False,
            reason="no sufficiently horizontal or vertical scene lines detected",
        )

    spread_degrees = float(np.degrees(spread))
    abs_rotation_degrees = abs(float(np.degrees(estimate)))

    if supporting_images >= max(min_supporting_images + 2, 5) and accepted_lines >= max(min_supporting_lines * 2, 16) and spread_degrees <= 2.5:
        confidence = "high"
    elif supporting_images >= min_supporting_images and accepted_lines >= min_supporting_lines and spread_degrees <= 5.0:
        confidence = "medium"
    else:
        confidence = "low"

    confidence_order = {"low": 0, "medium": 1, "high": 2}
    applied = confidence_order[confidence] >= confidence_order[apply_min_confidence]
    if abs_rotation_degrees < 0.25:
        applied = False
        reason = "scene is already approximately upright"
    elif not applied:
        reason = "line evidence is too weak or inconsistent for a safe correction"
    else:
        reason = "detected common horizontal/vertical line orientation"

    return UprightEstimate(
        rotation_degrees=float(np.degrees(estimate)) if applied else 0.0,
        confidence=confidence,
        supporting_lines=accepted_lines,
        supporting_images=supporting_images,
        spread_degrees=spread_degrees,
        applied=applied,
        reason=reason,
    )


def apply_global_rotation(
    transforms: Sequence[np.ndarray],
    rotation_degrees: float,
    center: tuple[float, float],
) -> list[np.ndarray]:
    """Rotate a complete set of global transforms around one common canvas point."""
    if not transforms:
        return []
    radians = np.deg2rad(float(rotation_degrees))
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    cx, cy = map(float, center)

    rotation = np.array(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    to_origin = np.array(
        [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    from_origin = np.array(
        [[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    correction = from_origin @ rotation @ to_origin

    result: list[np.ndarray] = []
    for transform in transforms:
        matrix = correction @ np.vstack([np.asarray(transform, dtype=np.float64), [0.0, 0.0, 1.0]])
        result.append(matrix[:2, :])
    return result
