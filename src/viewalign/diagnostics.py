from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .alignment import AlignmentResult


def draw_matches(path: Path, result: AlignmentResult, reference: np.ndarray, image: np.ndarray) -> None:
    if not result.good_matches or result.keypoints_reference is None or result.keypoints_image is None:
        return
    selected = result.good_matches
    canvas = cv2.drawMatches(
        image,
        result.keypoints_image,
        reference,
        result.keypoints_reference,
        selected,
        None,
        matchesMask=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def save_overlay(path: Path, reference: np.ndarray, aligned: np.ndarray) -> None:
    # Both images are in the reference coordinate system here.
    blend = cv2.addWeighted(reference, 0.5, aligned, 0.5, 0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), blend)


def write_metrics(path: Path, metrics: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
