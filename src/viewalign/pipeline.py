from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .alignment import AlignmentResult, SIFTAligner, scale_affine_to_full_resolution
from .derived import RenderedFrame, create_average, create_mosaic
from .diagnostics import draw_matches, save_overlay, write_metrics
from .images import discover_images, load_oriented_bgr, resize_for_working
from .ordering import order_images
from .video import create_video


def _save_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError(f"Could not write image: {path}")


def _working_to_full_crop(crop: tuple[int, int, int, int], reference_scale: float,
                          output_width: int, output_height: int) -> tuple[int, int, int, int]:
    x, y, w, h = crop
    x0 = max(0, int(np.floor(x / reference_scale)))
    y0 = max(0, int(np.floor(y / reference_scale)))
    x1 = min(output_width, int(np.ceil((x + w) / reference_scale)))
    y1 = min(output_height, int(np.ceil((y + h) / reference_scale)))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _compute_fit_canvas(
    transforms: list[tuple[np.ndarray, int, int]],
) -> tuple[int, int, float, float]:
    """Return a canvas that fully contains all transformed source images.

    Each transform maps source-image coordinates into the reference coordinate
    system. The returned origin is expressed in reference coordinates; callers
    translate all transforms by (-min_x, -min_y) before warping.
    """
    if not transforms:
        raise ValueError("No transforms available for fit canvas")

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for matrix, width, height in transforms:
        corners = np.array(
            [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        transformed = cv2.transform(corners, matrix).reshape(-1, 2)
        min_x = min(min_x, float(transformed[:, 0].min()))
        min_y = min(min_y, float(transformed[:, 1].min()))
        max_x = max(max_x, float(transformed[:, 0].max()))
        max_y = max(max_y, float(transformed[:, 1].max()))

    origin_x = float(np.floor(min_x))
    origin_y = float(np.floor(min_y))
    canvas_width = max(1, int(np.ceil(max_x) - origin_x))
    canvas_height = max(1, int(np.ceil(max_y) - origin_y))
    return canvas_width, canvas_height, origin_x, origin_y


def _compute_intersection_crop(
    valid_masks: list[np.ndarray],
    reference_scale: float,
    output_width: int,
    output_height: int,
) -> tuple[int, int, int, int] | None:
    if not valid_masks:
        return None
    combined = valid_masks[0].copy()
    for mask in valid_masks[1:]:
        combined = cv2.bitwise_and(combined, mask)
    coords = cv2.findNonZero(combined)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return _working_to_full_crop((x, y, w, h), reference_scale, output_width, output_height)


def _resolve_reference(paths: list[Path], reference: str) -> Path:
    if reference in {"first", "middle"}:
        return paths[0] if reference == "first" else paths[len(paths) // 2]
    candidate = Path(reference)
    if not candidate.is_absolute() and not candidate.exists():
        candidate = paths[0].parent / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Reference image does not exist: {reference}")
    candidate = candidate.resolve()
    matches = [p for p in paths if p.resolve() == candidate]
    if not matches:
        raise ValueError("Reference image must be one of the input images")
    return matches[0]


def run_alignment(
    input_dir: Path,
    output_dir: Path,
    reference: str = "first",
    order: str = "filename",
    working_size: int = 1600,
    crop: str = "fit",
    diagnostics: bool = False,
    jpeg_quality: int = 95,
    min_matches: int = 8,
    ratio_threshold: float = 0.75,
    ransac_threshold: float = 4.0,
    duration: float = 0.5,
    mosaic_polygons: int = 50,
    mosaic_seed: int | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    paths = discover_images(input_dir)
    if len(paths) < 2:
        raise ValueError("Need at least two supported images")
    if duration <= 0:
        raise ValueError("duration must be greater than zero")
    if mosaic_polygons < 0:
        raise ValueError("mosaic_polygons must be non-negative")

    ordered, missing_timestamps = order_images(paths, order)
    if missing_timestamps and order == "timestamp":
        log(f"Warning: {len(missing_timestamps)} image(s) have no EXIF timestamp; they are sorted last")
    reference_path = _resolve_reference(ordered, reference)

    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir = output_dir / "aligned"
    diagnostics_dir = output_dir / "diagnostics"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    if diagnostics:
        (diagnostics_dir / "matches").mkdir(parents=True, exist_ok=True)
        (diagnostics_dir / "overlays").mkdir(parents=True, exist_ok=True)

    loaded: dict[Path, np.ndarray] = {}
    working: dict[Path, np.ndarray] = {}
    scales: dict[Path, float] = {}
    metrics: list[dict] = []

    log(f"Found {len(ordered)} images")
    log(f"Reference: {reference_path.name}")

    for path in ordered:
        try:
            image = load_oriented_bgr(path)
        except ValueError as exc:
            log(f"[FAIL] {path.name}: {exc}")
            metrics.append({"file": path.name, "status": "failed", "reason": str(exc)})
            continue
        loaded[path] = image
        work, scale = resize_for_working(image, working_size)
        working[path] = work
        scales[path] = scale

    if reference_path not in loaded:
        raise RuntimeError("Reference image could not be loaded")

    reference_image = loaded[reference_path]
    reference_work = working[reference_path]
    reference_scale = scales[reference_path]
    ref_height, ref_width = reference_image.shape[:2]

    aligner = SIFTAligner(
        ratio_threshold=ratio_threshold,
        ransac_threshold=ransac_threshold,
        min_matches=min_matches,
    )
    reference_keypoints, reference_descriptors = aligner.describe(reference_work)
    if reference_descriptors is None or len(reference_keypoints) < 2:
        raise RuntimeError("Reference image has insufficient SIFT features")

    successful: list[tuple[Path, AlignmentResult, np.ndarray]] = []
    valid_masks: list[np.ndarray] = []

    for path in ordered:
        if path not in loaded:
            continue
        if path == reference_path:
            matrix_work = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
            result = AlignmentResult(
                True,
                matrix_work,
                len(reference_keypoints),
                len(reference_keypoints),
                len(reference_keypoints),
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
            )
        else:
            result = aligner.align(reference_keypoints, reference_descriptors, working[path])

        item = {
            "file": path.name,
            "reference": path == reference_path,
            "status": "ok" if result.success else "rejected",
            "feature_count": result.feature_count,
            "match_count": result.match_count,
            "inlier_count": result.inlier_count,
            "inlier_ratio": result.inlier_ratio,
            "reprojection_error_px_working": result.reprojection_error,
            "rotation_degrees": result.rotation_degrees,
            "scale": result.scale,
            "translation_x_px_working": result.translation_x,
            "translation_y_px_working": result.translation_y,
        }

        if not result.success or result.transform is None:
            item["reason"] = result.reason
            metrics.append(item)
            log(f"[REJECT] {path.name}: {result.reason}")
            if diagnostics:
                draw_matches(
                    diagnostics_dir / "matches" / f"{path.stem}_matches.jpg",
                    result,
                    reference_work,
                    working[path],
                )
            continue

        transform_full = scale_affine_to_full_resolution(
            result.transform, scales[path], reference_scale
        )
        preview_warped = cv2.warpAffine(
            loaded[path],
            transform_full,
            (ref_width, ref_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        mask = np.full(working[path].shape[:2], 255, dtype=np.uint8)
        warped_mask = cv2.warpAffine(
            mask,
            result.transform,
            (reference_work.shape[1], reference_work.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_masks.append(warped_mask)
        successful.append((path, result, transform_full))
        item["transform_full_resolution"] = transform_full.tolist()
        metrics.append(item)
        if diagnostics:
            draw_matches(
                diagnostics_dir / "matches" / f"{path.stem}_matches.jpg",
                result,
                reference_work,
                working[path],
            )
            save_overlay(
                diagnostics_dir / "overlays" / f"{path.stem}_overlay.jpg",
                reference_image,
                preview_warped,
            )
        log(f"[ OK ] {path.name}: {result.inlier_count}/{result.match_count} inliers")

    if not successful:
        raise RuntimeError("No images could be aligned")

    crop_rect = None
    fit_canvas = None
    fit_origin = (0.0, 0.0)
    if crop == "intersection":
        crop_rect = _compute_intersection_crop(
            valid_masks, reference_scale, ref_width, ref_height
        )
        if crop_rect is None:
            raise RuntimeError("Could not compute a non-empty common intersection crop")
        log(
            f"Intersection crop: x={crop_rect[0]}, y={crop_rect[1]}, "
            f"w={crop_rect[2]}, h={crop_rect[3]}"
        )
    elif crop == "fit":
        fit_inputs = [
            (transform, loaded[path].shape[1], loaded[path].shape[0])
            for path, _, transform in successful
        ]
        fit_canvas = _compute_fit_canvas(fit_inputs)
        fit_width, fit_height, origin_x, origin_y = fit_canvas
        fit_origin = (origin_x, origin_y)
        log(
            f"Fit canvas: x={origin_x:.2f}, y={origin_y:.2f}, "
            f"w={fit_width}, h={fit_height}"
        )

    if crop_rect is not None:
        output_width, output_height = crop_rect[2], crop_rect[3]
    elif fit_canvas is not None:
        output_width, output_height = fit_canvas[0], fit_canvas[1]
    else:
        output_width, output_height = ref_width, ref_height
    output_size = (output_width, output_height)

    # Render aligned files in input order. The RenderedFrame metadata lets us
    # reconstruct valid-pixel masks later without storing all masks in memory.
    stem_counts: dict[str, int] = {}
    rendered_frames: list[RenderedFrame] = []
    for path, _, transform_full in successful:
        if crop_rect is not None:
            x, y, w, h = crop_rect
            warped = cv2.warpAffine(
                loaded[path],
                transform_full,
                (ref_width, ref_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            warped = warped[y : y + h, x : x + w]
        elif fit_canvas is not None:
            fit_width, fit_height, _, _ = fit_canvas
            offset = np.array(
                [[1.0, 0.0, -fit_origin[0]], [0.0, 1.0, -fit_origin[1]]],
                dtype=np.float64,
            )
            fit_transform = offset @ np.vstack([transform_full, [0.0, 0.0, 1.0]])
            warped = cv2.warpAffine(
                loaded[path],
                fit_transform[:2, :],
                (fit_width, fit_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        else:
            warped = cv2.warpAffine(
                loaded[path],
                transform_full,
                (ref_width, ref_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

        key = path.stem
        stem_counts[key] = stem_counts.get(key, 0) + 1
        suffix = "" if stem_counts[key] == 1 else f"_{stem_counts[key]}"
        out_path = aligned_dir / f"{key}{suffix}.jpg"
        _save_jpeg(out_path, warped, jpeg_quality)
        rendered_frames.append(
            RenderedFrame(
                source_path=path,
                output_path=out_path,
                transform_full=transform_full,
                source_width=loaded[path].shape[1],
                source_height=loaded[path].shape[0],
            )
        )

    # Derived outputs are always produced. Both image composites use geometric
    # masks, so black source content (e.g. night scenes) is not discarded.
    average, average_valid = create_average(
        rendered_frames,
        reference_size=(ref_width, ref_height),
        crop=crop,
        crop_rect=crop_rect,
        fit_origin=fit_origin,
        output_size=output_size,
    )
    average_path = output_dir / "average.jpg"
    _save_jpeg(average_path, average, jpeg_quality)
    log(f"Average image: {average_path.name} ({int(np.count_nonzero(average_valid > 0))} valid pixels)")

    mosaic, polygons_placed = create_mosaic(
        rendered_frames,
        reference_size=(ref_width, ref_height),
        crop=crop,
        crop_rect=crop_rect,
        fit_origin=fit_origin,
        output_size=output_size,
        polygon_count=mosaic_polygons,
        seed=mosaic_seed,
    )
    mosaic_path = output_dir / "mosaic.jpg"
    _save_jpeg(mosaic_path, mosaic, jpeg_quality)
    log(f"Mosaic image: {mosaic_path.name} ({polygons_placed} polygons placed)")

    video_path = output_dir / "video.mp4"
    create_video(
        [frame.output_path for frame in rendered_frames],
        video_path,
        duration=duration,
    )
    log(f"Video: {video_path.name} ({duration:g}s per image)")

    summary = {
        "input_directory": str(input_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "reference": reference_path.name,
        "order": order,
        "images_found": len(paths),
        "images_loaded": len(loaded),
        "images_aligned": len(successful),
        "images_rejected_or_failed": len(ordered) - len(successful),
        "missing_timestamps": [p.name for p in missing_timestamps],
        "working_size": working_size,
        "crop": crop,
        "output_resolution": [output_width, output_height],
        "output_origin_reference": [fit_origin[0], fit_origin[1]],
        "video": str(video_path.name),
        "duration_seconds_per_image": duration,
        "average": str(average_path.name),
        "average_valid_pixel_count": int(np.count_nonzero(average_valid > 0)),
        "mosaic": str(mosaic_path.name),
        "mosaic_polygons_requested": mosaic_polygons,
        "mosaic_polygons_placed": polygons_placed,
        "mosaic_seed": mosaic_seed,
    }
    write_metrics(output_dir / "metrics.json", metrics)
    write_metrics(output_dir / "summary.json", [summary])
    log(f"Aligned {len(successful)}/{len(ordered)} images")
    return summary
