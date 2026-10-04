from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .alignment import (
    AlignmentResult,
    SIFTAligner,
    SimilarityEdge,
    filter_global_similarity_edges,
    scale_affine_to_full_resolution,
    solve_global_similarity_transforms,
)
from .derived import (
    RenderedFrame,
    compute_coverage_crop,
    compute_valid_coverage,
    create_average,
    create_mosaic,
    crop_image,
)
from .diagnostics import draw_matches, save_overlay, write_metrics
from .images import discover_images, load_oriented_bgr, resize_for_working
from .ordering import order_images, order_rendered_frames
from .video import create_video
from .upright import UprightEstimate, apply_global_rotation, estimate_upright


def _generated_output_path(
    output_dir: Path,
    filename: str | None,
    *,
    suffix: str,
    default_stem: str,
) -> Path:
    """Resolve a derived-output filename inside output_dir/generated."""
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        name = f"{timestamp}_{default_stem}{suffix}"
    else:
        filename_path = Path(filename)
        if filename_path.is_absolute() or filename_path.name != filename or filename in {"", ".", ".."}:
            raise ValueError("filename must be a simple filename, not a path")
        name = filename
        existing_suffix = filename_path.suffix.lower()
        if existing_suffix and existing_suffix != suffix.lower():
            raise ValueError(f"filename must use the {suffix} extension")
        if not existing_suffix:
            name = f"{name}{suffix}"

    return generated_dir / name


def _save_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError(f"Could not write image: {path}")


def _compute_fit_canvas(
    transforms: list[tuple[np.ndarray, int, int]],
) -> tuple[int, int, float, float]:
    """Return a canvas that fully contains all transformed source images."""
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
    return x, y, w, h


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


def _edge_weight(result: AlignmentResult) -> float:
    """Convert pairwise match quality into a positive global-optimization weight."""
    reprojection = max(result.reprojection_error or 1.0, 0.5)
    return max(1.0, float(result.inlier_count)) * max(result.inlier_ratio, 0.05) / reprojection


def _connected_component(node_count: int, edges: list[SimilarityEdge], start: int) -> set[int]:
    adjacency = [set() for _ in range(node_count)]
    for edge in edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)

    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _global_components(node_count: int, edges: list[SimilarityEdge]) -> list[set[int]]:
    adjacency = [set() for _ in range(node_count)]
    for edge in edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)

    components: list[set[int]] = []
    remaining = set(range(node_count))
    while remaining:
        start = min(remaining)
        component = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in component:
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
        remaining -= component
    return components


def _pairwise_record(
    source: Path,
    target: Path,
    result: AlignmentResult,
    transform_full: np.ndarray | None,
    *,
    derived_inverse: bool = False,
) -> dict:
    record = {
        "source": source.name,
        "target": target.name,
        "status": "ok" if result.success and transform_full is not None else "rejected",
        "feature_count_source": result.feature_count,
        "match_count": result.match_count,
        "inlier_count": result.inlier_count,
        "inlier_ratio": result.inlier_ratio,
        "reprojection_error_px_working": result.reprojection_error,
        "rotation_degrees": result.rotation_degrees,
        "scale": result.scale,
        "translation_x_px_working": result.translation_x,
        "translation_y_px_working": result.translation_y,
        "derived_inverse": derived_inverse,
    }
    if transform_full is not None:
        record["transform_full_resolution"] = transform_full.tolist()
    if not result.success:
        record["reason"] = result.reason
    return record


def _invert_affine(matrix: np.ndarray) -> np.ndarray | None:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
        return None
    linear = matrix[:, :2]
    determinant = float(np.linalg.det(linear))
    if not np.isfinite(determinant) or abs(determinant) <= 1e-12:
        return None
    inverse = cv2.invertAffineTransform(matrix)
    if inverse is None or not np.all(np.isfinite(inverse)):
        return None
    return inverse


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _identity_result(feature_count: int) -> AlignmentResult:
    return AlignmentResult(
        True,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        feature_count,
        feature_count,
        feature_count,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
    )


def _default_upright_estimate() -> UprightEstimate:
    return UprightEstimate(
        rotation_degrees=0.0,
        confidence="low",
        supporting_lines=0,
        supporting_images=0,
        spread_degrees=0.0,
        applied=False,
        reason="not used",
    )


@dataclass
class _AlignmentComputation:
    reference_path: Path
    successful: list[tuple[Path, AlignmentResult, np.ndarray]]
    valid_masks: list[np.ndarray]
    metrics: list[dict]
    pairwise_matrix: list[list[dict | None]] | None = None
    pairwise_successful_edges: list[SimilarityEdge] = field(default_factory=list)
    upright_estimate: UprightEstimate = field(default_factory=_default_upright_estimate)


def _warp_mask_to_reference(
    image: np.ndarray,
    transform: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    mask = np.full(image.shape[:2], 255, dtype=np.uint8)
    return cv2.warpAffine(
        mask,
        transform,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _warp_to_reference(
    image: np.ndarray,
    transform: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    return cv2.warpAffine(
        image,
        transform,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _run_global_alignment(
    *,
    ordered: list[Path],
    loaded: dict[Path, np.ndarray],
    working: dict[Path, np.ndarray],
    scales: dict[Path, float],
    aligner: SIFTAligner,
    diagnostics_dir: Path,
    diagnostics: bool,
    upright: str,
    log: Callable[[str], None],
) -> _AlignmentComputation:
    log("Detecting SIFT features...")
    described: dict[Path, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {
        path: aligner.describe(working[path]) for path in loaded
    }
    valid_paths = [path for path in ordered if path in loaded]
    pairwise_matrix: list[list[dict | None]] = [[None for _ in valid_paths] for _ in valid_paths]
    pairwise_edges: list[SimilarityEdge] = []
    pair_count = len(valid_paths) * (len(valid_paths) - 1) // 2
    log(f"Computing pairwise alignments ({pair_count} pairs)...")

    for i, source in enumerate(valid_paths):
        source_kp, source_desc = described[source]
        for j in range(i + 1, len(valid_paths)):
            target = valid_paths[j]
            target_kp, target_desc = described[target]
            result = aligner.align_described(
                target_kp,
                target_desc if target_desc is not None else np.empty((0, 128), dtype=np.float32),
                source_kp,
                source_desc,
            )

            transform_full: np.ndarray | None = None
            if result.success and result.transform is not None and target_desc is not None:
                candidate_transform = scale_affine_to_full_resolution(
                    result.transform, scales[source], scales[target]
                )
                if _invert_affine(candidate_transform) is not None:
                    transform_full = candidate_transform
                    pairwise_edges.append(
                        SimilarityEdge(i, j, transform_full, _edge_weight(result), result)
                    )
                else:
                    result = AlignmentResult(
                        False,
                        candidate_transform,
                        result.feature_count,
                        result.match_count,
                        result.inlier_count,
                        result.inlier_ratio,
                        result.reprojection_error,
                        result.rotation_degrees,
                        result.scale,
                        result.translation_x,
                        result.translation_y,
                        reason="estimated similarity transform is not invertible",
                        good_matches=result.good_matches,
                        keypoints_reference=result.keypoints_reference,
                        keypoints_image=result.keypoints_image,
                    )

            pairwise_matrix[i][j] = _pairwise_record(source, target, result, transform_full)
            if transform_full is not None:
                inverse_transform = _invert_affine(transform_full)
                if inverse_transform is None:
                    raise RuntimeError("Internal error: accepted pairwise transform became non-invertible")
                reverse_result = AlignmentResult(
                    True,
                    inverse_transform,
                    result.feature_count,
                    result.match_count,
                    result.inlier_count,
                    result.inlier_ratio,
                    result.reprojection_error,
                    -float(result.rotation_degrees or 0.0),
                    float(1.0 / result.scale) if result.scale else None,
                    None,
                    None,
                )
                reverse_record = _pairwise_record(
                    target,
                    source,
                    reverse_result,
                    inverse_transform,
                    derived_inverse=True,
                )
                reverse_record["feature_count_source"] = len(target_kp)
                pairwise_matrix[j][i] = reverse_record
            else:
                pairwise_matrix[j][i] = {
                    "source": target.name,
                    "target": source.name,
                    "status": "rejected",
                    "derived_inverse": True,
                    "reason": result.reason,
                }

            if result.success and transform_full is not None:
                log(
                    f"[ OK ] {source.name} <-> {target.name}: "
                    f"{result.inlier_count}/{result.match_count} inliers"
                )
            else:
                log(f"[REJECT] {source.name} <-> {target.name}: {result.reason}")

            if diagnostics:
                draw_matches(
                    diagnostics_dir / "matches" / f"{i:04d}_{j:04d}_{source.stem}__{target.stem}_matches.jpg",
                    result,
                    working[target],
                    working[source],
                )

    node_scales = [
        float(np.sqrt(loaded[path].shape[0] * loaded[path].shape[1]))
        for path in valid_paths
    ]
    raw_edge_count = len(pairwise_edges)
    pairwise_edges = filter_global_similarity_edges(pairwise_edges, node_scales)
    filtered_edge_count = raw_edge_count - len(pairwise_edges)
    if filtered_edge_count:
        log(f"Global graph: filtered {filtered_edge_count} implausible pairwise links")

    components = _global_components(len(valid_paths), pairwise_edges)
    viable_components = [component for component in components if len(component) >= 2]
    if not viable_components:
        _write_json(
            diagnostics_dir.parent / "pairwise_metrics.json",
            {"images": [path.name for path in valid_paths], "matrix": pairwise_matrix},
        )
        raise RuntimeError("No connected global alignment component could be formed")

    viable_components.sort(key=lambda component: (-len(component), min(component)))
    selected = viable_components[0]
    anchor_index = min(selected)
    anchor_path = valid_paths[anchor_index]
    selected_edges = [
        edge for edge in pairwise_edges if edge.source in selected and edge.target in selected
    ]

    log(
        f"Global graph: {len(selected)}/{len(valid_paths)} loaded images connected; "
        f"{len(selected_edges)} successful pairwise links"
    )
    if len(components) > 1:
        log(f"Global graph: {len(components)} connected component(s); using the largest component")
    log(f"Global anchor: {anchor_path.name}")
    log("Solving global transforms...")
    global_transforms = solve_global_similarity_transforms(
        len(valid_paths), selected_edges, anchor_index, node_scales=node_scales
    )

    upright_estimate = _default_upright_estimate()
    if upright == "auto":
        log("Estimating upright orientation...")
        upright_images = {
            path: working[path]
            for index, path in enumerate(valid_paths)
            if index in selected
        }
        upright_transforms = {
            path: global_transforms[index]
            for index, path in enumerate(valid_paths)
            if index in selected
        }
        upright_scales = {
            path: scales[path]
            for index, path in enumerate(valid_paths)
            if index in selected
        }
        upright_estimate = estimate_upright(upright_images, upright_transforms, upright_scales)
        log(
            f"Upright orientation: {upright_estimate.rotation_degrees:+.2f}° "
            f"({upright_estimate.confidence} confidence, "
            f"{upright_estimate.supporting_lines} supporting lines across "
            f"{upright_estimate.supporting_images} images)"
        )
        if upright_estimate.applied:
            selected_indices = sorted(selected)
            corrected = apply_global_rotation(
                [global_transforms[index] for index in selected_indices],
                upright_estimate.rotation_degrees,
                center=(
                    loaded[anchor_path].shape[1] / 2.0,
                    loaded[anchor_path].shape[0] / 2.0,
                ),
            )
            for index, matrix in zip(selected_indices, corrected):
                global_transforms[index] = matrix
            log("Applied automatic global upright correction.")
        else:
            log(f"Upright correction not applied: {upright_estimate.reason}")
    else:
        log("Upright orientation: disabled")

    best_edge_by_node: dict[int, SimilarityEdge] = {}
    for edge in selected_edges:
        for node in (edge.source, edge.target):
            current = best_edge_by_node.get(node)
            if current is None or edge.weight > current.weight:
                best_edge_by_node[node] = edge

    successful: list[tuple[Path, AlignmentResult, np.ndarray]] = []
    valid_masks: list[np.ndarray] = []
    metrics: list[dict] = []
    reference_size = (loaded[anchor_path].shape[1], loaded[anchor_path].shape[0])
    for index, path in enumerate(valid_paths):
        if index not in selected:
            metrics.append({
                "file": path.name,
                "status": "rejected",
                "reason": "not connected to the selected global alignment component",
                "global_anchor": anchor_path.name,
                "pairwise_edges": sum(
                    1 for edge in pairwise_edges if edge.source == index or edge.target == index
                ),
            })
            log(f"[REJECT] {path.name}: not connected to global alignment component")
            continue

        transform_full = global_transforms[index]
        best = best_edge_by_node.get(index)
        result_for_record = (
            _identity_result(len(described[path][0]))
            if index == anchor_index
            else best.result if best else None
        )
        metrics.append({
            "file": path.name,
            "reference": False,
            "global_anchor": index == anchor_index,
            "global_reference": anchor_path.name,
            "status": "ok",
            "pairwise_edges": sum(
                1 for edge in selected_edges if edge.source == index or edge.target == index
            ),
            "best_pairwise_inliers": result_for_record.inlier_count if result_for_record else 0,
            "best_pairwise_matches": result_for_record.match_count if result_for_record else 0,
            "transform_full_resolution": transform_full.tolist(),
        })
        successful.append((
            path,
            result_for_record or _identity_result(len(described[path][0])),
            transform_full,
        ))
        valid_masks.append(_warp_mask_to_reference(loaded[path], transform_full, reference_size))

    return _AlignmentComputation(
        reference_path=anchor_path,
        successful=successful,
        valid_masks=valid_masks,
        metrics=metrics,
        pairwise_matrix=pairwise_matrix,
        pairwise_successful_edges=selected_edges,
        upright_estimate=upright_estimate,
    )


def _run_reference_alignment(
    *,
    ordered: list[Path],
    loaded: dict[Path, np.ndarray],
    working: dict[Path, np.ndarray],
    scales: dict[Path, float],
    reference_path: Path,
    aligner: SIFTAligner,
    diagnostics_dir: Path,
    diagnostics: bool,
    log: Callable[[str], None],
) -> _AlignmentComputation:
    if reference_path not in loaded:
        raise RuntimeError("Reference image could not be loaded")

    reference_image = loaded[reference_path]
    reference_work = working[reference_path]
    reference_scale = scales[reference_path]
    ref_height, ref_width = reference_image.shape[:2]
    reference_keypoints, reference_descriptors = aligner.describe(reference_work)
    if reference_descriptors is None or len(reference_keypoints) < 2:
        raise RuntimeError("Reference image has insufficient SIFT features")

    log("Aligning images...")
    successful: list[tuple[Path, AlignmentResult, np.ndarray]] = []
    valid_masks: list[np.ndarray] = []
    metrics: list[dict] = []

    for path in ordered:
        if path not in loaded:
            continue
        if path == reference_path:
            result = _identity_result(len(reference_keypoints))
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
        item["transform_full_resolution"] = transform_full.tolist()
        metrics.append(item)
        successful.append((path, result, transform_full))
        valid_masks.append(_warp_mask_to_reference(loaded[path], transform_full, (ref_width, ref_height)))

        if diagnostics:
            preview_warped = _warp_to_reference(loaded[path], transform_full, (ref_width, ref_height))
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

    return _AlignmentComputation(
        reference_path=reference_path,
        successful=successful,
        valid_masks=valid_masks,
        metrics=metrics,
    )


def _render_aligned_frames(
    *,
    successful: list[tuple[Path, AlignmentResult, np.ndarray]],
    loaded: dict[Path, np.ndarray],
    aligned_dir: Path,
    crop_rect: tuple[int, int, int, int] | None,
    fit_canvas: tuple[int, int, float, float] | None,
    fit_origin: tuple[float, float],
    reference_size: tuple[int, int],
    jpeg_quality: int,
    log: Callable[[str], None],
) -> list[RenderedFrame]:
    ref_width, ref_height = reference_size
    if crop_rect is not None:
        output_width, output_height = crop_rect[2], crop_rect[3]
    elif fit_canvas is not None:
        output_width, output_height = fit_canvas[0], fit_canvas[1]
    else:
        output_width, output_height = ref_width, ref_height

    output_size = (output_width, output_height)
    stem_counts: dict[str, int] = {}
    rendered_frames: list[RenderedFrame] = []
    for path, _, transform_full in successful:
        if crop_rect is not None:
            warped = _warp_to_reference(loaded[path], transform_full, reference_size)
            x, y, width, height = crop_rect
            warped = warped[y : y + height, x : x + width]
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
            warped = _warp_to_reference(loaded[path], transform_full, reference_size)

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

    log(f"Rendered {len(rendered_frames)} aligned images")
    return rendered_frames


def run_alignment(
    input_dir: Path,
    output_dir: Path,
    reference: str = "first",
    upright: str = "auto",
    order: str = "filename",
    working_size: int = 1600,
    crop: str = "fit",
    diagnostics: bool = False,
    jpeg_quality: int = 95,
    min_matches: int = 8,
    ratio_threshold: float = 0.75,
    ransac_threshold: float = 4.0,
    log: Callable[[str], None] = print,
) -> dict:
    paths = discover_images(input_dir)
    if len(paths) < 2:
        raise ValueError("Need at least two supported images")
    if reference != "global" and reference not in {"first", "middle"}:
        _resolve_reference(paths, reference)
    if upright not in {"auto", "none"}:
        raise ValueError(f"Unknown upright mode: {upright}")

    ordered, missing_timestamps = order_images(paths, order)
    if missing_timestamps and order == "timestamp":
        log(
            f"Warning: {len(missing_timestamps)} image(s) have no EXIF timestamp; "
            "they are sorted last"
        )

    reference_path: Path | None = None
    if reference != "global":
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
    load_failures: list[dict] = []

    log(f"Found {len(ordered)} images")
    if reference == "global":
        log("Reference: global (pairwise alignment graph)")
    else:
        assert reference_path is not None
        log(f"Reference: {reference_path.name}")
    log("Loading images...")
    for path in ordered:
        try:
            image = load_oriented_bgr(path)
        except ValueError as exc:
            log(f"[FAIL] {path.name}: {exc}")
            load_failures.append({"file": path.name, "status": "failed", "reason": str(exc)})
            continue
        loaded[path] = image
        work, scale = resize_for_working(image, working_size)
        working[path] = work
        scales[path] = scale

    if len(loaded) < 2:
        raise RuntimeError("Fewer than two images could be loaded")

    aligner = SIFTAligner(
        ratio_threshold=ratio_threshold,
        ransac_threshold=ransac_threshold,
        min_matches=min_matches,
    )

    if reference == "global":
        computation = _run_global_alignment(
            ordered=ordered,
            loaded=loaded,
            working=working,
            scales=scales,
            aligner=aligner,
            diagnostics_dir=diagnostics_dir,
            diagnostics=diagnostics,
            upright=upright,
            log=log,
        )
    else:
        assert reference_path is not None
        computation = _run_reference_alignment(
            ordered=ordered,
            loaded=loaded,
            working=working,
            scales=scales,
            reference_path=reference_path,
            aligner=aligner,
            diagnostics_dir=diagnostics_dir,
            diagnostics=diagnostics,
            log=log,
        )

    if not computation.successful:
        raise RuntimeError("No images could be aligned")

    reference_path = computation.reference_path
    reference_image = loaded[reference_path]
    ref_height, ref_width = reference_image.shape[:2]
    valid_masks = computation.valid_masks

    if reference == "global":
        assert computation.pairwise_matrix is not None
        valid_paths = [path for path in ordered if path in loaded]
        _write_json(
            output_dir / "pairwise_metrics.json",
            {
                "images": [path.name for path in valid_paths],
                "matrix": computation.pairwise_matrix,
                "successful_pairwise_links": len(computation.pairwise_successful_edges),
                "global_anchor": reference_path.name,
            },
        )
        if diagnostics:
            for path, _, transform_full in computation.successful:
                aligned_preview = _warp_to_reference(loaded[path], transform_full, (ref_width, ref_height))
                save_overlay(
                    diagnostics_dir / "overlays" / f"{path.stem}_global_overlay.jpg",
                    reference_image,
                    aligned_preview,
                )

    crop_rect = None
    fit_canvas = None
    fit_origin = (0.0, 0.0)
    if crop == "intersection":
        crop_rect = _compute_intersection_crop(valid_masks)
        if crop_rect is None:
            raise RuntimeError("Could not compute a non-empty common intersection crop")
        log(
            f"Intersection crop: x={crop_rect[0]}, y={crop_rect[1]}, "
            f"w={crop_rect[2]}, h={crop_rect[3]}"
        )
    elif crop == "fit":
        fit_inputs = [
            (transform, loaded[path].shape[1], loaded[path].shape[0])
            for path, _, transform in computation.successful
        ]
        fit_canvas = _compute_fit_canvas(fit_inputs)
        fit_width, fit_height, origin_x, origin_y = fit_canvas
        fit_origin = (origin_x, origin_y)
        log(
            f"Fit canvas: x={origin_x:.2f}, y={origin_y:.2f}, "
            f"w={fit_width}, h={fit_height}"
        )

    output_width, output_height = (
        crop_rect[2:4]
        if crop_rect is not None
        else fit_canvas[:2]
        if fit_canvas is not None
        else (ref_width, ref_height)
    )
    output_size = (int(output_width), int(output_height))

    log("Rendering aligned images...")
    rendered_frames = _render_aligned_frames(
        successful=computation.successful,
        loaded=loaded,
        aligned_dir=aligned_dir,
        crop_rect=crop_rect,
        fit_canvas=fit_canvas,
        fit_origin=fit_origin,
        reference_size=(ref_width, ref_height),
        jpeg_quality=jpeg_quality,
        log=log,
    )

    alignment_manifest = {
        "version": 1,
        "reference": reference_path.name,
        "reference_mode": reference,
        "order": order,
        "working_size": working_size,
        "crop": crop,
        "jpeg_quality": jpeg_quality,
        "reference_size": [ref_width, ref_height],
        "output_size": [output_width, output_height],
        "crop_rect": list(crop_rect) if crop_rect is not None else None,
        "fit_origin": [fit_origin[0], fit_origin[1]],
        "frames": [
            {
                "source": frame.source_path.name,
                "output": str(frame.output_path.relative_to(output_dir)),
                "source_width": frame.source_width,
                "source_height": frame.source_height,
                "transform_full_resolution": frame.transform_full.tolist(),
            }
            for frame in rendered_frames
        ],
    }
    if reference == "global":
        alignment_manifest.update({
            "global_anchor": reference_path.name,
            "successful_pairwise_links": len(computation.pairwise_successful_edges),
            "pairwise_metrics": "pairwise_metrics.json",
            "upright_mode": upright,
            "upright_rotation_degrees": computation.upright_estimate.rotation_degrees,
            "upright_confidence": computation.upright_estimate.confidence,
            "upright_supporting_lines": computation.upright_estimate.supporting_lines,
            "upright_supporting_images": computation.upright_estimate.supporting_images,
            "upright_spread_degrees": computation.upright_estimate.spread_degrees,
            "upright_applied": computation.upright_estimate.applied,
        })
    _write_json(output_dir / "alignment.json", alignment_manifest)

    metrics = load_failures + computation.metrics
    summary = {
        "input_directory": str(input_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "reference": reference_path.name,
        "reference_mode": reference,
        "order": order,
        "images_found": len(paths),
        "images_loaded": len(loaded),
        "images_aligned": len(computation.successful),
        "images_rejected_or_failed": len(ordered) - len(computation.successful),
        "missing_timestamps": [p.name for p in missing_timestamps],
        "working_size": working_size,
        "crop": crop,
        "output_resolution": [output_width, output_height],
        "output_origin_reference": [fit_origin[0], fit_origin[1]],
        "aligned_directory": "aligned",
        "alignment_manifest": "alignment.json",
    }
    if reference == "global":
        summary["global_anchor"] = reference_path.name
        summary["pairwise_metrics"] = "pairwise_metrics.json"
        summary["successful_pairwise_links"] = len(computation.pairwise_successful_edges)
        summary["upright_mode"] = upright
        summary["upright_rotation_degrees"] = computation.upright_estimate.rotation_degrees
        summary["upright_confidence"] = computation.upright_estimate.confidence
        summary["upright_supporting_lines"] = computation.upright_estimate.supporting_lines
        summary["upright_supporting_images"] = computation.upright_estimate.supporting_images
        summary["upright_spread_degrees"] = computation.upright_estimate.spread_degrees
        summary["upright_applied"] = computation.upright_estimate.applied

    write_metrics(output_dir / "metrics.json", metrics)
    write_metrics(output_dir / "summary.json", [summary])
    log(f"Aligned {len(computation.successful)}/{len(ordered)} images")
    log(f"Alignment outputs written to {output_dir}")
    log("Done.")
    return summary


def _load_alignment_data(output_dir: Path) -> tuple[dict, list[RenderedFrame]]:
    manifest_path = output_dir / "alignment.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Alignment manifest not found: {manifest_path}. Run 'viewalign align' first."
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid alignment manifest: {manifest_path}") from exc

    if manifest.get("version") != 1:
        raise ValueError(f"Unsupported alignment manifest version: {manifest.get('version')!r}")

    try:
        reference_size = tuple(int(value) for value in manifest["reference_size"])
        output_size = tuple(int(value) for value in manifest["output_size"])
        crop = str(manifest["crop"])
        crop_rect_raw = manifest.get("crop_rect")
        crop_rect = tuple(int(value) for value in crop_rect_raw) if crop_rect_raw is not None else None
        fit_origin = tuple(float(value) for value in manifest["fit_origin"])
        frame_records = manifest["frames"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Alignment manifest is missing required fields: {manifest_path}") from exc

    if len(reference_size) != 2 or len(output_size) != 2 or len(fit_origin) != 2:
        raise ValueError(f"Alignment manifest has invalid geometry: {manifest_path}")
    if crop_rect is not None and len(crop_rect) != 4:
        raise ValueError(f"Alignment manifest has invalid crop rectangle: {manifest_path}")
    if not isinstance(frame_records, list) or not frame_records:
        raise ValueError(f"Alignment manifest contains no aligned frames: {manifest_path}")

    frames: list[RenderedFrame] = []
    for record in frame_records:
        try:
            source = Path(record["source"])
            output_path = output_dir / Path(record["output"])
            source_width = int(record["source_width"])
            source_height = int(record["source_height"])
            transform = np.asarray(record["transform_full_resolution"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid frame record in {manifest_path}") from exc

        if transform.shape != (2, 3) or not np.all(np.isfinite(transform)):
            raise ValueError(f"Invalid transform for {source.name} in {manifest_path}")
        if not output_path.exists():
            raise FileNotFoundError(
                f"Aligned image not found: {output_path}. Re-run 'viewalign align' if files were moved or removed."
            )
        frames.append(RenderedFrame(source, output_path, transform, source_width, source_height))

    normalized = {
        "manifest": manifest,
        "reference_size": reference_size,
        "output_size": output_size,
        "crop": crop,
        "crop_rect": crop_rect,
        "fit_origin": fit_origin,
        "jpeg_quality": int(manifest.get("jpeg_quality", 95)),
    }
    return normalized, frames


def _coverage_crop_for_output(
    data: dict,
    frames: list[RenderedFrame],
    min_coverage: float,
    *,
    log: Callable[[str], None],
) -> tuple[int, int, int, int] | None:
    if not 0.0 <= min_coverage <= 100.0 or not np.isfinite(min_coverage):
        raise ValueError("min_coverage must be between 0 and 100")
    if min_coverage == 0.0:
        log("Minimum coverage: disabled (0%)")
        return None

    coverage_counts = compute_valid_coverage(
        frames,
        reference_size=data["reference_size"],
        crop=data["crop"],
        crop_rect=data["crop_rect"],
        fit_origin=data["fit_origin"],
        output_size=data["output_size"],
    )
    crop_rect, required_images, active_pixels = compute_coverage_crop(
        coverage_counts,
        image_count=len(frames),
        min_coverage=min_coverage,
    )
    log(
        f"Minimum coverage: {min_coverage:g}% "
        f"({required_images}/{len(frames)} images per pixel)"
    )
    if crop_rect is None:
        log(
            f"Warning: no pixel is covered by at least {min_coverage:g}% of images; "
            "leaving the original canvas unchanged."
        )
        return None

    x, y, width, height = crop_rect
    log(
        f"Coverage crop: x={x}, y={y}, w={width}, h={height} "
        f"({active_pixels:,} active pixels before rectangular crop)"
    )
    return crop_rect


def run_average(
    output_dir: Path,
    *,
    filename: str | None = None,
    min_coverage: float = 0.0,
    log: Callable[[str], None] = print,
) -> dict:
    data, frames = _load_alignment_data(output_dir)
    log(f"Loaded alignment: {len(frames)} images")
    log("Creating average image...")
    average, average_valid = create_average(
        frames,
        reference_size=data["reference_size"],
        crop=data["crop"],
        crop_rect=data["crop_rect"],
        fit_origin=data["fit_origin"],
        output_size=data["output_size"],
    )
    coverage_crop = _coverage_crop_for_output(data, frames, min_coverage, log=log)
    if coverage_crop is not None:
        average = crop_image(average, coverage_crop)
        average_valid = crop_image(average_valid, coverage_crop)
    output_path = _generated_output_path(
        output_dir, filename, suffix=".jpg", default_stem="average"
    )
    _save_jpeg(output_path, average, data["jpeg_quality"])
    valid_pixels = int(np.count_nonzero(average_valid > 0))
    log(f"Average image: {output_path.name} ({valid_pixels} valid pixels)")
    log("Done.")
    return {
        "operation": "average",
        "output": str(output_path.relative_to(output_dir)),
        "images": len(frames),
        "valid_pixel_count": valid_pixels,
        "min_coverage_percent": min_coverage,
        "coverage_crop": list(coverage_crop) if coverage_crop is not None else None,
    }


def run_mosaic(
    output_dir: Path,
    *,
    polygon_count: int = 50,
    seed: int | None = None,
    polygon_size: float = 1.0,
    filename: str | None = None,
    min_coverage: float = 0.0,
    log: Callable[[str], None] = print,
) -> dict:
    if polygon_count < 0:
        raise ValueError("polygon_count must be non-negative")
    if polygon_size <= 0:
        raise ValueError("polygon_size must be greater than zero")
    data, frames = _load_alignment_data(output_dir)
    log(f"Loaded alignment: {len(frames)} images")
    log(
        f"Creating mosaic image ({polygon_count} polygons, size {polygon_size:g}x)..."
    )
    mosaic, polygons_placed, coverable_pixels, fill_pixels = create_mosaic(
        frames,
        reference_size=data["reference_size"],
        crop=data["crop"],
        crop_rect=data["crop_rect"],
        fit_origin=data["fit_origin"],
        output_size=data["output_size"],
        polygon_count=polygon_count,
        seed=seed,
        polygon_size=polygon_size,
    )
    coverage_crop = _coverage_crop_for_output(data, frames, min_coverage, log=log)
    if coverage_crop is not None:
        mosaic = crop_image(mosaic, coverage_crop)
    output_path = _generated_output_path(
        output_dir, filename, suffix=".jpg", default_stem="mosaic"
    )
    _save_jpeg(output_path, mosaic, data["jpeg_quality"])
    log(
        f"Mosaic coverage: 100.0% of coverable area "
        f"({coverable_pixels:,} pixels; final fill {fill_pixels:,})"
    )
    log(f"Mosaic image: {output_path.name} ({polygons_placed} polygons placed)")
    log("Done.")
    return {
        "operation": "mosaic",
        "output": str(output_path.relative_to(output_dir)),
        "images": len(frames),
        "polygons_requested": polygon_count,
        "polygons_placed": polygons_placed,
        "polygon_size": polygon_size,
        "coverable_pixel_count": coverable_pixels,
        "coverage_fraction": 1.0,
        "final_fill_pixel_count": fill_pixels,
        "seed": seed,
        "min_coverage_percent": min_coverage,
        "coverage_crop": list(coverage_crop) if coverage_crop is not None else None,
    }


def run_video(
    output_dir: Path,
    *,
    duration: float = 0.5,
    order: str = "random",
    filename: str | None = None,
    min_coverage: float = 0.0,
    log: Callable[[str], None] = print,
) -> dict:
    if duration <= 0:
        raise ValueError("duration must be greater than zero")
    if order not in {"random", "brightness", "center"}:
        raise ValueError(f"Unknown video order: {order}")

    data, frames = _load_alignment_data(output_dir)
    log(f"Loaded alignment: {len(frames)} images")
    ordered_frames = order_rendered_frames(
        frames,
        order,
        reference_size=data["reference_size"],
        crop=data["crop"],
        crop_rect=data["crop_rect"],
        fit_origin=data["fit_origin"],
        output_size=data["output_size"],
    )
    coverage_crop = _coverage_crop_for_output(data, frames, min_coverage, log=log)
    video_crop = coverage_crop
    if video_crop is not None:
        x, y, width, height = video_crop
        even_width = width - (width % 2)
        even_height = height - (height % 2)
        if even_width <= 0 or even_height <= 0:
            log(
                "Warning: coverage crop is smaller than the minimum even video frame size; "
                "leaving the original canvas unchanged."
            )
            video_crop = None
        elif (even_width, even_height) != (width, height):
            video_crop = (x, y, even_width, even_height)
            log(
                f"Video crop adjusted to even dimensions: x={x}, y={y}, "
                f"w={even_width}, h={even_height}"
            )
    log(f"Video order: {order} ({len(ordered_frames)} images)")
    log("Creating video...")
    output_path = _generated_output_path(
        output_dir, filename, suffix=".mp4", default_stem=f"video_{order}"
    )
    create_video(
        [frame.output_path for frame in ordered_frames],
        output_path,
        duration=duration,
        crop_rect=video_crop,
    )
    log(f"Video: {output_path.name} ({duration:g}s per image)")
    log("Done.")
    return {
        "operation": "video",
        "output": str(output_path.relative_to(output_dir)),
        "images": len(ordered_frames),
        "order": order,
        "duration_seconds_per_image": duration,
        "min_coverage_percent": min_coverage,
        "coverage_crop": list(video_crop) if video_crop is not None else None,
    }
