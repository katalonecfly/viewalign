from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass
class AlignmentResult:
    success: bool
    transform: np.ndarray | None
    feature_count: int
    match_count: int
    inlier_count: int
    inlier_ratio: float
    reprojection_error: float | None
    rotation_degrees: float | None
    scale: float | None
    translation_x: float | None
    translation_y: float | None
    reason: str | None = None
    good_matches: list[cv2.DMatch] | None = None
    keypoints_reference: list[cv2.KeyPoint] | None = None
    keypoints_image: list[cv2.KeyPoint] | None = None


@dataclass(frozen=True)
class SimilarityEdge:
    """A successful source->target similarity measurement in full-resolution pixels."""

    source: int
    target: int
    transform: np.ndarray
    weight: float
    result: AlignmentResult


class SIFTAligner:
    def __init__(self, ratio_threshold: float = 0.75, ransac_threshold: float = 4.0,
                 min_matches: int = 8) -> None:
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
        self.min_matches = min_matches
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def describe(self, image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        return self.sift.detectAndCompute(self._gray(image), None)

    def align_described(
        self,
        reference_keypoints: list[cv2.KeyPoint],
        reference_descriptors: np.ndarray | None,
        image_keypoints: list[cv2.KeyPoint],
        image_descriptors: np.ndarray | None,
    ) -> AlignmentResult:
        feature_count = len(image_keypoints)
        if reference_descriptors is None or len(reference_keypoints) < 2:
            return AlignmentResult(
                False, None, feature_count, 0, 0, 0.0, None, None, None, None, None,
                reason="reference has insufficient features",
                keypoints_reference=reference_keypoints,
                keypoints_image=image_keypoints,
            )
        if image_descriptors is None or len(image_keypoints) < 2:
            return AlignmentResult(
                False, None, feature_count, 0, 0, 0.0, None, None, None, None, None,
                reason="insufficient features",
                keypoints_reference=reference_keypoints,
                keypoints_image=image_keypoints,
            )

        knn_matches = self.matcher.knnMatch(image_descriptors, reference_descriptors, k=2)
        good: list[cv2.DMatch] = []
        for pair in knn_matches:
            if len(pair) != 2:
                continue
            first, second = pair
            if first.distance < self.ratio_threshold * second.distance:
                good.append(first)

        match_count = len(good)
        if match_count < self.min_matches:
            return AlignmentResult(
                False, None, feature_count, match_count, 0, 0.0, None, None, None, None, None,
                reason=f"only {match_count} good matches (need {self.min_matches})",
                good_matches=good,
                keypoints_reference=reference_keypoints,
                keypoints_image=image_keypoints,
            )

        src = np.float32([image_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([reference_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            maxIters=5000,
            confidence=0.99,
            refineIters=20,
        )

        if matrix is None or inlier_mask is None:
            return AlignmentResult(
                False, None, feature_count, match_count, 0, 0.0, None, None, None, None, None,
                reason="could not estimate a similarity transform",
                good_matches=good,
                keypoints_reference=reference_keypoints,
                keypoints_image=image_keypoints,
            )

        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = int(inliers.sum())
        inlier_ratio = inlier_count / match_count if match_count else 0.0

        src_inliers = src.reshape(-1, 2)[inliers]
        dst_inliers = dst.reshape(-1, 2)[inliers]
        predicted = cv2.transform(src_inliers.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        errors = np.linalg.norm(predicted - dst_inliers, axis=1)
        reprojection_error = float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None

        a, b = float(matrix[0, 0]), float(matrix[0, 1])
        estimated_scale = float(np.hypot(a, b))
        rotation_degrees = float(np.degrees(np.arctan2(b, a)))
        translation_x = float(matrix[0, 2])
        translation_y = float(matrix[1, 2])

        # SIFT/RANSAC can occasionally find a numerically degenerate
        # similarity from a small, pathological inlier set. Such a transform
        # must not enter the global graph because it cannot be inverted safely.
        transform_linear = matrix[:, :2]
        determinant = float(np.linalg.det(transform_linear))
        if (
            not np.isfinite(estimated_scale)
            or estimated_scale <= 1e-6
            or not np.isfinite(determinant)
            or abs(determinant) <= 1e-12
            or not np.all(np.isfinite(matrix))
        ):
            return AlignmentResult(
                False, matrix, feature_count, match_count, inlier_count, inlier_ratio,
                reprojection_error, rotation_degrees, estimated_scale,
                translation_x, translation_y,
                reason="estimated similarity transform is numerically degenerate",
                good_matches=good,
                keypoints_reference=reference_keypoints,
                keypoints_image=image_keypoints,
            )

        if inlier_count < self.min_matches:
            return AlignmentResult(
                False, matrix, feature_count, match_count, inlier_count, inlier_ratio,
                reprojection_error, rotation_degrees, estimated_scale,
                translation_x, translation_y,
                reason=f"only {inlier_count} RANSAC inliers (need {self.min_matches})",
                good_matches=good,
                keypoints_reference=reference_keypoints,
                keypoints_image=image_keypoints,
            )

        return AlignmentResult(
            True, matrix, feature_count, match_count, inlier_count, inlier_ratio,
            reprojection_error, rotation_degrees, estimated_scale,
            translation_x, translation_y,
            good_matches=good,
            keypoints_reference=reference_keypoints,
            keypoints_image=image_keypoints,
        )

    def align(
        self,
        reference_keypoints: list[cv2.KeyPoint],
        reference_descriptors: np.ndarray,
        image: np.ndarray,
    ) -> AlignmentResult:
        keypoints, descriptors = self.describe(image)
        return self.align_described(
            reference_keypoints,
            reference_descriptors,
            keypoints,
            descriptors,
        )


def scale_affine_to_full_resolution(matrix: np.ndarray, source_scale: float,
                                    reference_scale: float) -> np.ndarray:
    """Convert a working-resolution source->reference affine transform to original pixels."""
    src_scale = np.array([[source_scale, 0.0, 0.0], [0.0, source_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ref_inv = np.array([[1.0 / reference_scale, 0.0, 0.0], [0.0, 1.0 / reference_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    work = np.vstack([matrix, [0.0, 0.0, 1.0]])
    full = ref_inv @ work @ src_scale
    return full[:2, :]


def scale_affine_to_working_resolution(matrix: np.ndarray, source_scale: float,
                                       reference_scale: float) -> np.ndarray:
    """Convert a full-resolution source->reference transform to working pixels."""
    src_inv = np.array([[1.0 / source_scale, 0.0, 0.0], [0.0, 1.0 / source_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ref_scale = np.array([[reference_scale, 0.0, 0.0], [0.0, reference_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    full = np.vstack([matrix, [0.0, 0.0, 1.0]])
    work = ref_scale @ full @ src_inv
    return work[:2, :]


def _decompose_similarity(matrix: np.ndarray) -> tuple[float, float, np.ndarray]:
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    scale = float(np.hypot(a, b))
    if scale <= 1e-12:
        raise ValueError("Similarity transform has zero scale")
    theta = float(np.arctan2(b, a))
    translation = np.array([float(matrix[0, 2]), float(matrix[1, 2])], dtype=np.float64)
    return float(np.log(scale)), theta, translation


def _compose_similarity(log_scale: float, theta: float, translation: np.ndarray) -> np.ndarray:
    scale = float(np.exp(log_scale))
    a = scale * float(np.cos(theta))
    b = scale * float(np.sin(theta))
    return np.array(
        [[a, b, float(translation[0])], [-b, a, float(translation[1])]],
        dtype=np.float64,
    )


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _weighted_lstsq(
    rows: list[tuple[dict[int, float], float, float]],
    variable_count: int,
) -> np.ndarray:
    if not rows:
        return np.zeros(variable_count, dtype=np.float64)
    matrix = np.zeros((len(rows), variable_count), dtype=np.float64)
    values = np.zeros(len(rows), dtype=np.float64)
    weights = np.ones(len(rows), dtype=np.float64)
    for row_index, (coefficients, value, weight) in enumerate(rows):
        for variable, coefficient in coefficients.items():
            matrix[row_index, variable] = coefficient
        values[row_index] = value
        weights[row_index] = max(float(weight), 1e-8)
    weighted_matrix = matrix * np.sqrt(weights)[:, None]
    weighted_values = values * np.sqrt(weights)
    solution, *_ = np.linalg.lstsq(weighted_matrix, weighted_values, rcond=None)
    return solution


def _solve_relative_scalar(
    node_count: int,
    anchor: int,
    edges: Sequence[SimilarityEdge],
    attribute: int,
) -> np.ndarray:
    variable_nodes = [index for index in range(node_count) if index != anchor]
    variable_map = {node: index for index, node in enumerate(variable_nodes)}
    rows: list[tuple[dict[int, float], float, float]] = []

    for edge in edges:
        log_scale, theta, _ = _decompose_similarity(edge.transform)
        value = log_scale if attribute == 0 else _wrap_angle(theta)
        coefficients: dict[int, float] = {}
        if edge.source != anchor:
            coefficients[variable_map[edge.source]] = 1.0
        if edge.target != anchor:
            coefficients[variable_map[edge.target]] = coefficients.get(variable_map[edge.target], 0.0) - 1.0
        rows.append((coefficients, value, edge.weight))

    values = _weighted_lstsq(rows, len(variable_nodes))
    result = np.zeros(node_count, dtype=np.float64)
    for node, variable in variable_map.items():
        result[node] = values[variable]
    return result


def _solve_relative_translation(
    node_count: int,
    anchor: int,
    edges: Sequence[SimilarityEdge],
    log_scales: np.ndarray,
    thetas: np.ndarray,
) -> np.ndarray:
    variable_nodes = [index for index in range(node_count) if index != anchor]
    variable_map = {node: index for index, node in enumerate(variable_nodes)}
    translations = np.zeros((node_count, 2), dtype=np.float64)

    for axis in range(2):
        rows: list[tuple[dict[int, float], float, float]] = []
        for edge in edges:
            _, _, edge_translation = _decompose_similarity(edge.transform)
            target_matrix = _compose_similarity(log_scales[edge.target], thetas[edge.target], np.zeros(2))
            rhs = target_matrix[:, :2] @ edge_translation
            coefficients: dict[int, float] = {}
            if edge.source != anchor:
                coefficients[variable_map[edge.source]] = 1.0
            if edge.target != anchor:
                coefficients[variable_map[edge.target]] = coefficients.get(variable_map[edge.target], 0.0) - 1.0
            rows.append((coefficients, float(rhs[axis]), edge.weight))

        values = _weighted_lstsq(rows, len(variable_nodes))
        for node, variable in variable_map.items():
            translations[node, axis] = values[variable]

    return translations


def _normalize_edge_transform(
    matrix: np.ndarray,
    source_scale: float,
    target_scale: float,
) -> np.ndarray:
    """Express a full-resolution pairwise transform in per-image normalized coordinates."""
    if source_scale <= 0.0 or target_scale <= 0.0:
        raise ValueError("Node coordinate scales must be positive")
    normalized = np.asarray(matrix, dtype=np.float64).copy()
    normalized[:, :2] *= source_scale / target_scale
    normalized[:, 2] /= target_scale
    return normalized


def filter_global_similarity_edges(
    edges: Sequence[SimilarityEdge],
    node_scales: Sequence[float] | None = None,
) -> list[SimilarityEdge]:
    """Filter pairwise edges that are implausible in normalized image coordinates.

    The pairwise matcher can occasionally find an apparently valid RANSAC solution with
    an extreme scale. Such an edge is especially dangerous for a global graph because
    scale errors propagate through every connected image. Normalizing by image size makes
    a same-scene edge's scale close to one, independent of the source/target resolutions.
    """
    normalized_coordinates = node_scales is not None
    if node_scales is None:
        node_scales = [1.0] * (max((max(e.source, e.target) for e in edges), default=-1) + 1)
    scales = [float(value) for value in node_scales]
    if any(value <= 0.0 or not np.isfinite(value) for value in scales):
        raise ValueError("Node coordinate scales must be finite and positive")

    filtered: list[SimilarityEdge] = []
    for edge in edges:
        if edge.source >= len(scales) or edge.target >= len(scales):
            continue
        matrix = np.asarray(edge.transform, dtype=np.float64)
        if matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
            continue
        try:
            normalized = _normalize_edge_transform(
                matrix,
                scales[edge.source],
                scales[edge.target],
            )
            log_scale, _, translation = _decompose_similarity(normalized)
        except (ValueError, FloatingPointError):
            continue
        normalized_scale = float(np.exp(log_scale))
        # Same-view photographs should not require orders of magnitude of scale change
        # after accounting for their pixel dimensions. This intentionally leaves a wide
        # range for different camera resolutions/crops while removing catastrophic RANSAC
        # outliers that can make the global canvas enormous.
        if normalized_coordinates:
            if not 0.1 <= normalized_scale <= 10.0:
                continue
            if float(np.linalg.norm(translation)) > 10.0:
                continue
        filtered.append(edge)
    return filtered


def _edge_residual(
    edge: SimilarityEdge,
    transforms: Sequence[np.ndarray],
) -> float:
    """Return a dimensionless residual for a normalized similarity edge."""
    source_log_scale, source_theta, source_translation = _decompose_similarity(transforms[edge.source])
    target_log_scale, target_theta, target_translation = _decompose_similarity(transforms[edge.target])
    edge_log_scale, edge_theta, edge_translation = _decompose_similarity(edge.transform)

    target_linear = _compose_similarity(target_log_scale, target_theta, np.zeros(2))[:, :2]
    predicted_translation = target_linear @ edge_translation + target_translation
    scale_residual = source_log_scale - (target_log_scale + edge_log_scale)
    angle_residual = _wrap_angle(source_theta - (target_theta + edge_theta))
    translation_residual = float(np.linalg.norm(source_translation - predicted_translation))
    # Global graph coordinates are normalized to roughly one image diagonal. Keep all
    # three residual terms on comparable dimensionless scales.
    normalized_translation = translation_residual / max(0.5, np.linalg.norm(transforms[edge.source][:, :2]))
    return float(np.sqrt(
        scale_residual * scale_residual
        + angle_residual * angle_residual
        + normalized_translation * normalized_translation
    ))


def solve_global_similarity_transforms(
    node_count: int,
    edges: Sequence[SimilarityEdge],
    anchor: int,
    node_scales: Sequence[float] | None = None,
) -> list[np.ndarray]:
    """Solve one full-resolution similarity transform per image from pairwise measurements.

    Pairwise measurements are first normalized by each image's characteristic pixel scale.
    The global solve then operates in a stable, roughly unit-sized coordinate system and is
    converted back to the anchor image's pixel coordinates at the end.
    """
    if node_count <= 0:
        raise ValueError("node_count must be positive")
    if not 0 <= anchor < node_count:
        raise ValueError("anchor index out of range")
    if not edges:
        raise ValueError("At least one successful pairwise alignment is required")

    if node_scales is None:
        scales = np.ones(node_count, dtype=np.float64)
    else:
        scales = np.asarray(node_scales, dtype=np.float64)
        if scales.shape != (node_count,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("node_scales must contain one finite positive value per image")

    valid_edges = filter_global_similarity_edges(edges, scales if node_scales is not None else None)
    if not valid_edges:
        raise ValueError("No numerically plausible pairwise alignments are available")

    normalized_edges: list[SimilarityEdge] = []
    for edge in valid_edges:
        normalized_edges.append(SimilarityEdge(
            edge.source,
            edge.target,
            _normalize_edge_transform(edge.transform, scales[edge.source], scales[edge.target]),
            float(max(edge.weight, 1e-8)),
            edge.result,
        ))

    working_edges = normalized_edges
    normalized_transforms: list[np.ndarray] = [
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        for _ in range(node_count)
    ]

    # A few IRLS passes are enough because the graph is dense. Normalize before the
    # first solve so one pathological full-resolution scale cannot poison the system.
    for _ in range(6):
        log_scales = _solve_relative_scalar(node_count, anchor, working_edges, attribute=0)
        thetas = _solve_relative_scalar(node_count, anchor, working_edges, attribute=1)
        translations = _solve_relative_translation(node_count, anchor, working_edges, log_scales, thetas)
        normalized_transforms = [
            _compose_similarity(log_scales[index], thetas[index], translations[index])
            for index in range(node_count)
        ]

        residuals = np.array([_edge_residual(edge, normalized_transforms) for edge in working_edges], dtype=np.float64)
        finite = residuals[np.isfinite(residuals)]
        if len(finite) == 0:
            break
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        scale = max(0.02, median + 3.0 * mad)
        if not np.isfinite(scale) or scale <= 0.0:
            break

        new_edges: list[SimilarityEdge] = []
        for edge, residual in zip(normalized_edges, residuals):
            if not np.isfinite(residual):
                robust = 1e-6
            elif residual <= scale:
                robust = 1.0
            else:
                robust = scale / residual
            new_edges.append(SimilarityEdge(
                edge.source,
                edge.target,
                edge.transform,
                max(edge.weight * robust, 1e-8),
                edge.result,
            ))
        working_edges = new_edges

    normalized_transforms[anchor] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    # Convert from normalized image coordinates back to the anchor image's full-resolution
    # pixel coordinate system. Anchor itself stays exactly identity.
    anchor_scale = float(scales[anchor])
    transforms: list[np.ndarray] = []
    for index, normalized in enumerate(normalized_transforms):
        if index == anchor:
            transforms.append(normalized.copy())
            continue
        matrix = normalized.copy()
        matrix[:, :2] *= anchor_scale / float(scales[index])
        matrix[:, 2] *= anchor_scale
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Global alignment produced a non-finite transform")
        transforms.append(matrix)
    return transforms

