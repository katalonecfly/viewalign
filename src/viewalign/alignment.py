from __future__ import annotations

from dataclasses import dataclass
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

    def align(
        self,
        reference_keypoints: list[cv2.KeyPoint],
        reference_descriptors: np.ndarray,
        image: np.ndarray,
    ) -> AlignmentResult:
        keypoints, descriptors = self.describe(image)
        feature_count = len(keypoints)
        if descriptors is None or len(keypoints) < 2:
            return AlignmentResult(False, None, feature_count, 0, 0, 0.0, None, None, None, None, None,
                                   reason="insufficient features", keypoints_image=keypoints)

        knn_matches = self.matcher.knnMatch(descriptors, reference_descriptors, k=2)
        good: list[cv2.DMatch] = []
        for pair in knn_matches:
            if len(pair) != 2:
                continue
            first, second = pair
            if first.distance < self.ratio_threshold * second.distance:
                good.append(first)

        match_count = len(good)
        if match_count < self.min_matches:
            return AlignmentResult(False, None, feature_count, match_count, 0, 0.0, None, None, None, None, None,
                                   reason=f"only {match_count} good matches (need {self.min_matches})",
                                   good_matches=good, keypoints_reference=reference_keypoints,
                                   keypoints_image=keypoints)

        src = np.float32([keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
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
            return AlignmentResult(False, None, feature_count, match_count, 0, 0.0, None, None, None, None, None,
                                   reason="could not estimate a similarity transform",
                                   good_matches=good, keypoints_reference=reference_keypoints,
                                   keypoints_image=keypoints)

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

        if inlier_count < self.min_matches:
            return AlignmentResult(False, matrix, feature_count, match_count, inlier_count, inlier_ratio,
                                   reprojection_error, rotation_degrees, estimated_scale,
                                   translation_x, translation_y,
                                   reason=f"only {inlier_count} RANSAC inliers (need {self.min_matches})",
                                   good_matches=good, keypoints_reference=reference_keypoints,
                                   keypoints_image=keypoints)

        return AlignmentResult(True, matrix, feature_count, match_count, inlier_count, inlier_ratio,
                               reprojection_error, rotation_degrees, estimated_scale,
                               translation_x, translation_y,
                               good_matches=good, keypoints_reference=reference_keypoints,
                               keypoints_image=keypoints)


def scale_affine_to_full_resolution(matrix: np.ndarray, source_scale: float,
                                    reference_scale: float) -> np.ndarray:
    """Convert a working-resolution source->reference affine transform to original pixels."""
    src_scale = np.array([[source_scale, 0.0, 0.0], [0.0, source_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ref_inv = np.array([[1.0 / reference_scale, 0.0, 0.0], [0.0, 1.0 / reference_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    work = np.vstack([matrix, [0.0, 0.0, 1.0]])
    full = ref_inv @ work @ src_scale
    return full[:2, :]
