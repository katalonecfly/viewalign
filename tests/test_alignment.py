import cv2
import numpy as np

from viewalign.alignment import SIFTAligner, scale_affine_to_full_resolution
from viewalign.pipeline import _compute_fit_canvas


def test_scale_affine_to_full_resolution_identity() -> None:
    m = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]])
    scaled = scale_affine_to_full_resolution(m, source_scale=0.5, reference_scale=0.25)
    assert np.allclose(scaled, np.array([[2.0, 0.0, 40.0], [0.0, 2.0, 80.0]]))


def test_sift_alignment_on_synthetic_translation() -> None:
    image = np.zeros((500, 700), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (300, 280), 255, 3)
    cv2.circle(image, (450, 350), 45, 180, 4)
    cv2.putText(image, "VIEWALIGN", (160, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
    reference = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    matrix = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -7.0]], dtype=np.float32)
    shifted_gray = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    shifted = cv2.cvtColor(shifted_gray, cv2.COLOR_GRAY2BGR)

    aligner = SIFTAligner(min_matches=6)
    kp_ref, desc_ref = aligner.describe(reference)
    result = aligner.align(kp_ref, desc_ref, shifted)

    assert result.success
    assert result.transform is not None
    assert abs(result.transform[0, 2] + 12.0) < 2.0
    assert abs(result.transform[1, 2] - 7.0) < 2.0


def test_compute_fit_canvas_contains_transformed_images() -> None:
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    shifted = np.array([[1.0, 0.0, -20.0], [0.0, 1.0, 10.0]], dtype=np.float64)

    width, height, origin_x, origin_y = _compute_fit_canvas([
        (identity, 100, 80),
        (shifted, 100, 80),
    ])

    assert (width, height) == (120, 90)
    assert origin_x == -20.0
    assert origin_y == 0.0
