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


from viewalign.alignment import (
    AlignmentResult,
    SimilarityEdge,
    filter_global_similarity_edges,
    solve_global_similarity_transforms,
)


def _similarity(scale: float, degrees: float, tx: float, ty: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    a = scale * np.cos(theta)
    b = scale * np.sin(theta)
    return np.array([[a, b, tx], [-b, a, ty]], dtype=np.float64)


def _inverse(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.vstack([matrix, [0.0, 0.0, 1.0]]))[:2, :]


def _identity_result() -> AlignmentResult:
    return AlignmentResult(True, np.eye(2, 3), 20, 20, 20, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def test_global_similarity_solver_recovers_consistent_transforms() -> None:
    expected = [
        _similarity(1.0, 0.0, 0.0, 0.0),
        _similarity(1.10, 4.0, 100.0, 30.0),
        _similarity(0.92, -3.0, -40.0, 70.0),
        _similarity(1.05, 7.0, 60.0, -50.0),
    ]
    identity_result = _identity_result()
    edges: list[SimilarityEdge] = []
    for source in range(len(expected)):
        for target in range(source + 1, len(expected)):
            source_to_target = _inverse(expected[target]) @ np.vstack([expected[source], [0.0, 0.0, 1.0]])
            edges.append(SimilarityEdge(source, target, source_to_target, 1.0, identity_result))

    actual = solve_global_similarity_transforms(len(expected), edges, anchor=0)

    for want, got in zip(expected, actual):
        assert np.allclose(got, want, atol=1e-6)


def test_global_similarity_solver_uses_multiple_edges_to_reduce_noise() -> None:
    expected = [
        _similarity(1.0, 0.0, 0.0, 0.0),
        _similarity(1.0, 2.0, 50.0, 0.0),
        _similarity(1.0, 4.0, 100.0, 0.0),
    ]
    result = _identity_result()
    edges: list[SimilarityEdge] = []
    for source, target in ((0, 1), (1, 2), (0, 2)):
        exact = _inverse(expected[target]) @ np.vstack([expected[source], [0.0, 0.0, 1.0]])
        if (source, target) == (0, 2):
            exact = exact.copy()
            exact[0, 2] += 12.0
        edges.append(SimilarityEdge(source, target, exact, 1.0, result))

    actual = solve_global_similarity_transforms(3, edges, anchor=0)

    assert abs(actual[2][0, 2] - expected[2][0, 2]) < 8.0


def test_global_pairwise_graph_keeps_only_connected_component_helpers() -> None:
    from viewalign.pipeline import _global_components

    result = _identity_result()
    edge = np.eye(2, 3)
    edges = [
        SimilarityEdge(0, 1, edge, 1.0, result),
        SimilarityEdge(1, 2, edge, 1.0, result),
        SimilarityEdge(3, 4, edge, 1.0, result),
    ]
    components = _global_components(6, edges)
    assert {frozenset(component) for component in components} == {
        frozenset({0, 1, 2}),
        frozenset({3, 4}),
        frozenset({5}),
    }


def test_global_similarity_solver_skips_singular_edges() -> None:
    result = _identity_result()
    good = SimilarityEdge(0, 1, np.eye(2, 3), 1.0, result)
    singular = SimilarityEdge(1, 2, np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 20.0]]), 1.0, result)

    actual = solve_global_similarity_transforms(3, [good, singular], anchor=0)

    assert np.allclose(actual[0], np.eye(2, 3))
    assert np.allclose(actual[1], np.eye(2, 3))
    assert np.all(np.isfinite(actual[2]))


def test_global_filter_rejects_catastrophic_normalized_scale() -> None:
    result = _identity_result()
    extreme = SimilarityEdge(
        0,
        1,
        _similarity(1e-5, 0.0, 10.0, -5.0),
        1.0,
        result,
    )

    filtered = filter_global_similarity_edges([extreme], [1000.0, 1000.0])

    assert filtered == []


def test_global_solver_denormalizes_to_anchor_pixels() -> None:
    result = _identity_result()
    expected = _similarity(0.8, 3.0, 250.0, -120.0)
    edge = SimilarityEdge(0, 1, _inverse(expected), 1.0, result)

    actual = solve_global_similarity_transforms(2, [edge], anchor=0, node_scales=[1000.0, 1250.0])

    assert np.allclose(actual[0], np.eye(2, 3), atol=1e-8)
    assert np.allclose(actual[1], expected, atol=1e-6)
