import numpy as np

from viewalign.upright import apply_global_rotation, estimate_rotation_from_angles


def _angle(degrees: float) -> float:
    return float(np.deg2rad(degrees))


def test_estimate_rotation_from_horizontal_and_vertical_lines() -> None:
    angles = [_angle(3.0), _angle(93.0), _angle(-2.0), _angle(88.0)]
    estimate, spread, count = estimate_rotation_from_angles(angles)

    assert estimate is not None
    assert abs(np.rad2deg(estimate) - 0.75) < 0.5
    assert np.rad2deg(spread) < 3.0
    assert count == 4


def test_estimate_rotation_ignores_diagonal_lines() -> None:
    angles = [_angle(3.0), _angle(93.0), _angle(40.0), _angle(-40.0)]
    estimate, _, count = estimate_rotation_from_angles(angles)

    assert estimate is not None
    assert abs(np.rad2deg(estimate) - 3.0) < 0.5
    assert count == 2


def test_estimate_rotation_returns_none_without_axis_like_lines() -> None:
    angles = [_angle(25.0), _angle(-25.0), _angle(40.0)]
    estimate, spread, count = estimate_rotation_from_angles(angles)

    assert estimate is None
    assert not np.isfinite(spread)
    assert count == 0


def test_apply_global_rotation_rotates_all_transforms_about_center() -> None:
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    corrected = apply_global_rotation([identity], 5.0, center=(50.0, 40.0))[0]

    center = np.array([[[50.0, 40.0]]], dtype=np.float64)
    transformed = np.array([
        corrected[0, 0] * 50.0 + corrected[0, 1] * 40.0 + corrected[0, 2],
        corrected[1, 0] * 50.0 + corrected[1, 1] * 40.0 + corrected[1, 2],
    ])
    assert np.allclose(transformed, center.reshape(2), atol=1e-8)

    expected = np.array([
        [np.cos(np.deg2rad(5.0)), np.sin(np.deg2rad(5.0)), 0.0],
        [-np.sin(np.deg2rad(5.0)), np.cos(np.deg2rad(5.0)), 0.0],
    ])
    assert np.allclose(corrected[:, :2], expected[:, :2], atol=1e-8)


def test_estimate_upright_detects_tilted_scene_lines(tmp_path) -> None:
    import cv2

    base = np.zeros((500, 700, 3), dtype=np.uint8)
    for y in (80, 180, 280, 380):
        cv2.line(base, (70, y), (630, y), (255, 255, 255), 4)
    for x in (130, 300, 470, 600):
        cv2.line(base, (x, 40), (x, 440), (255, 255, 255), 4)

    rotation = cv2.getRotationMatrix2D((350, 250), 5.0, 1.0)
    tilted = cv2.warpAffine(base, rotation, (700, 500), borderValue=(0, 0, 0))
    path = tmp_path / "tilted.jpg"

    from viewalign.upright import estimate_upright

    assert cv2.imwrite(str(path), tilted)
    estimate = estimate_upright(
        {path: tilted},
        {path: np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)},
        {path: 1.0},
        min_supporting_images=1,
        min_supporting_lines=4,
    )

    assert estimate.applied
    assert estimate.confidence in {"medium", "high"}
    assert abs(estimate.rotation_degrees + 5.0) < 1.0


def test_line_segments_accepts_flat_opencv_line_array(monkeypatch) -> None:
    import cv2
    from viewalign.upright import _line_segments

    class FakeDetector:
        def detect(self, gray):
            return np.array([[10.0, 20.0, 100.0, 20.0], [30.0, 40.0, 30.0, 140.0]], dtype=np.float32), None, None

    monkeypatch.setattr(cv2, "createLineSegmentDetector", lambda *_args: FakeDetector())
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    segments = _line_segments(image, max_lines=10, min_length=20.0)

    assert len(segments) == 2
    assert all(np.isfinite(length) for _, _, length in segments)
    assert {round(length, 6) for _, _, length in segments} == {90.0, 100.0}
