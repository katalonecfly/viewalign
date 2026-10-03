from pathlib import Path
import json

import cv2
import numpy as np

from viewalign.derived import RenderedFrame
from viewalign.pipeline import _load_alignment_data, run_average, run_mosaic, run_video


def _write_manifest(tmp_path: Path, values=(50, 100, 200)) -> Path:
    aligned = tmp_path / "aligned"
    aligned.mkdir()
    frames = []
    for index, value in enumerate(values):
        name = f"frame_{index}.jpg"
        image = np.full((20, 30, 3), value, dtype=np.uint8)
        path = aligned / name
        assert cv2.imwrite(str(path), image)
        frames.append({
            "source": name,
            "output": f"aligned/{name}",
            "source_width": 30,
            "source_height": 20,
            "transform_full_resolution": [[1.0, 0.0, float(index)], [0.0, 1.0, 0.0]],
        })

    manifest = {
        "version": 1,
        "reference": "frame_0.jpg",
        "reference_mode": "first",
        "order": "filename",
        "working_size": 1600,
        "crop": "none",
        "jpeg_quality": 95,
        "reference_size": [30, 20],
        "output_size": [30, 20],
        "crop_rect": None,
        "fit_origin": [0.0, 0.0],
        "frames": frames,
    }
    (tmp_path / "alignment.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_alignment_manifest_roundtrip(tmp_path: Path) -> None:
    output = _write_manifest(tmp_path)
    data, frames = _load_alignment_data(output)
    assert data["output_size"] == (30, 20)
    assert len(frames) == 3
    assert frames[1].source_width == 30
    assert frames[2].transform_full[0, 2] == 2.0


def test_derived_commands_consume_alignment_outputs(tmp_path: Path, monkeypatch) -> None:
    output = _write_manifest(tmp_path)

    average_result = run_average(output, log=lambda _: None)
    assert average_result["output"].startswith("generated/")
    assert average_result["output"].endswith("_average.jpg")
    assert (output / average_result["output"]).exists()

    mosaic_result = run_mosaic(output, polygon_count=1, polygon_size=0.5, seed=123, log=lambda _: None)
    assert mosaic_result["output"].startswith("generated/")
    assert mosaic_result["output"].endswith("_mosaic.jpg")
    assert mosaic_result["polygon_size"] == 0.5
    assert mosaic_result["coverage_fraction"] == 1.0
    assert (output / mosaic_result["output"]).exists()

    created = {}
    def fake_video(paths, destination, duration, crop_rect=None):
        created["paths"] = paths
        created["destination"] = destination
        created["duration"] = duration
        created["crop_rect"] = crop_rect

    monkeypatch.setattr("viewalign.pipeline.create_video", fake_video)
    video_result = run_video(output, duration=0.25, order="center", log=lambda _: None)
    assert video_result["output"].startswith("generated/")
    assert video_result["output"].endswith("_video_center.mp4")
    assert created["duration"] == 0.25
    assert created["destination"].name.endswith("_video_center.mp4")
    assert created["crop_rect"] is None
    assert [Path(path).name for path in created["paths"]] == ["frame_0.jpg", "frame_1.jpg", "frame_2.jpg"]


def test_derived_commands_apply_min_coverage_crop(tmp_path: Path, monkeypatch) -> None:
    output = _write_manifest(tmp_path)

    logs = []
    average_result = run_average(output, min_coverage=100.0, log=logs.append)
    assert average_result["coverage_crop"] == [2, 0, 28, 20]
    image = cv2.imread(str(output / average_result["output"]), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape[:2] == (20, 28)
    assert any("Coverage crop: x=2, y=0, w=28, h=20" in line for line in logs)

    created = {}
    def fake_video(paths, destination, duration, crop_rect=None):
        created["crop_rect"] = crop_rect

    monkeypatch.setattr("viewalign.pipeline.create_video", fake_video)
    run_video(output, min_coverage=100.0, order="random", log=lambda _: None)
    assert created["crop_rect"] == (2, 0, 28, 20)


def test_derived_commands_custom_filenames_go_into_generated(tmp_path: Path, monkeypatch) -> None:
    output = _write_manifest(tmp_path)

    average_result = run_average(output, filename="my_average", log=lambda _: None)
    assert average_result["output"] == "generated/my_average.jpg"
    assert (output / "generated/my_average.jpg").exists()

    mosaic_result = run_mosaic(output, polygon_count=1, filename="my_mosaic.jpg", seed=1, log=lambda _: None)
    assert mosaic_result["output"] == "generated/my_mosaic.jpg"
    assert (output / "generated/my_mosaic.jpg").exists()

    created = {}
    def fake_video(paths, destination, duration, crop_rect=None):
        created["destination"] = destination

    monkeypatch.setattr("viewalign.pipeline.create_video", fake_video)
    video_result = run_video(output, filename="my_video.mp4", order="brightness", log=lambda _: None)
    assert video_result["output"] == "generated/my_video.mp4"
    assert created["destination"].as_posix().endswith("generated/my_video.mp4")


def test_derived_commands_reject_wrong_filename_extension(tmp_path: Path) -> None:
    output = _write_manifest(tmp_path)
    import pytest

    with pytest.raises(ValueError, match=r"\.jpg extension"):
        run_average(output, filename="bad.png", log=lambda _: None)
    with pytest.raises(ValueError, match=r"\.mp4 extension"):
        run_video(output, filename="bad.mov", log=lambda _: None)
    with pytest.raises(ValueError, match="simple filename"):
        run_mosaic(output, filename="nested/foo.jpg", log=lambda _: None)
