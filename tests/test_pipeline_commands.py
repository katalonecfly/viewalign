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
    assert average_result["output"] == "average.jpg"
    assert (output / "average.jpg").exists()

    mosaic_result = run_mosaic(output, polygon_count=1, polygon_size=0.5, seed=123, log=lambda _: None)
    assert mosaic_result["output"] == "mosaic.jpg"
    assert mosaic_result["polygon_size"] == 0.5
    assert mosaic_result["coverage_fraction"] == 1.0
    assert (output / "mosaic.jpg").exists()

    created = {}
    def fake_video(paths, destination, duration):
        created["paths"] = paths
        created["destination"] = destination
        created["duration"] = duration

    monkeypatch.setattr("viewalign.pipeline.create_video", fake_video)
    video_result = run_video(output, duration=0.25, order="center", log=lambda _: None)
    assert video_result["output"] == "video_center.mp4"
    assert created["duration"] == 0.25
    assert created["destination"].name == "video_center.mp4"
    assert [Path(path).name for path in created["paths"]] == ["frame_0.jpg", "frame_1.jpg", "frame_2.jpg"]
