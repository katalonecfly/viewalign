from pathlib import Path
import subprocess

import cv2
import numpy as np
import imageio_ffmpeg

from viewalign.video import _build_concat_manifest, create_video


def test_manifest_repeats_last_file(tmp_path: Path) -> None:
    a = tmp_path / "a one.jpg"
    b = tmp_path / "b.jpg"
    manifest = tmp_path / "frames.txt"
    _build_concat_manifest([a, b], 0.5, manifest)
    text = manifest.read_text(encoding="utf-8")
    assert text.count("file ") == 2
    assert "duration 0.500000000000" in text
    assert text.rstrip().endswith("duration 0.500000000000")


def test_create_video(tmp_path: Path) -> None:
    paths = []
    for index, value in enumerate((30, 200)):
        # Odd dimensions exercise the yuv420p-compatible video padding path.
        image = np.full((31, 33, 3), value, dtype=np.uint8)
        path = tmp_path / f"frame {index}.jpg"
        assert cv2.imwrite(str(path), image)
        paths.append(path)

    output = tmp_path / "video.mp4"
    assert Path(imageio_ffmpeg.get_ffmpeg_exe()).exists()
    create_video(paths, output, duration=0.2)
    assert output.exists()
    assert output.stat().st_size > 0

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=width,height,nb_frames",
            "-of", "default=noprint_wrappers=1", str(output),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    assert "duration=0.400000" in probe
    assert "nb_frames=2" in probe
    assert "width=34" in probe
    assert "height=32" in probe


def test_create_video_applies_crop(tmp_path: Path) -> None:
    paths = []
    for index, value in enumerate((50, 150)):
        image = np.full((40, 60, 3), value, dtype=np.uint8)
        path = tmp_path / f"frame_{index}.jpg"
        assert cv2.imwrite(str(path), image)
        paths.append(path)

    output = tmp_path / "cropped.mp4"
    create_video(paths, output, duration=0.2, crop_rect=(10, 4, 40, 32))
    assert output.exists()

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=width,height",
            "-of", "default=noprint_wrappers=1", str(output),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    assert "width=40" in probe
    assert "height=32" in probe
