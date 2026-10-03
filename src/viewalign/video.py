from __future__ import annotations

import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import imageio_ffmpeg


def _concat_quote(path: Path) -> str:
    # FFmpeg concat demuxer uses single-quoted file entries. Escape embedded quotes.
    value = path.resolve().as_posix()
    return "'" + value.replace("'", "'\\''") + "'"


def _build_concat_manifest(paths: Sequence[Path], duration: float, manifest: Path) -> None:
    lines: list[str] = []
    for path in paths:
        lines.append(f"file {_concat_quote(path)}")
        lines.append(f"duration {duration:.12f}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_video(
    image_paths: Sequence[Path],
    output_path: Path,
    *,
    duration: float,
    crop_rect: tuple[int, int, int, int] | None = None,
) -> None:
    """Create a video where each input image is displayed for `duration` seconds."""
    if not image_paths:
        raise ValueError("Cannot create a video with no images")
    if duration <= 0:
        raise ValueError("duration must be greater than zero")
    try:
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError as exc:
        raise RuntimeError(
            "FFmpeg is not available. Install the project dependencies with `uv sync`."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="viewalign-ffmpeg-") as temp_dir:
        manifest = Path(temp_dir) / "frames.txt"
        _build_concat_manifest(image_paths, duration, manifest)
        fps = Fraction(1.0 / duration).limit_denominator(100000)
        filters: list[str] = []
        if crop_rect is not None:
            x, y, width, height = (int(value) for value in crop_rect)
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError("Invalid video crop rectangle")
            filters.append(f"crop={width}:{height}:{x}:{y}")
        filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")

        command = [
            executable,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
            "-fps_mode",
            "cfr",
            "-r",
            f"{fps.numerator}/{fps.denominator}",
            "-vf",
            ",".join(filters),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise RuntimeError(f"Could not start FFmpeg: {exc}") from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or "unknown FFmpeg error"
            raise RuntimeError(f"FFmpeg failed: {detail}")
