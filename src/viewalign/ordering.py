from __future__ import annotations

from pathlib import Path

from .images import image_timestamp


def order_images(paths: list[Path], mode: str) -> tuple[list[Path], list[Path]]:
    """Return (ordered paths, files missing a timestamp)."""
    if mode == "filename":
        return sorted(paths, key=lambda p: p.name.lower()), []

    missing: list[Path] = []
    keyed: list[tuple[str, str, Path]] = []
    for path in paths:
        timestamp = image_timestamp(path)
        if timestamp is None:
            missing.append(path)
            # Keep missing files deterministic and after timestamped files.
            keyed.append(("~", path.name.lower(), path))
        else:
            keyed.append((timestamp[1], path.name.lower(), path))

    keyed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in keyed], missing
