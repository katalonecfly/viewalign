from pathlib import Path

from viewalign.ordering import order_images


def test_filename_ordering() -> None:
    paths = [Path("b.jpg"), Path("A.jpg"), Path("c.png")]
    ordered, missing = order_images(paths, "filename")
    assert [p.name for p in ordered] == ["A.jpg", "b.jpg", "c.png"]
    assert missing == []
