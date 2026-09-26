from viewalign.cli import build_parser


def test_align_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["align", "--input", "photos", "--output", "out"])
    assert args.reference == "first"
    assert args.order == "filename"
    assert args.crop == "fit"
    assert args.duration == 0.5
    assert not hasattr(args, "ffmpeg_path")
    assert args.mosaic_polygons == 50
    assert args.diagnostics is False


def test_crop_modes() -> None:
    parser = build_parser()
    for crop in ("fit", "none", "intersection"):
        args = parser.parse_args([
            "align", "--input", "photos", "--output", "out", "--crop", crop,
        ])
        assert args.crop == crop
