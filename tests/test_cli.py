from viewalign.cli import build_parser


def test_align_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["align", "--input", "photos", "--output", "out"])
    assert args.command == "align"
    assert args.reference == "first"
    assert args.upright == "auto"
    assert args.order == "filename"
    assert args.crop == "fit"
    assert not hasattr(args, "duration")
    assert not hasattr(args, "video_order")
    assert not hasattr(args, "mosaic_polygons")
    assert not hasattr(args, "mosaic_seed")
    assert args.diagnostics is False


def test_derived_commands() -> None:
    parser = build_parser()
    average = parser.parse_args(["average", "--input", "out"])
    assert average.command == "average"
    assert average.min_coverage == 0.0
    assert average.filename is None

    mosaic = parser.parse_args(["mosaic", "--input", "out", "--polygons", "80", "--seed", "42"])
    assert mosaic.command == "mosaic"
    assert mosaic.filename is None
    assert mosaic.polygons == 80
    assert mosaic.polygon_size == 1.0
    assert mosaic.seed == 42
    assert mosaic.min_coverage == 0.0

    mosaic_sized = parser.parse_args([
        "mosaic", "--input", "out", "--polygons", "80", "--polygon-size", "1.5",
    ])
    assert mosaic_sized.polygon_size == 1.5

    video = parser.parse_args(["video", "--input", "out", "--duration", "0.75", "--order", "brightness"])
    assert video.command == "video"
    assert video.filename is None
    assert video.duration == 0.75
    assert video.order == "brightness"
    assert video.min_coverage == 0.0

    custom = parser.parse_args([
        "average", "--input", "out", "--min-coverage", "37.5",
    ])
    assert custom.min_coverage == 37.5


def test_crop_modes() -> None:
    parser = build_parser()
    for crop in ("fit", "none", "intersection"):
        args = parser.parse_args([
            "align", "--input", "photos", "--output", "out", "--crop", crop,
        ])
        assert args.crop == crop


def test_video_order_modes() -> None:
    parser = build_parser()
    for mode in ("random", "brightness", "center"):
        args = parser.parse_args([
            "video", "--input", "out", "--order", mode,
        ])
        assert args.order == mode


def test_global_reference_option() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "align", "--input", "photos", "--output", "out", "--reference", "global",
    ])
    assert args.reference == "global"


def test_min_coverage_parser_range_is_present() -> None:
    parser = build_parser()
    for command in ("average", "mosaic", "video"):
        args = parser.parse_args([command, "--input", "out", "--min-coverage", "100"])
        assert args.min_coverage == 100.0


def test_upright_modes() -> None:
    parser = build_parser()
    for mode in ("auto", "none"):
        args = parser.parse_args([
            "align", "--input", "photos", "--output", "out", "--upright", mode,
        ])
        assert args.upright == mode


def test_version_is_0_9_0() -> None:
    from viewalign import __version__

    assert __version__ == "0.9.1"


def test_derived_filename_parser() -> None:
    parser = build_parser()
    average = parser.parse_args(["average", "--input", "out", "--filename", "avg.jpg"])
    assert average.filename == "avg.jpg"
    mosaic = parser.parse_args(["mosaic", "--input", "out", "--filename", "mosaic-custom"])
    assert mosaic.filename == "mosaic-custom"
    video = parser.parse_args(["video", "--input", "out", "--filename", "video.mp4"])
    assert video.filename == "video.mp4"
