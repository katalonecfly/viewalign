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
    assert parser.parse_args(["average", "--input", "out"]).command == "average"

    mosaic = parser.parse_args(["mosaic", "--input", "out", "--polygons", "80", "--seed", "42"])
    assert mosaic.command == "mosaic"
    assert mosaic.polygons == 80
    assert mosaic.seed == 42

    video = parser.parse_args(["video", "--input", "out", "--duration", "0.75", "--order", "brightness"])
    assert video.command == "video"
    assert video.duration == 0.75
    assert video.order == "brightness"


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


def test_upright_modes() -> None:
    parser = build_parser()
    for mode in ("auto", "none"):
        args = parser.parse_args([
            "align", "--input", "photos", "--output", "out", "--upright", mode,
        ])
        assert args.upright == mode


def test_version_is_0_6_0() -> None:
    from viewalign import __version__

    assert __version__ == "0.6.0"
