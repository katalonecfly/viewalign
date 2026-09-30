from viewalign.cli import build_parser


def test_align_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["align", "--input", "photos", "--output", "out"])
    assert args.reference == "first"
    assert args.order == "filename"
    assert args.crop == "fit"
    assert args.duration == 0.5
    assert args.video_order == "random"
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


def test_video_order_modes() -> None:
    parser = build_parser()
    for mode in ("random", "brightness", "center"):
        args = parser.parse_args([
            "align", "--input", "photos", "--output", "out", "--video-order", mode,
        ])
        assert args.video_order == mode



def test_global_reference_option() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "align", "--input", "photos", "--output", "out", "--reference", "global",
    ])
    assert args.reference == "global"


def test_version_is_0_4_2() -> None:
    from viewalign import __version__

    assert __version__ == "0.4.2"
