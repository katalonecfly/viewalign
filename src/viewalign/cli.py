from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .pipeline import run_alignment, run_average, run_mosaic, run_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="viewalign",
        description="Align photos taken from approximately the same viewpoint and generate derived outputs.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    align = subparsers.add_parser("align", help="Align a directory of JPEG/PNG photos")
    align.add_argument("--input", type=Path, required=True, help="Input directory containing JPEG/PNG photos")
    align.add_argument("--output", type=Path, required=True, help="Alignment output directory")
    align.add_argument(
        "--reference",
        default="first",
        help="Reference: first, middle, global, or a path to one of the input images",
    )
    align.add_argument(
        "--upright", choices=("auto", "none"), default="auto",
        help="Global-mode upright correction: auto or none (default: auto)",
    )
    align.add_argument(
        "--order", choices=("filename", "timestamp"), default="filename",
        help="Input ordering used during alignment (default: filename)",
    )
    align.add_argument(
        "--working-size", type=int, default=1600,
        help="Maximum dimension used for feature matching (default: 1600)",
    )
    align.add_argument(
        "--crop", choices=("fit", "none", "intersection"), default="fit",
        help="Output crop mode: fit, none, or intersection (default: fit)",
    )
    align.add_argument("--diagnostics", action="store_true", help="Write feature-match and overlay images")
    align.add_argument(
        "--jpeg-quality", type=int, default=95,
        help="Output JPEG quality for aligned images, 1-100 (default: 95)",
    )
    align.add_argument(
        "--min-matches", type=int, default=8,
        help="Minimum good/inlier matches (default: 8)",
    )
    align.add_argument(
        "--ratio-threshold", type=float, default=0.75,
        help="SIFT Lowe ratio threshold (default: 0.75)",
    )
    align.add_argument(
        "--ransac-threshold", type=float, default=4.0,
        help="RANSAC reprojection threshold in working-image pixels (default: 4)",
    )

    average = subparsers.add_parser("average", help="Create average.jpg from an existing alignment")
    average.add_argument("--input", type=Path, required=True, help="Alignment output directory from 'align'")

    mosaic = subparsers.add_parser("mosaic", help="Create mosaic.jpg from an existing alignment")
    mosaic.add_argument("--input", type=Path, required=True, help="Alignment output directory from 'align'")
    mosaic.add_argument(
        "--polygons", type=int, default=50,
        help="Number of random polygons to place (default: 50)",
    )
    mosaic.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible mosaics")

    video = subparsers.add_parser("video", help="Create video.mp4 from an existing alignment")
    video.add_argument("--input", type=Path, required=True, help="Alignment output directory from 'align'")
    video.add_argument(
        "--duration", type=float, default=0.5,
        help="Seconds each aligned image is shown (default: 0.5)",
    )
    video.add_argument(
        "--order", choices=("random", "brightness", "center"), default="random",
        help="Video order: random, brightness, or center (default: random)",
    )
    return parser

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "align":
            if not 1 <= args.jpeg_quality <= 100:
                parser.error("--jpeg-quality must be between 1 and 100")
            if args.working_size <= 0:
                parser.error("--working-size must be greater than zero")
            if not 0 < args.ratio_threshold < 1:
                parser.error("--ratio-threshold must be between 0 and 1")
            if args.ransac_threshold <= 0:
                parser.error("--ransac-threshold must be greater than zero")
            run_alignment(
                input_dir=args.input,
                output_dir=args.output,
                reference=args.reference,
                upright=args.upright,
                order=args.order,
                working_size=args.working_size,
                crop=args.crop,
                diagnostics=args.diagnostics,
                jpeg_quality=args.jpeg_quality,
                min_matches=args.min_matches,
                ratio_threshold=args.ratio_threshold,
                ransac_threshold=args.ransac_threshold,
            )
        elif args.command == "average":
            run_average(args.input)
        elif args.command == "mosaic":
            if args.polygons < 0:
                parser.error("--polygons must be zero or greater")
            run_mosaic(args.input, polygon_count=args.polygons, seed=args.seed)
        elif args.command == "video":
            if args.duration <= 0:
                parser.error("--duration must be greater than zero")
            run_video(args.input, duration=args.duration, order=args.order)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"viewalign: error: {exc}\n")


if __name__ == "__main__":
    main()
