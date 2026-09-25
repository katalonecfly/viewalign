from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .pipeline import run_alignment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="viewalign",
        description="Align photos taken from approximately the same viewpoint to a common reference.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    align = subparsers.add_parser("align", help="Align a directory of JPEG/PNG photos")
    align.add_argument("--input", type=Path, required=True, help="Input directory")
    align.add_argument("--output", type=Path, required=True, help="Output directory")
    align.add_argument("--reference", default="first",
                       help="Reference: first, middle, or a path to one of the input images")
    align.add_argument("--order", choices=("filename", "timestamp"), default="filename",
                       help="Input ordering (default: filename)")
    align.add_argument("--working-size", type=int, default=1600,
                       help="Maximum dimension used for feature matching (default: 1600)")
    align.add_argument("--crop", choices=("fit", "none", "intersection"), default="fit",
                       help="Output crop mode: fit keeps every aligned image fully visible; none uses the reference canvas; intersection uses the common overlap (default: fit)")
    align.add_argument("--diagnostics", action="store_true",
                       help="Write feature-match and overlay images")
    align.add_argument("--jpeg-quality", type=int, default=95,
                       help="Output JPEG quality, 1-100 (default: 95)")
    align.add_argument("--min-matches", type=int, default=8,
                       help="Minimum good/inlier matches (default: 8)")
    align.add_argument("--ratio-threshold", type=float, default=0.75,
                       help="SIFT Lowe ratio threshold (default: 0.75)")
    align.add_argument("--ransac-threshold", type=float, default=4.0,
                       help="RANSAC reprojection threshold in working-image pixels (default: 4)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "align":
        if not 1 <= args.jpeg_quality <= 100:
            parser.error("--jpeg-quality must be between 1 and 100")
        try:
            run_alignment(
                input_dir=args.input,
                output_dir=args.output,
                reference=args.reference,
                order=args.order,
                working_size=args.working_size,
                crop=args.crop,
                diagnostics=args.diagnostics,
                jpeg_quality=args.jpeg_quality,
                min_matches=args.min_matches,
                ratio_threshold=args.ratio_threshold,
                ransac_threshold=args.ransac_threshold,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            parser.exit(1, f"viewalign: error: {exc}\n")


if __name__ == "__main__":
    main()
