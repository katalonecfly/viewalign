# viewalign

A small cross-platform CLI for aligning photos taken from approximately the same viewpoint to a common reference image.

The current version uses SIFT feature matching and a RANSAC-estimated similarity transform (`cv2.estimateAffinePartial2D`). It supports JPEG/PNG input, EXIF orientation normalization, filename/timestamp ordering, different input resolutions/orientations, JPEG output, alignment diagnostics, a black-aware average image, a random polygon mosaic, and FFmpeg video generation with configurable frame ordering.

## Install

```bash
uv sync
```

FFmpeg is provided by the `imageio-ffmpeg` Python dependency, so `uv sync` installs the runtime used by ViewAlign. No separate `--ffmpeg-path` option is needed.

## Usage

```bash
uv run viewalign align --input ./photos --output ./aligned
```

A run creates:

```text
output/
├── aligned/          # transformed JPEG frames, in input order
├── average.jpg       # per-pixel average, ignoring geometric border areas
├── mosaic.jpg        # random polygon collage built from aligned frames
├── video.mp4         # one frame per aligned image, 0.5s per image by default
├── metrics.json
└── summary.json
```

Useful options:

```bash
uv run viewalign align \
  --input ./photos \
  --output ./aligned \
  --reference first \
  --order timestamp \
  --working-size 1600 \
  --crop fit \
  --duration 0.75 \
  --video-order brightness \
  --mosaic-polygons 80 \
  --mosaic-seed 42 \
  --diagnostics
```

The command continues when an individual image cannot be aligned. A `metrics.json` file is written to the output directory.

## Crop modes

`fit` is the default. The output canvas is expanded as necessary so the entire transformed source image is visible, including portions that move outside the reference image bounds. Empty areas are filled with black.

`none` keeps the reference-sized canvas, so an image can be clipped at the edges.

`intersection` crops all outputs to the common valid area shared by the successfully aligned images.

## Average image

`average.jpg` averages the aligned output frames pixel-by-pixel. It uses geometric validity masks produced from the alignment transforms, rather than testing whether pixels are black. This means black content from genuine night images is still included, while black border areas created by warping are ignored.

## Mosaic image

`mosaic.jpg` is an independent composition made from random convex polygon patches from aligned frames. A polygon is only placed when nearly all of its area is valid for the selected source frame, so warped black borders are not copied into the collage.

The number of requested polygons can be changed with `--mosaic-polygons`. Use `--mosaic-seed` to make the random layout reproducible.

## Video

`video.mp4` is always generated after successful alignment. `--duration` controls how long each aligned image is displayed; the default is 0.5 seconds. `--video-order` controls the sequence independently of the input/alignment ordering:

- `random`: shuffle aligned images randomly.
- `brightness`: sort from highest to lowest mean luminance over geometrically valid pixels, so bright daytime/snowy images tend to come first and dark/night images tend to come last.
- `center`: sort by the transformed source-image center from left to right in the final output coordinate system.

FFmpeg's concat demuxer is used with the generated aligned filenames. The output video uses a constant frame rate derived from `--duration` so each image occupies the requested duration.

### FFmpeg

The project depends on `imageio-ffmpeg`. ViewAlign discovers its FFmpeg executable automatically at runtime, so there is no `--ffmpeg-path` CLI option.
