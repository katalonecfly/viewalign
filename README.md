# viewalign

A small cross-platform CLI for aligning photos taken from approximately the same viewpoint to a common reference image.

The current version uses SIFT feature matching and a RANSAC-estimated similarity transform (`cv2.estimateAffinePartial2D`). It supports JPEG/PNG input, EXIF orientation normalization, filename/timestamp ordering, different input resolutions/orientations, JPEG output, and optional alignment diagnostics.

## Install

```bash
uv sync
```

## Usage

```bash
uv run viewalign align --input ./photos --output ./aligned
```

Useful options:

```bash
uv run viewalign align \
  --input ./photos \
  --output ./aligned \
  --reference first \
  --order timestamp \
  --working-size 1600 \
  --crop intersection \
  --diagnostics
```

The command continues when an individual image cannot be aligned. A `metrics.json` file is written to the output directory.

## Crop modes

`fit` is the default. The output canvas is expanded as necessary so the entire transformed source image is visible, including portions that move outside the reference image bounds. Empty areas are filled with black.

`none` keeps the old reference-sized canvas, so an image can be clipped at the edges.

`intersection` crops all outputs to the common valid area shared by the successfully aligned images.

For example:

```bash
uv run viewalign align --input ./photos --output ./aligned --crop fit
```
