# viewalign

A small cross-platform CLI for aligning photos taken from approximately the same viewpoint to a common coordinate system.

The alignment step uses SIFT feature matching and a RANSAC-estimated similarity transform (`cv2.estimateAffinePartial2D`). It supports JPEG/PNG input, EXIF orientation normalization, filename/timestamp input ordering, different input resolutions/orientations, JPEG output, alignment diagnostics, reference-based or global alignment, and automatic upright correction for global mode.

Derived outputs are separate CLI operations. `align` only performs alignment and writes reusable alignment metadata; `average`, `mosaic`, and `video` read those alignment outputs and can be rerun independently without repeating SIFT matching.

## Install

```bash
uv sync
```

FFmpeg is provided by the `imageio-ffmpeg` Python dependency, so `uv sync` installs the runtime used by ViewAlign. No separate `--ffmpeg-path` option is needed.

## Workflow

First align the photos:

```bash
uv run viewalign align --input ./photos --output ./output
```

The alignment command creates:

```text
output/
├── aligned/              # transformed JPEG frames
├── diagnostics/          # only when --diagnostics is used
├── metrics.json          # per-image alignment metrics
├── summary.json          # alignment summary
├── alignment.json        # reusable geometry/transform manifest
└── pairwise_metrics.json # global mode only
```

Then create whichever derived outputs you want:

```bash
uv run viewalign average --input ./output
uv run viewalign mosaic --input ./output --polygons 80 --seed 42
uv run viewalign video --input ./output --duration 0.75 --order brightness
```

These commands write their results into the same alignment output directory:

```text
output/
├── aligned/
├── average.jpg
├── mosaic.jpg
└── video.mp4
```

This separation means you can experiment with different mosaic seeds, polygon counts, video durations, or video ordering without running alignment again.

## Align options

```bash
uv run viewalign align \
  --input ./photos \
  --output ./output \
  --reference first \
  --upright auto \
  --order timestamp \
  --working-size 1600 \
  --crop fit \
  --diagnostics
```

The command continues when an individual image cannot be aligned. `metrics.json` records successful, rejected, and failed images.

### Reference modes

By default, ViewAlign aligns every image directly to a chosen reference image:

- `--reference first` uses the first image after the selected input ordering.
- `--reference middle` uses the middle image.
- `--reference <path>` uses a specific input image.
- `--reference global` performs reference-free global alignment.

For global mode, `--upright auto` is enabled by default. It estimates one scene-wide roll correction from long horizontal/vertical line structure detected across the registered images, then applies that single correction to the whole global coordinate system. `--upright none` disables this step. Fixed-reference modes are not automatically rotated, so an explicitly chosen reference image continues to define the final horizon/orientation.

### Global alignment

`--reference global` first computes pairwise similarity transforms between every unordered pair of loaded images. The other direction is represented by the inverse transform, so `N` images require `N(N-1)/2` feature-matching attempts.

Successful pairwise matches form an alignment graph. ViewAlign finds its connected components, selects the largest component, and solves one global similarity transform per image using all successful pairwise measurements in that component. Pairwise coordinates are normalized by each image's size before the global solve, which prevents differences in image resolution and pathological scale estimates from making the global canvas explode. Higher-quality pairwise measurements receive higher weights, and robust reweighting reduces the influence of inconsistent edges.

This means an image does not have to match one particular reference image directly. It can still be included when it connects reliably through other images. Images outside the selected connected component are rejected instead of being forced into the common coordinate system.

The pairwise measurements are stored in `pairwise_metrics.json` as a two-dimensional matrix indexed by the listed image names. Each successful unordered pair has an estimated transform in one direction and a derived inverse in the other matrix cell.

Global mode chooses a deterministic anchor image only to define the final coordinate system; it is not used as the sole source of alignment measurements.

## Alignment manifest

`alignment.json` is the handoff between the alignment step and the derived-output commands. It stores the output geometry, crop mode, fit origin, reference size, and the final transform and source dimensions for every successfully aligned frame.

Derived commands use this manifest together with `aligned/` rather than recomputing alignment. Keeping the geometric validity information is important: average, mosaic, brightness ordering, and center ordering must distinguish legitimate black image content from black pixels created by warping outside the source image.

## Crop modes

`fit` is the default. The output canvas is expanded as necessary so the entire transformed source image is visible, including portions that move outside the reference image bounds. Empty areas are filled with black.

`none` keeps the reference-sized canvas, so an image can be clipped at the edges.

`intersection` crops all outputs to the common valid area shared by the successfully aligned images.

## Average image

Run:

```bash
uv run viewalign average --input ./output
```

`average.jpg` averages the aligned output frames pixel-by-pixel. It uses geometric validity masks from the alignment manifest, rather than testing whether pixels are black. This means black content from genuine night images is still included, while black border areas created by warping are ignored.

## Mosaic image

Run:

```bash
uv run viewalign mosaic --input ./output --polygons 50 --seed 42
```

`mosaic.jpg` is an independent composition made from random polygon patches from aligned frames. A polygon is only placed when nearly all of its area is valid for the selected source frame, so warped black borders are not copied into the collage.

The mosaic does not use `average.jpg` as a background and can be generated even when the average image does not exist.

## Video

Run:

```bash
uv run viewalign video --input ./output --duration 0.5 --order random
```

`video.mp4` is generated from the aligned images. `--duration` controls how long each image is displayed. `--order` controls the sequence independently of the alignment order:

- `random`: shuffle aligned images randomly.
- `brightness`: sort from highest to lowest mean luminance over geometrically valid pixels, so bright daytime/snowy images tend to come first and dark/night images tend to come last.
- `center`: sort by the transformed source-image center from left to right in the final output coordinate system.

## FFmpeg

The project depends on `imageio-ffmpeg`. ViewAlign discovers its FFmpeg executable automatically at runtime, so there is no `--ffmpeg-path` CLI option.
