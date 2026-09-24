# bw_evaluation

Deterministic measurement of how well a colour photograph would work in black and white, and batch-relative ranking of a folder of them.

The tool refuses to answer "is this a good black and white photograph?", because there is no published ground truth to calibrate an absolute verdict against. It answers two measurable questions instead:

1. **How much information does this image lose when chroma is discarded?**
2. **Is the surviving luminance structure strong enough to carry the image alone?**

Both are reported as percentiles _within the folder you pass in_. Move an image to a different folder and its position changes. That is the intended behaviour, not a limitation to work around.

## Install

```bash
uv sync
```

Requires Python 3.11+, `numpy`, `scipy`, `scikit-learn` and `Pillow`.

## Use

The input folder must contain at least 8 readable photographs. Results are
relative to that batch, so the same image can receive a different percentile
in a different folder.

```bash
uv run bw_evaluation ~/shoots/2026-09-porto
uv run bw_evaluation ~/photos --recursive --key structure_first --top 20
uv run bw_evaluation ~/photos --json out.json --csv out.csv
```

Reads JPEG, PNG, TIFF, WebP, AVIF and JPEG 2000. Raw files are refused with a
specific reason: this measures renderings, not negatives. Embedded ICC profiles
are honoured; files without one are treated as sRGB and say so.

The command options are:

- `--recursive` includes photographs in nested folders.
- `--key` chooses the ordering: `loss_first`, `structure_first`, or
  `pareto_then_loss` (the default). The Pareto ordering leads with the
  non-dominated layer and does not weight the two axes against each other.
- `--top N` prints only the first `N` rows.
- `--json PATH` writes the complete result as JSON in addition to the table.
- `--csv PATH` writes the complete result as CSV in addition to the table.

Use `uv run bw_evaluation --help` for the same command reference.

## What it measures

| Module              | Question                                                                                  |
| ------------------- | ----------------------------------------------------------------------------------------- |
| `conventions.py`    | BT.601 / 709 / 2020 luma, sRGB EOTF/OETF, CIELAB. Luma and luminance kept strictly apart. |
| `chroma.py`         | Hasler–Süsstrunk colourfulness, with the authors' own category anchors.                   |
| `decolorization.py` | CCPR, CCFR, E-score, C2G-SSIM, threshold-independent area, channel-weight search.         |
| `isoluminance.py`   | Čadík's canonical failure mode: contrast carried by chroma alone.                         |
| `tonal.py`          | Zone System binning, region segmentation, tonal merger between regions.                   |
| `measurement.py`    | Assembles the per-image evidence and the recommended channel recipe.                      |
| `ranking.py`        | Two axes, Pareto layers, percentile labels. Never a single combined score.                |
| `loading.py`        | Decoding, ICC conversion, EXIF, stable image ids.                                         |
| `cli.py`            | Folder in, ranked table / JSON / CSV out.                                                 |

Every public symbol carries an evidence level: `empirical` (measured against a
published dataset), `convention` (a standard or a documented practice),
`derived` (built here from published primitives) or `unsupported` (an advisory
heuristic with no citation behind it).

## Tests

```bash
uv run pytest
```

216 tests, using the standard `unittest` assertions under pytest. Roughly four minutes; the
1024 px working size dominates. The suite is written test-first against
the documented module contracts, and thresholds were never nudged to make a test pass.

## Known gap

The framework has **not** been validated against real photographs with known
human preferences. It agrees with itself and with the synthetic reference
catalogue. Whether the derived isoluminance metrics earn their place against a
photographer's eye is still an open question.
