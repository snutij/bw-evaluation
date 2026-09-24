# bw_evaluation

**Deterministic black-and-white candidacy measurement and batch-relative ranking for colour photographs.** This project does not pretend to answer "is this a good black-and-white
photograph?". There is no published ground truth for an absolute verdict.
Instead, it measures two explicit questions and ranks images **relative to the
folder being analysed**:

1. How much information is lost when chroma is discarded?
2. Is the surviving luminance structure strong enough to carry the image alone?

Move an image to a different folder and its percentile can change. That is the
intended behaviour.

## Contents

- [Quick start](#quick-start)
- [Command reference](#command-reference)
- [What it measures](#what-it-measures)
- [Supported inputs](#supported-inputs)
- [Output files](#output-files)
- [Development](#development)
- [Project layout](#project-layout)
- [Limitations](#limitations)
- [License](#license)

## Quick start

### Requirements

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/)
- At least 8 readable photographs in the input folder

```bash
git clone https://github.com/snutij/bw-evaluation.git
cd bw-evaluation
uv sync
uv run bw_evaluation --help
```

Analyse a folder:

```bash
uv run bw_evaluation ~/shoots/2026-09-porto
```

Rank recursively, choose the leading axis, and limit the displayed rows:

```bash
uv run bw_evaluation ~/photos \
  --recursive \
  --key structure_first \
  --top 20
```

Write machine-readable results alongside the terminal table:

```bash
uv run bw_evaluation ~/photos \
  --json out.json \
  --csv out.csv
```

The package is intentionally run with `uv run` so the command uses the locked
environment. After `uv sync`, the installed `bw_evaluation` entry point can
also be invoked directly.

## Command reference

```text
usage: bw_evaluation [-h] [--recursive]
                     [--key {loss_first,structure_first,pareto_then_loss}]
                     [--top TOP] [--json JSON_PATH] [--csv CSV_PATH]
                     folder
```

| Argument                 | Description                                                              |
| ------------------------ | ------------------------------------------------------------------------ |
| `folder`                 | Folder of photographs to analyse.                                        |
| `--recursive`            | Include photographs in nested folders.                                   |
| `--key loss_first`       | Order by chroma information loss first.                                  |
| `--key structure_first`  | Order by surviving luminance structure first.                            |
| `--key pareto_then_loss` | Default; show non-dominated Pareto layers first, then loss.              |
| `--top N`                | Display only the first `N` rows while retaining the full batch analysis. |
| `--json PATH`            | Write complete results as JSON.                                          |
| `--csv PATH`             | Write complete results as CSV.                                           |
| `-h`, `--help`           | Show the built-in command help.                                          |

Every ranking key orders the same two axes; there is no hidden absolute score or
default weighting between them.

## What it measures

| Module              | Responsibility                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------- |
| `conventions.py`    | BT.601/709/2020 luma, sRGB transfer functions, and CIELAB conversion.                     |
| `chroma.py`         | Hasler–Süsstrunk colourfulness and category anchors.                                      |
| `decolorization.py` | Conversion-loss metrics, C2G-SSIM, threshold-independent area, and channel-weight search. |
| `isoluminance.py`   | Detection of contrast carried by chroma rather than luminance.                            |
| `tonal.py`          | Zone System binning, region segmentation, and tonal-merger analysis.                      |
| `measurement.py`    | Per-image evidence and recommended channel recipe.                                        |
| `ranking.py`        | Batch axes, Pareto layers, percentiles, and ordering.                                     |
| `loading.py`        | Image decoding, ICC conversion, EXIF metadata, and stable image IDs.                      |
| `cli.py`            | Folder analysis and table/JSON/CSV rendering.                                             |

Public measurements identify their evidence level:

- `empirical`: measured against a published dataset
- `convention`: based on a standard or documented practice
- `derived`: built from published primitives
- `unsupported`: an explicitly labelled advisory heuristic

## Supported inputs

Supported rendered image formats are JPEG, PNG, TIFF, WebP, AVIF, and JPEG 2000. Raw camera files are refused because this tool evaluates rendered pixels,
not negatives. Embedded ICC profiles are converted to sRGB; files without an
embedded profile are treated as sRGB and reported accordingly. EXIF orientation
and selected capture metadata are preserved where available.

## Output files

The terminal report shows:

- image ID
- chroma-loss percentile
- surviving-structure percentile
- Pareto layer
- relative label
- recommended channel recipe
- strongest evidence

JSON contains the complete per-image measurements, metadata, skipped-file
reasons, ranking, and batch caveat. CSV flattens list-valued fields for
spreadsheet use. Both outputs describe the full analysed batch, even when
`--top` limits the terminal display.

## Development

Install the locked development environment and enable the repository hooks:

```bash
uv sync --group dev
uv run pre-commit install
```

Run the same checks used by CI:

```bash
uv run pre-commit run --all-files
```

Run the test suite directly:

```bash
uv run pytest
```

The suite uses `unittest` assertions under pytest and covers colour conventions,
conversion loss, isoluminance, tonal structure, photograph loading, batch
ranking, and the command-line interface. Synthetic fixtures keep the tests
deterministic and do not require personal photographs.

## Project layout

```text
src/bw_evaluation/
├── __init__.py       # public Python API
├── __main__.py       # python -m bw_evaluation
├── cli.py            # command-line interface
├── loading.py        # image decoding and metadata
├── measurement.py    # per-image evidence
├── ranking.py        # batch-relative ranking
└── ...               # colour, tonal, and conversion metrics
tests/                # deterministic unit and integration tests
assets/               # repository artwork
```

## Limitations

The framework has not been validated against real photographs with known human
preferences. It agrees with itself and with the synthetic reference catalogue.
Whether the derived isoluminance metrics earn their place against a
photographer's eye remains an open question. Treat the output as structured
evidence for editing decisions, not an aesthetic verdict.

## License

Released under the [MIT License](LICENSE).
