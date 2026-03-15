![bw_evaluation](assets/banner.svg)

[![Python 3.12](https://img.shields.io/badge/python-3.12-1a1a2e?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![OpenCV](https://img.shields.io/badge/opencv-4.13-1a1a2e?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-1a1a2e?style=flat-square)](LICENSE)

---

Score a folder of photos for B&W potential. Get a ranked `results.json` and an interactive HTML report — filter, sort, compare, all offline in your browser.

```bash
git clone https://github.com/snutij/bw-evaluation.git && cd bw-evaluation
uv sync
bw score -i ~/photos/
bw report-html          # open report.html in your browser
```

## How it scores

Each photo gets a **0–100 score** from five weighted dimensions:

| Dimension | Weight | Measures |
|:--|:-:|:--|
| **Contrast** | 35% | Dynamic range, black/white presence, tonal balance |
| **Texture** | 25% | Edge density, local variance, sharpness |
| **Channel separation** | 25% | RGB divergence — high = creative B&W mixing potential |
| **Saturation** | 10% | Less color = more naturally suited for B&W |
| **Composition** | 5% | Region luminosity variation, highlight distribution |

Weights are configurable via `--config config.json`. Scoring is fully deterministic — same photo, same score, every time.

## CLI reference

| Command | What it does |
|:--|:--|
| `bw score -i photos/` | Score photos, write `results.json` |
| `bw report-html` | Interactive HTML report with thumbnails |
| `bw report` | Text score distribution summary |

| Flag | Default | Description |
|:--|:-:|:--|
| `-i, --input-dir` | `photos/` | Input directory |
| `-o, --output` | `results.json` | Output file |
| `-w, --workers` | `1` | Parallel workers |
| `--config` | — | JSON config override |
| `-q, --quiet` | — | Suppress progress output |

## Development

```bash
uv sync --group dev && pre-commit install
```

Pre-commit runs ruff (all rules), mypy strict, and pytest on every commit. CI runs the same via `uv run pre-commit run --all-files`.

## Docker

```bash
docker build -t bw-eval .
docker run --rm -v "$PWD:/app" bw-eval score -i photos/
```

## License

[MIT](LICENSE)
