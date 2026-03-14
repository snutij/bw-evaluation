<div align="center">



**`bw_evaluation`**\
Score photos for black & white potential.\
Deterministic analysis — no AI/ML.

[![Python 3.12](https://img.shields.io/badge/python-3.12-1a1a2e?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![OpenCV](https://img.shields.io/badge/opencv-4.13-1a1a2e?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-1a1a2e?style=flat-square)](LICENSE)

</div>

---

A CLI tool for pre-filtering and ranking the best black & white candidates from large photo collections. Feed it a folder, get back a scored and sorted `results.json`.

> **Not a replacement for your eye** — this is a triage tool to surface the most promising shots from hundreds or thousands of photos. Manual review is still required.

<br>

## How scoring works

Each photo is scored **0–100** as a weighted sum of five dimensions:

| Dimension | Weight | What it measures |
|:--|:-:|:--|
| **Contrast** | 35% | Histogram spread, deep blacks & bright whites, dynamic range, light/dark balance |
| **Texture** | 25% | Edge density (Sobel + Canny), local variance, Laplacian sharpness |
| **Channel separation** | 25% | Per-pixel RGB divergence — high separation = more creative B&W mixing potential |
| **Saturation** | 10% | Inverse mean saturation — less color = more naturally suited for B&W |
| **Composition** | 5% | Region luminosity variation, highlight distribution |

All weights and sub-parameters are configurable via `--config config.json`.

<br>

## Quick start

```bash
pip install -r requirements.txt
```

**Score** a folder of photos:
```bash
python cli.py score -i photos/
```

**Generate a visual HTML report** with thumbnails:
```bash
python cli.py report-html
```
Opens in any browser, fully offline. Use `--max-photos N` to limit to top N.

**View** text score distribution:
```bash
python cli.py report
```

<details>
<summary><strong>All scoring options</strong></summary>

<br>

| Flag | Default | Description |
|:--|:-:|:--|
| `-i, --input-dir` | `photos/` | Input directory |
| `-o, --output` | `results.json` | Output file |
| `-w, --workers` | `1` | Parallel workers |
| `--config` | — | JSON config override |
| `-v, --verbose` | — | Debug output |
| `-q, --quiet` | — | Warnings only |

</details>

<br>

## Docker

```bash
docker build -t bw-eval .
docker run --rm -v "$PWD:/app" bw-eval score -i photos/
docker run --rm -v "$PWD:/app" bw-eval report-html
docker run --rm -v "$PWD:/app" bw-eval report
```

<br>

## License

[MIT](LICENSE)
