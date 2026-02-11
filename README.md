# BW Evaluation

A Python tool for pre-filtering and selecting best potential black & white photos from large collections, then converting them with scene-adaptive presets.

## Disclaimer

I'm not a photographer. This tool is designed to:

- **Pre-filter photos** based on basic criteria (contrast, texture, saturation, composition, metadata)
- **Help select the best photos** among a large set of pictures (e.g., from an entire year's collection)
- **Convert with adaptive styles** — portrait, landscape, high-contrast, street, architecture

### Limitations

- The scoring is not perfect and may miss some good photos or keep some bad ones
- **Manual human review is required** — this is a tool to assist, not replace, your judgment

## Usage

### Score photos

```bash
python cli.py score -i photos/
```

This analyzes all photos in `photos/` and writes `results.json` with scores, breakdowns, and detected scene types.

Options:
- `-w/--workers N` — parallel scoring (default: 1)
- `--config config.json` — override scoring parameters
- `-v/--verbose` — debug output
- `-q/--quiet` — warnings only

### Convert to B&W

```bash
python cli.py convert -n 50 --min-score 60
```

Options:
- `--style auto|portrait|landscape|high-contrast|street|architecture` — conversion preset (default: auto, based on detected scene type)
- `--format jpeg|png|tiff` — output format (default: jpeg)
- `--quality 1-100` — JPEG quality (default: 95)
- `--dry-run` — list what would be converted without writing files
- `--no-overwrite` — skip files that already exist in output
- `--min-contrast/--min-texture/--min-saturation/--min-composition/--min-metadata` — per-dimension filters

### Report

```bash
python cli.py report
```

Shows score distribution (min/max/mean/median, histogram) and scene type breakdown.

## Requirements

```bash
pip install -r requirements.txt
```

## Docker

```bash
# Build
docker build -t bw-eval .

# Score
docker run --rm -v "$PWD:/app" bw-eval score -i photos/

# Convert
docker run --rm -v "$PWD:/app" bw-eval convert -n 50 --min-score 60

# Report
docker run --rm -v "$PWD:/app" bw-eval report
```

## License

MIT License - see [LICENSE](LICENSE) for details.
