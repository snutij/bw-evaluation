# BW Evaluation

A Python tool for pre-filtering and selecting the best black & white photo candidates from large collections.

## Disclaimer

I'm not a photographer. This tool is designed to:

- **Pre-filter photos** based on basic criteria (contrast, texture, channel separation, saturation, composition)
- **Help select the best photos** among a large set of pictures (e.g., from an entire year's collection)

### Limitations

- The scoring is not perfect and may miss some good photos or keep some bad ones
- **Manual human review is required** — this is a tool to assist, not replace, your judgment

## Usage

### Score photos

```bash
python cli.py score -i photos/
```

This analyzes all photos in `photos/` and writes `results.json` with scores and breakdowns.

Options:
- `-w/--workers N` — parallel scoring (default: 1)
- `--config config.json` — override scoring parameters
- `-v/--verbose` — debug output
- `-q/--quiet` — warnings only

### Report

```bash
python cli.py report
```

Shows score distribution (min/max/mean/median, histogram) and channel separation stats.

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

# Report
docker run --rm -v "$PWD:/app" bw-eval report
```

## License

MIT License - see [LICENSE](LICENSE) for details.
