# BW Evaluation

A Python script for pre-filtering and selecting best potential black & white photos from large collections.

## Disclaimer

I'm not a photographer. This script is designed to:

- **Pre-filter photos** based on basic criteria (blurriness, brightness, contrast, etc.)
- **Help select the best photos** among a large set of pictures (e.g., from an entire year's collection)

### Limitations

- The script is not perfect and may miss some good photos or keep some bad ones
- **Manual human review is required** - this is a tool to assist, not replace, your judgment

## Purpose

This project was also an experiment to test the Ralph AI workflow (with mixed success).

## Usage

1. Place your photos in the `photos/` directory
2. Score photos for B&W potential:
   ```bash
   python bw_scorer.py
   ```
   This outputs `results.json` with scores for each photo.

3. Convert top-scoring photos to B&W:
   ```bash
   python convert_bw.py -n 50 --min-score 60
   ```
   This converts the top 50 photos (score ≥ 60) to `photos_bw/`

## Requirements

```bash
pip install -r requirements.txt
```

## Docker Usage

No Python setup required:

```bash
# Build once
docker build -t bw-eval .

# Score photos (outputs results.json)
docker run --rm -v "$PWD:/app" bw-eval bw_scorer.py

# Convert to B&W
docker run --rm -v "$PWD:/app" bw-eval convert_bw.py -n 50 --min-score 60
```

## License

MIT License - see [LICENSE](LICENSE) for details.
