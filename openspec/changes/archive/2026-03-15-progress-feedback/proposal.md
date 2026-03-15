## Why

When scoring hundreds of photos, the CLI prints "Analyzing N photos..." then goes silent until done. Users think the tool is stuck. `tqdm` is already in `requirements.txt` but unused.

## What Changes

- Add per-photo progress bar (via tqdm) to `score_photos()` in `bw_scorer.py`
- Add per-photo progress bar to `generate_html_report()` in `report_html.py`
- Suppress progress bar in quiet mode (`-q`)

## Capabilities

### New Capabilities
- `progress-bar`: tqdm-based per-photo progress feedback during scoring and HTML report generation

### Modified Capabilities

_None._

## Impact

- **bw_scorer.py**: `score_photos()` wraps iteration with tqdm
- **report_html.py**: `generate_html_report()` wraps iteration with tqdm
- **cli.py**: Pass quiet flag through to control progress bar visibility
