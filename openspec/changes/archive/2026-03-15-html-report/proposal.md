## Why

The tool outputs a JSON file and a text-based terminal summary. To actually review which photos are good B&W candidates, users must cross-reference filenames manually. A visual HTML report with embedded thumbnails would make the scoring results immediately actionable.

## What Changes

- New `report-html` CLI subcommand that generates a self-contained HTML file from `results.json`
- Photos displayed as thumbnails, sorted by score descending
- Each photo shows its overall score and per-dimension breakdown as visual bars
- Single-file output (images embedded as base64) — no external dependencies, works offline
- Clickable details to expand full scoring breakdown per photo

## Capabilities

### New Capabilities
- `html-report`: Self-contained HTML report generation from scoring results, with embedded thumbnails and visual score breakdowns

### Modified Capabilities

_None — this is a new subcommand alongside existing `score` and `report`._

## Impact

- **cli.py**: New `report-html` subcommand and `cmd_report_html` function
- **New file**: `report_html.py` — HTML generation logic (Jinja2-free, pure string templating to avoid new deps)
- **Dockerfile**: Add `report_html.py` to COPY line
- **Tests**: New `tests/test_report_html.py`
