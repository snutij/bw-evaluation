## Why

The HTML report is currently a static ranked list. Users can see scores but cannot explore their collection — there's no way to filter by score range, sort by a specific dimension, or understand *why* one photo outranks another. This makes the report a one-way output rather than a feedback tool. Adding client-side interactivity closes the loop: users explore dimension breakdowns, filter candidates, and learn which scoring weights matter for their photography style — all without leaving the single self-contained HTML file.

## What Changes

- Add a **sticky toolbar** at the top of the HTML report with score range slider, dimension sort dropdown, and a search/filter input for filenames.
- Add **hover/click tooltips** on each photo card showing full 5-dimension breakdown with a mini radar chart or enhanced bar visualization.
- Add a **side-by-side compare mode**: click two photos to view them together with dimension deltas highlighted.
- Add a **stats summary bar** showing count of visible photos, average score, and a score distribution mini-histogram.
- All interactivity is **pure vanilla JS/CSS** — no external dependencies, no CDN links. The report remains a single self-contained `.html` file that works offline.

## Capabilities

### New Capabilities
- `interactive-report`: Client-side filtering, sorting, comparison, and score exploration features for the HTML report.

### Modified Capabilities

## Impact

- **Code**: `report_html.py` — template expansion to inject JS/CSS for interactivity. No changes to scoring logic.
- **Dependencies**: None added. Pure vanilla JS/CSS.
- **Tests**: New tests in `test_report_html.py` for filter controls presence, data attributes on cards, and JS injection.
- **File size**: ~3-5KB additional JS/CSS per report (negligible vs. base64 thumbnails).
