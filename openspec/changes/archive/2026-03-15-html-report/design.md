## Context

Currently `cli.py report` prints a text summary to stdout. The JSON results file contains all scoring data but requires manual cross-referencing with actual photos. For a collection of 200+ photos, this is impractical.

## Goals / Non-Goals

**Goals:**
- Generate a single self-contained HTML file that works offline (no CDN, no external assets)
- Embed photo thumbnails as base64 data URIs
- Show scores visually (bars, not just numbers)
- Keep the report responsive and fast even with 500+ photos
- Zero new Python dependencies

**Non-Goals:**
- Interactive filtering/sorting in the browser (static ranked list is enough)
- Full-resolution image embedding (thumbnails only, ~300px wide)
- PDF export
- Server-side rendering or live dashboard

## Decisions

### 1. Pure Python string templating (no Jinja2)

The HTML structure is a single template with a repeated photo card. Python f-strings and `str.join()` are sufficient. Adding Jinja2 for one template is unjustified.

### 2. Base64-embedded thumbnails

Photos are resized to ~300px wide via OpenCV (already a dependency) and encoded as base64 JPEG data URIs. This produces a single portable `.html` file. Trade-off: file size grows ~15KB per photo, so a 500-photo report is ~7.5MB — acceptable for local use.

### 3. Thumbnail generation at report time, not score time

`score_photo` stays pure (no side effects). Thumbnails are generated when building the HTML report by re-reading images from the input directory. This requires the photos to still be accessible at report time.

### 4. New module `report_html.py`

Keeps HTML generation logic separate from `cli.py`. The module exposes a single function `generate_html_report(results, photos_dir, output_path, thumbnail_width)`.

### 5. CSS-only score bars

Each dimension rendered as a colored bar proportional to its score (0-100). No JavaScript needed. Uses inline styles for simplicity and self-containment.

## Risks / Trade-offs

- **Large file size for big collections** → Accept for now; add `--max-photos N` flag to limit report size
- **Photos must be accessible at HTML generation time** → Document this clearly; fail gracefully with placeholder for missing photos
- **Base64 encoding is slow for many large photos** → Resize to thumbnail first (fast with OpenCV), then encode; add `--workers` for parallel thumbnail generation
