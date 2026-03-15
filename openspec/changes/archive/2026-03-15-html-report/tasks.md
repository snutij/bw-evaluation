## 1. Core report module

- [x] 1.1 Create `report_html.py` with `generate_html_report(results, photos_dir, output_path, thumbnail_width=300, max_photos=None)` function
- [x] 1.2 Implement thumbnail generation: resize with OpenCV, encode as base64 JPEG data URI
- [x] 1.3 Implement placeholder for missing photos (grey box with filename)
- [x] 1.4 Build HTML template with CSS-only score bars (overall score + 5 dimension bars per card)
- [x] 1.5 Ensure output is a single self-contained HTML file (all CSS inline, no external refs)

## 2. CLI integration

- [x] 2.1 Add `report-html` subcommand to `build_parser()` in `cli.py` with flags: `-r/--results`, `-i/--input-dir`, `-o/--output`, `--max-photos`
- [x] 2.2 Add `cmd_report_html` function that wires CLI args to `generate_html_report`
- [x] 2.3 Update Dockerfile COPY to include `report_html.py`

## 3. Tests

- [x] 3.1 Test HTML generation with synthetic results and temp images (verify output contains expected structure)
- [x] 3.2 Test missing photo graceful fallback (placeholder rendered)
- [x] 3.3 Test `--max-photos` limits output to top N
- [x] 3.4 Test CLI argument parsing for `report-html` subcommand

## 4. Documentation

- [x] 4.1 Add `report-html` usage section to README
- [x] 4.2 Add Docker example for report-html

## 5. Verify

- [x] 5.1 Run full test suite
