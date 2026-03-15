## 1. Add progress bar to scoring

- [x] 1.1 Add `quiet` param to `score_photos()` and wrap photo iteration with `tqdm(disable=quiet)`
- [x] 1.2 Pass `quiet` from `cmd_score` in `cli.py`

## 2. Add progress bar to HTML report

- [x] 2.1 Add `quiet` param to `generate_html_report()` and wrap card generation with `tqdm(disable=quiet)`
- [x] 2.2 Pass `quiet` from `cmd_report_html` in `cli.py`

## 3. Housekeeping

- [x] 3.1 Add `report.html` to `.gitignore`

## 4. Verify

- [x] 4.1 Run full test suite
