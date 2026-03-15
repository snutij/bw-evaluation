## Context

`tqdm` is already a dependency. The scoring loop in `score_photos()` and thumbnail loop in `generate_html_report()` are the two hot paths that can take minutes on large collections.

## Goals / Non-Goals

**Goals:**
- Per-photo progress bar with ETA during `score` and `report-html`
- Silent when `-q` is passed

**Non-Goals:**
- Custom progress output format
- Logging-integrated progress (tqdm stderr is fine)

## Decisions

### 1. Add `quiet` parameter to `score_photos()` and `generate_html_report()`

Both functions get a `quiet: bool = False` parameter. When False, wrap the photo iteration with `tqdm`. When True, no progress bar.

### 2. tqdm on stderr (default)

tqdm writes to stderr by default, which avoids mixing with stdout output. No change needed.

## Risks / Trade-offs

- **Minimal** — tqdm is already a dependency, changes are 2-3 lines per function.
