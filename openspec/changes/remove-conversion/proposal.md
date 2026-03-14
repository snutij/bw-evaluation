## Why

The project is "bw_evaluation" — its value is scoring photos for B&W potential. The conversion feature (5 channel-mixing presets) is commodity work that any photo editor does better. Removing it sharpens the project's scope, eliminates dead code (`detect_scene_type` now only serves conversion), and drops the Pillow dependency entirely.

## What Changes

- **BREAKING**: Delete `convert_bw.py` entirely (5 presets, `process_photos`, style picking)
- **BREAKING**: Remove `convert` subcommand from CLI
- **BREAKING**: Remove `detect_scene_type` from `bw_scorer.py` (its only remaining use is conversion preset selection)
- Remove `scene_type` from `score_photo` output details (no longer needed)
- Remove Pillow from `requirements.txt` (was used for EXIF in deleted metadata scorer and for conversion)
- Update `report` subcommand to drop scene type breakdown
- Update README to remove conversion docs, Docker conversion examples
- Update Dockerfile to not copy `convert_bw.py`
- Delete `tests/test_convert_bw.py` and all scene detection tests

## Capabilities

### New Capabilities

- `scoring-only-cli`: The CLI retains only `score` and `report` subcommands focused purely on evaluation output.

### Modified Capabilities

None (no existing specs)

## Impact

- `convert_bw.py`: deleted
- `bw_scorer.py`: remove `detect_scene_type`, remove scene_type from `score_photo` output
- `cli.py`: remove `convert` subcommand, remove scene type display from `report` and `score` output, remove `convert_bw` import
- `config.py`: no changes (scene bonuses already removed)
- `requirements.txt`: remove Pillow
- `Dockerfile`: remove `convert_bw.py` from COPY
- `README.md`: remove conversion section, update disclaimer and examples
- `tests/`: delete `test_convert_bw.py`, delete scene detection tests, update CLI tests
