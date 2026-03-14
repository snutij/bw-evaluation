## 1. Delete conversion code

- [ ] 1.1 Delete `convert_bw.py`
- [ ] 1.2 Delete `tests/test_convert_bw.py`

## 2. Remove scene detection

- [ ] 2.1 Delete `detect_scene_type` function from `bw_scorer.py`
- [ ] 2.2 Remove `scene_type` assignment from `score_photo` output details
- [ ] 2.3 Delete scene detection tests from `tests/test_bw_scorer.py` (`TestDetectSceneType` class)
- [ ] 2.4 Delete scene detection tests from `tests/test_golden_references.py` (`TestSceneDetection` class)
- [x] 2.5 Remove `scene_type` fixture/assertion from any remaining test that checks for it

## 3. Update CLI

- [x] 3.1 Remove `convert` subcommand and `cmd_convert` function from `cli.py`
- [x] 3.2 Remove `from convert_bw import ...` from `cli.py`
- [x] 3.3 Remove scene type display from `cmd_score` log output
- [x] 3.4 Remove scene type breakdown from `cmd_report`
- [x] 3.5 Update CLI tests: remove convert-related tests, update remaining assertions

## 4. Drop Pillow dependency

- [x] 4.1 Remove Pillow from `requirements.txt`

## 5. Update docs and config

- [x] 5.1 Update README: remove conversion section, Docker conversion examples, update disclaimer and filter docs
- [x] 5.2 Update Dockerfile: remove `convert_bw.py` from COPY line
- [x] 5.3 Remove conftest fixtures only used by conversion tests (if any)

## 6. Verify

- [x] 6.1 Run full test suite and verify all tests pass
