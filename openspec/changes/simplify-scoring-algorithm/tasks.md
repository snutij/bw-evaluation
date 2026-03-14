## 1. Config cleanup

- [x] 1.1 Remove metadata weight and sub-weights from `ScoringConfig` (`weight_metadata`, `metadata_iso_weight`, `metadata_focal_weight`, `metadata_light_weight`, `metadata_dof_weight`)
- [x] 1.2 Remove scene bonus parameters from `ScoringConfig` (`scene_portrait_bonus`, `scene_landscape_bonus`, `scene_architecture_bonus`, `scene_street_bonus`)
- [x] 1.3 Remove saturation sub-penalty parameters from `ScoringConfig` (`high_sat_threshold`, `high_sat_avg_trigger`, `high_sat_ratio_trigger`, `high_sat_strong_penalty_factor`, `high_sat_weak_penalty_factor`, `low_sat_threshold`, `gamut_penalty_max`, `gamut_spread_divisor`)
- [x] 1.4 Remove composition sub-weights for deleted metrics (`composition_center_weight`, `composition_thirds_weight`)
- [x] 1.5 Add channel separation parameters to `ScoringConfig` (`weight_channel_separation=0.25`, `channel_sep_ceiling=50`)
- [x] 1.6 Update default weights: contrast=0.35, texture=0.25, channel_separation=0.25, saturation=0.10, composition=0.05
- [x] 1.7 Update weight validation to check new 5-weight sum equals 1.0

## 2. Scorer deletions

- [x] 2.1 Delete `analyze_metadata` function from `bw_scorer.py`
- [x] 2.2 Remove scene bonus application from `score_image` (keep `detect_scene_type` for conversion use)
- [x] 2.3 Simplify `analyze_colorimetry` to single formula: `max(0, 100 - (mean_sat / 255) * 100)`
- [x] 2.4 Remove rule-of-thirds and center luminosity from `analyze_composition`, keep only plane separation and highlight distribution
- [x] 2.5 Reweight composition sub-metrics (plane separation and highlight distribution, normalized to sum to 1.0)

## 3. Channel separation implementation

- [x] 3.1 Add `analyze_channel_separation(image, config)` function to `bw_scorer.py` — per-pixel std dev across BGR channels, normalized by configurable ceiling
- [x] 3.2 Update `ScoreBreakdown` dataclass to include `channel_separation` field and remove `metadata` field
- [x] 3.3 Integrate `analyze_channel_separation` into `score_image` weighted average calculation

## 4. CLI updates

- [x] 4.1 Remove `--min-metadata` filter from CLI if present
- [x] 4.2 Add `--min-channel-separation` filter to `convert` subcommand
- [x] 4.3 Update `report` subcommand to display channel separation stats instead of metadata

## 5. Test updates

- [x] 5.1 Delete metadata-related tests from `test_bw_scorer.py`
- [x] 5.2 Delete scene bonus scoring tests (keep scene detection tests if they exist for conversion)
- [x] 5.3 Update saturation tests to match simplified single-formula behavior
- [x] 5.4 Update composition tests to reflect removal of rule-of-thirds and center luminosity
- [x] 5.5 Add unit tests for `analyze_channel_separation` (high divergence, low divergence, solid color, normalization ceiling)
- [x] 5.6 Add config tests for new weight defaults and validation
- [x] 5.7 Rewrite golden reference tests to validate new scoring behavior
- [x] 5.8 Update CLI tests for new filter options
- [x] 5.9 Run full test suite and verify all tests pass
