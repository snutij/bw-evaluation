## Why

The current scoring algorithm includes components and heuristics that are not trustworthy predictors of B&W conversion potential. Scene detection bonuses swing scores by up to 8 points based on brittle threshold rules. The saturation component uses 6+ interdependent sub-penalties to work around a flawed premise (low saturation != good B&W candidate). The metadata component guesses aesthetic quality from EXIF data. Rather than patch untrusted logic, we delete it and replace what matters with a single metric that actually predicts B&W potential: channel separation.

## What Changes

- **BREAKING**: Remove scene detection bonuses from scoring (keep scene detection only for conversion preset selection)
- **BREAKING**: Replace the complex saturation scoring (6 sub-penalties) with a single-line inverse mean saturation, reweighted from 28% to 10%
- **BREAKING**: Delete the metadata scoring component entirely (EXIF/noise-based scoring removed)
- **BREAKING**: Simplify composition to only B&W-relevant sub-metrics (keep plane separation + highlight distribution, remove rule-of-thirds and center luminosity)
- Add new channel separation metric (25% weight) measuring RGB channel divergence — the actual predictor of B&W conversion potential
- Reweight remaining components: contrast 35%, texture 25%, channel separation 25%, saturation 10%, composition 5%
- Update `ScoringConfig` to remove deleted parameters and add channel separation parameters
- Update `ScoreBreakdown` to reflect new components

## Capabilities

### New Capabilities
- `channel-separation`: New scoring component measuring divergence between R/G/B luminance maps. High divergence means more creative options when choosing channel mixing weights for B&W conversion.

### Modified Capabilities
- None (no existing specs)

## Impact

- `bw_scorer.py`: Major rewrite — remove `analyze_metadata`, simplify `analyze_colorimetry`, simplify `analyze_composition`, remove scene bonus from final score, add `analyze_channel_separation`
- `config.py`: Remove ~15 parameters (metadata weights, scene bonuses, saturation sub-penalties), add channel separation parameters, update weight defaults and validation
- `cli.py`: Remove `--min-metadata` filter if present, add `--min-channel-separation` filter
- `tests/`: All test files need updates — golden reference tests will change significantly, score expectations shift
- `convert_bw.py`: Minimal impact — scene detection for preset selection is preserved
