## ADDED Requirements

### Requirement: Channel separation scoring component
The system SHALL compute a channel separation score (0-100) measuring the divergence between R, G, and B channels of an input image. The score SHALL be computed as the mean per-pixel standard deviation across the three color channels, normalized to a 0-100 scale. A higher score indicates greater channel divergence and thus more creative potential for B&W channel mixing.

#### Scenario: High channel separation image
- **WHEN** scoring an image with strongly divergent RGB channels (e.g., red object on green background)
- **THEN** the channel separation score SHALL be above 60

#### Scenario: Low channel separation image (near-monochrome)
- **WHEN** scoring an image where R, G, and B values are nearly identical per pixel (e.g., grey gradient)
- **THEN** the channel separation score SHALL be below 20

#### Scenario: Uniform color image
- **WHEN** scoring a single solid-color image (e.g., pure blue)
- **THEN** the channel separation score SHALL reflect the inherent channel spread of that color (blue = high R/G vs B divergence)

### Requirement: Channel separation included in final score
The system SHALL include the channel separation score in the final weighted score with a default weight of 0.25 (25%). The weight SHALL be configurable via `ScoringConfig`.

#### Scenario: Weight contribution
- **WHEN** computing the final score for any image
- **THEN** channel separation SHALL contribute `channel_separation_score * weight_channel_separation` to the weighted average

#### Scenario: Custom weight via config
- **WHEN** a `ScoringConfig` specifies `weight_channel_separation=0.30`
- **THEN** the system SHALL use 0.30 as the channel separation weight and all weights SHALL still sum to 1.0

### Requirement: Channel separation normalization
The system SHALL normalize the raw mean per-pixel standard deviation using a configurable ceiling value (default: 50). Scores SHALL be clamped to the 0-100 range.

#### Scenario: Normalization with default ceiling
- **WHEN** the mean per-pixel std dev is 25.0 and the ceiling is 50
- **THEN** the channel separation score SHALL be 50.0

#### Scenario: Saturation beyond ceiling
- **WHEN** the mean per-pixel std dev exceeds the ceiling (e.g., 60 with ceiling 50)
- **THEN** the channel separation score SHALL be clamped to 100

### Requirement: Channel separation in score breakdown
The system SHALL include the channel separation score in `ScoreBreakdown` and in all output formats (JSON, report).

#### Scenario: JSON output includes channel separation
- **WHEN** scoring an image with `--format json`
- **THEN** the output SHALL contain a `channel_separation` field in the breakdown

### Requirement: Channel separation CLI filter
The CLI SHALL support `--min-channel-separation` filter for the `convert` subcommand to skip images below a channel separation threshold.

#### Scenario: Filter applied
- **WHEN** running `convert --min-channel-separation 40` on an image with channel separation score of 30
- **THEN** the image SHALL be skipped and not converted

### Requirement: Removed scoring components
The system SHALL NOT include metadata scoring in the final score. The system SHALL NOT apply scene detection bonuses to the final score. The saturation component SHALL use only inverse mean saturation (no sub-penalties). The composition component SHALL use only plane separation and highlight distribution (no rule-of-thirds or center luminosity).

#### Scenario: No metadata in score
- **WHEN** scoring any image
- **THEN** the breakdown SHALL NOT contain a metadata score and EXIF data SHALL NOT influence the final score

#### Scenario: No scene bonus in score
- **WHEN** scoring an image detected as "portrait" scene type
- **THEN** the final score SHALL NOT include any scene-type bonus points

#### Scenario: Simplified saturation
- **WHEN** scoring a highly saturated image (mean saturation > 200)
- **THEN** the saturation score SHALL be approximately `100 - (mean_sat / 255) * 100` without additional penalties or bonuses

#### Scenario: Simplified composition
- **WHEN** scoring any image
- **THEN** the composition score SHALL be computed from only plane separation and highlight distribution sub-metrics

### Requirement: New default weights
The system SHALL use the following default weights: contrast=0.35, texture=0.25, channel_separation=0.25, saturation=0.10, composition=0.05. All weights SHALL sum to 1.0.

#### Scenario: Default weight validation
- **WHEN** creating a default `ScoringConfig`
- **THEN** the five weights SHALL sum to exactly 1.0

#### Scenario: Config validation rejects bad weights
- **WHEN** providing weights that do not sum to 1.0
- **THEN** the system SHALL raise a validation error
