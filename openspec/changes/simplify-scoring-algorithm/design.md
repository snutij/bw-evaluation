## Context

The BW Evaluation scoring algorithm currently has 5 weighted components (contrast 28%, texture 22%, saturation 28%, composition 12%, metadata 10%) plus scene detection bonuses (+0 to +8). Several components are not trustworthy predictors of B&W conversion potential:
- Saturation scoring uses 6+ interdependent sub-penalties to work around a flawed premise
- Metadata scoring guesses quality from EXIF data
- Scene detection bonuses apply up to 8 points based on brittle heuristic thresholds
- Composition includes generic photography metrics (rule-of-thirds, center luminosity) that aren't B&W-specific

The codebase is ~1,086 lines of production code with 75 tests. All scoring lives in `bw_scorer.py`, config in `config.py`.

## Goals / Non-Goals

**Goals:**
- Delete untrustworthy scoring logic rather than patching it
- Add channel separation as the missing metric that actually predicts B&W conversion potential
- Simplify `ScoringConfig` by removing ~15 parameters tied to deleted logic
- Maintain determinism, testability, and explainability of scores
- Keep conversion preset selection via scene detection (low-stakes usage)

**Non-Goals:**
- Introducing ML/AI-based scoring
- Validating weights against a labeled dataset (future work)
- Changing the conversion engine (`convert_bw.py` presets)
- Adding new CLI subcommands

## Decisions

### 1. Channel separation algorithm: per-pixel std dev across R/G/B channels

**Approach:** For each pixel, compute the standard deviation of its (R, G, B) values. Average across all pixels. Normalize to 0-100.

**Why this over alternatives:**
- *Alternative A: Correlation between channel histograms* — More complex, harder to interpret, and histogram-level analysis loses spatial information.
- *Alternative B: Mutual information between channels* — Computationally expensive, overkill for this use case.
- Per-pixel std dev is simple, fast (one NumPy operation), and directly measures what we care about: how much the channels diverge at each point. High divergence = more creative freedom when choosing B&W channel mix.

**Formula:**
```python
per_pixel_std = np.std(image, axis=2)  # std across B,G,R channels per pixel
mean_std = np.mean(per_pixel_std)
channel_sep_score = min(100, (mean_std / 50) * 100)  # normalize, 50 as empirical ceiling
```

### 2. Saturation simplification: single inverse mean

**Approach:** Replace 6 sub-penalties with `max(0, 100 - (mean_sat / 255) * 100)`.

**Why:** The sub-penalties (gamut spread, palette, strong/weak high-sat) attempted to model nuance that doesn't exist — low saturation simply correlates weakly with B&W suitability. One honest line is better than 50 lines of sophisticated wrongness.

**Weight reduction:** 28% → 10%. Channel separation now captures the "color potential" aspect that saturation was trying (and failing) to model.

### 3. Composition: keep only tonal sub-metrics

**Keep:** Plane separation (tonal zone contrast) and highlight distribution (exposure quality).
**Delete:** Rule-of-thirds (generic photography) and center luminosity (arbitrary ideal of 127).

**Why:** Plane separation directly measures whether an image has distinct tonal zones — the foundation of B&W aesthetics (Ansel Adams zone system). Highlight distribution catches over/under-exposure. The others don't predict B&W quality.

### 4. Scene detection: scoring-only removal

**Keep** `detect_scene_type` function — it's used by `convert_bw.py` to pick presets.
**Remove** the bonus application in `score_image`. Scene type is still computed and stored in results for informational purposes.

### 5. New weight distribution

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Contrast | 35% | Strongest validated predictor |
| Texture | 25% | Proven metric, slight uplift |
| Channel separation | 25% | New, captures conversion potential |
| Saturation | 10% | Honest about limited value |
| Composition | 5% | Only tonal sub-metrics remain |

Weights sum to 1.0. Contrast + texture = 60% (measurable image properties), channel separation = 25% (conversion potential), saturation + composition = 15% (secondary signals).

## Risks / Trade-offs

- **[Score discontinuity]** All existing scores become incomparable after this change. → Acceptable: no persistent score database exists. Document in changelog.
- **[Channel separation ceiling]** The normalization divisor (50) is empirical. → Start with 50, adjust after testing on real photos. Make it configurable in `ScoringConfig`.
- **[Over-simplification of saturation]** Single mean may lose some valid signal. → The 10% weight limits damage. Channel separation picks up the slack for colorful-but-good-B&W images.
- **[Test rewrite scope]** Most golden reference tests will break. → Rewrite them to validate new behavior rather than trying to preserve old assertions.
- **[Composition at 5%]** May be too low to meaningfully affect ranking. → Acceptable: composition is the weakest B&W-specific signal. Can increase later if validated.
