"""Contract: how much chromatic information exists, and what the numbers mean.

Hasler & Susstrunk (2003) is one of the very few places the literature hands
over calibrated numeric thresholds, so the anchors are part of the contract and
must not drift. The known blind spot, that a large flat colour region scores
high without doing any discriminative work, is pinned by an explicit test so a
future implementer cannot quietly promote this metric to a verdict input.
"""

from __future__ import annotations

import unittest

import numpy as np

from bw_evaluation import (
    COLOURFULNESS_ANCHORS,
    ColorConvention,
    chroma_statistics,
    colourfulness_category,
    hasler_susstrunk_colourfulness,
)

from . import fixtures


def as_8bit(image: np.ndarray) -> np.ndarray:
    return image * 255.0


class ColourfulnessScore(unittest.TestCase):
    def test_given_a_neutral_grey_image_when_measuring_colourfulness_then_the_score_is_zero(
        self,
    ):
        # Given an image with no chroma at all
        # When colourfulness is measured
        # Then the score is exactly zero
        score = hasler_susstrunk_colourfulness(as_8bit(fixtures.neutral_grey()))
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_given_a_fog_scene_when_measuring_colourfulness_then_it_falls_in_the_not_colourful_band(
        self,
    ):
        # Given a near-achromatic fog or concrete scene
        # When colourfulness is measured
        # Then it sits under the published "not colourful" ceiling of 15
        score = hasler_susstrunk_colourfulness(as_8bit(fixtures.foggy_textured_scene()))
        self.assertLess(score, COLOURFULNESS_ANCHORS["slightly colourful"])

    def test_given_a_saturated_scene_when_measuring_colourfulness_then_it_scores_far_above_the_fog_scene(
        self,
    ):
        # Given a strongly coloured image and a near-achromatic one
        # When both are measured
        # Then the ordering is unambiguous, not marginal
        saturated = hasler_susstrunk_colourfulness(
            as_8bit(fixtures.isoluminant_checkerboard())
        )
        fog = hasler_susstrunk_colourfulness(as_8bit(fixtures.foggy_textured_scene()))
        self.assertGreater(saturated, fog + 30.0)

    def test_given_progressively_desaturated_copies_when_measuring_then_the_score_decreases_monotonically(
        self,
    ):
        # Given one image desaturated in steps toward neutral
        # When each step is measured
        # Then colourfulness never increases
        source = fixtures.colour_diagnostic_landscape()
        grey = source.mean(axis=2, keepdims=True)
        scores = []
        for amount in (1.0, 0.75, 0.5, 0.25, 0.0):
            blended = grey + amount * (source - grey)
            scores.append(hasler_susstrunk_colourfulness(as_8bit(blended)))
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_given_a_large_flat_colour_region_when_measuring_then_it_still_scores_high(
        self,
    ):
        # Given a single uniform saturated colour, where chroma does no
        # discriminative work whatsoever
        # When colourfulness is measured
        # Then it is still high, documenting exactly why this metric may never
        # be used alone to decide against black and white
        flat = np.tile(np.array([0.9, 0.1, 0.1]), (64, 64, 1))
        self.assertGreater(hasler_susstrunk_colourfulness(as_8bit(flat)), 40.0)


class ColourfulnessCategories(unittest.TestCase):
    def test_given_a_score_below_fifteen_when_categorising_then_it_reads_not_colourful(
        self,
    ):
        # Given a score under the first published anchor
        # When categorised
        # Then the plain-language label is "not colourful"
        self.assertEqual(colourfulness_category(9.0), "not colourful")

    def test_given_a_score_above_one_hundred_and_nine_when_categorising_then_it_reads_extremely_colourful(
        self,
    ):
        # Given a score above the top published anchor
        # When categorised
        # Then the label is "extremely colourful"
        self.assertEqual(colourfulness_category(140.0), "extremely colourful")

    def test_given_a_score_exactly_on_an_anchor_when_categorising_then_the_higher_band_wins(
        self,
    ):
        # Given a score sitting exactly on a published boundary
        # When categorised
        # Then the boundary is inclusive of the band it opens, so the rule is unambiguous
        self.assertEqual(colourfulness_category(33.0), "moderately colourful")

    def test_given_a_negative_score_when_categorising_then_it_is_rejected(self):
        # Given an impossible score
        # When categorised
        # Then it raises rather than silently clamping
        with self.assertRaises(ValueError):
            colourfulness_category(-1.0)


class ChromaStatisticsContract(unittest.TestCase):
    def setUp(self):
        self.convention = ColorConvention(encoding="gamma")

    def test_given_a_neutral_image_when_summarising_chroma_then_every_statistic_is_zero(
        self,
    ):
        # Given an achromatic image
        # When chroma is summarised
        # Then every statistic is zero, and hue dispersion reads zero because
        # hue is undefined here rather than widely spread
        stats = chroma_statistics(fixtures.neutral_grey(), self.convention)
        self.assertAlmostEqual(stats.mean, 0.0, places=6)
        self.assertAlmostEqual(stats.std, 0.0, places=6)
        self.assertAlmostEqual(stats.p95, 0.0, places=6)
        self.assertAlmostEqual(stats.hue_dispersion, 0.0, places=6)

    def test_given_any_image_when_summarising_chroma_then_the_percentile_is_at_least_the_mean(
        self,
    ):
        # Given a chroma distribution
        # When summarised
        # Then the 95th percentile cannot fall below the mean
        stats = chroma_statistics(
            fixtures.colour_diagnostic_landscape(), self.convention
        )
        self.assertGreaterEqual(stats.p95, stats.mean)

    def test_given_several_hues_at_similar_saturation_when_compared_by_magnitude_then_chroma_alone_barely_separates_them(
        self,
    ):
        # Given one uniformly coloured patch and one image of three very
        # different hues that happen to share a saturation level, which is
        # exactly the colour-diagnostic condition
        # When both are summarised by chroma magnitude
        # Then mean and spread are nearly the same, documenting a blind spot
        # rather than papering over it
        flat = np.tile(np.array([0.2, 0.55, 0.35]), (64, 64, 1))
        varied = fixtures.colour_diagnostic_landscape()
        flat_stats = chroma_statistics(flat, self.convention)
        varied_stats = chroma_statistics(varied, self.convention)
        self.assertAlmostEqual(flat_stats.mean, varied_stats.mean, delta=5.0)
        self.assertLess(varied_stats.std, 5.0)

    def test_given_several_hues_at_similar_saturation_when_compared_by_hue_then_they_separate_clearly(
        self,
    ):
        # Given the same two images
        # When hue dispersion is compared instead of chroma magnitude
        # Then the separation is unambiguous, which is why hue angle is
        # reported and not only saturation
        flat = np.tile(np.array([0.2, 0.55, 0.35]), (64, 64, 1))
        varied = fixtures.colour_diagnostic_landscape()
        self.assertAlmostEqual(
            chroma_statistics(flat, self.convention).hue_dispersion, 0.0, places=6
        )
        self.assertGreater(
            chroma_statistics(varied, self.convention).hue_dispersion, 0.3
        )

    def test_given_hue_dispersion_when_measured_on_any_image_then_it_stays_a_proportion(
        self,
    ):
        # Given a circular dispersion measure
        # When computed on unrelated images
        # Then it always lies in [0, 1]
        for builder in (
            fixtures.foggy_textured_scene,
            fixtures.isoluminant_checkerboard,
            fixtures.linear_ramp,
        ):
            with self.subTest(image=builder.__name__):
                dispersion = chroma_statistics(
                    builder(), self.convention
                ).hue_dispersion
                self.assertGreaterEqual(dispersion, 0.0)
                self.assertLessEqual(dispersion, 1.0)


if __name__ == "__main__":
    unittest.main()
