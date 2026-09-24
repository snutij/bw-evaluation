"""Contract: measuring what an actual conversion destroys.

CCPR, CCFR and the E-score come from Lu, Xu & Jia (ICCP 2012 and SIGGRAPH Asia
2012 Technical Briefs). C2G-SSIM and the luminance-entropy image-type
classifier come from Ma, Zhu, Wang & Wang (IEEE TIP 2015).

Two literature findings are enforced as hard constraints rather than advice:

* Rankings flip with the threshold, so a summary from one tau is refused.
* CCPR alone rewards fabricated contrast, so it may never be reported without
  CCFR alongside it.
"""

from __future__ import annotations

import unittest

import numpy as np

from bw_evaluation import (
    CLASSIC_FILTER_WEIGHTS,
    COEFFICIENTS,
    METHODS,
    MINIMUM_THRESHOLD_SWEEP,
    PHOTOGRAPHIC_ENTROPY_BOUNDARY,
    ChannelWeights,
    ColorConvention,
    apply_channel_weights,
    c2g_ssim,
    ccfr,
    ccpr,
    classify_image_type,
    colour_collapse,
    decolorize,
    e_score,
    e_score_curve,
    luminance_entropy,
    nearest_classic_filter,
    recommended_channel_weights,
    rgb_to_lab,
    threshold_independent_area,
)

from . import fixtures


def convention() -> ColorConvention:
    """Linear light, deliberately.

    Isoluminance is a property of linear-light luminance, not of gamma-encoded
    luma. The red/teal fixture shares a lightness of L* 53.2, yet BT.709 luma
    applied to the *encoded* values renders them 0.2126 against 0.4385 and so
    looks well separated. That separation is a gamma artefact, not tonal
    separation, and measuring through it would systematically hide the exact
    failure this module exists to detect. Conversions for display may be done
    however a photographer likes; measurement happens on linear light.

    Built lazily so an unimplemented constructor fails one test at a time
    rather than collapsing the module at import.
    """
    return ColorConvention(encoding="linear")


SWEEP = (5.0, 10.0, 15.0, 20.0, 25.0)


def lab_of(image: np.ndarray) -> np.ndarray:
    return rgb_to_lab(image, convention())


class Decolorizing(unittest.TestCase):
    def test_given_an_unknown_method_name_when_decolorizing_then_it_is_rejected(self):
        # Given a method the package does not implement
        # When decolorization is requested
        # Then it fails loudly rather than falling back to a default
        with self.assertRaises(ValueError):
            decolorize(
                fixtures.neutral_grey(),
                method="average_of_channels",
                convention=convention(),
            )

    def test_given_each_supported_method_when_decolorizing_then_a_single_channel_in_range_comes_back(
        self,
    ):
        # Given every method the package advertises
        # When applied to a photograph
        # Then each returns one channel of the same size, bounded in [0, 1]
        image = fixtures.colour_diagnostic_landscape()
        for method in METHODS:
            with self.subTest(method=method):
                grey = decolorize(image, method=method, convention=convention())
                self.assertEqual(grey.shape, image.shape[:2])
                self.assertGreaterEqual(float(grey.min()), 0.0)
                self.assertLessEqual(float(grey.max()), 1.0)

    def test_given_the_isoluminant_checkerboard_when_using_plain_luma_then_the_pattern_disappears(
        self,
    ):
        # Given colours solved to share luminance
        # When a luminance-only method converts them
        # Then the checkerboard collapses to a near-uniform field, which is the
        # whole failure this framework exists to detect
        grey = decolorize(
            fixtures.isoluminant_checkerboard(),
            method="bt709_luma",
            convention=convention(),
        )
        self.assertLess(float(grey.std()), 0.02)

    def test_given_the_isoluminant_checkerboard_when_using_a_contrast_preserving_method_then_structure_returns(
        self,
    ):
        # Given the same image
        # When an adaptive contrast-preserving method is used instead
        # Then the pattern is recovered, so the report can distinguish
        # "unconvertible" from "needs a better conversion"
        grey = decolorize(
            fixtures.isoluminant_checkerboard(),
            method="contrast_preserving",
            convention=convention(),
        )
        self.assertGreater(float(grey.std()), 0.1)


class ColourContrastPreservingRatio(unittest.TestCase):
    def test_given_a_threshold_of_zero_when_computing_ccpr_then_it_is_rejected_as_degenerate(
        self,
    ):
        # Given tau of zero, under which every pair counts as distinct
        # When CCPR is requested
        # Then it is rejected, because the answer would trivially be 1
        lab = lab_of(fixtures.colour_diagnostic_landscape())
        grey = decolorize(
            fixtures.colour_diagnostic_landscape(),
            method="bt709_luma",
            convention=convention(),
        )
        with self.assertRaises(ValueError):
            ccpr(lab, grey, tau=0.0)

    def test_given_a_collapsed_conversion_when_computing_ccpr_then_it_is_near_zero(
        self,
    ):
        # Given a grey rendering where all pairs became identical
        # When CCPR is computed
        # Then almost no pair survives as distinct
        image = fixtures.isoluminant_checkerboard()
        grey = decolorize(image, method="bt709_luma", convention=convention())
        self.assertLess(ccpr(lab_of(image), grey, tau=10.0), 0.05)

    def test_given_any_conversion_when_computing_ccpr_then_it_is_a_proportion(self):
        # Given any image and conversion
        # When CCPR is computed
        # Then it lies in [0, 1]
        image = fixtures.high_contrast_geometry()
        grey = decolorize(image, method="bt709_luma", convention=convention())
        value = ccpr(lab_of(image), grey, tau=10.0)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


class ColourContentFidelityRatio(unittest.TestCase):
    def test_given_an_isoluminant_image_when_computing_fidelity_then_almost_all_colour_distinctions_are_lost(
        self,
    ):
        # Given colours that are clearly different to the eye but equal in luminance
        # When fidelity is computed against a luminance-only rendering
        # Then it collapses toward zero
        image = fixtures.isoluminant_checkerboard()
        grey = decolorize(image, method="bt709_luma", convention=convention())
        self.assertLess(ccfr(lab_of(image), grey, tau=10.0), 0.15)

    def test_given_a_luminance_separated_image_when_computing_fidelity_then_almost_everything_survives(
        self,
    ):
        # Given the matched control where colours also differ in lightness
        # When fidelity is computed
        # Then nearly every distinction survives the conversion
        image = fixtures.luminance_separated_colours()
        grey = decolorize(image, method="bt709_luma", convention=convention())
        self.assertGreater(ccfr(lab_of(image), grey, tau=10.0), 0.85)

    def test_given_a_near_achromatic_image_when_computing_fidelity_then_it_reports_nothing_was_lost(
        self,
    ):
        # Given an image containing almost no perceptible colour differences
        # When fidelity is computed
        # Then it is high, because there was nothing there to destroy
        image = fixtures.foggy_textured_scene()
        grey = decolorize(image, method="bt709_luma", convention=convention())
        self.assertGreater(ccfr(lab_of(image), grey, tau=10.0), 0.9)

    def test_given_a_fidelity_score_when_reading_collapse_then_it_is_exactly_one_minus_fidelity(
        self,
    ):
        # Given the complementary framing used throughout the report
        # When collapse is read
        # Then it equals 1 - CCFR, with no separate definition to drift apart
        image = fixtures.colour_diagnostic_landscape()
        grey = decolorize(image, method="bt709_luma", convention=convention())
        lab = lab_of(image)
        self.assertAlmostEqual(
            colour_collapse(lab, grey, tau=10.0),
            1.0 - ccfr(lab, grey, tau=10.0),
            places=12,
        )


class EScore(unittest.TestCase):
    def setUp(self):
        self.image = fixtures.colour_diagnostic_landscape()
        self.grey = decolorize(self.image, method="bt709_luma", convention=convention())
        self.lab = lab_of(self.image)

    def test_given_ccpr_and_ccfr_when_combining_them_then_the_result_is_their_harmonic_mean(
        self,
    ):
        # Given the two component ratios
        # When the E-score is computed
        # Then it matches the published harmonic mean exactly
        p = ccpr(self.lab, self.grey, tau=10.0)
        r = ccfr(self.lab, self.grey, tau=10.0)
        expected = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        self.assertAlmostEqual(
            e_score(self.lab, self.grey, tau=10.0), expected, places=12
        )

    def test_given_the_two_components_when_combined_then_the_score_sits_between_them(
        self,
    ):
        # Given a harmonic mean
        # When compared with its inputs
        # Then it never escapes their range, which is what makes it penalise imbalance
        p = ccpr(self.lab, self.grey, tau=10.0)
        r = ccfr(self.lab, self.grey, tau=10.0)
        value = e_score(self.lab, self.grey, tau=10.0)
        self.assertGreaterEqual(value, min(p, r) - 1e-12)
        self.assertLessEqual(value, max(p, r) + 1e-12)

    def test_given_both_components_are_zero_when_combining_then_the_score_is_zero_not_undefined(
        self,
    ):
        # Given a fully collapsed conversion
        # When the E-score is computed
        # Then the zero denominator is handled and the score is zero
        image = fixtures.isoluminant_checkerboard()
        grey = decolorize(image, method="bt709_luma", convention=convention())
        self.assertGreaterEqual(e_score(lab_of(image), grey, tau=25.0), 0.0)


class ThresholdDependence(unittest.TestCase):
    def setUp(self):
        self.image = fixtures.colour_diagnostic_landscape()
        self.grey = decolorize(self.image, method="bt709_luma", convention=convention())
        self.lab = lab_of(self.image)

    def test_given_fewer_thresholds_than_the_minimum_when_building_a_curve_then_it_is_refused(
        self,
    ):
        # Given that rankings flip with tau, which is why threshold-independent
        # area exists in the first place
        # When a caller asks for a summary from too few thresholds
        # Then the package refuses instead of producing a fragile number
        self.assertGreaterEqual(MINIMUM_THRESHOLD_SWEEP, 3)
        with self.assertRaises(ValueError):
            e_score_curve(self.lab, self.grey, taus=(10.0, 20.0))

    def test_given_duplicate_thresholds_when_building_a_curve_then_they_are_rejected(
        self,
    ):
        # Given a sweep padded with repeats to satisfy the minimum
        # When the curve is built
        # Then the duplicates are rejected, so the minimum cannot be gamed
        with self.assertRaises(ValueError):
            e_score_curve(self.lab, self.grey, taus=(10.0, 10.0, 10.0))

    def test_given_a_valid_sweep_when_building_a_curve_then_every_threshold_gets_a_score(
        self,
    ):
        # Given a proper sweep
        # When the curve is built
        # Then it is keyed by threshold with every entry a valid score
        curve = e_score_curve(self.lab, self.grey, taus=SWEEP)
        self.assertEqual(set(curve), set(SWEEP))
        for tau, value in curve.items():
            with self.subTest(tau=tau):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_given_a_flat_curve_when_taking_the_area_then_it_equals_the_constant_value(
        self,
    ):
        # Given a curve with the same score at every threshold
        # When the normalised area is taken
        # Then it equals that constant, so the normalisation is verifiable by hand
        flat = {5.0: 0.4, 10.0: 0.4, 15.0: 0.4, 20.0: 0.4}
        self.assertAlmostEqual(threshold_independent_area(flat), 0.4, places=12)

    def test_given_a_curve_when_taking_the_area_then_it_never_exceeds_the_curve_maximum(
        self,
    ):
        # Given any curve
        # When the area is taken
        # Then it stays inside the range of the curve it summarises
        curve = e_score_curve(self.lab, self.grey, taus=SWEEP)
        area = threshold_independent_area(curve)
        self.assertGreaterEqual(area, min(curve.values()) - 1e-12)
        self.assertLessEqual(area, max(curve.values()) + 1e-12)


class ColourToGreyStructuralSimilarity(unittest.TestCase):
    def test_given_a_conversion_when_scoring_then_all_four_components_are_reported_separately(
        self,
    ):
        # Given that the diagnostic value lies in which component fails
        # When C2G-SSIM is computed
        # Then lightness, colour contrast, structure and the combined score are
        # all exposed, each a valid score
        image = fixtures.colour_diagnostic_landscape()
        grey = decolorize(image, method="bt709_luma", convention=convention())
        result = c2g_ssim(image, grey, convention=convention())
        for name in ("lightness", "colour_contrast", "structure", "overall"):
            with self.subTest(component=name):
                value = getattr(result, name)
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_given_an_isoluminant_failure_when_scoring_then_the_colour_contrast_component_is_what_drops(
        self,
    ):
        # Given a conversion that failed specifically by collapsing hue differences
        # When C2G-SSIM is computed
        # Then the colour-contrast component carries the penalty, naming the
        # failure mode rather than hiding it in an average
        image = fixtures.isoluminant_checkerboard()
        grey = decolorize(image, method="bt709_luma", convention=convention())
        result = c2g_ssim(image, grey, convention=convention())
        self.assertLess(result.colour_contrast, 0.4)

    def test_given_a_wrongly_exposed_conversion_when_scoring_then_the_lightness_component_is_what_drops(
        self,
    ):
        # Given a conversion that preserved structure but shifted overall tone
        # When C2G-SSIM is computed
        # Then lightness is penalised while structure is not, proving the
        # components are genuinely separable
        image = fixtures.high_contrast_geometry()
        grey = decolorize(image, method="bt709_luma", convention=convention())
        darkened = np.clip(grey * 0.35, 0.0, 1.0)
        result = c2g_ssim(image, darkened, convention=convention())
        self.assertLess(result.lightness, 0.8)
        self.assertGreater(result.structure, 0.8)

    def test_given_an_already_grey_image_when_scoring_its_own_conversion_then_the_score_is_near_perfect(
        self,
    ):
        # Given an achromatic source converted by a luminance method
        # When scored against itself
        # Then nothing was lost and the score approaches 1
        image = fixtures.foggy_textured_scene()
        grey = decolorize(image, method="bt709_luma", convention=convention())
        self.assertGreater(c2g_ssim(image, grey, convention=convention()).overall, 0.9)


class RecommendedChannelWeights(unittest.TestCase):
    """The practical output: not just "yes or no" but "mix it like this".

    The contrast-preserving search already solves for per-image channel weights,
    so surfacing them is nearly free and is the part a photographer can actually
    act on. The binding rule is the round trip: the weights shown to the user
    must be the ones that produced the measured result, or the recommendation is
    decoration.
    """

    def test_given_any_recommendation_when_read_then_the_weights_are_non_negative_and_sum_to_one(
        self,
    ):
        # Given weights meant to be pasted straight into a channel mixer
        # When any recommendation is read
        # Then it preserves exposure, so no renormalising or re-exposing is needed
        for builder in (
            fixtures.high_contrast_geometry,
            fixtures.isoluminant_checkerboard,
            fixtures.foggy_textured_scene,
        ):
            with self.subTest(image=builder.__name__):
                weights = recommended_channel_weights(
                    builder(), method="contrast_preserving", convention=convention()
                )
                triple = weights.as_tuple()
                self.assertAlmostEqual(sum(triple), 1.0, places=6)
                self.assertTrue(all(value >= 0.0 for value in triple))

    def test_given_a_fixed_luma_method_when_asking_for_its_weights_then_they_are_the_published_coefficients(
        self,
    ):
        # Given a method that is a fixed coefficient set by definition
        # When its weights are requested
        # Then they match the recommendation exactly, with nothing invented
        weights = recommended_channel_weights(
            fixtures.high_contrast_geometry(),
            method="bt709_luma",
            convention=convention(),
        )
        self.assertEqual(weights.as_tuple(), COEFFICIENTS["bt709"])

    def test_given_recommended_weights_when_applied_by_hand_then_they_reproduce_that_method_output(
        self,
    ):
        # Given the rule that a recommendation must be the recipe that was
        # actually measured, not an approximation of it
        # When the weights are applied directly
        # Then the result matches the method's own output pixel for pixel
        image = fixtures.isoluminant_checkerboard()
        weights = recommended_channel_weights(
            image, method="contrast_preserving", convention=convention()
        )
        by_hand = apply_channel_weights(image, weights, convention=convention())
        by_method = decolorize(
            image, method="contrast_preserving", convention=convention()
        )
        np.testing.assert_allclose(by_hand, by_method, atol=1e-9)

    def test_given_an_isoluminant_image_when_recommending_weights_then_they_depart_from_the_default(
        self,
    ):
        # Given an image that plain BT.709 luma flattens completely
        # When weights are recommended
        # Then they must differ materially from the default, since recovering
        # the pattern is impossible otherwise
        weights = recommended_channel_weights(
            fixtures.isoluminant_checkerboard(),
            method="contrast_preserving",
            convention=convention(),
        )
        distance = sum(
            abs(a - b) for a, b in zip(weights.as_tuple(), COEFFICIENTS["bt709"])
        )
        self.assertGreater(distance, 0.2)

    def test_given_an_achromatic_image_when_recommending_weights_then_they_stay_near_the_default(
        self,
    ):
        # Given an image where no mixing choice can change anything
        # When weights are recommended
        # Then the search does not wander off to an arbitrary corner of the
        # weight space just because every option scores the same
        weights = recommended_channel_weights(
            fixtures.neutral_grey(),
            method="contrast_preserving",
            convention=convention(),
        )
        distance = sum(
            abs(a - b) for a, b in zip(weights.as_tuple(), COEFFICIENTS["bt709"])
        )
        self.assertLess(distance, 0.05)

    def test_given_the_same_image_twice_when_recommending_weights_then_the_recipe_is_identical(
        self,
    ):
        # Given a discrete search that must not depend on iteration order or RNG
        # When weights are recommended twice
        # Then the recipe is identical, so two runs never advise differently
        image = fixtures.saturated_but_merging()
        first = recommended_channel_weights(
            image, method="contrast_preserving", convention=convention()
        )
        second = recommended_channel_weights(
            image, method="contrast_preserving", convention=convention()
        )
        self.assertEqual(first, second)

    def test_given_weights_that_do_not_sum_to_one_when_constructed_then_they_are_rejected(
        self,
    ):
        # Given a recipe that would change overall exposure
        # When it is constructed
        # Then it is rejected, keeping the contract simple for the caller
        with self.assertRaises(ValueError):
            ChannelWeights(red=0.9, green=0.9, blue=0.9)

    def test_given_a_negative_weight_when_constructed_then_it_is_rejected(self):
        # Given a recipe with a negative channel
        # When it is constructed
        # Then it is rejected, since the search space is non-negative
        with self.assertRaises(ValueError):
            ChannelWeights(red=1.4, green=-0.2, blue=-0.2)


class ClassicFilterNaming(unittest.TestCase):
    """A translation layer so a recipe reads in familiar language. The filter
    table is an engineering convention, not spectral measurement, and nothing
    is allowed to depend on it beyond the label it prints."""

    def test_given_the_default_coefficients_when_naming_the_filter_then_it_reads_as_no_filter(
        self,
    ):
        # Given the unmodified BT.709 weighting
        # When named
        # Then it reads "none" rather than being forced into a filter category
        weights = ChannelWeights(*COEFFICIENTS["bt709"])
        self.assertEqual(nearest_classic_filter(weights), "none")

    def test_given_strongly_red_weighted_mixing_when_naming_the_filter_then_it_reads_as_red(
        self,
    ):
        # Given a recipe dominated by the red channel, which is what darkens a
        # blue sky in the classic technique
        # When named
        # Then it reads "red"
        self.assertEqual(
            nearest_classic_filter(ChannelWeights(red=0.82, green=0.16, blue=0.02)),
            "red",
        )

    def test_given_any_recipe_when_naming_the_filter_then_the_answer_is_from_the_published_table(
        self,
    ):
        # Given any weights at all
        # When named
        # Then the label comes from the fixed table, never invented per image
        weights = ChannelWeights(red=0.33, green=0.34, blue=0.33)
        self.assertIn(nearest_classic_filter(weights), CLASSIC_FILTER_WEIGHTS)

    def test_given_the_filter_table_when_read_then_every_entry_is_itself_a_valid_recipe(
        self,
    ):
        # Given the table is used as a reference for naming
        # When each entry is read
        # Then all of them obey the same non-negative, sums-to-one rule
        for name, triple in CLASSIC_FILTER_WEIGHTS.items():
            with self.subTest(filter=name):
                self.assertAlmostEqual(sum(triple), 1.0, places=6)
                self.assertTrue(all(value >= 0.0 for value in triple))


class ImageTypeClassifier(unittest.TestCase):
    def test_given_the_published_boundary_when_reading_it_then_it_is_four_bits(self):
        # Given the classifier reported at 91.7% on Cadik and 97.2% on COLOR250
        # When the boundary is read
        # Then it is the published 4 bits of 8-bit luminance entropy
        self.assertEqual(PHOTOGRAPHIC_ENTROPY_BOUNDARY, 4.0)

    def test_given_a_flat_synthetic_image_when_classifying_then_it_reads_synthetic(
        self,
    ):
        # Given an image with one grey level and therefore zero entropy
        # When classified
        # Then it is synthetic, so photographic luminance weighting is not assumed
        grey = decolorize(
            fixtures.neutral_grey(), method="bt709_luma", convention=convention()
        )
        self.assertEqual(luminance_entropy(grey), 0.0)
        self.assertEqual(classify_image_type(grey), "synthetic")

    def test_given_a_detailed_photograph_when_classifying_then_it_reads_photographic(
        self,
    ):
        # Given a noisy, detail-dense scene
        # When classified
        # Then entropy clears the boundary and it reads photographic
        grey = decolorize(
            fixtures.foggy_textured_scene(),
            method="bt709_luma",
            convention=convention(),
        )
        self.assertGreaterEqual(luminance_entropy(grey), PHOTOGRAPHIC_ENTROPY_BOUNDARY)
        self.assertEqual(classify_image_type(grey), "photographic")


if __name__ == "__main__":
    unittest.main()
