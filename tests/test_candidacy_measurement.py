"""Contract: what a single-image measurement may and may not contain.

Most of this file encodes negative findings. The research turned up several
widely repeated beliefs with no empirical support, and the cheapest way to stop
them creeping into an implementation is to make their absence a failing test:

* Aspect ratio is a feature in Datta et al. and in OSCAR, and Wang, Lee & Lee
  (2022) find high-rated AVA photographs skew toward square and 3:2. Nothing
  links any ratio to monochrome suitability.
* "High ISO noise suits black and white" and "mixed white balance suits black
  and white" are conventional wisdom with zero published backing.
* "A good image uses the full tonal range" is an engineering convention, and a
  backlit silhouette is the standing counter-example.
* No published no-reference aesthetic model is validated for paired
  colour-versus-grey decisions.

The module also produces no verdict at all. That decision is in `ranking.py`.
"""

from __future__ import annotations

import unittest

import numpy as np

from bw_evaluation import (
    ACHROMATIC_INPUT_CEILING,
    ADVISORY_HEURISTIC_NAMES,
    CONTEXTUAL_FEATURE_NAMES,
    WORKING_LONG_EDGE,
    ColorConvention,
    measure_bw_candidacy,
)

from . import fixtures

SWEEP = (5.0, 10.0, 15.0, 20.0, 25.0)


def convention() -> ColorConvention:
    """Built lazily on purpose: with an unimplemented constructor this keeps the
    failure inside each test instead of collapsing the whole module at import."""
    return ColorConvention(encoding="gamma")


def measure(image: np.ndarray, **kwargs):
    params = {"convention": convention(), "taus": SWEEP}
    params.update(kwargs)
    return measure_bw_candidacy(image, **params)


class MeasurementShape(unittest.TestCase):
    def test_given_a_photograph_when_measured_then_every_evidence_group_is_present(
        self,
    ):
        # Given any photograph
        # When measured
        # Then chroma information, measured conversion loss, surviving structure,
        # context and advisories are all separated, so a reader can see which
        # kind of evidence says what
        result = measure(fixtures.high_contrast_geometry())
        self.assertTrue(result.chroma_information)
        self.assertTrue(result.conversion_loss)
        self.assertTrue(result.surviving_structure)
        self.assertIsNotNone(result.contextual_features)
        self.assertIsNotNone(result.advisory_heuristics)

    def test_given_a_measurement_when_looking_for_a_verdict_or_a_score_then_there_is_neither(
        self,
    ):
        # Given that no absolute threshold is supported by any literature
        # When a caller reaches for a judgement at the single-image level
        # Then neither a verdict nor an overall score exists to reach for
        result = measure(fixtures.high_contrast_geometry())
        for forbidden in ("verdict", "score", "overall_score", "label"):
            with self.subTest(attribute=forbidden):
                self.assertFalse(hasattr(result, forbidden))

    def test_given_a_measurement_when_inspecting_its_values_then_none_of_them_is_an_array(
        self,
    ):
        # Given that an array anywhere inside would make two measurements
        # incomparable with == and silently void the determinism guarantee
        # When every recorded value is inspected
        # Then all of them are scalars, strings or tuples
        result = measure(fixtures.colour_diagnostic_landscape())
        groups = [
            result.chroma_information,
            result.surviving_structure,
            result.contextual_features,
        ]
        groups.extend(result.conversion_loss.values())
        for group in groups:
            for name, value in group.items():
                with self.subTest(field=name):
                    self.assertNotIsInstance(value, np.ndarray)
                    self.assertIsInstance(value, (int, float, str, tuple, bool))

    def test_given_the_same_image_twice_when_measured_then_the_two_results_are_equal(
        self,
    ):
        # Given the determinism the whole framework claims
        # When the same image is measured twice
        # Then the results compare equal, with no sampling anywhere in the pipeline
        image = fixtures.colour_diagnostic_landscape()
        self.assertEqual(measure(image), measure(image))

    def test_given_a_measurement_when_read_then_it_records_the_convention_and_working_size(
        self,
    ):
        # Given that the same pixels give different greys under different
        # conventions, and that texture metrics move with resolution
        # When the measurement is read
        # Then both are recorded, so a result can be reproduced exactly
        result = measure(fixtures.high_contrast_geometry())
        self.assertEqual(result.convention, convention())
        self.assertEqual(result.working_long_edge, WORKING_LONG_EDGE)

    def test_given_the_same_photograph_at_two_export_sizes_when_measured_then_the_evidence_matches(
        self,
    ):
        # Given one image exported small and large
        # When both are measured, each resized to the fixed working size first
        # Then the structural evidence agrees, rather than drifting with pixel count
        small = fixtures.high_contrast_geometry()
        large = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
        a = measure(small).surviving_structure
        b = measure(large).surviving_structure
        for name in a:
            with self.subTest(field=name):
                self.assertAlmostEqual(
                    a[name], b[name], delta=0.05 * max(abs(a[name]), 1e-6)
                )


class ConversionEvidence(unittest.TestCase):
    def test_given_any_conversion_entry_when_read_then_fidelity_accompanies_preservation(
        self,
    ):
        # Given that CCPR alone rewards fabricated contrast
        # When a conversion entry is read
        # Then CCFR is always beside it, with the sweep and the separable
        # C2G-SSIM components recorded as flat scalars
        entry = measure(fixtures.colour_diagnostic_landscape()).conversion_loss[
            "bt709_luma"
        ]
        for key in (
            "ccpr",
            "ccfr",
            "colour_collapse",
            "e_score_curve",
            "threshold_independent_area",
            "c2g_lightness",
            "c2g_colour_contrast",
            "c2g_structure",
            "c2g_overall",
        ):
            with self.subTest(key=key):
                self.assertIn(key, entry)

    def test_given_a_recorded_sweep_when_read_then_it_is_a_tuple_of_threshold_and_score_pairs(
        self,
    ):
        # Given the no-arrays rule
        # When the E-score curve is read back
        # Then it is a tuple of (tau, score) pairs covering the requested sweep
        entry = measure(fixtures.colour_diagnostic_landscape()).conversion_loss[
            "bt709_luma"
        ]
        curve = entry["e_score_curve"]
        self.assertIsInstance(curve, tuple)
        self.assertEqual(tuple(tau for tau, _ in curve), SWEEP)

    def test_given_an_isoluminant_image_when_an_adaptive_conversion_recovers_it_then_both_are_recorded(
        self,
    ):
        # Given an image that plain luma destroys but contrast-preserving
        # decolorization rescues, exactly the Lu, Xu and Jia result
        # When both methods are measured
        # Then each has its own entry, so "needs a better conversion" stays
        # distinguishable from "cannot work in black and white"
        result = measure(
            fixtures.isoluminant_checkerboard(),
            methods=("bt709_luma", "contrast_preserving"),
        )
        naive = result.conversion_loss["bt709_luma"]["colour_collapse"]
        adaptive = result.conversion_loss["contrast_preserving"]["colour_collapse"]
        self.assertGreater(naive, adaptive)

    def test_given_an_image_with_one_merging_region_when_measured_then_the_worst_pair_drives_the_evidence(
        self,
    ):
        # Given a scene where two regions separate cleanly and one merges
        # When merger is summarised
        # Then the worst adjacent pair is what is recorded, because a single bad
        # merger ruins a monochrome rendering regardless of the others
        result = measure(fixtures.saturated_but_merging())
        self.assertIn("worst_region_merger", result.surviving_structure)
        self.assertLess(result.surviving_structure["worst_region_merger"], 0.5)


class RecommendedRecipe(unittest.TestCase):
    """A measurement says how to convert, never whether to. The verdict stays
    batch-relative and lives in `ranking.py`."""

    def test_given_a_photograph_when_measured_then_it_carries_a_channel_mixer_recipe(
        self,
    ):
        # Given that the useful output is a recipe, not only a judgement
        # When a photograph is measured
        # Then weights and a familiar filter name come back, as plain values
        result = measure(fixtures.colour_diagnostic_landscape())
        self.assertIn(result.recommended_method, ("bt709_luma", "contrast_preserving"))
        self.assertIsInstance(result.recommended_weights, tuple)
        self.assertAlmostEqual(sum(result.recommended_weights), 1.0, places=6)
        self.assertTrue(result.recommended_filter)

    def test_given_each_measured_method_when_read_then_its_own_weights_are_recorded_beside_its_scores(
        self,
    ):
        # Given several methods measured on one image
        # When each entry is read
        # Then the recipe that produced those scores sits next to them
        result = measure(fixtures.saturated_but_merging())
        for method, entry in result.conversion_loss.items():
            with self.subTest(method=method):
                self.assertIn("channel_weights", entry)
                self.assertAlmostEqual(sum(entry["channel_weights"]), 1.0, places=6)

    def test_given_several_methods_when_recommending_then_the_one_that_preserved_most_is_chosen(
        self,
    ):
        # Given competing conversions of the same image
        # When a recommendation is made
        # Then it is the method with the least measured colour collapse, so the
        # advice follows the evidence already gathered rather than a preference
        result = measure(fixtures.isoluminant_checkerboard())
        best = min(
            result.conversion_loss,
            key=lambda name: result.conversion_loss[name]["colour_collapse"],
        )
        self.assertEqual(result.recommended_method, best)
        self.assertEqual(
            result.recommended_weights, result.conversion_loss[best]["channel_weights"]
        )

    def test_given_a_poor_black_and_white_candidate_when_measured_then_a_recipe_is_still_offered(
        self,
    ):
        # Given an image the evidence will rank badly
        # When it is measured
        # Then a recipe is still produced, because "how" and "whether" are
        # separate questions and only the second is batch-relative
        result = measure(fixtures.colour_diagnostic_landscape())
        self.assertTrue(result.recommended_filter)
        self.assertFalse(hasattr(result, "verdict"))


class ThingsThatMayNotDriveARanking(unittest.TestCase):
    def test_given_aspect_ratio_when_checking_the_ranking_inputs_then_it_is_absent(
        self,
    ):
        # Given that no evidence links a ratio to monochrome suitability
        # When the ranking inputs are listed
        # Then aspect ratio does not appear among them
        self.assertNotIn(
            "aspect_ratio", measure(fixtures.high_contrast_geometry()).ranking_inputs
        )

    def test_given_aspect_ratio_when_reading_the_measurement_then_it_is_still_recorded_as_context(
        self,
    ):
        # Given that it is a legitimate descriptive feature in the aesthetics literature
        # When the measurement is read
        # Then it is present as context, documented without being trusted
        result = measure(fixtures.high_contrast_geometry())
        self.assertIn("aspect_ratio", result.contextual_features)
        self.assertIn("aspect_ratio", CONTEXTUAL_FEATURE_NAMES)

    def test_given_the_same_content_cropped_wide_and_tall_when_measured_then_the_ranking_evidence_matches(
        self,
    ):
        # Given identical content at 3:1 and at 1:3
        # When both are measured
        # Then every value that may drive a ranking agrees, so shape alone
        # cannot move a photograph up or down a batch
        wide, tall = fixtures.wide_and_tall_crops()
        a, b = measure(wide), measure(tall)
        for name in a.ranking_inputs:
            with self.subTest(field=name):
                self.assertAlmostEqual(
                    a.surviving_structure.get(
                        name, a.chroma_information.get(name, 0.0)
                    ),
                    b.surviving_structure.get(
                        name, b.chroma_information.get(name, 0.0)
                    ),
                    delta=0.05,
                )

    def test_given_the_unsupported_heuristics_when_checking_the_ranking_inputs_then_none_appear(
        self,
    ):
        # Given noise, mixed white balance and full tonal range, all unsupported
        # When the ranking inputs are listed
        # Then none of them contributed
        result = measure(fixtures.foggy_textured_scene())
        for name in ADVISORY_HEURISTIC_NAMES:
            with self.subTest(heuristic=name):
                self.assertNotIn(name, result.ranking_inputs)

    def test_given_an_advisory_heuristic_when_reported_then_it_carries_an_honest_evidence_label(
        self,
    ):
        # Given that these are surfaced anyway, because photographers use them
        # When each advisory is read
        # Then it states its evidence level, and none may claim to be empirical
        result = measure(fixtures.foggy_textured_scene())
        self.assertTrue(result.advisory_heuristics)
        for heuristic in result.advisory_heuristics:
            with self.subTest(heuristic=heuristic.name):
                self.assertIn(heuristic.evidence_level, {"convention", "unsupported"})

    def test_given_a_backlit_silhouette_when_measured_then_its_narrow_histogram_is_not_penalised(
        self,
    ):
        # Given content legitimately confined to the extreme zones
        # When measured
        # Then tonal range appears only as an advisory, never as a ranking input,
        # because there is no ideal histogram
        result = measure(fixtures.backlit_silhouette())
        self.assertNotIn("full_tonal_range", result.ranking_inputs)
        self.assertIn("full_tonal_range", {h.name for h in result.advisory_heuristics})

    def test_given_every_ranking_input_when_listed_then_each_one_traces_to_measured_evidence(
        self,
    ):
        # Given the requirement that any ordering be explainable
        # When the ranking inputs are listed
        # Then every one is a key that actually appears in the recorded evidence
        result = measure(fixtures.isoluminant_checkerboard())
        measured = (
            set(result.chroma_information)
            | set(result.surviving_structure)
            | set(result.conversion_loss)
        )
        self.assertTrue(result.ranking_inputs)
        for name in result.ranking_inputs:
            with self.subTest(field=name):
                self.assertIn(name, measured)


class LearnedAestheticSignals(unittest.TestCase):
    def test_given_no_aesthetic_model_is_supplied_when_measuring_then_the_evidence_is_still_complete(
        self,
    ):
        # Given that the framework must work with no learned model at all
        # When no delta is supplied
        # Then a full measurement is still produced and the field reads None
        result = measure(fixtures.high_contrast_geometry(), aesthetic_delta=None)
        self.assertTrue(result.conversion_loss)
        self.assertIsNone(result.aesthetic_delta)

    def test_given_a_learned_delta_when_supplied_then_it_is_recorded_but_is_not_a_ranking_input(
        self,
    ):
        # Given that no published model is validated for paired desaturation
        # judgements, and that all inherit colour and genre bias
        # When a delta is supplied
        # Then it is stored for inspection but cannot drive an ordering by default
        result = measure(fixtures.high_contrast_geometry(), aesthetic_delta=0.4)
        self.assertEqual(result.aesthetic_delta, 0.4)
        self.assertNotIn("aesthetic_delta", result.ranking_inputs)

    def test_given_two_images_differing_only_in_their_supplied_delta_when_measured_then_the_evidence_is_identical(
        self,
    ):
        # Given the same pixels with opposite learned opinions attached
        # When both are measured
        # Then every measured group matches, proving the delta is inert
        image = fixtures.colour_diagnostic_landscape()
        a = measure(image, aesthetic_delta=+1.0)
        b = measure(image, aesthetic_delta=-1.0)
        self.assertEqual(a.conversion_loss, b.conversion_loss)
        self.assertEqual(a.surviving_structure, b.surviving_structure)

    def test_given_a_delta_outside_the_expected_range_when_measuring_then_it_is_rejected(
        self,
    ):
        # Given a delta that is not a normalised difference in [-1, 1]
        # When measurement is attempted
        # Then it is rejected rather than silently rescaled
        with self.assertRaises(ValueError):
            measure(fixtures.high_contrast_geometry(), aesthetic_delta=12.0)


class InputValidation(unittest.TestCase):
    def test_given_a_single_threshold_when_measuring_then_it_is_refused(self):
        # Given that a one-threshold summary is what the literature warns against
        # When measurement is attempted with one tau
        # Then it is refused
        with self.assertRaises(ValueError):
            measure(fixtures.high_contrast_geometry(), taus=(10.0,))

    def test_given_an_already_achromatic_image_when_measured_then_it_warns_that_the_question_does_not_apply(
        self,
    ):
        # Given an input with no colour to discard
        # When measured
        # Then the measurement warns rather than pretending to weigh anything
        result = measure(fixtures.neutral_grey())
        self.assertTrue(
            any("achromatic" in warning.lower() for warning in result.warnings)
        )

    def test_given_a_merely_low_chroma_image_when_measured_then_it_is_not_dismissed_as_achromatic(
        self,
    ):
        # Given fog, which is low in chroma but genuinely a colour photograph
        # When measured
        # Then no achromatic warning fires, so the ceiling sits below the
        # published "not colourful" anchor rather than at it
        self.assertLess(ACHROMATIC_INPUT_CEILING, 15.0)
        result = measure(fixtures.foggy_textured_scene())
        self.assertFalse(
            any("achromatic" in warning.lower() for warning in result.warnings)
        )

    def test_given_a_single_channel_array_when_measuring_then_it_is_rejected(self):
        # Given an array that is not a three-channel image
        # When measurement is attempted
        # Then it is rejected with a clear error
        with self.assertRaises(ValueError):
            measure(np.zeros((16, 16)))

    def test_given_pixel_values_outside_zero_to_one_when_measuring_then_they_are_rejected(
        self,
    ):
        # Given 8-bit values passed in by mistake
        # When measurement is attempted
        # Then it is rejected rather than silently misinterpreted, since the
        # colourfulness anchors and the transfer functions assume a fixed scale
        with self.assertRaises(ValueError):
            measure(np.full((16, 16, 3), 255.0))


if __name__ == "__main__":
    unittest.main()
