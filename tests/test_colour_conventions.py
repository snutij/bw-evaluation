"""Contract: nothing is measurable until the colour convention is pinned down.

BT.601, BT.709 and BT.2020 disagree about red by roughly 40% and about blue by
roughly 58%, and all three are defined on gamma-encoded R'G'B'. A framework that
lets the caller stay vague about this is not deterministic, so these tests treat
vagueness as an error rather than a default.
"""

from __future__ import annotations

import unittest

import numpy as np

from bw_evaluation import (
    COEFFICIENTS,
    ColorConvention,
    luminance,
    rgb_to_lab,
    srgb_eotf,
    srgb_oetf,
)

from . import fixtures


class DeclaringTheConvention(unittest.TestCase):
    def test_given_no_encoding_is_stated_when_building_a_convention_then_construction_fails(
        self,
    ):
        # Given a caller who names a coefficient set but not the transfer function
        # When they try to build a convention
        # Then it must fail rather than silently pick one
        with self.assertRaises(TypeError):
            ColorConvention(coefficients="bt709")  # type: ignore[call-arg]

    def test_given_an_unknown_coefficient_set_when_building_a_convention_then_it_is_rejected(
        self,
    ):
        # Given a coefficient set that no recommendation defines
        # When a convention is built from it
        # Then it is rejected instead of falling back to a default
        with self.assertRaises(ValueError):
            ColorConvention(encoding="linear", coefficients="bt2100_invented")

    def test_given_an_unknown_encoding_when_building_a_convention_then_it_is_rejected(
        self,
    ):
        # Given an encoding outside {linear, gamma}
        # When a convention is built
        # Then it is rejected
        with self.assertRaises(ValueError):
            ColorConvention(encoding="log")  # type: ignore[arg-type]

    def test_given_the_three_recommendations_when_reading_the_coefficients_then_they_match_the_standards(
        self,
    ):
        # Given the published ITU-R tables
        # When the package exposes them
        # Then they match to the published precision
        self.assertEqual(COEFFICIENTS["bt601"], (0.299, 0.587, 0.114))
        self.assertEqual(COEFFICIENTS["bt709"], (0.2126, 0.7152, 0.0722))
        self.assertEqual(COEFFICIENTS["bt2020"], (0.2627, 0.6780, 0.0593))

    def test_given_each_coefficient_set_when_summed_then_it_normalises_to_one(self):
        # Given any of the coefficient sets
        # When the three weights are summed
        # Then white maps to white
        for name, weights in COEFFICIENTS.items():
            with self.subTest(recommendation=name):
                self.assertAlmostEqual(sum(weights), 1.0, places=4)


class TransferFunctions(unittest.TestCase):
    def test_given_a_linear_value_when_encoded_and_decoded_then_the_original_is_recovered(
        self,
    ):
        # Given linear-light values spanning the range
        # When passed through the OETF and back through the EOTF
        # Then the round trip is lossless to floating point tolerance
        linear = np.linspace(0.0, 1.0, 257)
        np.testing.assert_allclose(srgb_eotf(srgb_oetf(linear)), linear, atol=1e-10)

    def test_given_middle_grey_when_encoded_then_it_lands_near_the_familiar_118_of_255(
        self,
    ):
        # Given 18% linear reflectance
        # When sRGB encoded
        # Then it sits close to the 118/255 value photographers expect
        encoded = float(srgb_oetf(np.array(0.18)))
        self.assertAlmostEqual(encoded * 255, 118.0, delta=1.5)


class ChoosingLumaOrLuminance(unittest.TestCase):
    def setUp(self):
        self.pure_red = np.zeros((4, 4, 3))
        self.pure_red[..., 0] = 1.0

    def test_given_pure_red_when_weighted_by_bt601_and_bt709_then_the_greys_differ_materially(
        self,
    ):
        # Given a saturated red patch
        # When converted under BT.601 and under BT.709
        # Then the two greys differ by the full 0.299 vs 0.2126 gap, not a rounding error
        bt601 = luminance(
            self.pure_red, ColorConvention(encoding="gamma", coefficients="bt601")
        )
        bt709 = luminance(
            self.pure_red, ColorConvention(encoding="gamma", coefficients="bt709")
        )
        self.assertGreater(abs(float(bt601.mean()) - float(bt709.mean())), 0.08)

    def test_given_the_same_pixels_when_weighted_as_luma_and_as_luminance_then_the_results_differ(
        self,
    ):
        # Given one mid-tone colour image
        # When the same weights are applied to gamma-encoded and to linear values
        # Then the outputs differ, which is why the encoding must be declared
        image = fixtures.colour_diagnostic_landscape()
        as_luma = luminance(
            image, ColorConvention(encoding="gamma", coefficients="bt709")
        )
        as_luminance = luminance(
            image, ColorConvention(encoding="linear", coefficients="bt709")
        )
        self.assertGreater(
            abs(float(as_luma.mean()) - float(as_luminance.mean())), 0.05
        )

    def test_given_a_neutral_grey_when_weighted_by_any_recommendation_then_the_result_is_that_grey(
        self,
    ):
        # Given an achromatic patch
        # When any coefficient set is applied to gamma-encoded values
        # Then the luma equals the input, because the weights sum to one
        grey = fixtures.neutral_grey()
        for name in COEFFICIENTS:
            with self.subTest(recommendation=name):
                result = luminance(
                    grey, ColorConvention(encoding="gamma", coefficients=name)
                )
                np.testing.assert_allclose(result, 0.5, atol=1e-9)

    def test_given_any_photograph_when_converted_to_luminance_then_values_stay_inside_zero_to_one(
        self,
    ):
        # Given several unrelated fixtures
        # When luminance is computed
        # Then no value escapes the unit range
        convention = ColorConvention(encoding="linear")
        for builder in (
            fixtures.high_contrast_geometry,
            fixtures.foggy_textured_scene,
            fixtures.linear_ramp,
        ):
            with self.subTest(image=builder.__name__):
                result = luminance(builder(), convention)
                self.assertGreaterEqual(float(result.min()), 0.0)
                self.assertLessEqual(float(result.max()), 1.0)


class LabConversion(unittest.TestCase):
    def test_given_a_neutral_patch_when_converted_to_lab_then_chroma_is_zero(self):
        # Given an achromatic patch
        # When converted to CIELAB
        # Then a* and b* are zero, so chroma is zero
        lab = rgb_to_lab(fixtures.neutral_grey(), ColorConvention(encoding="gamma"))
        np.testing.assert_allclose(lab[..., 1:], 0.0, atol=1e-6)

    def test_given_reference_white_when_converted_to_lab_then_lightness_is_one_hundred(
        self,
    ):
        # Given pure white
        # When converted to CIELAB under D65
        # Then L* is 100
        white = np.ones((2, 2, 3))
        lab = rgb_to_lab(white, ColorConvention(encoding="gamma"))
        np.testing.assert_allclose(lab[..., 0], 100.0, atol=1e-4)

    def test_given_the_isoluminant_fixture_when_converted_to_lab_then_the_two_patches_share_lightness(
        self,
    ):
        # Given the red/teal checkerboard solved for equal BT.709 luminance
        # When converted to CIELAB
        # Then the L* spread across the image is small while a*/b* differ strongly,
        # which is the precondition every isoluminance test below relies on
        lab = rgb_to_lab(
            fixtures.isoluminant_checkerboard(), ColorConvention(encoding="gamma")
        )
        self.assertLess(float(lab[..., 0].std()), 2.0)
        self.assertGreater(float(lab[..., 1].std()), 20.0)


if __name__ == "__main__":
    unittest.main()
