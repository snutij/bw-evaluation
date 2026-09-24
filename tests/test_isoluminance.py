"""Contract: detecting chromatic contrast that luminance does not carry.

This is the single most decisive signal in the framework. Cadik (2008) names
isoluminant colour as the canonical failure mode of luminance-only conversion,
and Mullen (1985) explains why the effect is concentrated at coarse scales:
chromatic contrast sensitivity falls off with spatial frequency far faster than
achromatic sensitivity, so fine detail is a luminance phenomenon.

Because no canonical named metric exists for this, the tests also pin the
honesty requirement: the module must advertise itself as derived, not cited.
"""

from __future__ import annotations

import unittest

import numpy as np

from bw_evaluation import ColorConvention, rgb_to_lab
from bw_evaluation.isoluminance import (
    EVIDENCE_LEVEL,
    chromatic_gradient,
    chromatic_gradient_energy_ratio,
    coarse_scale_chroma_contribution,
    isoluminant_edge_fraction,
    luminance_gradient,
)

from . import fixtures


def convention() -> ColorConvention:
    """Built lazily on purpose: with an unimplemented constructor this keeps the
    failure inside each test instead of collapsing the whole module at import."""
    return ColorConvention(encoding="gamma")


def lab_of(image: np.ndarray) -> np.ndarray:
    return rgb_to_lab(image, convention())


class ProvenanceOfTheseMetrics(unittest.TestCase):
    def test_given_these_metrics_are_not_in_the_literature_when_inspected_then_they_declare_themselves_derived(
        self,
    ):
        # Given that no published paper defines a standard isoluminance ratio
        # When the module is inspected
        # Then it labels itself derived, so the report can never present it as cited
        self.assertEqual(EVIDENCE_LEVEL, "derived")


class GradientChannels(unittest.TestCase):
    def test_given_a_flat_image_when_computing_either_gradient_then_both_are_zero(self):
        # Given an image with no spatial variation
        # When both gradients are computed
        # Then both are zero everywhere
        lab = lab_of(fixtures.neutral_grey())
        np.testing.assert_allclose(luminance_gradient(lab), 0.0, atol=1e-9)
        np.testing.assert_allclose(chromatic_gradient(lab), 0.0, atol=1e-9)

    def test_given_the_isoluminant_checkerboard_when_computing_gradients_then_only_the_chromatic_one_responds(
        self,
    ):
        # Given edges that exist purely in hue
        # When both gradients are computed
        # Then the chromatic gradient fires strongly and the luminance gradient barely moves
        lab = lab_of(fixtures.isoluminant_checkerboard())
        self.assertGreater(float(chromatic_gradient(lab).max()), 40.0)
        self.assertLess(float(luminance_gradient(lab).max()), 5.0)

    def test_given_a_monochrome_texture_when_computing_gradients_then_only_the_luminance_one_responds(
        self,
    ):
        # Given the mirror-image case, detail carried entirely by luminance
        # When both gradients are computed
        # Then the roles are reversed
        lab = lab_of(fixtures.foggy_textured_scene())
        self.assertGreater(float(luminance_gradient(lab).max()), 10.0)
        self.assertLess(float(chromatic_gradient(lab).mean()), 2.0)

    def test_given_any_image_when_computing_gradients_then_the_maps_keep_the_pixel_shape(
        self,
    ):
        # Given any image
        # When gradients are computed
        # Then the maps are per-pixel and non-negative, so they can be masked and pooled
        lab = lab_of(fixtures.high_contrast_geometry())
        for gradient in (luminance_gradient(lab), chromatic_gradient(lab)):
            self.assertEqual(gradient.shape, lab.shape[:2])
            self.assertGreaterEqual(float(gradient.min()), 0.0)


class IsoluminantEdgeFraction(unittest.TestCase):
    def test_given_no_thresholds_are_supplied_when_measuring_then_the_call_fails(self):
        # Given that both thresholds are noise-dependent with no defensible default
        # When the caller omits them
        # Then the call fails rather than inventing values
        with self.assertRaises(TypeError):
            isoluminant_edge_fraction(lab_of(fixtures.neutral_grey()))  # type: ignore[call-arg]

    def test_given_the_isoluminant_checkerboard_when_measuring_then_most_edges_are_chromatic_only(
        self,
    ):
        # Given the canonical failure case
        # When the fraction is measured
        # Then the large majority of detected edges have no luminance support
        fraction = isoluminant_edge_fraction(
            lab_of(fixtures.isoluminant_checkerboard()), tau_chroma=5.0, tau_luma=2.0
        )
        self.assertGreater(fraction, 0.8)

    def test_given_the_luminance_separated_control_when_measuring_then_almost_no_edges_are_chromatic_only(
        self,
    ):
        # Given the same geometry and comparable saturation but strong L* separation
        # When the fraction is measured
        # Then it is near zero, proving the metric tracks isoluminance and not saturation
        fraction = isoluminant_edge_fraction(
            lab_of(fixtures.luminance_separated_colours()), tau_chroma=5.0, tau_luma=2.0
        )
        self.assertLess(fraction, 0.1)

    def test_given_a_flat_image_with_no_edges_at_all_when_measuring_then_the_result_is_zero_not_undefined(
        self,
    ):
        # Given an image where the denominator would be empty
        # When the fraction is measured
        # Then it returns zero instead of dividing by zero
        fraction = isoluminant_edge_fraction(
            lab_of(fixtures.neutral_grey()), tau_chroma=5.0, tau_luma=2.0
        )
        self.assertEqual(fraction, 0.0)

    def test_given_any_image_when_measuring_then_the_fraction_stays_within_zero_and_one(
        self,
    ):
        # Given a range of unrelated images
        # When the fraction is measured
        # Then it is always a proportion
        for builder in (
            fixtures.high_contrast_geometry,
            fixtures.colour_diagnostic_landscape,
            fixtures.foggy_textured_scene,
            fixtures.linear_ramp,
        ):
            with self.subTest(image=builder.__name__):
                fraction = isoluminant_edge_fraction(
                    lab_of(builder()), tau_chroma=5.0, tau_luma=2.0
                )
                self.assertGreaterEqual(fraction, 0.0)
                self.assertLessEqual(fraction, 1.0)


class ChromaticEnergyRatio(unittest.TestCase):
    def test_given_a_monochrome_image_when_measuring_the_chromatic_share_then_it_is_near_zero(
        self,
    ):
        # Given an image whose structure is entirely achromatic
        # When the chromatic share of gradient energy is measured
        # Then it is negligible
        self.assertLess(
            chromatic_gradient_energy_ratio(lab_of(fixtures.foggy_textured_scene())),
            0.05,
        )

    def test_given_an_isoluminant_image_when_measuring_the_chromatic_share_then_it_dominates(
        self,
    ):
        # Given an image whose structure is entirely chromatic
        # When the share is measured
        # Then it dominates the total gradient energy
        self.assertGreater(
            chromatic_gradient_energy_ratio(
                lab_of(fixtures.isoluminant_checkerboard())
            ),
            0.8,
        )

    def test_given_a_single_scale_is_requested_when_measuring_then_the_result_still_lies_between_zero_and_one(
        self,
    ):
        # Given a caller who asks for one scale only
        # When the ratio is measured
        # Then it remains a well-formed proportion
        ratio = chromatic_gradient_energy_ratio(
            lab_of(fixtures.colour_diagnostic_landscape()), scales=(1,)
        )
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)


class CoarseScaleChromaContribution(unittest.TestCase):
    def test_given_broad_colour_bands_when_measuring_coarse_scale_chroma_then_the_contribution_is_high(
        self,
    ):
        # Given a landscape whose category is signalled by large colour regions,
        # the Oliva and Schyns diagnostic-colour condition
        # When coarse-scale chromatic energy is measured
        # Then the contribution is high, flagging that colour carries the scene gist
        contribution = coarse_scale_chroma_contribution(
            lab_of(fixtures.colour_diagnostic_landscape()), cycles_per_image=8.0
        )
        self.assertGreater(contribution, 0.6)

    def test_given_fine_chromatic_noise_when_measuring_coarse_scale_chroma_then_the_contribution_is_low(
        self,
    ):
        # Given chroma that exists only as fine high-frequency variation, which
        # Mullen shows the visual system resolves poorly
        # When coarse-scale chromatic energy is measured
        # Then the contribution is low, so such colour counts as discardable
        rng = np.random.default_rng(seed=7)
        image = np.clip(0.5 + 0.08 * rng.standard_normal((128, 128, 3)), 0.0, 1.0)
        contribution = coarse_scale_chroma_contribution(
            lab_of(image), cycles_per_image=8.0
        )
        self.assertLess(contribution, 0.3)

    def test_given_the_same_image_twice_when_measuring_then_the_two_results_are_identical(
        self,
    ):
        # Given the deterministic requirement that motivates this whole framework
        # When any metric is called twice on identical input
        # Then the outputs are bit-identical, with no sampling or RNG anywhere
        lab = lab_of(fixtures.colour_diagnostic_landscape())
        first = coarse_scale_chroma_contribution(lab, cycles_per_image=8.0)
        second = coarse_scale_chroma_contribution(lab, cycles_per_image=8.0)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
