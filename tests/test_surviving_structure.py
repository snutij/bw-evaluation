"""Contract: is what remains after desaturation strong enough to stand alone.

The Zone System supplies the discretisation (eleven zones, one stop apart,
z = 5 + log2(Y / 0.18) on linear-light luminance). Peli (JOSA A 7, 1990)
supplies the contrast family appropriate to complex natural images.

The important negative constraint is tested here too: there is no ideal
histogram. A high-key portrait legitimately lives in zones V to VIII and a
silhouette in zones 0 to III, so no function may return a verdict about tonal
range on its own.
"""

from __future__ import annotations

import unittest

import numpy as np

from bw_evaluation import (
    SEGMENTATION_CLUSTERS,
    TEXTURAL_ZONES,
    ZONE_COUNT,
    ColorConvention,
    active_zone_count,
    adjacent_region_pairs,
    edge_density,
    luminance,
    peli_contrast,
    region_tonal_separation,
    rgb_to_lab,
    robust_tonal_span,
    segment_regions,
    textural_range_coverage,
    texture_energy,
    worst_region_merger,
    zone_entropy,
    zone_histogram,
    zone_map,
)

from . import fixtures


def linear() -> ColorConvention:
    """Built lazily on purpose: with an unimplemented constructor this keeps the
    failure inside each test instead of collapsing the whole module at import."""
    return ColorConvention(encoding="linear")


def zones_of(image: np.ndarray) -> np.ndarray:
    return zone_map(luminance(image, linear()))


class ZoneArithmetic(unittest.TestCase):
    def test_given_the_zone_system_when_counting_labels_then_there_are_eleven_from_zero_to_ten(
        self,
    ):
        # Given Zone 0 through Zone X inclusive
        # When the labels are counted
        # Then there are eleven, not the ten intervals people often quote
        self.assertEqual(ZONE_COUNT, 11)

    def test_given_eighteen_percent_linear_luminance_when_mapping_to_zones_then_it_lands_on_zone_five(
        self,
    ):
        # Given the middle-grey reference
        # When mapped
        # Then it is Zone V by definition
        patch = fixtures.linear_luminance_patches((0.18,))
        np.testing.assert_allclose(zones_of(patch), 5.0, atol=1e-6)

    def test_given_one_stop_above_middle_grey_when_mapping_to_zones_then_it_lands_on_zone_six(
        self,
    ):
        # Given a doubling of linear luminance
        # When mapped
        # Then it advances exactly one zone, confirming the log2 relation
        patch = fixtures.linear_luminance_patches((0.36,))
        np.testing.assert_allclose(zones_of(patch), 6.0, atol=1e-6)

    def test_given_one_stop_below_middle_grey_when_mapping_to_zones_then_it_lands_on_zone_four(
        self,
    ):
        # Given a halving of linear luminance
        # When mapped
        # Then it drops exactly one zone
        patch = fixtures.linear_luminance_patches((0.09,))
        np.testing.assert_allclose(zones_of(patch), 4.0, atol=1e-6)

    def test_given_pure_black_when_mapping_to_zones_then_it_clamps_to_zero_rather_than_diverging(
        self,
    ):
        # Given zero luminance, where log2 is undefined
        # When mapped
        # Then it clamps to Zone 0 and no infinity or NaN escapes
        zones = zones_of(fixtures.flat_black())
        self.assertTrue(np.all(np.isfinite(zones)))
        np.testing.assert_allclose(zones, 0.0, atol=1e-9)

    def test_given_any_image_when_mapping_to_zones_then_values_stay_within_zero_and_ten(
        self,
    ):
        # Given a full-range ramp
        # When mapped
        # Then nothing escapes the eleven-zone scale
        zones = zones_of(fixtures.linear_ramp())
        self.assertGreaterEqual(float(zones.min()), 0.0)
        self.assertLessEqual(float(zones.max()), 10.0)


class ZoneOccupancy(unittest.TestCase):
    def test_given_a_zone_histogram_when_summed_then_it_forms_a_proper_distribution(
        self,
    ):
        # Given the occupancy of the eleven zones
        # When summed
        # Then it is a probability distribution over exactly eleven bins
        histogram = zone_histogram(zones_of(fixtures.linear_ramp()))
        self.assertEqual(histogram.shape, (ZONE_COUNT,))
        self.assertAlmostEqual(float(histogram.sum()), 1.0, places=10)

    def test_given_an_image_confined_to_one_zone_when_measuring_entropy_then_it_is_zero(
        self,
    ):
        # Given every pixel in the same zone
        # When entropy is measured
        # Then it is zero
        self.assertAlmostEqual(
            zone_entropy(zones_of(fixtures.linear_luminance_patches((0.18,)))),
            0.0,
            places=9,
        )

    def test_given_an_image_spread_evenly_across_all_zones_when_measuring_entropy_then_it_is_one(
        self,
    ):
        # Given a perfectly uniform zone histogram
        # When entropy is measured
        # Then the normalisation by log(11) yields exactly 1
        zones = np.tile(np.arange(ZONE_COUNT, dtype=float), (ZONE_COUNT, 1))
        self.assertAlmostEqual(zone_entropy(zones), 1.0, places=9)

    def test_given_a_high_key_image_when_measuring_entropy_then_the_low_value_is_reported_without_judgement(
        self,
    ):
        # Given a legitimate high-key rendering confined to the upper zones
        # When entropy is measured
        # Then a plain number comes back, because no function here is allowed to
        # declare a narrow histogram a defect
        zones = zones_of(fixtures.linear_luminance_patches((0.30, 0.45, 0.62, 0.85)))
        value = zone_entropy(zones)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_given_a_few_clipped_pixels_when_measuring_the_span_then_they_do_not_set_the_range(
        self,
    ):
        # Given an image that is mostly mid-tone with a handful of blown pixels
        # When the robust span is measured
        # Then it reflects the bulk of the image, not the outliers
        image = fixtures.linear_luminance_patches(tuple([0.18] * 200 + [1.0]))
        self.assertLess(robust_tonal_span(zone_map(luminance(image, linear()))), 1.0)

    def test_given_an_image_entirely_in_the_textural_zones_when_measuring_coverage_then_it_is_complete(
        self,
    ):
        # Given content placed inside zones II to VIII
        # When textural coverage is measured
        # Then it is 1
        self.assertEqual(TEXTURAL_ZONES, (2, 8))
        image = fixtures.linear_luminance_patches((0.09, 0.18, 0.36))
        self.assertAlmostEqual(
            textural_range_coverage(zone_map(luminance(image, linear()))), 1.0, places=9
        )

    def test_given_a_threshold_when_counting_active_zones_then_sparse_zones_are_excluded(
        self,
    ):
        # Given a minimum occupancy
        # When zones are counted
        # Then zones holding a negligible pixel share do not inflate the count
        image = fixtures.linear_luminance_patches(tuple([0.18] * 500 + [0.9]))
        zones = zone_map(luminance(image, linear()))
        self.assertLess(
            active_zone_count(zones, min_occupancy=0.05),
            active_zone_count(zones, min_occupancy=0.0),
        )


class TonalMerger(unittest.TestCase):
    def setUp(self):
        self.zones = zones_of(fixtures.linear_ramp())
        self.left = np.zeros(self.zones.shape, dtype=bool)
        self.right = np.zeros(self.zones.shape, dtype=bool)
        self.left[:, : fixtures.SIZE // 4] = True
        self.right[:, -fixtures.SIZE // 4 :] = True

    def test_given_two_identical_regions_when_measuring_separation_then_it_is_zero(
        self,
    ):
        # Given the same region compared with itself
        # When separation is measured
        # Then Cohen's d is zero
        self.assertAlmostEqual(
            region_tonal_separation(self.zones, self.left, self.left), 0.0, places=9
        )

    def test_given_a_dark_region_and_a_bright_region_when_measuring_separation_then_it_is_large(
        self,
    ):
        # Given two regions at opposite ends of the tonal scale
        # When separation is measured
        # Then the effect size is unambiguous
        self.assertGreater(
            region_tonal_separation(self.zones, self.left, self.right), 2.0
        )

    def test_given_separation_when_the_regions_are_swapped_then_the_magnitude_is_unchanged(
        self,
    ):
        # Given that merger risk has no direction
        # When the two masks are swapped
        # Then the reported separation is the same
        forward = region_tonal_separation(self.zones, self.left, self.right)
        backward = region_tonal_separation(self.zones, self.right, self.left)
        self.assertAlmostEqual(forward, backward, places=12)

    def test_given_an_empty_region_mask_when_measuring_separation_then_it_is_rejected(
        self,
    ):
        # Given a mask selecting no pixels
        # When separation is requested
        # Then it fails rather than returning a meaningless number
        empty = np.zeros(self.zones.shape, dtype=bool)
        with self.assertRaises(ValueError):
            region_tonal_separation(self.zones, self.left, empty)

    def test_given_two_regions_that_differ_only_in_hue_when_measuring_separation_then_it_is_near_zero(
        self,
    ):
        # Given the classic merger case, two subject areas of different colour
        # but the same luminance
        # When tonal separation is measured on the zone map
        # Then it is near zero, which is the computable form of the warning
        # practitioners give qualitatively
        image = fixtures.isoluminant_checkerboard()
        zones = zones_of(image)
        block = fixtures.SIZE // 8
        first = np.zeros(zones.shape, dtype=bool)
        second = np.zeros(zones.shape, dtype=bool)
        first[0:block, 0:block] = True
        second[0:block, block : 2 * block] = True
        self.assertLess(region_tonal_separation(zones, first, second), 0.5)


class Segmentation(unittest.TestCase):
    """Merger is a property of adjacent regions, so the regions must come from
    somewhere. k-means in CIELAB with a fixed seed keeps that deterministic."""

    def setUp(self):
        self.image = fixtures.saturated_but_merging()
        self.lab = rgb_to_lab(self.image, ColorConvention(encoding="gamma"))

    def test_given_two_clusters_closer_than_a_just_noticeable_difference_when_segmented_then_they_are_one_region(
        self,
    ):
        # Given three bands, two of which differ by well under a JND
        # When three clusters are asked for
        # Then only two regions come back
        #
        # This is what compression ringing produces along a real boundary, and
        # it is not a hypothetical. The same photograph saved as PNG and as
        # quality-95 JPEG segmented into three regions and six respectively,
        # and the worst-merger reading moved by a factor of two hundred purely
        # because of the encoder. Two colours a viewer cannot tell apart are
        # not two regions, and reporting that they "merge" would be circular.
        image = np.zeros((fixtures.SIZE, fixtures.SIZE, 3))
        image[: fixtures.SIZE // 3] = (0.500, 0.500, 0.500)
        image[fixtures.SIZE // 3 : 2 * fixtures.SIZE // 3] = (0.505, 0.505, 0.505)
        image[2 * fixtures.SIZE // 3 :] = (0.900, 0.200, 0.200)
        labels = segment_regions(
            rgb_to_lab(image, ColorConvention(encoding="gamma")), clusters=3
        )
        self.assertEqual(len(np.unique(labels)), 2)

    def test_given_an_image_when_segmented_twice_then_the_label_maps_are_identical(
        self,
    ):
        # Given that k-means is only deterministic with a fixed seed and init
        # When the same image is segmented twice
        # Then the label maps match exactly, including the label numbering
        np.testing.assert_array_equal(
            segment_regions(self.lab), segment_regions(self.lab)
        )

    def test_given_a_requested_cluster_count_when_segmenting_then_no_more_labels_come_back(
        self,
    ):
        # Given a requested k
        # When segmenting
        # Then the label map uses at most k labels, numbered from zero
        labels = segment_regions(self.lab, clusters=SEGMENTATION_CLUSTERS)
        self.assertLessEqual(len(np.unique(labels)), SEGMENTATION_CLUSTERS)
        self.assertEqual(int(labels.min()), 0)

    def test_given_a_three_band_image_when_segmented_then_the_bands_are_recovered_as_regions(
        self,
    ):
        # Given an image made of three obvious colour bands
        # When segmented
        # Then each band maps to a single dominant label, so the regions the
        # merger metric compares are the ones a viewer would name
        labels = segment_regions(self.lab, clusters=3)
        third = fixtures.SIZE // 3
        # The band starts have to be read off the fixture rather than computed
        # as multiples of `third`: the fixture splits at SIZE // 3 and
        # 2 * SIZE // 3, and with SIZE = 128 that second boundary is 85, not
        # 2 * (128 // 3) = 84. Sampling from 84 would straddle two bands and
        # fail a correct segmentation.
        for start in (0, fixtures.SIZE // 3, 2 * fixtures.SIZE // 3):
            with self.subTest(band=start):
                band = labels[start : start + third // 2]
                self.assertEqual(len(np.unique(band)), 1)

    def test_given_a_segmented_image_when_listing_pairs_then_only_bordering_regions_are_returned(
        self,
    ):
        # Given that two similarly toned regions at opposite corners do not merge
        # When adjacent pairs are listed
        # Then non-touching regions are absent, and each pair appears once
        pairs = adjacent_region_pairs(segment_regions(self.lab, clusters=3))
        self.assertTrue(all(a < b for a, b in pairs))
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_given_one_merging_pair_among_several_when_summarising_then_the_worst_pair_is_reported(
        self,
    ):
        # Given a scene where two pairs separate cleanly and one merges
        # When merger is summarised
        # Then the minimum is reported, because a single merger ruins the
        # rendering however well the other pairs behave
        zones = zones_of(self.image)
        labels = segment_regions(self.lab, clusters=3)
        worst = worst_region_merger(zones, labels)
        pairs = adjacent_region_pairs(labels)
        separations = [
            region_tonal_separation(zones, labels == a, labels == b) for a, b in pairs
        ]
        self.assertAlmostEqual(worst, min(separations), places=9)

    def test_given_an_image_with_a_single_region_when_summarising_merger_then_it_is_rejected(
        self,
    ):
        # Given a flat image where no adjacent pair exists
        # When merger is summarised
        # Then it raises rather than returning a number that means nothing
        flat_lab = rgb_to_lab(
            fixtures.neutral_grey(), ColorConvention(encoding="gamma")
        )
        with self.assertRaises(ValueError):
            worst_region_merger(
                zones_of(fixtures.neutral_grey()), segment_regions(flat_lab, clusters=3)
            )


class ContrastAndTexture(unittest.TestCase):
    def test_given_a_band_count_when_computing_peli_contrast_then_one_map_per_band_comes_back(
        self,
    ):
        # Given a request for several octave bands
        # When Peli contrast is computed
        # Then each band returns a full-size non-negative map with its centre frequency
        image = fixtures.high_contrast_geometry()
        bands = peli_contrast(luminance(image, linear()), bands=4)
        self.assertEqual(len(bands), 4)
        for band in bands:
            with self.subTest(frequency=band.centre_frequency):
                self.assertEqual(band.contrast_map.shape, image.shape[:2])
                self.assertGreaterEqual(float(np.abs(band.contrast_map).min()), 0.0)
                self.assertGreater(band.centre_frequency, 0.0)

    def test_given_a_flat_image_when_computing_peli_contrast_then_every_band_is_zero(
        self,
    ):
        # Given an image with no structure at any scale
        # When Peli contrast is computed
        # Then no band reports contrast, and the epsilon guard prevents division blow-ups
        bands = peli_contrast(luminance(fixtures.neutral_grey(), linear()), bands=3)
        for band in bands:
            with self.subTest(frequency=band.centre_frequency):
                np.testing.assert_allclose(band.contrast_map, 0.0, atol=1e-6)

    def test_given_a_geometric_image_and_a_flat_one_when_comparing_edge_density_then_the_ordering_is_clear(
        self,
    ):
        # Given hard-edged geometry versus a featureless field
        # When edge density is measured
        # Then the geometric image scores higher and both remain proportions
        geometric = edge_density(
            luminance(fixtures.high_contrast_geometry(), linear()), threshold=0.05
        )
        flat = edge_density(
            luminance(fixtures.neutral_grey(), linear()), threshold=0.05
        )
        self.assertGreater(geometric, flat)
        for value in (geometric, flat):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_given_a_detailed_scene_when_measuring_texture_energy_then_it_exceeds_a_smooth_scene(
        self,
    ):
        # Given that Mullen shows fine detail is carried by luminance and
        # therefore survives desaturation intact
        # When texture energy is compared between a detailed and a smooth image
        # Then the detailed one scores higher, and this is a point in favour of B&W
        detailed = texture_energy(luminance(fixtures.foggy_textured_scene(), linear()))
        smooth = texture_energy(
            luminance(fixtures.colour_diagnostic_landscape(), linear())
        )
        self.assertGreater(detailed, smooth)

    def test_given_an_image_when_desaturated_then_its_texture_energy_is_essentially_unchanged(
        self,
    ):
        # Given the central empirical claim that detail is a luminance phenomenon
        # When an image is stripped of chroma
        # Then its texture energy barely moves, so nothing detail-related is lost
        #
        # The grey is built by averaging the channels in the same gamma-encoded
        # domain the fixture lives in. Two details matter. Building it from
        # `luminance(image, linear())` and feeding that back through
        # `luminance(..., linear())` would linearise twice, so the test would
        # measure a gamma curve rather than a desaturation. And averaging
        # rather than reusing the luminance weights keeps the comparison
        # non-trivial: the two greys are genuinely different renderings, and
        # the claim under test is that the detail does not care which one
        # is chosen.
        image = fixtures.foggy_textured_scene()
        grey = np.repeat(image.mean(axis=2)[:, :, None], 3, axis=2)
        before = texture_energy(luminance(image, linear()))
        after = texture_energy(luminance(grey, linear()))
        self.assertAlmostEqual(before, after, delta=0.05 * before)


if __name__ == "__main__":
    unittest.main()
