"""Contract: judgement happens only relative to a batch, and never for free.

Two constraints pull against each other here and both have to hold.

Ranking needs an ordering. But Ma et al. show the diagnostic value is in the
separable components, and Zeger et al. show metric rankings openly conflict
across images, so the package must not invent a weighting between "how little
was lost" and "how much survives". The resolution is Pareto layering, which is
a real ordering that needs no weights, plus a `total_order` function that
refuses to run until the caller names the weighting and owns it.

Labels are percentiles within the submitted batch. A test below deliberately
asserts that the same photograph can be labelled differently in two different
batches, so that nobody later reads a label as an absolute claim.
"""

from __future__ import annotations

import functools
import unittest

from bw_evaluation import (
    MINIMUM_BATCH_SIZE,
    POOR_CANDIDATE_PERCENTILE,
    RANKING_KEYS,
    STRONG_CANDIDATE_PERCENTILE,
    ColorConvention,
    measure_bw_candidacy,
    rank_batch,
    total_order,
)

from . import fixtures

SWEEP = (5.0, 10.0, 15.0, 20.0, 25.0)


def convention() -> ColorConvention:
    """Built lazily on purpose: with an unimplemented constructor this keeps the
    failure inside each test instead of collapsing the whole module at import."""
    return ColorConvention(encoding="gamma")


@functools.lru_cache(maxsize=1)
def _measured_catalogue_once() -> tuple[tuple[str, object], ...]:
    return tuple(
        (name, measure_bw_candidacy(image, convention=convention(), taus=SWEEP))
        for name, image in fixtures.catalogue().items()
    )


def measured_catalogue() -> dict:
    """The nine-image catalogue, measured once and reused.

    Measuring one image at the 1024px working size costs a few seconds, and
    this helper is called from roughly twenty `setUp` methods, so measuring
    afresh each time would put the suite into the tens of minutes. Nothing is
    weakened by the cache: measurement determinism is asserted in
    `test_candidacy_measurement.py` against uncached calls, and the ranking
    determinism test here still runs `rank_batch` twice over the same inputs,
    which is precisely the property it claims to check. A fresh dict is handed
    out each call so no test can disturb another.
    """
    return dict(_measured_catalogue_once())


class BatchSize(unittest.TestCase):
    def test_given_a_batch_smaller_than_the_minimum_when_ranking_then_it_is_refused(
        self,
    ):
        # Given that a percentile over three images is theatre
        # When a tiny batch is submitted
        # Then ranking is refused rather than producing meaningless positions
        self.assertGreaterEqual(MINIMUM_BATCH_SIZE, 8)
        measurements = measured_catalogue()
        tiny = {name: measurements[name] for name in list(measurements)[:3]}
        with self.assertRaises(ValueError):
            rank_batch(tiny)

    def test_given_a_single_image_when_ranking_then_it_is_refused_rather_than_labelled(
        self,
    ):
        # Given one photograph and no distribution to place it in
        # When ranking is attempted
        # Then it is refused, because a batch-relative label needs a batch
        measurements = measured_catalogue()
        with self.assertRaises(ValueError):
            rank_batch({"only": measurements["fog"]})


class RankingShape(unittest.TestCase):
    def setUp(self):
        self.ranking = rank_batch(measured_catalogue())

    def test_given_a_batch_when_ranked_then_every_image_gets_a_position_on_both_axes(
        self,
    ):
        # Given a full catalogue
        # When ranked
        # Then each image carries a loss percentile and a structure percentile,
        # both proper proportions, plus the Pareto layer between them
        self.assertEqual(len(self.ranking.images), len(fixtures.catalogue()))
        for entry in self.ranking.images:
            with self.subTest(image=entry.image_id):
                self.assertGreaterEqual(entry.loss_percentile, 0.0)
                self.assertLessEqual(entry.loss_percentile, 1.0)
                self.assertGreaterEqual(entry.structure_percentile, 0.0)
                self.assertLessEqual(entry.structure_percentile, 1.0)
                self.assertGreaterEqual(entry.pareto_layer, 0)

    def test_given_a_ranking_when_read_then_it_states_that_its_labels_are_batch_relative(
        self,
    ):
        # Given that nothing here is an absolute claim
        # When the ranking is read
        # Then it says so in its own data, not only in documentation
        self.assertTrue(self.ranking.labels_are_batch_relative)
        self.assertEqual(self.ranking.batch_size, len(fixtures.catalogue()))

    def test_given_an_entry_when_read_then_it_names_the_evidence_that_placed_it_there(
        self,
    ):
        # Given that a ranking is only useful if it can be argued with
        # When an entry is read
        # Then it names the dominant piece of evidence behind its position
        for entry in self.ranking.images:
            with self.subTest(image=entry.image_id):
                self.assertTrue(entry.dominant_evidence)

    def test_given_the_same_batch_twice_when_ranked_then_the_two_rankings_are_equal(
        self,
    ):
        # Given the determinism the framework claims end to end
        # When the same batch is ranked twice
        # Then the results compare equal, including the ordering of ties
        self.assertEqual(
            rank_batch(measured_catalogue()), rank_batch(measured_catalogue())
        )


class RelativePositions(unittest.TestCase):
    def setUp(self):
        self.ranking = rank_batch(measured_catalogue())

    def test_given_a_near_achromatic_detailed_scene_when_ranked_then_it_reaches_the_pareto_front(
        self,
    ):
        # Given fog: almost no chroma to lose, dense luminance detail to keep
        # When the batch is ranked
        # Then it sits on the non-dominated front
        self.assertEqual(self.ranking["fog"].pareto_layer, 0)

    def test_given_high_contrast_geometry_when_ranked_then_it_reaches_the_pareto_front(
        self,
    ):
        # Given hard shapes and wide luminance separation with little colour
        # When the batch is ranked
        # Then it too is non-dominated
        self.assertEqual(self.ranking["geometry"].pareto_layer, 0)

    def test_given_an_isoluminant_image_when_ranked_then_it_sits_behind_every_clear_candidate(
        self,
    ):
        # Given an image whose entire structure lives in hue
        # When the batch is ranked
        # Then it is dominated by both clear candidates
        self.assertGreater(self.ranking["isoluminant"].pareto_layer, 0)
        self.assertLess(
            self.ranking["isoluminant"].loss_percentile,
            self.ranking["fog"].loss_percentile,
        )

    def test_given_a_colour_diagnostic_landscape_when_ranked_then_it_loses_more_than_the_fog_scene(
        self,
    ):
        # Given broad colour bands carrying scene category at coarse scale,
        # the Oliva and Schyns condition
        # When the batch is ranked
        # Then it sits below fog on the loss axis
        self.assertLess(
            self.ranking["landscape"].loss_percentile,
            self.ranking["fog"].loss_percentile,
        )

    def test_given_saturated_colours_that_also_separate_tonally_when_ranked_then_saturation_alone_does_not_sink_them(
        self,
    ):
        # Given a strongly coloured image whose colours differ in lightness too
        # When ranked against a genuinely isoluminant one
        # Then it places higher, proving position tracks lost distinctions
        # rather than raw colourfulness
        self.assertGreater(
            self.ranking["separated"].loss_percentile,
            self.ranking["isoluminant"].loss_percentile,
        )

    def test_given_a_backlit_silhouette_when_ranked_then_its_narrow_histogram_does_not_sink_it(
        self,
    ):
        # Given content confined to the extreme zones, which is a legitimate
        # rendering and not a defect
        # When ranked
        # Then it is not forced into the bottom quarter on structure
        self.assertGreater(
            self.ranking["silhouette"].structure_percentile, POOR_CANDIDATE_PERCENTILE
        )


class LabelsAreNotAbsolute(unittest.TestCase):
    def test_given_the_percentile_boundaries_when_read_then_they_split_the_batch_into_three(
        self,
    ):
        # Given the two boundaries
        # When read
        # Then they are ordered and leave a genuine middle band
        self.assertLess(POOR_CANDIDATE_PERCENTILE, STRONG_CANDIDATE_PERCENTILE)

    def test_given_one_photograph_in_two_different_batches_when_labelled_then_the_label_may_change(
        self,
    ):
        # Given the same image ranked among strong company and among weak company
        # When both labels are read
        # Then they differ, which is the honest consequence of having no ground
        # truth and is asserted here so no reader mistakes a label for a verdict
        measurements = measured_catalogue()
        strong_company = {
            name: measurements[name]
            for name in (
                "fog",
                "geometry",
                "silhouette",
                "ramp",
                "separated",
                "portrait",
                "merging",
                "landscape",
            )
        }
        weak_company = {
            name: measurements[name]
            for name in (
                "portrait",
                "isoluminant",
                "landscape",
                "merging",
                "fog",
                "geometry",
                "silhouette",
                "ramp",
            )
        }
        self.assertNotEqual(
            rank_batch(strong_company)["portrait"].label,
            rank_batch(weak_company)["portrait"].label,
        )

    def test_given_a_ranked_batch_when_labels_are_read_then_each_is_one_of_the_three_stated_bands(
        self,
    ):
        # Given the three bands
        # When labels are read
        # Then "uncertain" is a first-class outcome for the middle of the batch
        labels = {entry.label for entry in rank_batch(measured_catalogue()).images}
        self.assertTrue(labels <= {"strong_candidate", "uncertain", "poor_candidate"})
        self.assertIn("uncertain", labels)


class TotalOrder(unittest.TestCase):
    def setUp(self):
        self.ranking = rank_batch(measured_catalogue())

    def test_given_no_key_when_flattening_to_one_sequence_then_it_is_refused(self):
        # Given that collapsing two axes into one is a weighting decision
        # When a caller asks for a single ordering without naming the weighting
        # Then it is refused, which is how the no-single-score rule survives
        # the introduction of ranking
        with self.assertRaises(TypeError):
            total_order(self.ranking)  # type: ignore[call-arg]

    def test_given_an_unknown_key_when_flattening_then_it_is_rejected(self):
        # Given a key the package does not define
        # When a total order is requested
        # Then it is rejected instead of falling back to a default
        with self.assertRaises(ValueError):
            total_order(self.ranking, key="whatever_looks_best")

    def test_given_each_named_key_when_flattening_then_every_image_appears_exactly_once(
        self,
    ):
        # Given every ordering the package offers
        # When each is applied
        # Then the result is a permutation of the batch, losing nothing
        for key in RANKING_KEYS:
            with self.subTest(key=key):
                ordered = total_order(self.ranking, key=key)
                self.assertEqual(
                    sorted(entry.image_id for entry in ordered),
                    sorted(entry.image_id for entry in self.ranking.images),
                )

    def test_given_two_different_keys_when_flattening_then_the_orderings_may_disagree(
        self,
    ):
        # Given that the two axes genuinely conflict for some images
        # When the batch is ordered by loss first and by structure first
        # Then the sequences differ, which is the visible reason the package
        # refuses to pick one on the caller's behalf
        by_loss = [
            entry.image_id for entry in total_order(self.ranking, key="loss_first")
        ]
        by_structure = [
            entry.image_id for entry in total_order(self.ranking, key="structure_first")
        ]
        self.assertNotEqual(by_loss, by_structure)

    def test_given_a_caller_supplied_key_when_flattening_then_their_own_weighting_is_honoured(
        self,
    ):
        # Given a caller who wants to weigh the two axes themselves, including
        # folding in a learned aesthetic delta
        # When they pass a callable
        # Then it is used, so the escape hatch exists without the package
        # ever choosing a weighting itself
        ordered = total_order(
            self.ranking, key=lambda entry: entry.structure_percentile
        )
        self.assertEqual(len(ordered), self.ranking.batch_size)

    def test_given_tied_entries_when_flattening_then_the_order_is_stable_and_deterministic(
        self,
    ):
        # Given ties on the chosen key
        # When the same batch is ordered twice
        # Then the sequences are identical, broken deterministically on image id
        first = [
            entry.image_id
            for entry in total_order(self.ranking, key="pareto_then_loss")
        ]
        second = [
            entry.image_id
            for entry in total_order(self.ranking, key="pareto_then_loss")
        ]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
