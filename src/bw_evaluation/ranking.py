"""Batch-relative ranking. The only place in the package that orders anything.

Why relative, and why here
--------------------------
No public dataset pairs a colour original with its monochrome rendering and a
human preference between them, so there is nothing to calibrate an absolute
threshold against. "This photograph scores 0.78, convert it" would be a number
with no referent. What *can* be said without inventing evidence is "of these
two hundred frames, these twelve lose the least and keep the most", and that is
what this module says.

Every percentile here is a position inside the batch that was handed in. Move
the same photograph into different company and its numbers move. That is not a
defect to be engineered away, it is the honest consequence of having no ground
truth, and `BatchRanking.labels_are_batch_relative` states it in the data so
that no downstream reader has to have read this docstring.

The two axes are never combined
-------------------------------
Loss and structure are different questions, and weighting them against each
other would be exactly the arbitrary single score the package refuses to
produce. They are held apart in three ways:

* Pareto layering orders the batch without any weighting at all. Layer 0 is
  the set of images nothing else beats on both axes at once.
* A label is the *worse* of an image's two positions, never their average.
  Strong means strong on both counts; poor means weak on at least one. The
  minimum is the one aggregator that never trades a gain on one axis against a
  loss on the other.
* :func:`total_order` refuses to run without an explicit key, because
  flattening two axes into one sequence *is* the weighting decision. A caller
  may pass their own callable and fold in whatever they like, including a
  learned aesthetic delta. The package will not choose on their behalf.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np

from .measurement import (
    LOSS_AXIS_INPUTS,
    RANKING_INPUT_POLARITY,
    STRUCTURE_AXIS_INPUTS,
    CandidacyMeasurement,
)

Percentiles: TypeAlias = dict[str, float]

#: Smallest batch a percentile can honestly be computed over. A position
#: inside three images is theatre, not a statistic.
MINIMUM_BATCH_SIZE = 8

#: Top and bottom quartile boundaries for the three label bands.
STRONG_CANDIDATE_PERCENTILE = 0.75
POOR_CANDIDATE_PERCENTILE = 0.25

#: The orderings the package is willing to name. Anything else is the caller's
#: own weighting and must arrive as a callable.
RANKING_KEYS: tuple[str, ...] = ("loss_first", "structure_first", "pareto_then_loss")

_STRONG: Final[str] = "strong_candidate"
_UNCERTAIN: Final[str] = "uncertain"
_POOR: Final[str] = "poor_candidate"


@dataclass(frozen=True)
class RankedImage:
    """One image's position in a batch. Carries no absolute claim."""

    image_id: str
    loss_percentile: float
    structure_percentile: float
    pareto_layer: int
    label: str
    dominant_evidence: str


@dataclass(frozen=True)
class BatchRanking:
    """A ranked batch, which knows and says that its labels are relative."""

    images: tuple[RankedImage, ...]
    batch_size: int
    labels_are_batch_relative: bool = True

    def __getitem__(self, image_id: str) -> RankedImage:
        for entry in self.images:
            if entry.image_id == image_id:
                return entry
        raise KeyError(f"{image_id!r} is not in this batch")


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Positions of `values` within themselves, ties sharing the mean position.

    Ties must share a position rather than be broken arbitrarily: two images
    that measured identically have not earned an ordering, and inventing one
    would leak the input's dict order into the result.
    """
    array = np.asarray(values, dtype=np.float64)
    return [
        float(
            np.count_nonzero(array < value)
            + 0.5 * (np.count_nonzero(array == value) - 1)
        )
        for value in array
    ]


def _oriented(
    evidence: Mapping[str, Mapping[str, float]], names: Sequence[str], key: str
) -> list[float]:
    """One ranking input across the batch, flipped so that larger is always better."""
    sign = 1.0 if RANKING_INPUT_POLARITY[key] else -1.0
    values: list[float] = []
    for name in names:
        if key not in evidence[name]:
            raise ValueError(
                f"{name!r} records no value for the ranking input {key!r}; "
                "every measurement in a batch must carry the same evidence"
            )
        values.append(sign * float(evidence[name][key]))
    return values


def _axis_percentiles(
    evidence: Mapping[str, Mapping[str, float]],
    names: Sequence[str],
    inputs: Sequence[str],
) -> Percentiles:
    """Batch position on one axis, in [0, 1], from that axis's inputs.

    Two rankings, and both are doing work.

    The first makes the inputs commensurable. Edge density is a proportion,
    texture energy a small dimensionless contrast, and worst-region merger an
    unbounded effect size that ran from 0.003 to 7.4 across the reference
    catalogue. Averaging those raw would be an unstated weighting in favour of
    whichever happened to have the widest spread, and min-max normalising them
    first is worse, because one outlier then compresses everything else into
    the bottom of the range: in testing, an image third-best of nine on merger
    normalised to 0.24 purely because two others were far away.

    The second turns the composite back into a position. A mean of ranks
    clusters toward the middle, so dividing it by the batch size yields numbers
    that never reach either end: on the reference catalogue the best image on
    the loss axis scored 0.72, and nothing scored 0 or 1. Ranking the composite
    restores the property a percentile is supposed to have, that the best image
    in the batch sits at 1 and the worst at 0.
    """
    per_input = [_average_ranks(_oriented(evidence, names, key)) for key in inputs]
    composite = [float(np.mean(column)) for column in zip(*per_input)]
    span = float(len(names) - 1)
    return {name: rank / span for name, rank in zip(names, _average_ranks(composite))}


def _dominates(first: tuple[float, float], second: tuple[float, float]) -> bool:
    """True when `first` is at least as good on both axes and better on one."""
    return (
        first[0] >= second[0]
        and first[1] >= second[1]
        and (first[0] > second[0] or first[1] > second[1])
    )


def _pareto_layers(positions: Mapping[str, tuple[float, float]]) -> dict[str, int]:
    """Non-dominated sorting. Layer 0 is the front, and needs no weighting."""
    remaining = set(positions)
    layers: dict[str, int] = {}
    layer = 0
    while remaining:
        front = [
            candidate
            for candidate in remaining
            if not any(
                other != candidate
                and _dominates(positions[other], positions[candidate])
                for other in remaining
            )
        ]
        for candidate in front:
            layers[candidate] = layer
        remaining -= set(front)
        layer += 1
    return layers


def _label(loss: float, structure: float) -> str:
    """Three bands, decided by the worse of the two axes.

    A minimum rather than a mean. Averaging would let a strong showing on one
    axis pay for a weak one on the other, which is precisely the trade this
    package declines to make on the caller's behalf. Strong therefore means
    strong on both counts, and poor means weak on at least one.
    """
    worst = min(loss, structure)
    if worst >= STRONG_CANDIDATE_PERCENTILE:
        return _STRONG
    if worst <= POOR_CANDIDATE_PERCENTILE:
        return _POOR
    return _UNCERTAIN


def _dominant_evidence(
    name: str,
    evidence: Mapping[str, Mapping[str, float]],
    input_percentiles: Mapping[str, Percentiles],
) -> str:
    """The single most extreme measurement behind an image's position.

    Furthest from the middle of the batch in either direction, so the sentence
    explains a poor placing as readily as a good one. Ties fall to the
    alphabetically first name, because a rendered explanation that changed
    between runs would undermine the determinism the rest of the package works
    to provide.
    """
    ordered = sorted(
        input_percentiles,
        key=lambda key: (-abs(input_percentiles[key][name] - 0.5), key),
    )
    strongest = ordered[0]
    percentile = input_percentiles[strongest][name]
    return (
        f"{strongest} = {evidence[name][strongest]:.4g}, "
        f"at the {percentile:.0%} mark of this batch"
    )


def rank_batch(measurements: Mapping[str, CandidacyMeasurement]) -> BatchRanking:
    """Place every measurement relative to the others. No absolute verdicts.

    Refuses batches below :data:`MINIMUM_BATCH_SIZE`, because the output is
    made entirely of positions within the batch and a position among three
    images carries no information.
    """
    if len(measurements) < MINIMUM_BATCH_SIZE:
        raise ValueError(
            f"ranking needs at least {MINIMUM_BATCH_SIZE} images, got {len(measurements)}. "
            "Every number this module produces is a position within the batch, so a batch "
            "this small would yield positions that say more about the sample than the photographs."
        )

    names = sorted(measurements)
    expected = measurements[names[0]].ranking_inputs
    for name in names:
        if measurements[name].ranking_inputs != expected:
            raise ValueError(
                f"{name!r} declares ranking inputs {measurements[name].ranking_inputs}, "
                f"which differ from {expected}; a batch must be measured consistently"
            )

    evidence: dict[str, Mapping[str, float]] = {
        name: {
            **measurements[name].chroma_information,
            **measurements[name].surviving_structure,
        }
        for name in names
    }

    loss = _axis_percentiles(evidence, names, LOSS_AXIS_INPUTS)
    structure = _axis_percentiles(evidence, names, STRUCTURE_AXIS_INPUTS)
    positions = {name: (loss[name], structure[name]) for name in names}
    layers = _pareto_layers(positions)

    span = float(len(names) - 1)
    input_percentiles: dict[str, Percentiles] = {
        key: dict(
            zip(
                names,
                (
                    rank / span
                    for rank in _average_ranks(_oriented(evidence, names, key))
                ),
            )
        )
        for key in expected
    }

    images = tuple(
        RankedImage(
            image_id=name,
            loss_percentile=loss[name],
            structure_percentile=structure[name],
            pareto_layer=layers[name],
            label=_label(loss[name], structure[name]),
            dominant_evidence=_dominant_evidence(name, evidence, input_percentiles),
        )
        # Sorted by image id, so two equal batches produce equal rankings
        # rather than merely equivalent ones.
        for name in names
    )
    return BatchRanking(
        images=images, batch_size=len(names), labels_are_batch_relative=True
    )


def total_order(
    ranking: BatchRanking,
    *,
    key: str | Callable[[RankedImage], float],
) -> tuple[RankedImage, ...]:
    """Flatten a two-axis ranking into one sequence, under a named weighting.

    `key` is keyword-only and has no default on purpose. Collapsing two axes
    into a single list is a weighting decision, and the package will not make
    one silently; a caller who wants a list has to say which trade-off they
    want. A callable is accepted so that any weighting at all remains possible,
    including one that folds in an external aesthetic model, without the
    package ever endorsing a particular one.

    Ties always fall back to the image id, so the sequence is reproducible
    rather than dependent on the input's iteration order.
    """
    if callable(key):
        return tuple(
            sorted(ranking.images, key=lambda entry: (key(entry), entry.image_id))
        )
    if not isinstance(key, str):
        raise TypeError(
            f"key must be a name from {RANKING_KEYS} or a callable, got {type(key)!r}"
        )
    if key not in RANKING_KEYS:
        raise ValueError(
            f"unknown ranking key {key!r}; expected one of {RANKING_KEYS} or a callable. "
            "There is deliberately no default, because the default would be a weighting."
        )
    if key == "loss_first":
        return tuple(
            sorted(
                ranking.images,
                key=lambda e: (-e.loss_percentile, -e.structure_percentile, e.image_id),
            )
        )
    if key == "structure_first":
        return tuple(
            sorted(
                ranking.images,
                key=lambda e: (-e.structure_percentile, -e.loss_percentile, e.image_id),
            )
        )
    return tuple(
        sorted(
            ranking.images,
            key=lambda e: (e.pareto_layer, -e.loss_percentile, e.image_id),
        )
    )
