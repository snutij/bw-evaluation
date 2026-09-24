"""Is the surviving luminance structure strong enough to carry the image alone.

Research anchors
----------------
Zone System (Ansel Adams, *The Negative*): eleven zones, one stop apart. On a
digital image, with linear-light luminance Y and an 18% middle-grey reference:

    z = 5 + log2(Y / 0.18), clamped to [0, 10]

Peli, "Contrast in complex images", JOSA A 7, 1990, DOI 10.1364/JOSAA.7.002032,
gives the band-limited local contrast C_b = I_b / (I_LP,b + eps), which is the
defensible contrast family for complex natural photographs. Michelson is built
for gratings and global RMS conflates noise with contrast, so neither is exposed
as a primary metric here.

Deliberate omission: there is no "ideal histogram". A high-key portrait
legitimately occupies zones V-VIII and a silhouette zones 0-III, so no function
here returns a "tonal range is good" boolean.

Two implementation notes
------------------------
*Zone binning.* A zone is a band one stop wide *centred* on its label, not the
interval starting at it, so the continuous map is rounded to the nearest
integer rather than floored. Flooring would put 18% grey at the bottom edge of
Zone V instead of its centre and shift every occupancy statistic by half a stop.

*Merger with no variance.* Cohen's d is undefined between two perfectly flat
regions, which is exactly the isoluminant case this metric exists to catch. The
pooled spread is therefore floored at :data:`MINIMUM_POOLED_SPREAD` rather than
guarded with a special case, so a tiny but real tonal difference between two
flat regions produces a tiny separation instead of an infinite one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

Array: TypeAlias = NDArray[np.float64]
LabelArray: TypeAlias = NDArray[np.intp]
MaskArray: TypeAlias = NDArray[np.bool_]

#: Zones conventionally associated with visible texture in a print.
TEXTURAL_ZONES: tuple[int, int] = (2, 8)

#: Number of discrete Zone System labels, Zone 0 to Zone X inclusive.
ZONE_COUNT = 11

#: Default k for CIELAB k-means segmentation. Tonal merger is a property of
#: adjacent *regions*, so the metric needs regions. k-means in Lab with a fixed
#: seed keeps the pipeline deterministic and needs only scikit-learn; a fixed
#: grid would mostly measure the grid rather than the picture.
SEGMENTATION_CLUSTERS = 6

#: Reference reflectance for Zone V.
MIDDLE_GREY: Final[float] = 0.18

#: Floor on the pooled spread used by :func:`region_tonal_separation`, in zones.
#: One zone, because one zone is one stop and the whole premise of the Zone
#: System is that a stop is the meaningful step between print values. The floor
#: has a clean reading rather than being a fudge factor: between two regions
#: flat enough that their own spread is under a stop, Cohen's d reduces exactly
#: to the difference between their zones, measured in stops.
MINIMUM_POOLED_SPREAD: Final[float] = 1.0

#: Pixels sampled to fit the segmentation k-means. Lloyd's algorithm on a
#: megapixel frame costs well over a second and buys nothing: cluster centres
#: in Lab are a property of the palette, and a palette is estimated perfectly
#: well from tens of thousands of pixels. Every pixel is still *assigned* a
#: label; only the fit is subsampled, and by a fixed stride rather than at
#: random, so the result stays bit-identical between runs.
SEGMENTATION_FIT_SAMPLES: Final[int] = 40_000

#: Fixed seed for the segmentation k-means. Determinism is a contract
#: requirement of this framework, not an implementation detail.
SEGMENTATION_SEED: Final[int] = 20260923

#: Clusters whose centres are closer than this in CIELAB are one region.
#: 2.3 is the classic CIE Delta-E*ab just-noticeable difference: below it, two
#: colours are not distinguishable, so calling them two regions and then
#: reporting that they "merge" would be circular.
#:
#: This is load-bearing on real files rather than theoretical. Asking k-means
#: for six clusters on an image that only contains three spends the spare ones
#: on JPEG ringing along the real boundaries. Measured on one reference image
#: saved as PNG and as quality-95 JPEG, the JPEG segmented into six regions
#: instead of three, and the extra ones were slivers of 1.6% to 2.3% of the
#: frame whose mean tone differed from their parent region by 0.002 zones. The
#: worst-merger reading fell from 0.300 to 0.0016, a two-hundred-fold change
#: caused entirely by the encoder. Centre separations make the two cases easy
#: to tell apart: across the reference catalogue the closest genuine pair of
#: centres is 6.3, while the compression artefacts sat between 1.2 and 2.4, so
#: the threshold falls in a wide gap rather than on a boundary.
PERCEPTUAL_MERGE_DELTA_E: Final[float] = 2.3

#: Half-power relation between a Gaussian sigma in pixels and a cutoff in
#: cycles per image, shared with `isoluminance` so a stated frequency means the
#: same thing in both modules.
_HALF_POWER_SIGMA: Final[float] = float(np.sqrt(np.log(2.0) / (2.0 * np.pi**2)))

#: Centre frequency of the finest Peli band, as a divisor of the long edge.
#: Four samples per cycle keeps the finest band clear of the Nyquist limit,
#: where a Gaussian difference measures aliasing rather than picture detail.
_FINEST_BAND_DIVISOR: Final[float] = 4.0

#: Smallest luminance treated as non-zero before the zone logarithm.
_LUMINANCE_FLOOR: Final[float] = 1e-12


def _as_map(values: Array) -> Array:
    """Validate and normalise an incoming single-channel map to float64 (H, W)."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            f"expected a single-channel map of shape (H, W), got {array.shape}"
        )
    return array


def _as_lab(lab: Array) -> Array:
    """Validate and normalise an incoming CIELAB image to float64 (H, W, 3)."""
    array = np.asarray(lab, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"expected a CIELAB image of shape (H, W, 3), got {array.shape}"
        )
    return array


@dataclass(frozen=True)
class PeliBand:
    """One octave band of Peli local contrast."""

    centre_frequency: float
    contrast_map: Array


def zone_map(linear_luminance: Array, *, middle_grey: float = MIDDLE_GREY) -> Array:
    """Continuous zone value in [0, 10]. Zero luminance clamps to 0, never -inf."""
    values = _as_map(linear_luminance)
    if middle_grey <= 0.0:
        raise ValueError(f"middle_grey must be positive, got {middle_grey}")
    floored = np.maximum(values, _LUMINANCE_FLOOR)
    zones = 5.0 + np.log2(floored / middle_grey)
    return np.clip(zones, 0.0, float(ZONE_COUNT - 1))


def _zone_labels(zones: Array) -> LabelArray:
    """Round a continuous zone map onto the eleven integer Zone System labels."""
    values = _as_map(zones)
    rounded = np.rint(np.clip(values, 0.0, float(ZONE_COUNT - 1)))
    return rounded.astype(np.intp)


def zone_histogram(zones: Array) -> Array:
    """Occupancy proportions of the eleven integer zones; sums to 1."""
    labels = _zone_labels(zones)
    counts = np.bincount(labels.ravel(), minlength=ZONE_COUNT).astype(np.float64)
    total = float(counts.sum())
    if total == 0.0:
        raise ValueError("cannot build a zone histogram from an empty image")
    return counts / total


def active_zone_count(zones: Array, *, min_occupancy: float) -> int:
    """Number of zones holding more than `min_occupancy` of the pixels."""
    histogram = zone_histogram(zones)
    return int(np.count_nonzero(histogram > min_occupancy))


def zone_entropy(zones: Array) -> float:
    """Zone-histogram entropy normalised by log(11), in [0, 1].

    Reported as a descriptor. It is explicitly NOT an objective to maximise.
    """
    histogram = zone_histogram(zones)
    occupied = histogram[histogram > 0.0]
    # abs() rather than a bare negation: entropy is non-negative by
    # construction, but a single fully occupied zone evaluates -1 * log(1) to
    # -0.0, and max(-0.0, 0.0) returns -0.0 in Python. A signed zero reaching a
    # report or a percentile is a small lie about the sign of the quantity.
    entropy = abs(float(-np.sum(occupied * np.log(occupied))))
    return entropy / float(np.log(ZONE_COUNT))


def robust_tonal_span(zones: Array, *, low: float = 0.01, high: float = 0.99) -> float:
    """Percentile span in zones, so a handful of clipped pixels cannot set the range."""
    values = _as_map(zones)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"expected 0 <= low < high <= 1, got low={low}, high={high}")
    lower, upper = np.percentile(values, [low * 100.0, high * 100.0])
    return float(upper - lower)


def textural_range_coverage(zones: Array) -> float:
    """Share of pixels inside :data:`TEXTURAL_ZONES`, in [0, 1]."""
    labels = _zone_labels(zones)
    lowest, highest = TEXTURAL_ZONES
    inside = (labels >= lowest) & (labels <= highest)
    return float(np.count_nonzero(inside) / labels.size)


def segment_regions(lab: Array, *, clusters: int = SEGMENTATION_CLUSTERS) -> LabelArray:
    """Label map from k-means over CIELAB, with a fixed seed and fixed init.

    Deterministic by construction: identical input must give an identical label
    map, including identical label numbering, so downstream merger scores are
    reproducible. Two steps secure that beyond the seed. The requested k is
    capped at the number of distinct colours present, because asking k-means
    for more clusters than there are points leaves empty clusters whose
    numbering is an artefact of the initialisation. Labels are then renumbered
    by ascending centroid L*, a*, b*, so the numbering is a function of the
    picture rather than of the order the solver happened to converge in.
    """
    values = _as_lab(lab)
    if clusters < 1:
        raise ValueError(f"clusters must be at least 1, got {clusters}")
    height, width = values.shape[:2]
    samples = values.reshape(-1, 3)
    stride = max(1, int(np.ceil(samples.shape[0] / SEGMENTATION_FIT_SAMPLES)))
    fitting = samples[::stride]
    # The distinct-colour cap is computed on the fitting subsample, not the
    # whole frame. That is the set k-means actually sees, so it is the set the
    # cap is about, and np.unique over a megapixel is a sort of a million rows
    # that costs more than the clustering it guards.
    distinct = np.unique(fitting, axis=0)
    effective = min(clusters, int(distinct.shape[0]))
    if effective <= 1:
        return np.zeros((height, width), dtype=np.intp)
    model = KMeans(
        n_clusters=effective,
        n_init=1,
        init="k-means++",
        algorithm="lloyd",
        random_state=SEGMENTATION_SEED,
    )
    model.fit(fitting)
    centres = np.asarray(model.cluster_centers_, dtype=np.float64)
    # Nearest-centroid assignment written out rather than model.predict(): the
    # expansion ||x||^2 - 2 x.c + ||c||^2 is the same arithmetic without
    # scikit-learn's per-call validation, which dominates the cost on a
    # megapixel frame. Verified to give identical labels.
    raw = np.argmin(
        (samples**2).sum(axis=1)[:, None]
        - 2.0 * (samples @ centres.T)
        + (centres**2).sum(axis=1)[None, :],
        axis=1,
    )
    groups, centres = _merge_indistinguishable(centres)
    raw = groups[raw]
    # lexsort takes the primary key last: ascending L*, then a*, then b*.
    order = np.lexsort((centres[:, 2], centres[:, 1], centres[:, 0]))
    canonical = np.empty(centres.shape[0], dtype=np.intp)
    canonical[order] = np.arange(centres.shape[0], dtype=np.intp)
    relabelled = canonical[raw]
    # Renumber consecutively from zero in case the solver emptied a cluster.
    _, compact = np.unique(relabelled, return_inverse=True)
    return compact.astype(np.intp).reshape(height, width)


def _merge_indistinguishable(centres: Array) -> tuple[LabelArray, Array]:
    """Collapse cluster centres that no viewer could tell apart.

    Single linkage under :data:`PERCEPTUAL_MERGE_DELTA_E`, so a chain of
    centres each within a just-noticeable difference of the next becomes one
    region. That is the right closure for this problem: such a chain is a
    smooth gradient that k-means has cut into arbitrary slices, not a set of
    distinct areas.

    Returns a map from original cluster index to merged index, and the merged
    centres.
    """
    count = centres.shape[0]
    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for first in range(count):
        for second in range(first + 1, count):
            if (
                float(np.linalg.norm(centres[first] - centres[second]))
                < PERCEPTUAL_MERGE_DELTA_E
            ):
                left, right = find(first), find(second)
                if left != right:
                    parent[max(left, right)] = min(left, right)

    roots = sorted({find(index) for index in range(count)})
    position = {root: index for index, root in enumerate(roots)}
    groups = np.array([position[find(index)] for index in range(count)], dtype=np.intp)
    merged = np.array(
        [centres[groups == index].mean(axis=0) for index in range(len(roots))],
        dtype=np.float64,
    )
    return groups, merged


def adjacent_region_pairs(labels: LabelArray) -> tuple[tuple[int, int], ...]:
    """Pairs of region labels that actually share a border.

    Merger only matters between neighbours. Two similarly toned regions at
    opposite corners of the frame do not merge into anything.
    """
    array = np.asarray(labels)
    if array.ndim != 2:
        raise ValueError(f"expected a label map of shape (H, W), got {array.shape}")
    pairs: set[tuple[int, int]] = set()
    for left, right in (
        (array[:, :-1], array[:, 1:]),
        (array[:-1, :], array[1:, :]),
    ):
        differing = left != right
        for first, second in zip(left[differing].tolist(), right[differing].tolist()):
            pairs.add((min(first, second), max(first, second)))
    return tuple(sorted(pairs))


def region_tonal_separation(
    zones: Array, mask_a: MaskArray, mask_b: MaskArray
) -> float:
    """Cohen's d between the zone distributions of two regions.

    This is the computable form of the classic tonal-merger problem: two
    adjacent subject areas of different colour but similar luminance.
    """
    values = _as_map(zones)
    first = np.asarray(mask_a, dtype=bool)
    second = np.asarray(mask_b, dtype=bool)
    for name, mask in (("mask_a", first), ("mask_b", second)):
        if mask.shape != values.shape:
            raise ValueError(
                f"{name} must match the zone map shape {values.shape}, got {mask.shape}"
            )
        if not mask.any():
            raise ValueError(
                f"{name} selects no pixels, so tonal separation is undefined"
            )
    sample_a = values[first]
    sample_b = values[second]
    difference = abs(float(sample_a.mean()) - float(sample_b.mean()))
    count_a, count_b = sample_a.size, sample_b.size
    if count_a + count_b <= 2:
        pooled = 0.0
    else:
        variance_a = float(sample_a.var(ddof=1)) if count_a > 1 else 0.0
        variance_b = float(sample_b.var(ddof=1)) if count_b > 1 else 0.0
        pooled = float(
            np.sqrt(
                ((count_a - 1) * variance_a + (count_b - 1) * variance_b)
                / float(count_a + count_b - 2)
            )
        )
    return difference / max(pooled, MINIMUM_POOLED_SPREAD)


def worst_region_merger(zones: Array, labels: LabelArray) -> float:
    """Smallest tonal separation among all adjacent region pairs.

    The summary statistic that reaches the measurement: one bad merger ruins a
    monochrome rendering even when every other pair separates cleanly, so this
    is a minimum rather than a mean.
    """
    values = _as_map(zones)
    array = np.asarray(labels)
    if array.shape != values.shape:
        raise ValueError(
            f"labels must match the zone map shape {values.shape}, got {array.shape}"
        )
    pairs = adjacent_region_pairs(array)
    if not pairs:
        raise ValueError(
            "no adjacent region pair exists, so there is nothing that could merge; "
            "a single-region image has no merger risk to report"
        )
    # Pairs where *both* regions sit outside the textural zones are set aside.
    # Zone 0 is maximum black and Zone X is paper white, both defined as
    # carrying no detail, so two areas that both print there do not "merge":
    # there was never any rendered detail between them to lose. Without this,
    # a backlit silhouette scores a perfect merger failure purely because its
    # two shadow masses both clamp to Zone 0, which would read as a defect
    # when it is the normal appearance of a low-key photograph.
    lowest, highest = TEXTURAL_ZONES
    mean_zone = {
        int(label): float(values[array == label].mean()) for label in np.unique(array)
    }

    def textural(label: int) -> bool:
        return lowest <= mean_zone[label] <= highest

    candidates = [pair for pair in pairs if textural(pair[0]) or textural(pair[1])]
    if not candidates:
        # Every region prints without detail. Nothing here is a merger in the
        # sense above, but reporting the unfiltered worst is more honest than
        # inventing a good score for an image with no rendered tone at all.
        candidates = list(pairs)
    return min(
        region_tonal_separation(values, array == a, array == b) for a, b in candidates
    )


def peli_contrast(
    luminance_map: Array, *, bands: int, epsilon: float = 1e-6
) -> list[PeliBand]:
    """Band-limited local contrast, one map per octave band.

    Following Peli (1990), band `b` is the difference of two Gaussian
    low-passes an octave apart, divided by the low-pass that carries everything
    below the band, which is the local mean luminance the band rides on. Bands
    run from fine to coarse: the first is centred a quarter of the long edge
    up in frequency, and each subsequent band halves that.
    """
    values = _as_map(luminance_map)
    if bands < 1:
        raise ValueError(f"bands must be at least 1, got {bands}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    long_edge = float(max(values.shape))
    results: list[PeliBand] = []
    for index in range(bands):
        centre_frequency = long_edge / (_FINEST_BAND_DIVISOR * float(2**index))
        sigma_band = _HALF_POWER_SIGMA * long_edge / centre_frequency
        sigma_below = sigma_band * 2.0
        upper = gaussian_filter(values, sigma=sigma_band, mode="nearest")
        lower = gaussian_filter(values, sigma=sigma_below, mode="nearest")
        results.append(
            PeliBand(
                centre_frequency=centre_frequency,
                contrast_map=(upper - lower) / (lower + epsilon),
            )
        )
    return results


def edge_density(luminance_map: Array, *, threshold: float) -> float:
    """Share of pixels whose luminance gradient magnitude exceeds `threshold`, in [0, 1]."""
    values = _as_map(luminance_map)
    vertical, horizontal = np.gradient(values)
    magnitude = np.hypot(vertical, horizontal)
    return float(np.count_nonzero(magnitude > threshold) / magnitude.size)


def texture_energy(luminance_map: Array) -> float:
    """High-frequency luminance energy, the detail that survives desaturation intact.

    The RMS of the finest Peli band, so it is a local *contrast* rather than a
    raw amplitude and a dark textured area is not penalised for being dark.
    Mullen (1985) is the reason this is worth reporting at all: fine detail is
    carried almost entirely by luminance, so whatever this measures is exactly
    the part of the picture that desaturation leaves untouched.
    """
    finest = peli_contrast(luminance_map, bands=1)[0]
    return float(np.sqrt(np.mean(finest.contrast_map**2)))
