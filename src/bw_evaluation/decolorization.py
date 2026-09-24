"""Measured loss caused by an actual conversion.

Research anchors
----------------
Lu, Xu & Jia, "Contrast Preserving Decolorization" (ICCP 2012) and
"Real-time Contrast Preserving Decolorization" (SIGGRAPH Asia 2012 Technical
Briefs, DOI 10.1145/2407746.2407780) define CCPR, CCFR and the E-score, and
search a low-dimensional space of channel weights per image.

Ma, Zhu, Wang & Wang, "Objective Quality Assessment for Color-to-Gray Image
Conversion" (IEEE TIP, 2015) define C2G-SSIM with separable lightness, colour
contrast and structure components, plus a luminance-entropy image-type
classifier (photographic when T >= 4).

Ayunts & Agaian introduce threshold-independent area because CCPR/CCFR/E-score
rankings flip with the threshold. The contract below therefore refuses to
summarise from a single threshold.

Two implementation notes, both load-bearing
-------------------------------------------
*Units.* A grey rendering is compared against a CIELAB colour difference, so
the grey has to be expressed in the same units. Every comparison here converts
the grey image to CIELAB L* first via :func:`linear_to_lightness`. Comparing a
[0,1] grey value against a Delta-E would be meaningless.

*Fidelity.* CCPR and CCFR are the recall and precision of a conversion:
CCPR asks how many originally visible colour differences survived, CCFR asks
how many of the differences visible in the grey rendering were real rather than
fabricated. The E-score is their harmonic mean, which is why it penalises a
method that scores well on only one of them.

C2G-SSIM here is a reimplementation in the spirit of the paper rather than a
port of the authors' code, which is not available offline. It keeps the
separable three-component structure, which is the property the contract
depends on, and the module states this rather than implying an exact match.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter

from .conventions import (
    COEFFICIENTS,
    ColorConvention,
    Encoding,
    linear_to_lightness,
    rgb_to_lab,
    srgb_eotf,
)

Array: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.intp]

#: Conversion methods the framework knows how to generate.
METHODS: Final[tuple[str, ...]] = (
    "bt601_luma",
    "bt709_luma",
    "bt2020_luma",
    "contrast_preserving",
)

#: Minimum number of distinct thresholds required before any summary is produced.
MINIMUM_THRESHOLD_SWEEP: Final[int] = 3

#: Pixel separations at which pairs are sampled for CCPR / CCFR.
#: Neighbour pairs at several dilations rather than a random sample: fully
#: deterministic with no RNG, and the long dilations are what catch a subject
#: merging with its background across the frame rather than only at its edge.
PAIR_DILATIONS: Final[tuple[int, ...]] = (1, 4, 16, 64)

#: Upper bound on sampled pairs, enforced by a fixed stride so the selection
#: stays deterministic and the cost stays bounded on full-resolution images.
MAXIMUM_SAMPLED_PAIRS: Final[int] = 200_000

#: Encoding a bare grey array is assumed to carry when no other information is
#: available. Linear is not a convenience default: isoluminance is a
#: linear-light property, and measuring it through a gamma-encoded luma
#: under-detects exactly the failure case the framework exists to catch.
DEFAULT_GREY_ENCODING: Final[Encoding] = "linear"

#: Classifier boundary from Ma et al.: 8-bit luminance entropy T >= 4 => photographic.
PHOTOGRAPHIC_ENTROPY_BOUNDARY: Final[float] = 4.0

#: Step of the channel-weight search grid, as a denominator. Lu et al. search a
#: small discrete space rather than optimising continuously; a tenth-step
#: simplex gives 66 candidates, which is exhaustive and reproducible.
_WEIGHT_GRID_STEPS: Final[int] = 10

#: Threshold sweep the weight search optimises over. A sweep rather than a
#: single tau for the same reason the public summaries demand one: rankings
#: flip with the threshold, and a recipe chosen at one tau would inherit that.
SEARCH_TAU_SWEEP: Final[tuple[float, ...]] = (5.0, 10.0, 15.0, 20.0, 25.0)

#: Two candidate weightings count as tied within this much objective difference.
#: Not a numerical epsilon: a recipe has to earn its departure from the default
#: by a margin a viewer could plausibly notice. At 1e-9 a fog scene picked up
#: BT.2020 coefficients over BT.709 on a rounding-level gain, which is advice
#: driven by noise.
_WEIGHT_TIE_TOLERANCE: Final[float] = 0.005

#: Channel weights must sum to 1 within this tolerance.
_WEIGHT_SUM_TOLERANCE: Final[float] = 1e-9

#: SSIM-style stabilisers, on the L* scale where the dynamic range is 100.
_SSIM_C1: Final[float] = (0.01 * 100.0) ** 2
_SSIM_C2: Final[float] = (0.03 * 100.0) ** 2
_SSIM_C3: Final[float] = _SSIM_C2 / 2.0

#: Side of the local window used by the C2G-SSIM components, in pixels.
_SSIM_WINDOW: Final[int] = 11

#: Local colour contrast below which a window is treated as having no colour
#: information to preserve, and is therefore given no weight in the pooling.
_COLOUR_CONTRAST_FLOOR: Final[float] = 1e-6

#: Conventional approximations of classic B&W contrast filters on panchromatic
#: film, normalised to sum to 1. These are NOT spectral measurements: they are
#: the engineering convention photographers already reason with, kept only so a
#: recommendation can be named in familiar language. Evidence level: convention.
CLASSIC_FILTER_WEIGHTS: Final[dict[str, tuple[float, float, float]]] = {
    "none": (0.2126, 0.7152, 0.0722),
    "yellow": (0.40, 0.55, 0.05),
    "orange": (0.60, 0.38, 0.02),
    "red": (0.85, 0.15, 0.00),
    "green": (0.15, 0.75, 0.10),
    "blue": (0.10, 0.30, 0.60),
}


@dataclass(frozen=True)
class ChannelWeights:
    """A channel-mixer recipe: non-negative, summing to 1.

    Constrained to sum to 1 so that applying it preserves overall exposure and
    the numbers can be pasted straight into a channel mixer without the user
    having to renormalise or re-expose.
    """

    red: float
    green: float
    blue: float

    def __post_init__(self) -> None:
        """Rejects negative channels and any recipe not summing to 1."""
        triple = (self.red, self.green, self.blue)
        if any(value < 0.0 for value in triple):
            raise ValueError(f"channel weights must be non-negative, got {triple}")
        total = sum(triple)
        if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"channel weights must sum to 1 to preserve exposure, got {total}"
            )

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.red, self.green, self.blue)


@dataclass(frozen=True)
class C2GSSIM:
    """The three separable components plus the combined score, each in [0, 1]."""

    lightness: float
    colour_contrast: float
    structure: float
    overall: float


def _as_image(rgb: Array) -> Array:
    values = np.asarray(rgb, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(
            f"expected a three-channel image of shape (H, W, 3), got {values.shape}"
        )
    return values


def apply_channel_weights(
    rgb: Array, weights: ChannelWeights, *, convention: ColorConvention
) -> Array:
    """Render to grey with an explicit channel-mixer recipe."""
    values = _as_image(rgb)
    if convention.encoding == "linear":
        values = srgb_eotf(values)
    mixed = np.tensordot(
        values, np.asarray(weights.as_tuple(), dtype=np.float64), axes=([2], [0])
    )
    return np.clip(mixed, 0.0, 1.0)


def _candidate_weights(convention: ColorConvention) -> tuple[ChannelWeights, ...]:
    """The discrete search space: a tenth-step simplex plus the named standards.

    The convention's own coefficients are always included so that an image with
    nothing to gain from remixing can return the default exactly rather than
    the nearest grid point.
    """
    candidates: list[tuple[float, float, float]] = []
    for red in range(_WEIGHT_GRID_STEPS + 1):
        for green in range(_WEIGHT_GRID_STEPS + 1 - red):
            blue = _WEIGHT_GRID_STEPS - red - green
            candidates.append(
                (
                    red / _WEIGHT_GRID_STEPS,
                    green / _WEIGHT_GRID_STEPS,
                    blue / _WEIGHT_GRID_STEPS,
                )
            )
    named = [COEFFICIENTS[name] for name in COEFFICIENTS]
    # The convention's own weighting goes first so it wins any exact tie.
    ordered = [convention.weights] + named + candidates
    seen: dict[tuple[float, float, float], None] = {}
    for triple in ordered:
        seen.setdefault(triple, None)
    return tuple(ChannelWeights(*triple) for triple in seen)


def _pair_indices(shape: tuple[int, int]) -> tuple[IndexArray, IndexArray]:
    """Flat index arrays for neighbour pairs at every dilation.

    Deterministic by construction. When the full set exceeds
    :data:`MAXIMUM_SAMPLED_PAIRS` it is thinned by a fixed stride rather than
    sampled, so two runs on the same image always compare the same pairs.
    """
    height, width = shape
    grid = np.arange(height * width, dtype=np.intp).reshape(height, width)
    first: list[IndexArray] = []
    second: list[IndexArray] = []
    for dilation in PAIR_DILATIONS:
        if dilation < width:
            first.append(grid[:, :-dilation].ravel())
            second.append(grid[:, dilation:].ravel())
        if dilation < height:
            first.append(grid[:-dilation, :].ravel())
            second.append(grid[dilation:, :].ravel())
    if not first:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    left = np.concatenate(first)
    right = np.concatenate(second)
    if left.size > MAXIMUM_SAMPLED_PAIRS:
        stride = int(np.ceil(left.size / MAXIMUM_SAMPLED_PAIRS))
        left, right = left[::stride], right[::stride]
    return left, right


def _grey_to_lightness(grey: Array, encoding: Encoding) -> Array:
    """Express a [0,1] grey rendering as CIELAB L*, so it is comparable to Delta-E."""
    values = np.asarray(grey, dtype=np.float64)
    linear = srgb_eotf(values) if encoding == "gamma" else values
    return linear_to_lightness(linear)


def _pair_differences(
    lab: Array, grey: Array, encoding: Encoding
) -> tuple[Array, Array]:
    """Per-pair CIELAB colour difference and grey lightness difference."""
    lightness = _grey_to_lightness(grey, encoding)
    if lab.shape[:2] != lightness.shape:
        raise ValueError(
            f"image and grey rendering disagree on shape: {lab.shape[:2]} vs {lightness.shape}"
        )
    left, right = _pair_indices(lightness.shape)
    flat_lab = lab.reshape(-1, 3)
    flat_grey = lightness.reshape(-1)
    delta_colour = np.linalg.norm(flat_lab[left] - flat_lab[right], axis=1)
    delta_grey = np.abs(flat_grey[left] - flat_grey[right])
    return delta_colour, delta_grey


def _ccpr_from_differences(delta_colour: Array, delta_grey: Array, tau: float) -> float:
    """CCPR from precomputed pair differences. Single source of truth for recall."""
    visible = delta_colour >= tau
    if not bool(visible.any()):
        return 1.0
    return float(np.mean(delta_grey[visible] >= tau))


def _ccfr_from_differences(delta_colour: Array, delta_grey: Array, tau: float) -> float:
    """CCFR from precomputed pair differences. Single source of truth for precision."""
    rendered = delta_grey >= tau
    if not bool(rendered.any()):
        return 0.0 if bool((delta_colour >= tau).any()) else 1.0
    return float(np.mean(delta_colour[rendered] >= tau))


def _harmonic_mean(preserved: float, fidelity: float) -> float:
    total = preserved + fidelity
    return 2.0 * preserved * fidelity / total if total > 0.0 else 0.0


def _e_score_from_differences(
    delta_colour: Array,
    delta_grey: Array,
    tau: float,
    visible: NDArray[np.bool_],
) -> float:
    """E-score from precomputed pair differences, for use inside the search."""
    rendered = delta_grey >= tau
    preserved = float(np.mean(rendered[visible])) if bool(visible.any()) else 1.0
    if bool(rendered.any()):
        fidelity = float(np.mean(visible[rendered]))
    else:
        fidelity = 0.0 if bool(visible.any()) else 1.0
    return _harmonic_mean(preserved, fidelity)


def _loss_scalars(
    lab: Array,
    grey: Array,
    *,
    taus: tuple[float, ...],
    grey_encoding: Encoding = DEFAULT_GREY_ENCODING,
) -> dict[str, dict[float, float]]:
    """Every threshold-dependent loss scalar from a single pass over the pairs.

    Purely an efficiency shim over the public metrics, and it goes through the
    same two helpers they do, so the numbers are identical by construction
    rather than by coincidence. It exists because measuring one image calls for
    CCPR, CCFR, collapse and the E-score at five thresholds for each of two
    methods, and routing each of those through the public entry points would
    recompute the same pair differences twenty times over.
    """
    values = _validated_sweep(taus)
    delta_colour, delta_grey = _pair_differences(lab, grey, grey_encoding)
    preserved: dict[float, float] = {}
    fidelity: dict[float, float] = {}
    curve: dict[float, float] = {}
    for tau in values:
        threshold = _validated_tau(tau)
        preserved[tau] = _ccpr_from_differences(delta_colour, delta_grey, threshold)
        fidelity[tau] = _ccfr_from_differences(delta_colour, delta_grey, threshold)
        curve[tau] = _harmonic_mean(preserved[tau], fidelity[tau])
    return {
        "ccpr": preserved,
        "ccfr": fidelity,
        "colour_collapse": {tau: 1.0 - value for tau, value in fidelity.items()},
        "e_score": curve,
    }


def _validated_sweep(taus: tuple[float, ...]) -> tuple[float, ...]:
    """Shared validation for a threshold sweep."""
    values = tuple(float(tau) for tau in taus)
    if len(values) < MINIMUM_THRESHOLD_SWEEP:
        raise ValueError(
            f"at least {MINIMUM_THRESHOLD_SWEEP} thresholds are required, got {len(values)}. "
            "Rankings flip with tau, so a single-threshold summary is not meaningful."
        )
    if len(set(values)) != len(values):
        raise ValueError(f"thresholds must be distinct, got {values}")
    return values


def _validated_tau(tau: float) -> float:
    value = float(tau)
    if value <= 0.0:
        raise ValueError(
            f"tau must be strictly positive, got {value}. At tau = 0 every pair counts as "
            "distinct and the ratios are trivially 1."
        )
    return value


def decolorize(rgb: Array, *, method: str, convention: ColorConvention) -> Array:
    """Produce a single-channel rendering in [0, 1] using one of :data:`METHODS`."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    weights = recommended_channel_weights(rgb, method=method, convention=convention)
    return apply_channel_weights(rgb, weights, convention=convention)


def recommended_channel_weights(
    rgb: Array,
    *,
    method: str,
    convention: ColorConvention,
) -> ChannelWeights:
    """The weights a given method actually used for this image.

    For the fixed luma methods this is simply the coefficient set. For
    `contrast_preserving` it is the winner of the discrete weight search, which
    is the useful output: "yes, and mix it roughly like this".

    The objective is the threshold-independent area under the E-score curve,
    which is exactly what the rest of the module measures. Two properties make
    it the right choice, and both were arrived at by discarding a worse one.

    A least-squares fit of grey difference to colour difference was tried
    first. It is unbounded above, so it does not preserve contrast, it
    maximises it: it recommended a green filter for a near-achromatic concrete
    scene and a pure blue channel for a landscape. The E-score saturates at 1,
    so once a weighting has preserved everything preservable, nothing is gained
    by pushing further and the tie-break sends the answer back to the default.

    Ties resolve toward the convention's own weighting, so an image with
    nothing to gain from remixing is never sent somewhere arbitrary.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method != "contrast_preserving":
        return ChannelWeights(*COEFFICIENTS[method.removesuffix("_luma")])

    values = _as_image(rgb)
    lab = rgb_to_lab(values, convention)
    left, right = _pair_indices(values.shape[:2])
    flat_lab = lab.reshape(-1, 3)
    delta_colour = np.linalg.norm(flat_lab[left] - flat_lab[right], axis=1)
    visible = {tau: delta_colour >= tau for tau in SEARCH_TAU_SWEEP}

    best: ChannelWeights | None = None
    best_area = -np.inf
    best_distance = np.inf
    default = np.asarray(convention.weights, dtype=np.float64)
    for candidate in _candidate_weights(convention):
        grey = apply_channel_weights(values, candidate, convention=convention)
        lightness = _grey_to_lightness(grey, convention.encoding).reshape(-1)
        delta_grey = np.abs(lightness[left] - lightness[right])
        curve = {
            tau: _e_score_from_differences(delta_colour, delta_grey, tau, visible[tau])
            for tau in SEARCH_TAU_SWEEP
        }
        area = threshold_independent_area(curve)
        distance = float(np.abs(np.asarray(candidate.as_tuple()) - default).sum())
        if area > best_area + _WEIGHT_TIE_TOLERANCE or (
            area >= best_area - _WEIGHT_TIE_TOLERANCE and distance < best_distance
        ):
            best, best_area, best_distance = candidate, max(area, best_area), distance
    assert best is not None
    return best


def nearest_classic_filter(weights: ChannelWeights) -> str:
    """Name the closest entry in :data:`CLASSIC_FILTER_WEIGHTS`.

    Purely a translation layer so a recommendation reads as "roughly a red
    filter" rather than three decimals. Carries no evidential weight. Ties
    resolve to the first entry in the table, so the answer is deterministic.
    """
    target = np.asarray(weights.as_tuple(), dtype=np.float64)
    return min(
        CLASSIC_FILTER_WEIGHTS,
        key=lambda name: float(
            np.sum((np.asarray(CLASSIC_FILTER_WEIGHTS[name]) - target) ** 2)
        ),
    )


def ccpr(
    lab: Array,
    grey: Array,
    *,
    tau: float,
    grey_encoding: Encoding = DEFAULT_GREY_ENCODING,
) -> float:
    """Colour Contrast Preserving Ratio at threshold `tau`, in [0, 1], higher better.

    Of the pairs that were visibly different in colour, the share still visibly
    different in grey. Recall.
    """
    threshold = _validated_tau(tau)
    delta_colour, delta_grey = _pair_differences(lab, grey, grey_encoding)
    return _ccpr_from_differences(delta_colour, delta_grey, threshold)


def ccfr(
    lab: Array,
    grey: Array,
    *,
    tau: float,
    grey_encoding: Encoding = DEFAULT_GREY_ENCODING,
) -> float:
    """Colour Content Fidelity Ratio at threshold `tau`, in [0, 1], higher better.

    Of the pairs rendered visibly different in grey, the share that were
    genuinely different in colour. Precision, so a method cannot score well by
    fabricating contrast.

    Returns 0.0 when the rendering contains no visible differences at all: no
    colour content was carried through, so fidelity is zero rather than
    vacuously perfect.
    """
    threshold = _validated_tau(tau)
    delta_colour, delta_grey = _pair_differences(lab, grey, grey_encoding)
    return _ccfr_from_differences(delta_colour, delta_grey, threshold)


def e_score(
    lab: Array,
    grey: Array,
    *,
    tau: float,
    grey_encoding: Encoding = DEFAULT_GREY_ENCODING,
) -> float:
    """Harmonic mean of CCPR and CCFR at threshold `tau`."""
    preserved = ccpr(lab, grey, tau=tau, grey_encoding=grey_encoding)
    fidelity = ccfr(lab, grey, tau=tau, grey_encoding=grey_encoding)
    total = preserved + fidelity
    if total <= 0.0:
        return 0.0
    return 2.0 * preserved * fidelity / total


def colour_collapse(
    lab: Array,
    grey: Array,
    *,
    tau: float,
    grey_encoding: Encoding = DEFAULT_GREY_ENCODING,
) -> float:
    """1 - CCFR: the share of originally visible colour distinctions destroyed."""
    return 1.0 - ccfr(lab, grey, tau=tau, grey_encoding=grey_encoding)


def e_score_curve(
    lab: Array,
    grey: Array,
    *,
    taus: tuple[float, ...],
    grey_encoding: Encoding = DEFAULT_GREY_ENCODING,
) -> dict[float, float]:
    """E-score evaluated across a threshold sweep. Requires >= MINIMUM_THRESHOLD_SWEEP taus."""
    values = _validated_sweep(taus)
    return {
        tau: e_score(lab, grey, tau=tau, grey_encoding=grey_encoding) for tau in values
    }


def threshold_independent_area(curve: Mapping[float, float]) -> float:
    """Trapezoidal area under an E-score curve, normalised by the threshold range."""
    if len(curve) < 2:
        raise ValueError(f"an area needs at least two thresholds, got {len(curve)}")
    taus = np.asarray(sorted(curve), dtype=np.float64)
    scores = np.asarray([curve[float(tau)] for tau in taus], dtype=np.float64)
    span = float(taus[-1] - taus[0])
    if span <= 0.0:
        raise ValueError("threshold range must be positive")
    return float(np.trapezoid(scores, taus) / span)


def _local_mean(values: Array) -> Array:
    return uniform_filter(values, size=_SSIM_WINDOW, mode="nearest")


def _local_variance(values: Array) -> Array:
    mean = _local_mean(values)
    return np.maximum(_local_mean(values * values) - mean * mean, 0.0)


def c2g_ssim(rgb: Array, grey: Array, *, convention: ColorConvention) -> C2GSSIM:
    """Full-reference colour-to-grey structural similarity with separable components.

    Reimplemented in the spirit of Ma et al. rather than ported from their
    code. The property the contract relies on is that the three components fail
    independently, so a caller can tell a tonal error from a collapsed hue from
    a destroyed structure.

    Pooling for the colour-contrast term is weighted by how much colour
    contrast a window actually had. An image region with no colour variation
    cannot lose any, and averaging such regions in unweighted would drown the
    signal from the regions that do carry colour.
    """
    values = _as_image(rgb)
    lab = rgb_to_lab(values, convention)
    reference_lightness = lab[..., 0]
    rendered_lightness = _grey_to_lightness(grey, convention.encoding)
    if reference_lightness.shape != rendered_lightness.shape:
        raise ValueError(
            f"image and grey rendering disagree on shape: "
            f"{reference_lightness.shape} vs {rendered_lightness.shape}"
        )

    # Lightness: mean absolute tonal error, on the L* scale where 100 is the
    # full range. A plain SSIM luminance term proved far too forgiving of a
    # wholesale exposure shift to be diagnostic.
    lightness = (
        1.0 - float(np.mean(np.abs(reference_lightness - rendered_lightness))) / 100.0
    )
    lightness = float(np.clip(lightness, 0.0, 1.0))

    # Colour contrast: local colour variation includes chroma, so a hue-only
    # edge counts as contrast that the grey rendering is expected to carry.
    colour_variance = (
        _local_variance(lab[..., 0])
        + _local_variance(lab[..., 1])
        + _local_variance(lab[..., 2])
    )
    colour_sigma = np.sqrt(colour_variance)
    grey_sigma = np.sqrt(_local_variance(rendered_lightness))
    contrast_map = (2.0 * colour_sigma * grey_sigma + _SSIM_C2) / (
        colour_sigma**2 + grey_sigma**2 + _SSIM_C2
    )
    weights = colour_sigma
    total_weight = float(weights.sum())
    if total_weight <= _COLOUR_CONTRAST_FLOOR:
        colour_contrast = 1.0
    else:
        colour_contrast = float((contrast_map * weights).sum() / total_weight)

    # Structure: the usual SSIM correlation term, between reference and
    # rendered lightness. Invariant to gain, which is what makes it possible
    # for structure to stay high while lightness fails.
    mean_reference = _local_mean(reference_lightness)
    mean_rendered = _local_mean(rendered_lightness)
    covariance = (
        _local_mean(reference_lightness * rendered_lightness)
        - mean_reference * mean_rendered
    )
    sigma_reference = np.sqrt(np.maximum(_local_variance(reference_lightness), 0.0))
    sigma_rendered = np.sqrt(np.maximum(_local_variance(rendered_lightness), 0.0))
    structure_map = (covariance + _SSIM_C3) / (
        sigma_reference * sigma_rendered + _SSIM_C3
    )
    structure = float(np.clip(structure_map.mean(), 0.0, 1.0))

    overall = float(np.cbrt(lightness * colour_contrast * structure))
    return C2GSSIM(
        lightness=lightness,
        colour_contrast=float(np.clip(colour_contrast, 0.0, 1.0)),
        structure=structure,
        overall=float(np.clip(overall, 0.0, 1.0)),
    )


def luminance_entropy(grey: Array) -> float:
    """Shannon entropy of the 8-bit quantised luminance histogram, in bits."""
    values = np.asarray(grey, dtype=np.float64)
    quantised = np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)
    counts = np.bincount(quantised.ravel(), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def classify_image_type(grey: Array) -> str:
    """'photographic' when luminance entropy >= 4 bits, otherwise 'synthetic'."""
    return (
        "photographic"
        if luminance_entropy(grey) >= PHOTOGRAPHIC_ENTROPY_BOUNDARY
        else "synthetic"
    )
