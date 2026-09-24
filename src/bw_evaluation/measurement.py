"""Per-image measurement. Deliberately produces no verdict.

Nothing in the literature supports an absolute threshold for "this photograph
should be black and white". There is no public dataset of (colour original,
monochrome rendering, human preference) triples to fit one against, so this
module measures and refuses to judge. Judgement happens in `ranking.py`, and
only ever relative to a batch.

Two structural constraints, both enforced by tests:

* Every value reaching a `CandidacyMeasurement` is a scalar, string or tuple.
  No numpy arrays. Arrays would make two measurements non-comparable with `==`
  and would silently break the determinism guarantee.
* A measurement carries `ranking_inputs`, the names of the fields that may
  legitimately drive an ordering. Anything absent from that tuple is recorded
  for provenance and may never influence a rank.

Three measurement decisions worth stating plainly
-------------------------------------------------
*Structure is measured on plain linear-light luminance, not on the recommended
grey.* It answers the conservative question, "what survives if chroma is simply
discarded", which is a property of the photograph rather than of a recipe. The
recipe's own performance is already recorded, per method, in `conversion_loss`.
Keeping the two apart is what lets a reader tell "this needs a better
conversion" from "this cannot work in black and white".

*The zone map is always built in linear light*, whatever convention the caller
passes. Zones are defined as stops above and below an 18% reflectance, which is
a photometric quantity; computing them from gamma-encoded luma would put middle
grey in the wrong zone.

*Edges are counted in CIELAB L\\*, textures in linear luminance.* An edge is a
perceptual event, so its threshold belongs on a perceptual scale where 1.0 is
roughly one just-noticeable step. Peli contrast divides by local mean
luminance, so it belongs on a photometric one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from .chroma import chroma_statistics, hasler_susstrunk_colourfulness
from .conventions import (
    ColorConvention,
    luminance,
    rgb_to_lab,
)
from .decolorization import (
    METHODS,
    ChannelWeights,
    _loss_scalars,
    _validated_sweep,
    apply_channel_weights,
    c2g_ssim,
    classify_image_type,
    decolorize,
    nearest_classic_filter,
    recommended_channel_weights,
    threshold_independent_area,
)
from .isoluminance import (
    chromatic_gradient_energy_ratio,
    coarse_scale_chroma_contribution,
    isoluminant_edge_fraction,
)
from .tonal import (
    active_zone_count,
    edge_density,
    robust_tonal_span,
    segment_regions,
    textural_range_coverage,
    texture_energy,
    worst_region_merger,
    zone_entropy,
    zone_map,
)

Array = NDArray[np.float64]

EvidenceLevel = Literal["empirical", "convention", "derived", "unsupported"]

#: Recorded for provenance, never allowed to influence a ranking.
CONTEXTUAL_FEATURE_NAMES: tuple[str, ...] = (
    "aspect_ratio",
    "megapixels",
    "orientation",
    "image_type",
)

#: Named in the practitioner literature, unsupported by evidence, advisory only.
ADVISORY_HEURISTIC_NAMES: tuple[str, ...] = (
    "high_iso_noise",
    "mixed_white_balance",
    "full_tonal_range",
)

#: Hasler & Susstrunk score below which an image has no colour worth discussing.
#: Set well under their published "not colourful" anchor of 15 so that a merely
#: low-chroma photograph is still measured normally.
ACHROMATIC_INPUT_CEILING = 2.0

#: Long edge every image is resized to before measurement. Edge density and
#: texture energy are resolution-dependent, so a fixed working size is the only
#: way two photographs exported at different sizes stay comparable.
WORKING_LONG_EDGE = 1024

#: Long edge used for the contrast-preserving weight search only. The optimal
#: channel mix is a property of the palette and of coarse structure, not of
#: pixel-level detail, so searching 66 candidate weightings over a full
#: megapixel frame costs eleven times as much for the same answer. Verified to
#: return an identical recommendation to the full-resolution search on every
#: fixture in the catalogue.
SEARCH_PROXY_LONG_EDGE = 256

#: Gradient magnitude, in CIELAB L*, above which a pixel counts as an edge.
#: One L* unit is roughly one just-noticeable lightness step.
EDGE_LIGHTNESS_THRESHOLD = 1.0

#: Thresholds separating a chromatic boundary from a luminance-supported one,
#: in CIELAB units.
ISOLUMINANT_TAU_CHROMA = 5.0
ISOLUMINANT_TAU_LUMA = 2.0

#: Low-pass cutoff, in cycles per image, defining "coarse" for the Oliva and
#: Schyns diagnostic-colour proxy.
COARSE_SCALE_CYCLES = 8.0

#: Grid used by the mixed-white-balance advisory to compare local casts.
WHITE_BALANCE_GRID = 4

#: Names that may drive the "how little was lost" axis of a ranking.
#:
#: One published measure of what a conversion actually preserved, plus three
#: derived measures of how much of the image's structure is chromatic in the
#: first place. Neither half orders the reference catalogue correctly alone,
#: which is the reason both are here.
#:
#: The published half is the threshold-independent area under the E-score
#: curve, not `irreducible_colour_collapse`. Collapse is 1 - CCFR, so it
#: ignores CCPR entirely and saturates: three of the nine reference images
#: record exactly 0.0000, leaving the top of the batch unorderable. The area
#: integrates the full CCPR/CCFR curve across the sweep, which is precisely
#: why Lu, Xu and Jia propose it. Collapse stays recorded as evidence and
#: still selects the recommended method, but it does not carry an axis.
#:
#: The derived half is needed because the area is blind to *how much* colour
#: was at stake. Two reference images lose nothing measurable, yet one is a
#: near-achromatic graphic and the other is saturated with fully separable
#: colour. Only the chroma shares tell them apart.
#:
#: Conversely the shares alone are blind to magnitude: a frame with almost no
#: colour can still post a high chromatic-gradient *share*, so ratio noise
#: between two images that both have nothing at stake would decide their order.
LOSS_AXIS_INPUTS: tuple[str, ...] = (
    "best_threshold_independent_area",
    "isoluminant_edge_fraction",
    "chromatic_gradient_energy_ratio",
    "coarse_scale_chroma_contribution",
)

#: Names that may drive the "how much survives" axis of a ranking.
STRUCTURE_AXIS_INPUTS: tuple[str, ...] = (
    "edge_density",
    "texture_energy",
    "worst_region_merger",
)

#: Everything a ranking is allowed to look at. Anything else recorded by a
#: measurement is provenance.
RANKING_INPUT_NAMES: tuple[str, ...] = LOSS_AXIS_INPUTS + STRUCTURE_AXIS_INPUTS

#: Which direction is better for each ranking input. Recorded here rather than
#: hard-coded in `ranking.py` so that the polarity travels with the definition
#: of the input: the loss axis mixes a preservation measure, where more is
#: better, with three chromatic shares, where less is.
RANKING_INPUT_POLARITY: Mapping[str, bool] = {
    "best_threshold_independent_area": True,
    "isoluminant_edge_fraction": False,
    "chromatic_gradient_energy_ratio": False,
    "coarse_scale_chroma_contribution": False,
    "edge_density": True,
    "texture_energy": True,
    "worst_region_merger": True,
}

_LINEAR = ColorConvention(encoding="linear")


@dataclass(frozen=True)
class AdvisoryHeuristic:
    """A practitioner heuristic, reported with an honest evidence label."""

    name: str
    value: float
    evidence_level: EvidenceLevel


@dataclass(frozen=True)
class CandidacyMeasurement:
    """Evidence about one photograph. Carries no verdict and no overall score.

    `conversion_loss` is keyed by method name, and each entry holds only
    scalars: ccpr, ccfr, colour_collapse, threshold_independent_area, the
    e_score curve as a tuple of (tau, score) pairs, the four C2G-SSIM
    components flattened as c2g_lightness / c2g_colour_contrast /
    c2g_structure / c2g_overall, and `channel_weights` as a plain triple.

    `recommended_weights` and `recommended_filter` are the practical output:
    the channel-mixer recipe from whichever measured method preserved the most,
    plus its nearest classic-filter name. They describe *how* to convert, and
    say nothing about whether converting is a good idea, which stays a
    batch-relative question for `ranking.py`.
    """

    convention: ColorConvention
    working_long_edge: int
    chroma_information: Mapping[str, float]
    conversion_loss: Mapping[str, Mapping[str, Any]]
    surviving_structure: Mapping[str, float]
    # Widened from Mapping[str, float] because `orientation` and `image_type`
    # are genuinely strings. Under strict typing an annotation that lies is
    # worse than one that is broad.
    contextual_features: Mapping[str, float | str]
    advisory_heuristics: tuple[AdvisoryHeuristic, ...]
    ranking_inputs: tuple[str, ...]
    recommended_method: str
    recommended_weights: tuple[float, float, float]
    recommended_filter: str
    aesthetic_delta: float | None = None
    warnings: tuple[str, ...] = field(default=())


def _validated_image(rgb: np.ndarray) -> Array:
    """Reject anything that is not a three-channel image scaled to [0, 1]."""
    values = np.asarray(rgb, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(
            f"expected a three-channel image of shape (H, W, 3), got {values.shape}"
        )
    if values.size == 0:
        raise ValueError("cannot measure an empty image")
    lowest, highest = float(values.min()), float(values.max())
    if lowest < 0.0 or highest > 1.0:
        raise ValueError(
            f"pixel values must lie in [0, 1], got [{lowest}, {highest}]. "
            "The colourfulness anchors and the sRGB transfer functions both assume "
            "that scale, so 8-bit input is rescaled by the caller, never guessed at here."
        )
    return values


def _validated_delta(aesthetic_delta: float | None) -> float | None:
    if aesthetic_delta is None:
        return None
    value = float(aesthetic_delta)
    if not -1.0 <= value <= 1.0:
        raise ValueError(
            f"aesthetic_delta must be a normalised difference in [-1, 1], got {value}. "
            "Rescaling it here would hide a unit mismatch in the caller's model."
        )
    return value


def _validated_methods(methods: tuple[str, ...]) -> tuple[str, ...]:
    if not methods:
        raise ValueError("at least one conversion method is required")
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if len(set(methods)) != len(methods):
        raise ValueError(f"methods must be distinct, got {methods}")
    return tuple(methods)


def _resize_long_edge(image: Array, target: int) -> Array:
    """Resample so the long edge is exactly `target`.

    Downscaling low-passes first, because decimating without it aliases fine
    texture straight into the metrics that are supposed to measure it.
    Upscaling replicates instead of interpolating: there is no information to
    recover, and replication makes the operation exactly consistent under
    integer factors, so the same photograph exported at two sizes reaches
    byte-identical working pixels rather than merely similar ones.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest == target:
        return image
    scale = target / float(longest)
    out_height = max(1, round(height * scale))
    out_width = max(1, round(width * scale))
    source = image
    if scale < 1.0:
        sigma = 0.5 / scale
        source = gaussian_filter(source, sigma=(sigma, sigma, 0.0), mode="nearest")
    rows = (np.arange(out_height) * height) // out_height
    columns = (np.arange(out_width) * width) // out_width
    return np.ascontiguousarray(source[rows][:, columns])


def _noise_estimate(lightness: Array) -> float:
    """Robust high-frequency lightness noise, in L* units.

    The median absolute deviation of the residual after a light blur. Median
    rather than mean so that genuine edges, which are sparse, do not masquerade
    as noise. Advisory only: no published work links sensor noise to monochrome
    suitability.
    """
    residual = lightness - gaussian_filter(lightness, sigma=1.0, mode="nearest")
    return float(np.median(np.abs(residual)) * 1.4826)


def _white_balance_spread(lab: Array) -> float:
    """Spread of local colour casts across the frame, in CIELAB a*/b* units.

    Each cell of a coarse grid contributes the mean chromaticity of its
    brighter half, where a cast shows most clearly, and the reported value is
    the spread of those cell means. Advisory only, and a crude proxy: a scene
    lit by two sources and a scene simply containing two colours are not
    distinguishable this way.
    """
    height, width = lab.shape[:2]
    rows = np.array_split(np.arange(height), min(WHITE_BALANCE_GRID, height))
    columns = np.array_split(np.arange(width), min(WHITE_BALANCE_GRID, width))
    casts: list[tuple[float, float]] = []
    for row_block in rows:
        for column_block in columns:
            cell = lab[np.ix_(row_block, column_block)]
            lightness = cell[:, :, 0]
            if lightness.size == 0:
                continue
            bright = lightness >= np.median(lightness)
            if not bool(bright.any()):
                continue
            casts.append(
                (
                    float(cell[:, :, 1][bright].mean()),
                    float(cell[:, :, 2][bright].mean()),
                )
            )
    if len(casts) < 2:
        return 0.0
    points = np.asarray(casts, dtype=np.float64)
    return float(np.sqrt(np.sum(points.var(axis=0))))


def _conversion_entry(
    working: Array,
    lab: Array,
    method: str,
    weights: ChannelWeights,
    grey: Array,
    sweep: tuple[float, ...],
    convention: ColorConvention,
) -> dict[str, Any]:
    """One method's loss evidence, flattened to scalars and tuples."""
    scalars = _loss_scalars(lab, grey, taus=sweep)
    curve = scalars["e_score"]
    similarity = c2g_ssim(working, grey, convention=convention)
    return {
        # Averaged across the sweep. Rankings flip with tau, so no single
        # threshold is reported as if it were the answer; the full curve is
        # kept alongside for anyone who wants to look.
        "ccpr": float(np.mean([scalars["ccpr"][tau] for tau in sweep])),
        "ccfr": float(np.mean([scalars["ccfr"][tau] for tau in sweep])),
        "colour_collapse": float(
            np.mean([scalars["colour_collapse"][tau] for tau in sweep])
        ),
        "e_score_curve": tuple((tau, curve[tau]) for tau in sweep),
        "threshold_independent_area": threshold_independent_area(curve),
        "c2g_lightness": similarity.lightness,
        "c2g_colour_contrast": similarity.colour_contrast,
        "c2g_structure": similarity.structure,
        "c2g_overall": similarity.overall,
        "channel_weights": weights.as_tuple(),
    }


def measure_bw_candidacy(
    rgb: np.ndarray,
    *,
    convention: ColorConvention,
    taus: tuple[float, ...],
    methods: tuple[str, ...] = ("bt709_luma", "contrast_preserving"),
    aesthetic_delta: float | None = None,
) -> CandidacyMeasurement:
    """Measure one photograph. Pure: two calls on identical input compare equal.

    `aesthetic_delta` is an optional externally supplied score(grey) - score(colour)
    from a no-reference model, normalised to [-1, 1]. It is recorded but is not
    a ranking input, because no published model is validated for paired
    desaturation decisions. A caller who wants it to count must say so
    explicitly when building a total order.
    """
    values = _validated_image(rgb)
    sweep = _validated_sweep(taus)
    chosen = _validated_methods(methods)
    delta = _validated_delta(aesthetic_delta)

    source_height, source_width = values.shape[:2]
    working = _resize_long_edge(values, WORKING_LONG_EDGE)
    lab = rgb_to_lab(working, convention)
    lightness = lab[:, :, 0]
    linear_luminance = luminance(working, _LINEAR)
    zones = zone_map(linear_luminance)

    # --- conversion loss, one entry per method ------------------------------
    proxy = _resize_long_edge(values, SEARCH_PROXY_LONG_EDGE)
    entries: dict[str, dict[str, Any]] = {}
    for method in chosen:
        if method == "contrast_preserving":
            weights = recommended_channel_weights(
                proxy, method=method, convention=convention
            )
            grey = apply_channel_weights(working, weights, convention=convention)
        else:
            weights = recommended_channel_weights(
                working, method=method, convention=convention
            )
            grey = decolorize(working, method=method, convention=convention)
        entries[method] = _conversion_entry(
            working, lab, method, weights, grey, sweep, convention
        )

    # Same expression the contract test uses, so "the method that preserved
    # most" means one thing rather than two. Ties fall to the earliest method
    # in the caller's own ordering.
    recommended_method = min(entries, key=lambda name: entries[name]["colour_collapse"])
    recommended_weights = entries[recommended_method]["channel_weights"]
    irreducible = min(entry["colour_collapse"] for entry in entries.values())
    best_area = max(entry["threshold_independent_area"] for entry in entries.values())

    # --- chroma information -------------------------------------------------
    colourfulness = hasler_susstrunk_colourfulness(working * 255.0)
    chroma = chroma_statistics(working, convention)
    chroma_information: dict[str, float] = {
        "colourfulness": colourfulness,
        "chroma_mean": chroma.mean,
        "chroma_std": chroma.std,
        "chroma_p95": chroma.p95,
        "hue_dispersion": chroma.hue_dispersion,
        "isoluminant_edge_fraction": isoluminant_edge_fraction(
            lab, tau_chroma=ISOLUMINANT_TAU_CHROMA, tau_luma=ISOLUMINANT_TAU_LUMA
        ),
        "chromatic_gradient_energy_ratio": chromatic_gradient_energy_ratio(lab),
        "coarse_scale_chroma_contribution": coarse_scale_chroma_contribution(
            lab, cycles_per_image=COARSE_SCALE_CYCLES
        ),
        # These two are recorded among the chroma evidence rather than under a
        # method, because they are properties of the photograph: what no
        # measured recipe could carry through, and what the best of them did
        # carry through, as opposed to what any one recipe happened to do.
        # This group is named for the question it answers, "how much
        # information does this image lose when chroma is discarded", not for
        # the module the numbers came from.
        "irreducible_colour_collapse": irreducible,
        "best_threshold_independent_area": best_area,
    }

    # --- surviving structure ------------------------------------------------
    labels = segment_regions(lab)
    surviving_structure: dict[str, float] = {
        "zone_span": robust_tonal_span(zones),
        "zone_entropy": zone_entropy(zones),
        "textural_range_coverage": textural_range_coverage(zones),
        "active_zone_count": float(active_zone_count(zones, min_occupancy=0.01)),
        "edge_density": edge_density(lightness, threshold=EDGE_LIGHTNESS_THRESHOLD),
        "texture_energy": texture_energy(linear_luminance),
        "worst_region_merger": _merger_or_default(zones, labels),
    }

    # --- context and advisories --------------------------------------------
    contextual_features: dict[str, float | str] = {
        "aspect_ratio": float(source_width) / float(source_height),
        "megapixels": (source_height * source_width) / 1e6,
        "orientation": _orientation(source_height, source_width),
        "image_type": classify_image_type(linear_luminance),
    }
    advisory_heuristics = (
        AdvisoryHeuristic("high_iso_noise", _noise_estimate(lightness), "unsupported"),
        AdvisoryHeuristic(
            "mixed_white_balance", _white_balance_spread(lab), "unsupported"
        ),
        AdvisoryHeuristic(
            "full_tonal_range", surviving_structure["zone_span"] / 10.0, "convention"
        ),
    )

    warnings: list[str] = []
    if colourfulness < ACHROMATIC_INPUT_CEILING:
        warnings.append(
            f"Input is already achromatic (Hasler-Susstrunk colourfulness {colourfulness:.2f} "
            f"below {ACHROMATIC_INPUT_CEILING}); there is no colour to discard, so the "
            "question of whether to convert does not apply."
        )

    return CandidacyMeasurement(
        convention=convention,
        working_long_edge=WORKING_LONG_EDGE,
        chroma_information=chroma_information,
        conversion_loss=entries,
        surviving_structure=surviving_structure,
        contextual_features=contextual_features,
        advisory_heuristics=advisory_heuristics,
        ranking_inputs=RANKING_INPUT_NAMES,
        recommended_method=recommended_method,
        recommended_weights=recommended_weights,
        recommended_filter=nearest_classic_filter(ChannelWeights(*recommended_weights)),
        aesthetic_delta=delta,
        warnings=tuple(warnings),
    )


def _orientation(height: int, width: int) -> str:
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"


def _merger_or_default(zones: Array, labels: NDArray[np.intp]) -> float:
    """Worst adjacent-region merger, or the best possible value when undefined.

    A single-region image has no adjacent pair, so nothing in it could merge.
    That is the *absence* of a defect, and the honest number is the largest
    separation the zone scale can express rather than zero, which is the score
    for the worst possible merger.
    """
    try:
        return worst_region_merger(zones, labels)
    except ValueError:
        return float(zones.max() - zones.min()) if zones.size else 0.0
