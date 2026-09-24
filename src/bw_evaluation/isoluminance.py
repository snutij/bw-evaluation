"""Chromatic contrast that is not accompanied by luminance contrast.

Research anchor
---------------
Cadik, "Perceptual Evaluation of Color-to-Grayscale Image Conversions",
Computer Graphics Forum 27(7), 2008, identifies isoluminant colour as *the*
canonical failure mode of luminance-only conversion.

Oliva & Schyns, Cognitive Psychology 41(2), 2000, show the useful chromatic
contribution is concentrated at coarse spatial scales, and Mullen, J. Physiol.
359, 1985, shows chromatic contrast sensitivity falls off with spatial
frequency far faster than achromatic sensitivity, so fine detail is a
luminance phenomenon. Both findings are why the measures below are computed
across scales rather than at native resolution only.

No canonical named scalar metric exists in the literature for "percentage of
isoluminant content". The two measures below are therefore explicitly marked as
derived, not standard, and every result carries `evidence_level="derived"` when
it reaches the report.

Two implementation notes
------------------------
*Units.* Everything here runs on CIELAB. L*, a* and b* share one perceptual
scale, so a luminance gradient and a chromatic gradient can be compared
directly. The same comparison on raw RGB would be meaningless.

*Pooling.* :func:`chromatic_gradient_energy_ratio` pools squared magnitudes,
because both of its terms are measured at the same scale and "energy" is the
usual squared quantity. :func:`coarse_scale_chroma_contribution` pools absolute
magnitudes instead, and that difference is load-bearing rather than cosmetic:
it compares a blurred numerator against an unblurred denominator, and blurring
preserves the total gradient *mass* across a step edge while destroying its
squared energy. Pooling squares there would have measured the blur kernel
rather than the picture.
"""

from __future__ import annotations

from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

Array: TypeAlias = NDArray[np.float64]

#: Marks these metrics as constructed for this framework rather than cited.
EVIDENCE_LEVEL: Final[str] = "derived"

#: Half-power relation between a Gaussian sigma in pixels and a cutoff in
#: cycles per image: the response exp(-2 pi^2 sigma^2 f^2) reaches 1/2 when
#: sigma = sqrt(ln 2 / (2 pi^2)) / f. Shared with the Peli bands in `tonal`
#: so that "8 cycles per image" means the same thing in both modules.
_HALF_POWER_SIGMA: Final[float] = float(np.sqrt(np.log(2.0) / (2.0 * np.pi**2)))

#: Below this total gradient mass an image is treated as having no structure at
#: all, and every ratio returns 0 rather than dividing by approximately nothing.
_STRUCTURE_FLOOR: Final[float] = 1e-12


def _as_lab(lab: Array) -> Array:
    """Validate and normalise an incoming CIELAB image to float64 (H, W, 3)."""
    values = np.asarray(lab, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(
            f"expected a CIELAB image of shape (H, W, 3), got {values.shape}"
        )
    return values


def _jacobian_norm(channels: Array) -> Array:
    """Frobenius norm of the spatial Jacobian of a stack of channels.

    Di Zenzo's multi-channel gradient takes the leading eigenvalue of the
    structure tensor instead. The Frobenius norm is its upper bound, is
    cheaper, and needs no eigen-decomposition per pixel; the metrics here only
    ever compare it against a threshold or pool it, so the distinction does not
    change any ordering. Documented rather than silently substituted.
    """
    squared = np.zeros(channels.shape[:2], dtype=np.float64)
    for index in range(channels.shape[2]):
        vertical, horizontal = np.gradient(channels[:, :, index])
        squared += vertical**2 + horizontal**2
    return np.sqrt(squared)


def _sigma_for_cutoff(shape: tuple[int, int], cycles_per_image: float) -> float:
    """Gaussian sigma in pixels whose half-power point sits at `cycles_per_image`.

    "Per image" is resolved against the long edge. On a non-square frame that
    is a choice, not a fact, so it is stated here rather than left implicit.
    """
    if cycles_per_image <= 0.0:
        raise ValueError(f"cycles_per_image must be positive, got {cycles_per_image}")
    long_edge = float(max(shape))
    return _HALF_POWER_SIGMA * long_edge / float(cycles_per_image)


def luminance_gradient(lab: Array) -> Array:
    """Gradient magnitude of the CIELAB L* channel."""
    values = _as_lab(lab)
    return _jacobian_norm(values[:, :, :1])


def chromatic_gradient(lab: Array) -> Array:
    """Gradient magnitude over the CIELAB a*/b* channels jointly."""
    values = _as_lab(lab)
    return _jacobian_norm(values[:, :, 1:])


def isoluminant_edge_fraction(
    lab: Array,
    *,
    tau_chroma: float,
    tau_luma: float,
) -> float:
    """Fraction of detected edges that are chromatic but not luminance-supported.

    Returns a value in [0, 1]. Both thresholds are required and noise-dependent;
    there is no defensible default, so none is supplied.

    An edge is any pixel where either channel exceeds its threshold, so the
    denominator counts everything a viewer could see a boundary at. The
    numerator counts the subset that a luminance-only conversion would erase.
    """
    values = _as_lab(lab)
    chromatic = chromatic_gradient(values)
    luminous = luminance_gradient(values)
    edges = (chromatic >= tau_chroma) | (luminous >= tau_luma)
    total = int(np.count_nonzero(edges))
    if total == 0:
        return 0.0
    unsupported = (chromatic >= tau_chroma) & (luminous < tau_luma)
    return float(np.count_nonzero(unsupported) / total)


def chromatic_gradient_energy_ratio(
    lab: Array,
    *,
    scales: tuple[int, ...] = (1, 2, 4),
) -> float:
    """Chromatic share of total gradient energy, averaged over scales, in [0, 1].

    Multi-scale because Oliva & Schyns (Cognitive Psychology, 2000) show the
    chromatic contribution is concentrated at coarse spatial scales, and Mullen
    (J. Physiol., 1985) shows fine detail is carried almost entirely by
    luminance. Scale `s` means the image pre-filtered and decimated by `s`, so
    scale 1 is the native resolution.
    """
    values = _as_lab(lab)
    if not scales:
        raise ValueError("at least one scale is required")
    ratios: list[float] = []
    for scale in scales:
        if scale < 1:
            raise ValueError(f"scales must be positive integers, got {scale}")
        if scale == 1:
            sampled = values
        else:
            blurred = gaussian_filter(
                values, sigma=(scale / 2.0, scale / 2.0, 0.0), mode="nearest"
            )
            sampled = blurred[::scale, ::scale, :]
        if min(sampled.shape[:2]) < 2:
            continue
        chromatic = float(np.sum(chromatic_gradient(sampled) ** 2))
        luminous = float(np.sum(luminance_gradient(sampled) ** 2))
        total = chromatic + luminous
        ratios.append(0.0 if total <= _STRUCTURE_FLOOR else chromatic / total)
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def coarse_scale_chroma_contribution(
    lab: Array,
    *,
    cycles_per_image: float,
) -> float:
    """Chromatic gradient energy restricted to low spatial frequencies, in [0, 1].

    This is the computable proxy for "colour is carrying scene category rather
    than detail", which is the Oliva & Schyns diagnostic-colour effect.

    The numerator is the chromatic gradient mass surviving a low-pass at
    `cycles_per_image`; the denominator is the full gradient mass, chromatic
    and achromatic, at native resolution. Broad colour fields therefore score
    high while fine chromatic noise, which Mullen shows the visual system
    resolves poorly, scores near zero even when it is the larger share of the
    raw chroma.
    """
    values = _as_lab(lab)
    height, width = values.shape[:2]
    sigma = _sigma_for_cutoff((height, width), cycles_per_image)
    low_passed = gaussian_filter(values, sigma=(sigma, sigma, 0.0), mode="nearest")
    coarse_chroma = float(np.sum(chromatic_gradient(low_passed)))
    total = float(
        np.sum(chromatic_gradient(values)) + np.sum(luminance_gradient(values))
    )
    if total <= _STRUCTURE_FLOOR:
        return 0.0
    return float(min(coarse_chroma / total, 1.0))
