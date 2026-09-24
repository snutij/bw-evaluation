"""How much chromatic information is present at all.

Research anchor
---------------
Hasler & Susstrunk, "Measuring Colourfulness in Natural Images",
SPIE/IS&T Human Vision and Electronic Imaging VIII, 2003.

    rg = R - G
    yb = 0.5 * (R + G) - B
    C_HS = sqrt(var_rg + var_yb) + 0.3 * sqrt(mean_rg^2 + mean_yb^2)

Their published psychophysical anchors are the rare case where the literature
supplies real numeric thresholds, so they are encoded here rather than invented.

Caveat carried into the contract: C_HS cannot distinguish a chromatic edge that
does discriminative work from one large flat coloured region. It is therefore a
screening signal only, never a verdict input on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .conventions import ColorConvention, rgb_to_lab

Array: TypeAlias = NDArray[np.float64]

#: Category -> lower bound of the Hasler & Susstrunk score, 8-bit RGB scale.
COLOURFULNESS_ANCHORS: Final[dict[str, float]] = {
    "not colourful": 0.0,
    "slightly colourful": 15.0,
    "moderately colourful": 33.0,
    "averagely colourful": 45.0,
    "quite colourful": 59.0,
    "highly colourful": 82.0,
    "extremely colourful": 109.0,
}

#: Weight on the opponent-channel means, from the published formulation.
_MEAN_WEIGHT: Final[float] = 0.3

#: Percentile reported alongside mean and spread of CIELAB chroma.
_CHROMA_PERCENTILE: Final[float] = 95.0

#: Total chroma below which hue angle is undefined rather than merely small.
_NEUTRAL_CHROMA_FLOOR: Final[float] = 1e-9


@dataclass(frozen=True)
class ChromaStatistics:
    """Summary of the CIELAB chroma channel C*ab = sqrt(a*^2 + b*^2).

    `hue_dispersion` is here because the three magnitude statistics turned out
    to be nearly blind to the case that matters most. A scene of several
    distinct hues at similar saturation, which is precisely the colour-diagnostic
    condition Oliva & Schyns describe, has almost the same chroma mean and a
    negligible chroma spread compared with a single flat colour patch. Hue
    angle is what separates them.
    """

    mean: float
    std: float
    p95: float
    hue_dispersion: float


def _opponent_channels(rgb_8bit: Array) -> tuple[Array, Array]:
    """The simple opponent approximations the 2003 paper is defined on."""
    values = np.asarray(rgb_8bit, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(
            f"expected a three-channel image of shape (H, W, 3), got {values.shape}"
        )
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    return red - green, 0.5 * (red + green) - blue


def hasler_susstrunk_colourfulness(rgb_8bit: Array) -> float:
    """Colourfulness on the 8-bit RGB scale the published anchors were fitted on."""
    rg, yb = _opponent_channels(rgb_8bit)
    spread = float(np.hypot(rg.std(), yb.std()))
    offset = float(np.hypot(rg.mean(), yb.mean()))
    return spread + _MEAN_WEIGHT * offset


def colourfulness_category(score: float) -> str:
    """Map a Hasler & Susstrunk score onto one of :data:`COLOURFULNESS_ANCHORS`.

    Each anchor is the inclusive lower bound of the band it opens, so a score
    landing exactly on a boundary belongs to the higher band and the rule is
    never ambiguous.
    """
    value = float(score)
    if value < 0.0:
        raise ValueError(f"colourfulness cannot be negative, got {value}")
    label = "not colourful"
    for name, lower_bound in COLOURFULNESS_ANCHORS.items():
        if value >= lower_bound:
            label = name
    return label


def _hue_dispersion(lab: Array) -> float:
    """Chroma-weighted circular dispersion of CIELAB hue angle, in [0, 1].

    Hue is an angle, so ordinary variance is meaningless on it. This is the
    standard circular measure 1 - R, where R is the resultant length of the
    unit hue vectors. Weighting by chroma stops near-neutral pixels, whose hue
    angle is numerically unstable, from dominating.

    Zero for a single hue, rising toward 1 as hues spread around the wheel.
    Returns 0.0 for an achromatic image, where hue is undefined rather than
    dispersed.
    """
    a, b = lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    total = float(chroma.sum())
    if total <= _NEUTRAL_CHROMA_FLOOR:
        return 0.0
    # Unit hue vectors are (a, b) / C, so weighting each by C leaves (a, b).
    resultant = float(np.hypot(a.sum(), b.sum())) / total
    return 1.0 - resultant


def chroma_statistics(rgb: Array, convention: ColorConvention) -> ChromaStatistics:
    """Mean, spread, 95th percentile and circular hue dispersion of CIELAB chroma."""
    lab = rgb_to_lab(rgb, convention)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    return ChromaStatistics(
        mean=float(chroma.mean()),
        std=float(chroma.std()),
        p95=float(np.percentile(chroma, _CHROMA_PERCENTILE)),
        hue_dispersion=_hue_dispersion(lab),
    )
