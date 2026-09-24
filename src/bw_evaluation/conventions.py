"""Colour conventions.

Research anchor
---------------
ITU-R BT.601-7 (0.299 / 0.587 / 0.114), BT.709-6 (0.2126 / 0.7152 / 0.0722)
and BT.2020-2 (0.2627 / 0.6780 / 0.0593) specify *luma* coefficients applied to
gamma-encoded R'G'B'. Applying the same weights to linear-light RGB yields
*luminance*, which is a materially different grey for the same pixel.

Contract consequence: no metric in this package accepts an image without an
explicit, fully specified `ColorConvention`. A "deterministic criterion" that
does not pin the transfer function and the coefficient set is not deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, get_args

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]

Encoding: TypeAlias = Literal["linear", "gamma"]

#: Luma / luminance coefficient sets, keyed by the recommendation that defines them.
COEFFICIENTS: Final[dict[str, tuple[float, float, float]]] = {
    "bt601": (0.299, 0.587, 0.114),
    "bt709": (0.2126, 0.7152, 0.0722),
    "bt2020": (0.2627, 0.6780, 0.0593),
}

# sRGB (IEC 61966-2-1) primaries to CIE XYZ, D65.
_SRGB_TO_XYZ: Final[Array] = np.array(
    [
        [0.4123907992659595, 0.3575843393838780, 0.1804807884018343],
        [0.2126390058715104, 0.7151686787677559, 0.0721923153607337],
        [0.0193308187155919, 0.1191947797946260, 0.9505321522496606],
    ],
    dtype=np.float64,
)

# Taken as the matrix applied to (1, 1, 1) rather than a rounded table value, so
# that a neutral patch maps to exactly a* = b* = 0 and white to exactly L* = 100.
_WHITE_POINT: Final[Array] = _SRGB_TO_XYZ.sum(axis=1)

# CIELAB piecewise transfer constants: delta = 6/29.
_LAB_DELTA: Final[float] = 6.0 / 29.0
_LAB_DELTA_CUBED: Final[float] = _LAB_DELTA**3

# sRGB transfer function breakpoints (IEC 61966-2-1).
_OETF_LINEAR_THRESHOLD: Final[float] = 0.0031308
_EOTF_ENCODED_THRESHOLD: Final[float] = 0.04045
_TRANSFER_SLOPE: Final[float] = 12.92
_TRANSFER_GAIN: Final[float] = 1.055
_TRANSFER_OFFSET: Final[float] = 0.055
_TRANSFER_EXPONENT: Final[float] = 2.4


@dataclass(frozen=True)
class ColorConvention:
    """Fully specifies how pixel values are to be interpreted.

    `encoding` has no default on purpose: the caller must state whether the
    weighted sum produces luma (gamma-encoded input) or luminance (linear input).
    """

    encoding: Encoding
    coefficients: str = "bt709"
    working_space: str = "srgb"
    white_point: str = "D65"

    def __post_init__(self) -> None:
        if self.encoding not in get_args(Encoding):
            raise ValueError(
                f"encoding must be one of {get_args(Encoding)}, got {self.encoding!r}. "
                "Luma on gamma-encoded values and luminance on linear values are "
                "different greys; the caller has to say which one is meant."
            )
        if self.coefficients not in COEFFICIENTS:
            raise ValueError(
                f"coefficients must be one of {tuple(COEFFICIENTS)}, got {self.coefficients!r}"
            )
        if self.working_space != "srgb":
            raise ValueError(
                f"only the sRGB working space is supported, got {self.working_space!r}"
            )
        if self.white_point != "D65":
            raise ValueError(
                f"only the D65 white point is supported, got {self.white_point!r}"
            )

    @property
    def weights(self) -> tuple[float, float, float]:
        """The three channel weights this convention applies."""
        return COEFFICIENTS[self.coefficients]


def srgb_eotf(encoded: Array) -> Array:
    """sRGB electro-optical transfer function: gamma-encoded [0,1] -> linear [0,1]."""
    values = np.asarray(encoded, dtype=np.float64)
    return np.where(
        values <= _EOTF_ENCODED_THRESHOLD,
        values / _TRANSFER_SLOPE,
        ((values + _TRANSFER_OFFSET) / _TRANSFER_GAIN) ** _TRANSFER_EXPONENT,
    )


def srgb_oetf(linear: Array) -> Array:
    """Inverse of :func:`srgb_eotf`: linear [0,1] -> gamma-encoded [0,1]."""
    values = np.asarray(linear, dtype=np.float64)
    return np.where(
        values <= _OETF_LINEAR_THRESHOLD,
        values * _TRANSFER_SLOPE,
        _TRANSFER_GAIN * np.power(np.maximum(values, 0.0), 1.0 / _TRANSFER_EXPONENT)
        - _TRANSFER_OFFSET,
    )


def _as_image(rgb: Array) -> Array:
    """Validate and normalise an incoming image to float64 (H, W, 3)."""
    values = np.asarray(rgb, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(
            f"expected a three-channel image of shape (H, W, 3), got {values.shape}"
        )
    return values


def luminance(rgb: Array, convention: ColorConvention) -> Array:
    """Weighted achromatic channel.

    Returns luma when `convention.encoding == "gamma"` and linear-light
    luminance when `convention.encoding == "linear"`. Input is always
    gamma-encoded sRGB in [0,1]; the convention decides whether it is
    linearised before weighting.
    """
    values = _as_image(rgb)
    if convention.encoding == "linear":
        values = srgb_eotf(values)
    weights = np.asarray(convention.weights, dtype=np.float64)
    return np.tensordot(values, weights, axes=([2], [0]))


def _lab_transfer(ratio: Array) -> Array:
    """The CIELAB piecewise cube-root companding function."""
    return np.where(
        ratio > _LAB_DELTA_CUBED,
        np.cbrt(ratio),
        ratio / (3.0 * _LAB_DELTA**2) + 4.0 / 29.0,
    )


def linear_to_lightness(linear_luminance: Array) -> Array:
    """Linear-light luminance in [0,1] -> CIELAB L* in [0,100].

    Exposed because any comparison between a grey rendering and a CIELAB colour
    difference has to happen in the same units. Comparing a [0,1] grey value
    against a Delta-E would be meaningless.
    """
    values = np.asarray(linear_luminance, dtype=np.float64)
    return 116.0 * _lab_transfer(np.clip(values, 0.0, 1.0)) - 16.0


def rgb_to_lab(rgb: Array, convention: ColorConvention) -> Array:
    """Gamma-encoded sRGB in [0,1] -> CIELAB (L* in [0,100], a*/b* signed).

    The convention is required for provenance and validation. CIELAB is defined
    on linear-light tristimulus values, so the input is always linearised here
    regardless of `convention.encoding`; that field governs the weighted
    achromatic channel, not this colorimetric transform.
    """
    values = _as_image(rgb)
    linear = srgb_eotf(values)
    xyz = np.tensordot(linear, _SRGB_TO_XYZ.T, axes=([2], [0]))
    companded = _lab_transfer(xyz / _WHITE_POINT)
    fx, fy, fz = companded[..., 0], companded[..., 1], companded[..., 2]
    return np.stack(
        [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)],
        axis=-1,
    )
