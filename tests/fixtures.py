"""Synthetic images that isolate one research finding each.

These builders deliberately depend on nothing from `bw_evaluation`. If the
fixtures used the implementation's own colour maths, a bug in that maths would
cancel out and the tests would pass for the wrong reason.

Every image is returned as gamma-encoded sRGB float in [0, 1], shape (H, W, 3),
and every builder is deterministic: no RNG without a fixed seed.
"""

from __future__ import annotations

import numpy as np

SIZE = 128


def _oetf(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055
    )


def _flat(colour: tuple[float, float, float], size: int = SIZE) -> np.ndarray:
    return np.tile(np.asarray(colour, dtype=float), (size, size, 1))


def neutral_grey() -> np.ndarray:
    """Perfectly achromatic. Colourfulness must be exactly zero."""
    return _flat((0.5, 0.5, 0.5))


def isoluminant_checkerboard() -> np.ndarray:
    """Red and teal patches carrying equal BT.709 linear luminance (0.2126).

    Teal 0/142/142 was solved for, not guessed: 0.7152+0.0722 times the
    linearised value of 142/255 equals the linear luminance of pure red.

    This is Cadik's canonical failure case. Under any luminance-only conversion
    the pattern must vanish entirely.
    """
    red = np.array([1.0, 0.0, 0.0])
    teal = np.array([0.0, 142 / 255, 142 / 255])
    image = np.zeros((SIZE, SIZE, 3))
    block = SIZE // 8
    for row in range(0, SIZE, block):
        for col in range(0, SIZE, block):
            odd = ((row // block) + (col // block)) % 2
            image[row : row + block, col : col + block] = teal if odd else red
    return image


def luminance_separated_colours() -> np.ndarray:
    """Same checkerboard geometry, but the two colours differ strongly in L*.

    The control condition for :func:`isoluminant_checkerboard`: equally
    saturated, equally structured, but nothing is lost in grey.
    """
    dark_red = np.array([0.35, 0.0, 0.0])
    light_yellow = np.array([1.0, 0.95, 0.55])
    image = np.zeros((SIZE, SIZE, 3))
    block = SIZE // 8
    for row in range(0, SIZE, block):
        for col in range(0, SIZE, block):
            odd = ((row // block) + (col // block)) % 2
            image[row : row + block, col : col + block] = (
                light_yellow if odd else dark_red
            )
    return image


def foggy_textured_scene() -> np.ndarray:
    """Near-achromatic, detail-dense: fog, snow, concrete, overcast.

    Mullen (1985): fine detail is carried almost entirely by luminance, so this
    image should lose essentially nothing when desaturated.
    """
    rng = np.random.default_rng(seed=20260923)
    base = 0.45 + 0.18 * rng.standard_normal((SIZE, SIZE))
    base += 0.12 * np.sin(np.linspace(0, 24 * np.pi, SIZE))[None, :]
    base = np.clip(base, 0.02, 0.98)
    image = np.repeat(base[:, :, None], 3, axis=2)
    # A faint but real cool cast. It has to clear the achromatic guard, because
    # this fixture stands for a low-chroma *colour* photograph, not a greyscale
    # file. Uniform on purpose, so it adds no chromatic gradient.
    image[:, :, 2] += 0.035
    return np.clip(image, 0.0, 1.0)


def colour_diagnostic_landscape() -> np.ndarray:
    """Coarse chromatic bands with weak luminance separation.

    Oliva & Schyns (2000): colour carries scene category at coarse spatial
    scales. Blue sky over green foliage over ochre ground, all placed close in
    lightness so that luminance alone cannot reconstruct the layout.
    """
    image = np.zeros((SIZE, SIZE, 3))
    image[: SIZE // 3] = (0.35, 0.52, 0.78)
    image[SIZE // 3 : 2 * SIZE // 3] = (0.34, 0.58, 0.30)
    image[2 * SIZE // 3 :] = (0.66, 0.50, 0.26)
    return image


def high_contrast_geometry() -> np.ndarray:
    """Strong luminance contrast, hard shapes, minimal chroma: the archetypal
    strong candidate that practitioner literature describes qualitatively."""
    image = np.full((SIZE, SIZE, 3), 0.95)
    image[:, :] = (0.942, 0.950, 0.962)
    image[20:60, 20:100] = (0.045, 0.040, 0.038)
    image[70:110, 40:60] = (0.098, 0.100, 0.106)
    image[70:110, 75:118] = (0.575, 0.550, 0.515)
    return image


def flat_black() -> np.ndarray:
    """Zero luminance everywhere. Guards against log(0) in the zone mapping."""
    return _flat((0.0, 0.0, 0.0))


def linear_ramp() -> np.ndarray:
    """Linear-light ramp from black to white, used for zone arithmetic."""
    linear = np.linspace(0.0, 1.0, SIZE)[None, :].repeat(SIZE, axis=0)
    encoded = _oetf(linear)
    return np.repeat(encoded[:, :, None], 3, axis=2)


def linear_luminance_patches(values: tuple[float, ...]) -> np.ndarray:
    """A row of patches with exactly the requested *linear* luminance values."""
    encoded = _oetf(np.asarray(values, dtype=float))
    return np.repeat(np.repeat(encoded[None, :, None], SIZE, axis=0), 3, axis=2)


def muted_portrait_tones() -> np.ndarray:
    """Deliberately ambiguous: modest chroma, modest luminance separation.

    Skin-like tones in the upper zones. Neither clearly wins nor clearly loses
    from desaturation, so it exists to give the middle of a batch something
    real to sit in rather than leaving `uncertain` as whatever falls through.
    """
    image = np.zeros((SIZE, SIZE, 3))
    image[:, :] = (0.78, 0.66, 0.58)
    image[SIZE // 3 : 2 * SIZE // 3] = (0.70, 0.57, 0.50)
    image[2 * SIZE // 3 :] = (0.62, 0.52, 0.47)
    return image


def saturated_but_merging() -> np.ndarray:
    """Strong colour where only *some* pairs are isoluminant.

    The hard case for any metric that looks at colourfulness alone: two of the
    three regions separate tonally, the third merges with its neighbour.
    """
    image = np.zeros((SIZE, SIZE, 3))
    image[: SIZE // 3] = (0.92, 0.88, 0.20)
    image[SIZE // 3 : 2 * SIZE // 3] = (0.85, 0.20, 0.18)
    image[2 * SIZE // 3 :] = (0.0, 142 / 255, 142 / 255)
    return image


def backlit_silhouette() -> np.ndarray:
    """Content confined to the extreme zones, which is legitimate, not a defect.

    Guards the rule that a narrow or bimodal histogram must never be penalised
    on its own.
    """
    image = np.full((SIZE, SIZE, 3), 0.97)
    image[:, :] = (0.985, 0.968, 0.935)  # warm backlight
    image[70:, 30:95] = (0.030, 0.029, 0.034)
    image[60:75, 45:80] = (0.052, 0.050, 0.057)
    return image


def catalogue() -> dict[str, np.ndarray]:
    """A batch large enough for percentiles to mean anything.

    Spans the full range on purpose: two clear strong candidates, two clear
    poor ones, and four that sit somewhere in between.
    """
    return {
        "fog": foggy_textured_scene(),
        "geometry": high_contrast_geometry(),
        "silhouette": backlit_silhouette(),
        "ramp": linear_ramp(),
        "portrait": muted_portrait_tones(),
        "merging": saturated_but_merging(),
        "landscape": colour_diagnostic_landscape(),
        "isoluminant": isoluminant_checkerboard(),
        "separated": luminance_separated_colours(),
    }


def wide_and_tall_crops() -> tuple[np.ndarray, np.ndarray]:
    """The same content at 3:1 and 1:3, for the aspect-ratio neutrality test."""
    source = high_contrast_geometry()
    wide = source[48:80, :, :]
    tall = source[:, 48:80, :]
    return wide, tall
