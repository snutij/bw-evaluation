#!/usr/bin/env python3
"""
Adaptive B&W converter with multiple style presets.

Each preset tunes channel mix, contrast, shadow lift, and optional post-processing
for a particular scene type.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

STYLE_NAMES = ("portrait", "landscape", "high-contrast", "street", "architecture")

# Map scene_type → default conversion style
SCENE_TO_STYLE: dict[str, str] = {
    "portrait": "portrait",
    "landscape": "landscape",
    "architecture": "architecture",
    "street": "street",
    "generic": "portrait",  # safe default
}


@dataclass(frozen=True)
class ConversionPreset:
    """Parameters for a B&W conversion style."""

    red: float
    green: float
    blue: float
    contrast: float
    shadow_lift: int
    grain: float = 0.0       # standard deviation for additive grain
    unsharp: bool = False     # apply unsharp mask post-conversion


PRESETS: dict[str, ConversionPreset] = {
    "portrait": ConversionPreset(
        red=0.35, green=0.45, blue=0.20, contrast=1.15, shadow_lift=10,
    ),
    "landscape": ConversionPreset(
        red=0.30, green=0.60, blue=0.10, contrast=1.25, shadow_lift=0,
    ),
    "high-contrast": ConversionPreset(
        red=0.40, green=0.35, blue=0.25, contrast=1.40, shadow_lift=0,
    ),
    "street": ConversionPreset(
        red=0.33, green=0.36, blue=0.31, contrast=1.10, shadow_lift=5,
        grain=3.0,
    ),
    "architecture": ConversionPreset(
        red=0.25, green=0.50, blue=0.25, contrast=1.30, shadow_lift=0,
        unsharp=True,
    ),
}


def convert_bw(img: Image.Image, preset: ConversionPreset) -> Image.Image:
    """Convert an RGB image to B&W using the given preset."""
    rgb = np.array(img.convert("RGB"), dtype=np.float32)

    # Channel mixing
    bw = preset.red * rgb[:, :, 0] + preset.green * rgb[:, :, 1] + preset.blue * rgb[:, :, 2]
    bw = np.clip(bw, 0, 255)

    # Contrast around midpoint
    midpoint = 128.0
    bw = (bw - midpoint) * preset.contrast + midpoint

    # Shadow lift
    if preset.shadow_lift > 0:
        shadow_mask = np.clip(1 - (bw / 80), 0, 1)
        bw = bw + preset.shadow_lift * shadow_mask

    # Optional grain
    if preset.grain > 0:
        rng = np.random.default_rng()
        noise = rng.normal(0, preset.grain, bw.shape).astype(np.float32)
        bw = bw + noise

    bw = np.clip(bw, 0, 255).astype(np.uint8)
    result = Image.fromarray(bw, mode="L")

    # Optional unsharp mask (architecture sharpening)
    if preset.unsharp:
        result = result.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=3))

    return result


def pick_style(scene_type: str | None, forced_style: str | None) -> str:
    """Return the conversion style name to use."""
    if forced_style and forced_style != "auto":
        return forced_style
    return SCENE_TO_STYLE.get(scene_type or "", "portrait")


def process_photos(
    input_dir: Path,
    output_dir: Path,
    filenames: list[str],
    style: str = "auto",
    scene_types: dict[str, str] | None = None,
    fmt: str = "jpeg",
    quality: int = 95,
    no_overwrite: bool = False,
) -> int:
    """
    Convert a list of photos to B&W.

    Returns the number of successfully converted files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_types = scene_types or {}
    converted = 0

    for filename in filenames:
        input_path = input_dir / filename
        if not input_path.exists():
            logger.warning("SKIP: %s (not found)", filename)
            continue

        # Determine output format and extension
        stem = Path(filename).stem
        if fmt == "original":
            # Keep the original file's format
            orig_ext = Path(filename).suffix.lower()
            ext_to_fmt = {
                ".jpg": "jpeg", ".jpeg": "jpeg",
                ".png": "png",
                ".tiff": "tiff", ".tif": "tiff",
                ".bmp": "png", ".webp": "png",  # no lossless BMP/WebP in PIL, fallback to PNG
            }
            file_fmt = ext_to_fmt.get(orig_ext, "png")
            out_ext = orig_ext if orig_ext in {".jpg", ".jpeg", ".png", ".tiff", ".tif"} else ".png"
        else:
            file_fmt = fmt
            ext_map = {"jpeg": ".jpg", "png": ".png", "tiff": ".tiff"}
            out_ext = ext_map.get(fmt, ".png")
        output_path = output_dir / f"{stem}{out_ext}"

        if no_overwrite and output_path.exists():
            logger.info("SKIP: %s (already exists)", filename)
            continue

        try:
            chosen_style = pick_style(scene_types.get(filename), style)
            preset = PRESETS[chosen_style]

            img = Image.open(input_path)
            bw_img = convert_bw(img, preset)

            save_kwargs: dict = {}
            if file_fmt == "jpeg":
                save_kwargs["quality"] = quality
            pil_format = {"jpeg": "JPEG", "png": "PNG", "tiff": "TIFF"}[file_fmt]
            bw_img.save(output_path, pil_format, **save_kwargs)

            logger.info("OK: %s [%s]", filename, chosen_style)
            converted += 1
        except Exception as e:
            logger.error("ERROR: %s - %s", filename, e)

    return converted
