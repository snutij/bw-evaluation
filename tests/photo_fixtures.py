"""File-based fixtures: real image files written to disk, decoded by the real decoder.

Separate from `fixtures.py`, which stays a pure array module importing nothing
from the package. Everything here touches the filesystem and Pillow, so it is
kept apart rather than slowing the array fixtures down with image-library
imports.

The wide-gamut profile is built by patching the colorant tags of Pillow's
built-in sRGB profile rather than shipping a binary. littlecms cannot
synthesise an AdobeRGB profile and no ICC file is available offline, but an ICC
profile is a tag table, so overwriting the three XYZ colorant tags in place
leaves a structurally valid profile with different primaries. That is enough
for a genuine test: a saturated red round-trips 220 -> 254 through it, so a
conversion that silently did nothing could not pass.
"""

from __future__ import annotations

import struct
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from . import fixtures

#: AdobeRGB (1998) colorants, D50-adapted, as they appear in an ICC profile.
ADOBE_RGB_COLORANTS: dict[bytes, tuple[float, float, float]] = {
    b"rXYZ": (0.60974, 0.31111, 0.01947),
    b"gXYZ": (0.20528, 0.62567, 0.06087),
    b"bXYZ": (0.14919, 0.06322, 0.74457),
}


def srgb_profile_bytes() -> bytes:
    """Pillow's built-in sRGB profile, serialised."""
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def wide_gamut_profile_bytes() -> bytes:
    """An sRGB profile with its colorants replaced by AdobeRGB's."""
    data = bytearray(srgb_profile_bytes())
    tag_count = struct.unpack(">I", data[128:132])[0]
    for index in range(tag_count):
        entry = 132 + index * 12
        signature = bytes(data[entry : entry + 4])
        offset, _size = struct.unpack(">II", data[entry + 4 : entry + 12])
        if signature in ADOBE_RGB_COLORANTS:
            assert bytes(data[offset : offset + 4]) == b"XYZ "
            for axis, value in enumerate(ADOBE_RGB_COLORANTS[signature]):
                struct.pack_into(
                    ">i", data, offset + 8 + 4 * axis, round(value * 65536)
                )
    return bytes(data)


def corrupt_profile_bytes() -> bytes:
    """Bytes that claim to be an ICC profile and are not."""
    return b"\x00\x00\x02\x4c" + b"definitely not a colour profile" * 8


def sample_exif(*, orientation: int = 1) -> bytes:
    """A plausible camera EXIF block, serialised."""
    exif = Image.Exif()
    exif[0x010F] = "Fujifilm"
    exif[0x0110] = "X-T5"
    exif[0x0112] = orientation
    exif.get_ifd(0x8769).update(
        {
            0x829A: (1, 250),  # ExposureTime
            0x829D: (28, 10),  # FNumber
            0x8827: 1600,  # ISOSpeedRatings
            0x920A: (35, 1),  # FocalLength
            0x9003: "2026:09:23 14:05:00",  # DateTimeOriginal
            0xA434: "XF 35mm F1.4 R",  # LensModel
        }
    )
    return exif.tobytes()


def as_bytes(image: np.ndarray) -> np.ndarray:
    """Quantise a [0, 1] float image to 8-bit, the way an exporter would."""
    return np.clip(np.rint(np.asarray(image) * 255.0), 0, 255).astype(np.uint8)


def write_png(path: Path, image: np.ndarray, **save: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(as_bytes(image)).save(path, "PNG", **save)
    return path


def write_jpeg(path: Path, image: np.ndarray, **save: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(as_bytes(image)).save(path, "JPEG", quality=95, **save)
    return path


def write_catalogue(folder: Path) -> dict[str, Path]:
    """The nine reference images, written out as real PNG files."""
    folder.mkdir(parents=True, exist_ok=True)
    return {
        name: write_png(folder / f"{name}.png", image)
        for name, image in fixtures.catalogue().items()
    }


def solid(colour: tuple[float, float, float], size: int = 32) -> np.ndarray:
    return np.tile(np.asarray(colour, dtype=np.float64), (size, size, 1))


def profile_description(data: bytes) -> str:
    return ImageCms.getProfileDescription(
        ImageCms.getOpenProfile(BytesIO(data))
    ).strip()
