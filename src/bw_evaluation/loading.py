"""Reading photographs off disk in the form the rest of the package contracts for.

Everything above this module assumes float64 RGB in [0, 1], colorimetrically
sRGB, the right way up. This module is the only place responsible for making
that true, and three of its jobs are easy to get wrong silently.

*Colour management.* A large share of exported JPEGs carry an AdobeRGB or
Display P3 profile. Reading those as sRGB inflates every chroma measurement in
the package, which would bias a whole ranking toward whoever exported
wide-gamut. Embedded profiles are converted with littlecms; an absent profile
is assumed sRGB, and the assumption is recorded rather than left implicit.

*Bit depth.* A 16-bit file read naively lands in [0, 65535] and every
Hasler-Susstrunk anchor becomes meaningless. The divisor tracks the dtype.

*Orientation.* A portrait frame stored as landscape with an EXIF rotation flag
is the ordinary output of most cameras. Measuring it unrotated measures a
different photograph.

Known gaps, stated rather than hidden
-------------------------------------
No camera raw. Decoding CR2, NEF, ARW or DNG needs a raw library that is not
installed, so raw files are refused with that reason named. This matters: a raw
file is where the colour decisions have not been made yet, and the honest
answer is that this tool measures renderings, not negatives.

No HEIF/HEIC. Pillow here is built without it.

Both lists are explicit, and :data:`UNSUPPORTED_RAW_SUFFIXES` exists so that
the refusal can explain itself instead of saying "unsupported suffix" at a
photographer holding a folder of raws.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageCms, ImageOps
from PIL.ExifTags import Base as ExifBase

from .measurement import WORKING_LONG_EDGE

Array: TypeAlias = NDArray[np.float64]

#: Formats this build of Pillow can decode and this package can interpret.
SUPPORTED_SUFFIXES: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".avif",
    ".jp2",
)

#: Formats a photographer will plausibly point this at and which are refused
#: with a specific reason rather than a generic one.
UNSUPPORTED_RAW_SUFFIXES: tuple[str, ...] = (
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".dng",
    ".raf",
    ".orf",
    ".rw2",
    ".pef",
    ".srw",
    ".heic",
    ".heif",
)

#: Long edge a file is decoded down to. Twice the measurement working size, so
#: that the measurement's own resize always governs a texture statistic and
#: this cap never quietly becomes the thing being measured. Its purpose is
#: only to keep a folder of 45-megapixel frames out of memory at full size.
MAXIMUM_DECODE_LONG_EDGE = 2 * WORKING_LONG_EDGE

_ASSUMED_SRGB: Final[str] = "sRGB (assumed, no embedded profile)"
_RECOVERED_SRGB: Final[str] = "sRGB (assumed, embedded profile unreadable)"


@dataclass(frozen=True)
class PhotographMetadata:
    """What the file says about itself. Never inferred, never guessed."""

    image_id: str
    path: str
    width: int
    height: int
    displayed_width: int
    displayed_height: int
    colour_profile: str
    camera: str | None = None
    lens: str | None = None
    iso: int | None = None
    shutter: str | None = None
    aperture: float | None = None
    focal_length: float | None = None
    captured_at: str | None = None


@dataclass(frozen=True)
class LoadedPhotograph:
    """Pixels in the package's contracted form, plus what the file said."""

    metadata: PhotographMetadata
    rgb: Array


def _validated_path(path: Path | str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"no such file: {resolved}")
    if not resolved.is_file():
        raise IsADirectoryError(f"{resolved} is not a file")
    suffix = resolved.suffix.lower()
    if suffix in UNSUPPORTED_RAW_SUFFIXES:
        raise ValueError(
            f"{resolved.name} is a camera raw or HEIF file, and no raw decoder is installed. "
            "This package measures renderings rather than negatives: export to JPEG or TIFF "
            f"first. Readable formats are {', '.join(SUPPORTED_SUFFIXES)}."
        )
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"{resolved.name} has suffix {suffix or '(none)'}, which is not a readable image. "
            f"Supported formats are {', '.join(SUPPORTED_SUFFIXES)}."
        )
    return resolved


def _rational(value: object) -> float | None:
    """EXIF numbers arrive as rationals, floats or tuples depending on the tag."""
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        numerator, denominator = value
        return float(numerator) / float(denominator) if denominator else None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _shutter_text(value: object) -> str | None:
    """Render an exposure time the way a photographer reads it."""
    seconds = _rational(value)
    if seconds is None or seconds <= 0.0:
        return None
    if seconds >= 1.0:
        return f"{seconds:g}s"
    fraction = Fraction(seconds).limit_denominator(8000)
    return f"{fraction.numerator}/{fraction.denominator}"


def _clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("\x00").strip()
    return text or None


def _exif_fields(image: Image.Image) -> dict[str, object]:
    """Pull the settings a photographer would group by, tolerating absence."""
    try:
        exif = image.getexif()
    except Exception:  # noqa: BLE001 - Pillow raises assorted types here
        return {}
    if not exif:
        return {}
    detail = exif.get_ifd(ExifBase.ExifOffset.value)
    make = _clean(exif.get(ExifBase.Make.value))
    model = _clean(exif.get(ExifBase.Model.value))
    camera = " ".join(part for part in (make, model) if part) or None
    # 0x8827 is ISOSpeedRatings in EXIF 2.2 and PhotographicSensitivity in 2.3;
    # same tag, and Pillow only names the older one. 0x8833 is the separate
    # ISOSpeed tag some bodies write instead. Both are read by number so this
    # does not depend on which spelling a given Pillow release exposes.
    iso = detail.get(ExifBase.ISOSpeedRatings.value) or detail.get(0x8833)
    if isinstance(iso, (tuple, list)):
        iso = iso[0] if iso else None
    return {
        "camera": camera,
        "lens": _clean(detail.get(ExifBase.LensModel.value)),
        "iso": int(iso) if isinstance(iso, (int, float)) else None,
        "shutter": _shutter_text(detail.get(ExifBase.ExposureTime.value)),
        "aperture": _rational(detail.get(ExifBase.FNumber.value)),
        "focal_length": _rational(detail.get(ExifBase.FocalLength.value)),
        "captured_at": _clean(detail.get(ExifBase.DateTimeOriginal.value)),
    }


def _to_srgb(image: Image.Image) -> tuple[Image.Image, str]:
    """Convert to sRGB using the embedded profile, or say what was assumed."""
    raw_profile = image.info.get("icc_profile")
    if not raw_profile:
        return image, _ASSUMED_SRGB
    try:
        source = ImageCms.getOpenProfile(BytesIO(raw_profile))
        description = (
            ImageCms.getProfileDescription(source).strip() or "embedded profile"
        )
        converted = ImageCms.profileToProfile(
            image, source, ImageCms.createProfile("sRGB"), outputMode="RGB"
        )
    except Exception:  # noqa: BLE001 - recover from malformed embedded profiles
        # A damaged ICC block is common in edited exports and is not a reason
        # to drop the photograph. Fall back to sRGB and record that it happened.
        return image, _RECOVERED_SRGB
    if converted is None:
        return image, _RECOVERED_SRGB
    return converted, f"{description} -> sRGB"


def _to_unit_array(image: Image.Image) -> Array:
    """Pixels as float64 RGB in [0, 1], with the divisor tracking the bit depth."""
    if image.mode in ("I;16", "I;16B", "I;16L", "I"):
        values = np.asarray(image).astype(np.float64)
        peak = 65535.0 if values.max() > 255.0 else 255.0
        grey = values / peak
        return np.repeat(grey[:, :, None], 3, axis=2)
    if image.mode == "F":
        values = np.asarray(image, dtype=np.float64)
        return np.clip(np.repeat(values[:, :, None], 3, axis=2), 0.0, 1.0)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.float64) / 255.0


def read_metadata(path: Path | str) -> PhotographMetadata:
    """Read what a file says about itself, without decoding it fully."""
    resolved = _validated_path(path)
    with Image.open(resolved) as handle:
        stored_width, stored_height = handle.size
        upright = ImageOps.exif_transpose(handle)
        displayed_width, displayed_height = upright.size if upright else handle.size
        fields = _exif_fields(handle)
        raw_profile = handle.info.get("icc_profile")
        if raw_profile:
            try:
                description = ImageCms.getProfileDescription(
                    ImageCms.getOpenProfile(BytesIO(raw_profile))
                ).strip()
                profile = f"{description or 'embedded profile'} -> sRGB"
            except Exception:  # noqa: BLE001 - recover from malformed embedded profiles
                profile = _RECOVERED_SRGB
        else:
            profile = _ASSUMED_SRGB
    return PhotographMetadata(
        image_id=resolved.stem,
        path=str(resolved),
        width=stored_width,
        height=stored_height,
        displayed_width=displayed_width,
        displayed_height=displayed_height,
        colour_profile=profile,
        **fields,  # type: ignore[arg-type]
    )


def load_photograph(path: Path | str) -> LoadedPhotograph:
    """Decode one photograph into the form `measure_bw_candidacy` requires."""
    resolved = _validated_path(path)
    with Image.open(resolved) as handle:
        handle.load()
        stored_width, stored_height = handle.size
        fields = _exif_fields(handle)
        upright = ImageOps.exif_transpose(handle) or handle
        # The profile has to be carried across the transpose by hand: Pillow
        # keeps `info` on the original object, and losing it here would make
        # every rotated wide-gamut frame silently read as sRGB.
        if "icc_profile" in handle.info and "icc_profile" not in upright.info:
            upright.info["icc_profile"] = handle.info["icc_profile"]
        if upright.mode in ("RGB", "RGBA", "L", "LA", "P", "CMYK"):
            managed, profile = _to_srgb(upright.convert("RGB"))
        else:
            managed, profile = upright, _ASSUMED_SRGB
        if max(managed.size) > MAXIMUM_DECODE_LONG_EDGE:
            managed = managed.copy()
            managed.thumbnail(
                (MAXIMUM_DECODE_LONG_EDGE, MAXIMUM_DECODE_LONG_EDGE),
                Image.Resampling.LANCZOS,
            )
        rgb = _to_unit_array(managed)
    height, width = rgb.shape[:2]
    metadata = PhotographMetadata(
        image_id=resolved.stem,
        path=str(resolved),
        width=stored_width,
        height=stored_height,
        displayed_width=width,
        displayed_height=height,
        colour_profile=profile,
        **fields,  # type: ignore[arg-type]
    )
    return LoadedPhotograph(metadata=metadata, rgb=rgb)


def discover_photographs(
    folder: Path | str, *, recursive: bool = False
) -> tuple[Path, ...]:
    """Every readable photograph in a folder, in a stable order.

    Sorted rather than left in filesystem order, because every number this
    package produces is a position within the batch, and a batch that changed
    composition order between runs would not be reproducible.

    Non-recursive by default: an exports or rejects subfolder swept in
    silently would change the batch the photographer thought they asked for.
    """
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"no such folder: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is a file, not a folder")
    candidates: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    return tuple(
        sorted(
            (
                p
                for p in candidates
                if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
            ),
            key=lambda p: p.as_posix(),
        )
    )


def unique_image_ids(paths: Sequence[Path], *, root: Path | str) -> dict[Path, str]:
    """Short, stable, collision-free names for a batch.

    File stems where they are distinct, because `DSCF1234` is what the
    photographer recognises. Relative paths where they are not: recursing into
    a year of work reliably turns up the same stem in two shoot folders, and a
    batch keyed by colliding ids would silently drop one of the photographs.
    """
    base = Path(root)
    stems: dict[str, int] = {}
    for path in paths:
        stems[path.stem] = stems.get(path.stem, 0) + 1

    def relative(path: Path) -> str:
        try:
            return path.relative_to(base).with_suffix("").as_posix()
        except ValueError:
            return path.with_suffix("").as_posix()

    return {
        path: (path.stem if stems[path.stem] == 1 else relative(path)) for path in paths
    }
