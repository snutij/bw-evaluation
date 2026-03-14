#!/usr/bin/env python3
"""Generate self-contained HTML report from B&W scoring results."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visual config
# ---------------------------------------------------------------------------

_DIMENSION_META = [
    ("contrast",          "Contrast",   "#d4d4d4"),
    ("texture",           "Texture",    "#93c5fd"),
    ("channel_separation","Ch.Sep.",    "#fbbf24"),
    ("saturation",        "Saturation", "#86efac"),
    ("composition",       "Comp.",      "#f9a8d4"),
]

_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f0f;color:#ccc;font-family:'Courier New',monospace;padding:2rem}
header{max-width:1440px;margin:0 auto 2.5rem;border-bottom:1px solid #222;padding-bottom:1.2rem}
h1{font-size:1.1rem;font-weight:normal;letter-spacing:.2em;color:#fff;text-transform:uppercase}
.meta{font-size:.75rem;color:#444;margin-top:.4rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:1.25rem;max-width:1440px;margin:0 auto}
.card{background:#161616;border:1px solid #1f1f1f;overflow:hidden;transition:border-color .15s}
.card:hover{border-color:#333}
.thumb{width:100%;display:block;aspect-ratio:3/2;object-fit:cover}
.placeholder{width:100%;aspect-ratio:3/2;display:flex;align-items:center;justify-content:center;background:#1a1a1a;font-size:.65rem;color:#333;word-break:break-all;padding:.5rem;text-align:center}
.body{padding:.85rem}
.fname{font-size:.7rem;color:#555;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:.65rem}
.score{font-size:2rem;font-weight:bold;color:#fff;line-height:1}
.score span{font-size:.65rem;color:#444;margin-left:1px}
.bars{margin-top:.85rem;display:flex;flex-direction:column;gap:5px}
.bar-row{display:flex;align-items:center;gap:6px}
.bar-lbl{width:58px;font-size:.6rem;color:#555;flex-shrink:0}
.bar-track{flex:1;height:2px;background:#222}
.bar-fill{height:100%}
.bar-val{width:22px;font-size:.6rem;color:#444;text-align:right;flex-shrink:0}
"""


# ---------------------------------------------------------------------------
# Thumbnail helpers
# ---------------------------------------------------------------------------

def _encode_thumbnail(filepath: Path, width: int) -> str | None:
    """Resize photo and return base64 JPEG data URI, or None on failure."""
    img = cv2.imread(str(filepath))
    if img is None:
        return None
    h, w = img.shape[:2]
    new_h = max(1, int(h * width / w))
    thumb = cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _photo_card(result: dict[str, Any], photos_dir: Path, thumb_width: int) -> str:
    filename = result["filename"]
    score = result.get("score", 0)
    breakdown = result.get("breakdown", {})

    # Thumbnail or placeholder
    data_uri = _encode_thumbnail(photos_dir / filename, thumb_width)
    if data_uri:
        media = f'<img class="thumb" src="{data_uri}" alt="{filename}" loading="lazy">'
    else:
        logger.warning("Photo not found, using placeholder: %s", filename)
        media = f'<div class="placeholder">{filename}<br>not found</div>'

    # Dimension bars
    bars = ""
    for key, label, color in _DIMENSION_META:
        val = breakdown.get(key, 0)
        bars += (
            f'<div class="bar-row">'
            f'<span class="bar-lbl">{label}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{val}%;background:{color}"></div></div>'
            f'<span class="bar-val">{val}</span>'
            f"</div>"
        )

    return (
        f'<div class="card">'
        f"{media}"
        f'<div class="body">'
        f'<div class="fname">{filename}</div>'
        f'<div class="score">{score}<span>/100</span></div>'
        f'<div class="bars">{bars}</div>'
        f"</div>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_html_report(
    results: list[dict[str, Any]],
    photos_dir: Path,
    output_path: Path,
    thumbnail_width: int = 300,
    max_photos: int | None = None,
    quiet: bool = False,
) -> int:
    """
    Generate a self-contained HTML report from scoring results.

    Returns the number of photo cards written.
    """
    sorted_results = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
    if max_photos is not None:
        sorted_results = sorted_results[:max_photos]

    cards = "".join(
        _photo_card(r, photos_dir, thumbnail_width)
        for r in tqdm(sorted_results, desc="Building report", unit="photo", disable=quiet)
    )

    n = len(sorted_results)
    total = len(results)
    meta_str = f"{n} of {total} photos" if max_photos and n < total else f"{n} photos"

    html = (
        "<!DOCTYPE html>"
        '<html lang="en">'
        "<head>"
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>B&amp;W Report</title>"
        f"<style>{_CSS}</style>"
        "</head>"
        "<body>"
        "<header>"
        "<h1>B&amp;W Evaluation Report</h1>"
        f'<p class="meta">{meta_str} &middot; sorted by score &darr;</p>'
        "</header>"
        f'<div class="grid">{cards}</div>'
        "</body>"
        "</html>"
    )

    output_path.write_text(html, encoding="utf-8")
    return n
