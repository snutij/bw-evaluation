"""B&W Potential Evaluation Script.

Scores photos (0-100) based on their potential for a successful black & white filter.
Deterministic and statistical analysis only - no AI/ML for scoring.
"""

from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np
from tqdm import tqdm

from bw_evaluation.config import ScoringConfig

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ── Scoring algorithm constants ───────────────────────────────────────────────
_MIDTONE_THRESHOLD: int = 128  # pixel value separating light from dark zones
_HIGHLIGHT_THRESHOLD: int = 200  # pixel value considered a bright highlight
# highlights above this fraction of pixels penalise the composition score
_HIGHLIGHT_OVEREXPOSED_RATIO: float = 0.3


class ScoreBreakdown(TypedDict):
    """Per-dimension scores for a single photo."""

    contrast: int
    texture: int
    saturation: int
    composition: int
    channel_separation: int


class PhotoResult(TypedDict):
    """Full scoring result for a single photo."""

    filename: str
    score: int
    breakdown: ScoreBreakdown
    details: dict[str, Any]


def analyze_tonal_contrast(
    gray: np.ndarray,
    cfg: ScoringConfig | None = None,
) -> tuple[int, dict[str, Any]]:
    """Analyze tonal contrast: histogram distribution, std dev, blacks/whites presence.

    Returns score (0-100) and details dict.
    """
    cfg = cfg or ScoringConfig()

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_normalized = hist / hist.sum()

    # Luminance standard deviation (higher = more contrast)
    std_dev = np.std(gray)
    std_score = min(100, (std_dev / 80) * 100)

    # Presence of deep blacks (0-15) and pure whites (240-255)
    black_ratio = hist_normalized[:16].sum()
    white_ratio = hist_normalized[240:].sum()
    extremes_score = min(100, (black_ratio + white_ratio) * 500)

    # Dynamic range (difference between 5th and 95th percentile)
    p5 = np.percentile(gray, 5)
    p95 = np.percentile(gray, 95)
    dynamic_range = p95 - p5
    range_score = min(100, (dynamic_range / 200) * 100)

    # Light/dark zones ratio (ideal is balanced)
    dark_pixels = (gray < _MIDTONE_THRESHOLD).sum()
    light_pixels = (gray >= _MIDTONE_THRESHOLD).sum()
    balance = min(dark_pixels, light_pixels) / max(dark_pixels, light_pixels)
    balance_score = balance * 100

    # Combined score
    weighted = (
        cfg.contrast_std_weight * std_score
        + cfg.contrast_extremes_weight * extremes_score
        + cfg.contrast_range_weight * range_score
        + cfg.contrast_balance_weight * balance_score
    )
    score = round(weighted)

    details: dict[str, Any] = {
        "std_dev": round(float(std_dev), 2),
        "black_ratio": round(float(black_ratio), 4),
        "white_ratio": round(float(white_ratio), 4),
        "dynamic_range": round(float(dynamic_range), 2),
        "light_dark_balance": round(float(balance), 3),
    }

    return score, details


def analyze_texture_details(
    gray: np.ndarray,
    cfg: ScoringConfig | None = None,
) -> tuple[int, dict[str, Any]]:
    """Analyze texture and details: edge detection, local variance, sharpness.

    Returns score (0-100) and details dict.
    """
    cfg = cfg or ScoringConfig()

    # Edge detection with Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_density = (sobel_mag > cfg.sobel_threshold).sum() / gray.size
    edge_score = min(100, edge_density * 500)

    # Canny edge detection for finer details
    canny = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
    canny_density = canny.sum() / (255 * gray.size)
    canny_score = min(100, canny_density * 400)

    # Local variance (textural richness)
    kernel_size = 5
    local_mean = cv2.blur(gray.astype(np.float64), (kernel_size, kernel_size))
    local_sq_mean = cv2.blur((gray.astype(np.float64)) ** 2, (kernel_size, kernel_size))
    local_var = local_sq_mean - local_mean**2
    avg_local_var = np.mean(local_var)
    texture_score = min(100, (avg_local_var / 1000) * 100)

    # Sharpness via Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    sharpness_score = min(100, (sharpness / 500) * 100)

    # Combined score
    weighted = (
        cfg.texture_edge_weight * edge_score
        + cfg.texture_canny_weight * canny_score
        + cfg.texture_variance_weight * texture_score
        + cfg.texture_sharpness_weight * sharpness_score
    )
    score = round(weighted)

    details: dict[str, Any] = {
        "edge_density": round(float(edge_density), 4),
        "canny_density": round(float(canny_density), 4),
        "avg_local_variance": round(float(avg_local_var), 2),
        "laplacian_variance": round(float(sharpness), 2),
    }

    return score, details


def analyze_colorimetry(
    img_bgr: np.ndarray,
    _cfg: ScoringConfig | None = None,
) -> tuple[int, dict[str, Any]]:
    """Analyze colorimetry: inverse mean saturation.

    Low saturation = better for B&W (higher score).
    Returns score (0-100) and details dict.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    avg_sat = float(np.mean(img_hsv[:, :, 1]))
    score = max(0, min(100, round(100 - (avg_sat / 255) * 100)))
    details: dict[str, Any] = {"avg_saturation": round(avg_sat, 2)}
    return score, details


def analyze_tonal_composition(
    gray: np.ndarray,
    cfg: ScoringConfig | None = None,
) -> tuple[int, dict[str, Any]]:
    """Analyze tonal composition: plane separation and highlight distribution.

    Returns score (0-100) and details dict.
    """
    cfg = cfg or ScoringConfig()
    h, w = gray.shape

    # Plane separation by luminosity (divide into regions)
    grid_h, grid_w = h // 3, w // 3
    region_means = []
    for i in range(3):
        for j in range(3):
            region = gray[i * grid_h : (i + 1) * grid_h, j * grid_w : (j + 1) * grid_w]
            region_means.append(np.mean(region))

    region_std = np.std(region_means)
    separation_score = min(100, (region_std / 40) * 100)

    # Highlight detection (bright focused areas)
    highlights = (gray > _HIGHLIGHT_THRESHOLD).sum() / gray.size
    highlight_score = (
        min(100, highlights * 300)
        if highlights < _HIGHLIGHT_OVEREXPOSED_RATIO
        else max(0, 100 - (highlights - _HIGHLIGHT_OVEREXPOSED_RATIO) * 200)
    )

    # Combined score
    weighted = (
        cfg.composition_separation_weight * separation_score
        + cfg.composition_highlight_weight * highlight_score
    )
    score = round(weighted)

    details: dict[str, Any] = {
        "region_luminosity_std": round(float(region_std), 2),
        "highlight_ratio": round(float(highlights), 4),
    }

    return score, details


def analyze_channel_separation(
    img_bgr: np.ndarray,
    cfg: ScoringConfig | None = None,
) -> tuple[int, dict[str, Any]]:
    """Analyze RGB channel separation: mean per-pixel std dev across B, G, R channels.

    High separation = more creative potential for B&W channel mixing.
    Returns score (0-100) and details dict.
    """
    cfg = cfg or ScoringConfig()
    per_pixel_std = np.std(img_bgr.astype(np.float64), axis=2)
    mean_std = float(np.mean(per_pixel_std))
    score = min(100, round((mean_std / cfg.channel_sep_ceiling) * 100))
    details: dict[str, Any] = {"mean_channel_std": round(mean_std, 2)}
    return score, details


def score_photo(
    filepath: Path,
    cfg: ScoringConfig | None = None,
) -> PhotoResult:
    """Score a single photo for B&W potential.

    Returns PhotoResult with score, breakdown and details.
    """
    cfg = cfg or ScoringConfig()

    img_bgr = cv2.imread(str(filepath))
    if img_bgr is None:
        msg = f"Could not read image: {filepath}"
        raise ValueError(msg)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Run all analyses
    contrast_score, contrast_details = analyze_tonal_contrast(gray, cfg)
    texture_score, texture_details = analyze_texture_details(gray, cfg)
    saturation_score, saturation_details = analyze_colorimetry(img_bgr, cfg)
    composition_score, composition_details = analyze_tonal_composition(gray, cfg)
    channel_sep_score, channel_sep_details = analyze_channel_separation(img_bgr, cfg)

    breakdown = ScoreBreakdown(
        contrast=contrast_score,
        texture=texture_score,
        saturation=saturation_score,
        composition=composition_score,
        channel_separation=channel_sep_score,
    )
    all_details: dict[str, Any] = {
        "contrast": contrast_details,
        "texture": texture_details,
        "saturation": saturation_details,
        "composition": composition_details,
        "channel_separation": channel_sep_details,
    }

    # Weighted final score
    final_score = max(
        0,
        min(
            100,
            round(
                cfg.weight_contrast * contrast_score
                + cfg.weight_texture * texture_score
                + cfg.weight_saturation * saturation_score
                + cfg.weight_composition * composition_score
                + cfg.weight_channel_separation * channel_sep_score
            ),
        ),
    )

    return PhotoResult(
        filename=filepath.name,
        score=final_score,
        breakdown=breakdown,
        details=all_details,
    )


def score_photos(
    photos: list[Path],
    cfg: ScoringConfig | None = None,
    workers: int = 1,
    *,
    quiet: bool = False,
) -> list[PhotoResult]:
    """Score multiple photos, optionally in parallel."""
    cfg = cfg or ScoringConfig()

    def _score(p: Path) -> PhotoResult | None:
        try:
            return score_photo(p, cfg)
        except Exception:
            logger.exception("Failed to score %s", p.name)
            return None

    results: list[PhotoResult] = []

    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_score, p): p for p in photos}
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Scoring",
                unit="photo",
                disable=quiet,
            ):
                result = future.result()
                if result:
                    results.append(result)
    else:
        for photo in tqdm(photos, desc="Scoring", unit="photo", disable=quiet):
            result = _score(photo)
            if result:
                results.append(result)

    return results
