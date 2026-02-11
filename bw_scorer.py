#!/usr/bin/env python3
"""
B&W Potential Evaluation Script

Scores photos (0-100) based on their potential for a successful black & white filter.
Deterministic and statistical analysis only - no AI/ML for scoring.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import sys
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

from config import ScoringConfig

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class ScoreBreakdown(TypedDict):
    contrast: int
    texture: int
    saturation: int
    composition: int
    metadata: int


class PhotoResult(TypedDict):
    filename: str
    score: int
    breakdown: ScoreBreakdown
    details: dict[str, Any]


def analyze_tonal_contrast(
    gray: np.ndarray, cfg: ScoringConfig | None = None,
) -> tuple[int, dict[str, Any]]:
    """
    Analyze tonal contrast: histogram distribution, std dev, blacks/whites presence.
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
    dark_pixels = (gray < 128).sum()
    light_pixels = (gray >= 128).sum()
    balance = min(dark_pixels, light_pixels) / max(dark_pixels, light_pixels)
    balance_score = balance * 100

    # Combined score
    weighted = (
        cfg.contrast_std_weight * std_score
        + cfg.contrast_extremes_weight * extremes_score
        + cfg.contrast_range_weight * range_score
        + cfg.contrast_balance_weight * balance_score
    )
    score = int(round(weighted))

    details = {
        "std_dev": round(float(std_dev), 2),
        "black_ratio": round(float(black_ratio), 4),
        "white_ratio": round(float(white_ratio), 4),
        "dynamic_range": round(float(dynamic_range), 2),
        "light_dark_balance": round(float(balance), 3),
    }

    return score, details


def analyze_texture_details(
    gray: np.ndarray, cfg: ScoringConfig | None = None,
) -> tuple[int, dict[str, Any]]:
    """
    Analyze texture and details: edge detection, local variance, sharpness.
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
    score = int(round(weighted))

    details = {
        "edge_density": round(float(edge_density), 4),
        "canny_density": round(float(canny_density), 4),
        "avg_local_variance": round(float(avg_local_var), 2),
        "laplacian_variance": round(float(sharpness), 2),
    }

    return score, details


def analyze_colorimetry(
    img_bgr: np.ndarray, cfg: ScoringConfig | None = None,
) -> tuple[int, dict[str, Any]]:
    """
    Analyze colorimetry: saturation distribution, quasi-monochrome detection, gamut.
    Returns score (0-100) and details dict.
    Low saturation = better for B&W (higher score).
    """
    cfg = cfg or ScoringConfig()

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation = img_hsv[:, :, 1]

    # Average saturation (lower is better for B&W)
    avg_sat = float(np.mean(saturation))
    sat_score = max(0.0, 100 - (avg_sat / 255) * 100)

    # High saturation penalty (only for very saturated pixels)
    high_sat_ratio = float((saturation > cfg.high_sat_threshold).sum()) / saturation.size
    if avg_sat > cfg.high_sat_avg_trigger and high_sat_ratio > cfg.high_sat_ratio_trigger:
        high_sat_penalty = high_sat_ratio * cfg.high_sat_strong_penalty_factor
    else:
        high_sat_penalty = high_sat_ratio * cfg.high_sat_weak_penalty_factor

    # Saturation standard deviation (lower variance = more uniform = bonus)
    sat_std = float(np.std(saturation))
    uniformity_bonus = max(0.0, 20 - (sat_std / 50) * 20)

    # Quasi-monochrome detection (very low saturation throughout)
    low_sat_ratio = float((saturation < cfg.low_sat_threshold).sum()) / saturation.size
    mono_bonus = low_sat_ratio * 35

    # Color dominance vs varied palette
    hue = img_hsv[:, :, 0]
    meaningful_mask = saturation > cfg.low_sat_threshold
    if meaningful_mask.sum() > 0:
        meaningful_hues = hue[meaningful_mask]
        hue_std = float(np.std(meaningful_hues))
        palette_penalty = min(20.0, (hue_std / 50) * 20)
    else:
        palette_penalty = 0.0

    # Used gamut (spread of saturation values) — penalizes mixed saturated/desaturated
    sat_p10 = float(np.percentile(saturation, 10))
    sat_p90 = float(np.percentile(saturation, 90))
    gamut_spread = sat_p90 - sat_p10
    gamut_penalty = min(
        cfg.gamut_penalty_max,
        (gamut_spread / cfg.gamut_spread_divisor) * cfg.gamut_penalty_max,
    )

    # Combined score
    combined = (
        float(sat_score)
        + float(uniformity_bonus)
        + float(mono_bonus)
        - float(palette_penalty)
        - float(high_sat_penalty)
        - float(gamut_penalty)
    )
    score = max(0, min(100, int(round(combined))))

    details = {
        "avg_saturation": round(float(avg_sat), 2),
        "saturation_std": round(float(sat_std), 2),
        "low_saturation_ratio": round(float(low_sat_ratio), 4),
        "high_saturation_ratio": round(float(high_sat_ratio), 4),
        "gamut_spread": round(float(gamut_spread), 2),
    }

    return score, details


def analyze_tonal_composition(
    gray: np.ndarray, cfg: ScoringConfig | None = None,
) -> tuple[int, dict[str, Any]]:
    """
    Analyze tonal composition: plane separation, gray gradients, highlight zones,
    rule-of-thirds energy.
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

    # Natural gray gradients (smooth transitions)
    grad_x = np.diff(gray.astype(np.float64), axis=1)
    grad_y = np.diff(gray.astype(np.float64), axis=0)
    grad_smoothness = 1 / (1 + np.std(grad_x) / 10 + np.std(grad_y) / 10)
    gradient_score = grad_smoothness * 100

    # Center-weighted luminosity
    center_region = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    center_mean = np.mean(center_region)
    center_exposure = 100 - abs(center_mean - 127) * 0.7
    center_score = max(0, center_exposure)

    # Rule-of-thirds: measure edge energy along 1/3 and 2/3 lines
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    band = max(1, min(h, w) // 30)  # ~3% of smallest dimension
    h3, h23 = h // 3, 2 * h // 3
    w3, w23 = w // 3, 2 * w // 3

    thirds_energy = float(np.mean([
        np.mean(edge_mag[max(0, h3 - band) : h3 + band, :]),
        np.mean(edge_mag[max(0, h23 - band) : h23 + band, :]),
        np.mean(edge_mag[:, max(0, w3 - band) : w3 + band]),
        np.mean(edge_mag[:, max(0, w23 - band) : w23 + band]),
    ]))
    thirds_score = min(100, (thirds_energy / 40) * 100)

    # Highlight detection (bright focused areas)
    highlights = (gray > 200).sum() / gray.size
    highlight_score = (
        min(100, highlights * 300)
        if highlights < 0.3
        else max(0, 100 - (highlights - 0.3) * 200)
    )

    # Combined score
    weighted = (
        cfg.composition_separation_weight * separation_score
        + cfg.composition_gradient_weight * gradient_score
        + cfg.composition_center_weight * center_score
        + cfg.composition_thirds_weight * thirds_score
        + cfg.composition_highlight_weight * highlight_score
    )
    score = int(round(weighted))

    details = {
        "region_luminosity_std": round(float(region_std), 2),
        "gradient_smoothness": round(float(grad_smoothness), 4),
        "center_mean_luminosity": round(float(center_mean), 2),
        "thirds_energy": round(float(thirds_energy), 2),
        "highlight_ratio": round(float(highlights), 4),
    }

    return score, details


def _estimate_noise_level(gray: np.ndarray) -> float:
    """Estimate sensor noise from image content (proxy for ISO when EXIF missing)."""
    # Median absolute deviation of Laplacian — robust noise estimator
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sigma = float(np.median(np.abs(laplacian)) * 1.4826)
    return sigma


def analyze_metadata(
    filepath: Path,
    gray: np.ndarray | None = None,
    cfg: ScoringConfig | None = None,
) -> tuple[int, dict[str, Any]]:
    """
    Analyze EXIF metadata: ISO, focal length, light conditions, depth of field.
    When EXIF is missing, estimates noise from pixel data instead of returning flat 50.
    Returns score (0-100) and details dict.
    """
    cfg = cfg or ScoringConfig()
    details: dict[str, Any] = {}
    has_exif = False

    try:
        with Image.open(filepath) as img:
            exif_data = img.getexif()
            if exif_data is None:
                has_exif = False
            else:
                exif = {}
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif[tag] = value

                iso_score = 50
                focal_score = 50
                light_score = 50
                dof_score = 50

                # High ISO (grain = B&W aesthetic bonus)
                if "ISOSpeedRatings" in exif:
                    has_exif = True
                    iso = exif["ISOSpeedRatings"]
                    if isinstance(iso, tuple):
                        iso = iso[0]
                    details["iso"] = iso
                    if iso >= 1600:
                        iso_score = 80
                    elif iso >= 800:
                        iso_score = 65
                    elif iso >= 400:
                        iso_score = 55
                    else:
                        iso_score = 45

                # Focal length
                if "FocalLength" in exif:
                    has_exif = True
                    focal = exif["FocalLength"]
                    if hasattr(focal, "numerator"):
                        focal = focal.numerator / focal.denominator
                    details["focal_length"] = round(float(focal), 1)
                    if 50 <= focal <= 135:
                        focal_score = 70  # Portrait range
                    elif focal < 35:
                        focal_score = 65  # Wide angle
                    else:
                        focal_score = 50

                # Exposure time for light condition hints
                if "ExposureTime" in exif:
                    has_exif = True
                    exp = exif["ExposureTime"]
                    if hasattr(exp, "numerator"):
                        exp_val = exp.numerator / exp.denominator
                    else:
                        exp_val = float(exp)
                    details["exposure_time"] = round(exp_val, 6)
                    if exp_val >= 1:
                        light_score = 70

                # F-number for depth of field estimation
                if "FNumber" in exif:
                    has_exif = True
                    fnum = exif["FNumber"]
                    if hasattr(fnum, "numerator"):
                        fnum = fnum.numerator / fnum.denominator
                    details["f_number"] = round(float(fnum), 1)
                    if fnum <= 2.8:
                        dof_score = 75  # Shallow DOF, subject isolation
                    elif fnum >= 11:
                        dof_score = 70  # Deep DOF, landscapes
                    else:
                        dof_score = 50

                if has_exif:
                    details["available"] = True
                    weighted = (
                        cfg.metadata_iso_weight * iso_score
                        + cfg.metadata_focal_weight * focal_score
                        + cfg.metadata_light_weight * light_score
                        + cfg.metadata_dof_weight * dof_score
                    )
                    return int(round(weighted)), details

    except Exception as e:
        details["error"] = str(e)

    # No useful EXIF — estimate from pixel noise
    details["available"] = False
    if gray is not None:
        noise = _estimate_noise_level(gray)
        details["estimated_noise"] = round(noise, 2)
        # High noise → likely high ISO → slight B&W bonus
        if noise > 15:
            return 60, details
        elif noise > 8:
            return 55, details
    return 50, details


def detect_scene_type(
    breakdown: ScoreBreakdown,
    details: dict[str, Any],
    cfg: ScoringConfig | None = None,
) -> tuple[str, int]:
    """
    Classify scene type from scoring breakdown and return (type, bonus).

    Rules (evaluated in order):
      portrait:      texture < 20, saturation >= 50, composition >= 50  → +8
      architecture:  texture >= 60, contrast >= 50, edge_density > 0.15 → +6
      landscape:     texture >= 50, contrast >= 60, composition >= 40   → +5
      street:        dynamic_range > 150, 30 <= texture <= 70           → +4
      generic:       fallback                                           → +0
    """
    cfg = cfg or ScoringConfig()
    texture = breakdown["texture"]
    contrast = breakdown["contrast"]
    saturation = breakdown["saturation"]
    composition = breakdown["composition"]

    edge_density = details.get("texture", {}).get("edge_density", 0)
    dynamic_range = details.get("contrast", {}).get("dynamic_range", 0)

    if texture < 20 and saturation >= 50 and composition >= 50:
        return "portrait", cfg.scene_portrait_bonus
    if texture >= 60 and contrast >= 50 and edge_density > 0.15:
        return "architecture", cfg.scene_architecture_bonus
    if texture >= 50 and contrast >= 60 and composition >= 40:
        return "landscape", cfg.scene_landscape_bonus
    if dynamic_range > 150 and 30 <= texture <= 70:
        return "street", cfg.scene_street_bonus
    return "generic", 0


def score_photo(
    filepath: Path, cfg: ScoringConfig | None = None,
) -> PhotoResult:
    """
    Score a single photo for B&W potential.
    Returns PhotoResult with score, breakdown, scene_type and details.
    """
    cfg = cfg or ScoringConfig()

    img_bgr = cv2.imread(str(filepath))
    if img_bgr is None:
        raise ValueError(f"Could not read image: {filepath}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Run all analyses
    contrast_score, contrast_details = analyze_tonal_contrast(gray, cfg)
    texture_score, texture_details = analyze_texture_details(gray, cfg)
    saturation_score, saturation_details = analyze_colorimetry(img_bgr, cfg)
    composition_score, composition_details = analyze_tonal_composition(gray, cfg)
    metadata_score, metadata_details = analyze_metadata(filepath, gray, cfg)

    breakdown = ScoreBreakdown(
        contrast=contrast_score,
        texture=texture_score,
        saturation=saturation_score,
        composition=composition_score,
        metadata=metadata_score,
    )
    all_details = {
        "contrast": contrast_details,
        "texture": texture_details,
        "saturation": saturation_details,
        "composition": composition_details,
        "metadata": metadata_details,
    }

    # Weighted final score
    final_score = int(round(
        cfg.weight_contrast * contrast_score
        + cfg.weight_texture * texture_score
        + cfg.weight_saturation * saturation_score
        + cfg.weight_composition * composition_score
        + cfg.weight_metadata * metadata_score
    ))

    # Scene detection + bonus
    scene_type, bonus = detect_scene_type(breakdown, all_details, cfg)
    final_score = min(100, final_score + bonus)

    all_details["scene_type"] = scene_type
    if bonus:
        all_details["scene_bonus"] = bonus

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
) -> list[PhotoResult]:
    """Score multiple photos, optionally in parallel."""
    cfg = cfg or ScoringConfig()

    def _score(p: Path) -> PhotoResult | None:
        try:
            return score_photo(p, cfg)
        except Exception as e:
            logger.error("Failed to score %s: %s", p.name, e)
            return None

    results: list[PhotoResult] = []

    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_score, p): p for p in photos}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
    else:
        for photo in photos:
            result = _score(photo)
            if result:
                results.append(result)

    return results
