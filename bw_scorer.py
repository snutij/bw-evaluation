#!/usr/bin/env python3
"""
B&W Potential Evaluation Script

Scores photos (0-100) based on their potential for a successful black & white filter.
Deterministic and statistical analysis only - no AI/ML for scoring.
"""

import json
import sys
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


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


def analyze_tonal_contrast(gray: np.ndarray) -> tuple[int, dict[str, Any]]:
    """
    Analyze tonal contrast: histogram distribution, std dev, blacks/whites presence.
    Returns score (0-100) and details dict.
    """
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
    total = gray.size
    balance = min(dark_pixels, light_pixels) / max(dark_pixels, light_pixels)
    balance_score = balance * 100

    # Combined score
    score = int(0.35 * std_score + 0.25 * extremes_score + 0.25 * range_score + 0.15 * balance_score)

    details = {
        "std_dev": round(float(std_dev), 2),
        "black_ratio": round(float(black_ratio), 4),
        "white_ratio": round(float(white_ratio), 4),
        "dynamic_range": round(float(dynamic_range), 2),
        "light_dark_balance": round(float(balance), 3),
    }

    return score, details


def analyze_texture_details(gray: np.ndarray) -> tuple[int, dict[str, Any]]:
    """
    Analyze texture and details: edge detection, local variance, sharpness.
    Returns score (0-100) and details dict.
    """
    # Edge detection with Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_density = (sobel_mag > 50).sum() / gray.size
    edge_score = min(100, edge_density * 500)

    # Canny edge detection for finer details
    canny = cv2.Canny(gray, 100, 200)
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
    score = int(0.25 * edge_score + 0.20 * canny_score + 0.30 * texture_score + 0.25 * sharpness_score)

    details = {
        "edge_density": round(float(edge_density), 4),
        "canny_density": round(float(canny_density), 4),
        "avg_local_variance": round(float(avg_local_var), 2),
        "laplacian_variance": round(float(sharpness), 2),
    }

    return score, details


def analyze_colorimetry(img_bgr: np.ndarray) -> tuple[int, dict[str, Any]]:
    """
    Analyze colorimetry: saturation distribution, quasi-monochrome detection, gamut.
    Returns score (0-100) and details dict.
    Low saturation = better for B&W (higher score).
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation = img_hsv[:, :, 1]

    # Average saturation (lower is better for B&W)
    avg_sat = float(np.mean(saturation))
    sat_score = max(0.0, 100 - (avg_sat / 255) * 100)

    # High saturation penalty (only for very saturated pixels > 150)
    high_sat_ratio = float((saturation > 150).sum()) / saturation.size
    # Stronger penalty when both avg_sat is high AND high_sat_ratio is significant
    if avg_sat > 100 and high_sat_ratio > 0.3:
        high_sat_penalty = high_sat_ratio * 40  # Food/colorful scenes
    else:
        high_sat_penalty = high_sat_ratio * 15

    # Saturation standard deviation (lower variance = more uniform = bonus)
    sat_std = float(np.std(saturation))
    uniformity_bonus = max(0.0, 20 - (sat_std / 50) * 20)

    # Quasi-monochrome detection (very low saturation throughout)
    low_sat_ratio = float((saturation < 30).sum()) / saturation.size
    mono_bonus = low_sat_ratio * 35  # Slight increase for quasi-monochrome

    # Color dominance vs varied palette
    hue = img_hsv[:, :, 0]
    # Only consider pixels with meaningful saturation
    meaningful_mask = saturation > 30
    if meaningful_mask.sum() > 0:
        meaningful_hues = hue[meaningful_mask]
        hue_std = float(np.std(meaningful_hues))
        # High hue variance = varied palette = slightly worse for B&W
        palette_penalty = min(20.0, (hue_std / 50) * 20)
    else:
        palette_penalty = 0.0

    # Used gamut (spread of saturation values)
    sat_p10 = float(np.percentile(saturation, 10))
    sat_p90 = float(np.percentile(saturation, 90))
    gamut_spread = sat_p90 - sat_p10

    # Combined score
    score = int(float(sat_score) + float(uniformity_bonus) + float(mono_bonus) - float(palette_penalty) - float(high_sat_penalty))
    score = max(0, min(100, score))

    details = {
        "avg_saturation": round(float(avg_sat), 2),
        "saturation_std": round(float(sat_std), 2),
        "low_saturation_ratio": round(float(low_sat_ratio), 4),
        "high_saturation_ratio": round(float(high_sat_ratio), 4),
        "gamut_spread": round(float(gamut_spread), 2),
    }

    return score, details


def analyze_tonal_composition(gray: np.ndarray) -> tuple[int, dict[str, Any]]:
    """
    Analyze tonal composition: plane separation, gray gradients, highlight zones.
    Returns score (0-100) and details dict.
    """
    h, w = gray.shape

    # Plane separation by luminosity (divide into regions)
    # Split into 3x3 grid and measure luminosity differences
    grid_h, grid_w = h // 3, w // 3
    region_means = []
    for i in range(3):
        for j in range(3):
            region = gray[i * grid_h : (i + 1) * grid_h, j * grid_w : (j + 1) * grid_w]
            region_means.append(np.mean(region))

    region_std = np.std(region_means)
    separation_score = min(100, (region_std / 40) * 100)

    # Natural gray gradients (smooth transitions)
    # Measure gradient smoothness via derivative variance
    grad_x = np.diff(gray.astype(np.float64), axis=1)
    grad_y = np.diff(gray.astype(np.float64), axis=0)
    grad_smoothness = 1 / (1 + np.std(grad_x) / 10 + np.std(grad_y) / 10)
    gradient_score = grad_smoothness * 100

    # Center-weighted luminosity (for portraits: center often contains subject)
    center_region = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    center_mean = np.mean(center_region)
    overall_mean = np.mean(gray)
    # Good if center is well-exposed (not too dark, not too bright)
    center_exposure = 100 - abs(center_mean - 127) * 0.7
    center_score = max(0, center_exposure)

    # Highlight detection (bright focused areas)
    highlights = (gray > 200).sum() / gray.size
    highlight_score = min(100, highlights * 300) if highlights < 0.3 else max(0, 100 - (highlights - 0.3) * 200)

    # Combined score
    score = int(0.30 * separation_score + 0.25 * gradient_score + 0.25 * center_score + 0.20 * highlight_score)

    details = {
        "region_luminosity_std": round(float(region_std), 2),
        "gradient_smoothness": round(float(grad_smoothness), 4),
        "center_mean_luminosity": round(float(center_mean), 2),
        "highlight_ratio": round(float(highlights), 4),
    }

    return score, details


def analyze_metadata(filepath: Path) -> tuple[int, dict[str, Any]]:
    """
    Analyze EXIF metadata: ISO, focal length, light conditions.
    Returns score (0-100) and details dict.
    """
    score = 50  # Default neutral score if no metadata
    details: dict[str, Any] = {}

    try:
        with Image.open(filepath) as img:
            exif_data = img.getexif()
            if exif_data is None:
                return score, {"available": False}

            exif = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                exif[tag] = value

            details["available"] = True
            iso_score = 50
            focal_score = 50
            light_score = 50

            # High ISO (grain = B&W aesthetic bonus)
            if "ISOSpeedRatings" in exif:
                iso = exif["ISOSpeedRatings"]
                if isinstance(iso, tuple):
                    iso = iso[0]
                details["iso"] = iso
                # Higher ISO = more grain = slight bonus for B&W
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
                focal = exif["FocalLength"]
                if hasattr(focal, "numerator"):
                    focal = focal.numerator / focal.denominator
                details["focal_length"] = round(float(focal), 1)
                # Portrait range (50-135mm) and wide (< 35mm) good for B&W
                if 50 <= focal <= 135:
                    focal_score = 70  # Portrait range
                elif focal < 35:
                    focal_score = 65  # Wide angle
                else:
                    focal_score = 50

            # Exposure time for light condition hints
            if "ExposureTime" in exif:
                exp = exif["ExposureTime"]
                if hasattr(exp, "numerator"):
                    exp_val = exp.numerator / exp.denominator
                else:
                    exp_val = float(exp)
                details["exposure_time"] = round(exp_val, 6)
                # Very long exposures can be artistic in B&W
                if exp_val >= 1:
                    light_score = 70

            # F-number for depth of field estimation
            if "FNumber" in exif:
                fnum = exif["FNumber"]
                if hasattr(fnum, "numerator"):
                    fnum = fnum.numerator / fnum.denominator
                details["f_number"] = round(float(fnum), 1)

            score = int(0.40 * iso_score + 0.30 * focal_score + 0.30 * light_score)

    except Exception as e:
        details["error"] = str(e)

    return score, details


def score_photo(filepath: Path) -> PhotoResult:
    """
    Score a single photo for B&W potential.
    Returns PhotoResult with score and breakdown.
    """
    img_bgr = cv2.imread(str(filepath))
    if img_bgr is None:
        raise ValueError(f"Could not read image: {filepath}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Run all analyses
    contrast_score, contrast_details = analyze_tonal_contrast(gray)
    texture_score, texture_details = analyze_texture_details(gray)
    saturation_score, saturation_details = analyze_colorimetry(img_bgr)
    composition_score, composition_details = analyze_tonal_composition(gray)
    metadata_score, metadata_details = analyze_metadata(filepath)

    # Weighted final score
    # Weights: Contrast 28%, Texture 22%, Saturation 28%, Composition 12%, Metadata 10%
    # Slightly increased saturation weight (food photos need more penalty)
    # Slightly increased texture weight (undervalued detail scenes)
    final_score = int(
        0.28 * contrast_score
        + 0.22 * texture_score
        + 0.28 * saturation_score
        + 0.12 * composition_score
        + 0.10 * metadata_score
    )

    # Portrait bonus: low-texture images with good saturation and composition
    # These are often intimate close-ups (skin, hands, faces) that work well in B&W
    # Pattern: smooth skin + low saturation + good composition = portrait-like
    if texture_score < 20 and saturation_score >= 50 and composition_score >= 50:
        # Smooth portrait-like image with good B&W characteristics
        portrait_bonus = 8
        final_score = min(100, final_score + portrait_bonus)

    return PhotoResult(
        filename=filepath.name,
        score=final_score,
        breakdown=ScoreBreakdown(
            contrast=contrast_score,
            texture=texture_score,
            saturation=saturation_score,
            composition=composition_score,
            metadata=metadata_score,
        ),
        details={
            "contrast": contrast_details,
            "texture": texture_details,
            "saturation": saturation_details,
            "composition": composition_details,
            "metadata": metadata_details,
        },
    )


def main() -> None:
    """Main entry point."""
    photos_dir = Path("./photos")
    if not photos_dir.exists():
        print("Error: ./photos directory not found")
        sys.exit(1)

    # Find all image files
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    photos = sorted([f for f in photos_dir.iterdir() if f.suffix.lower() in extensions])

    if not photos:
        print("Error: No photos found in ./photos directory")
        sys.exit(1)

    print(f"Analyzing {len(photos)} photos...\n")

    results: list[PhotoResult] = []
    for photo in photos:
        try:
            result = score_photo(photo)
            results.append(result)

            # Print result in expected format
            breakdown = result["breakdown"]
            sat_label = "Low saturation" if breakdown["saturation"] >= 50 else "High saturation"
            print(f"{result['filename']}: {result['score']}/100")
            print(
                f"  - Contrast: {breakdown['contrast']} | "
                f"Texture: {breakdown['texture']} | "
                f"{sat_label}: {breakdown['saturation']} | "
                f"Composition: {breakdown['composition']}"
            )
            print()
        except Exception as e:
            print(f"{photo.name}: ERROR - {e}\n")

    # Sort results by score (highest first)
    results.sort(key=lambda x: x["score"], reverse=True)

    # Export to JSON
    output_path = Path("results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults exported to {output_path}")
    print(f"Top 5 photos for B&W conversion:")
    for i, r in enumerate(results[:5], 1):
        print(f"  {i}. {r['filename']}: {r['score']}/100")


if __name__ == "__main__":
    main()
