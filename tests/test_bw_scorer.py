"""Unit tests for bw_scorer.py"""

import numpy as np
import cv2
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from bw_scorer import (
    analyze_tonal_contrast,
    analyze_texture_details,
    analyze_colorimetry,
    analyze_tonal_composition,
    analyze_channel_separation,
    score_photo,
    ScoreBreakdown,
    PhotoResult,
)


class TestAnalyzeTonalContrast:
    """Tests for tonal contrast analysis."""

    def test_high_contrast_scores_high(self, high_contrast_gray: np.ndarray) -> None:
        """High contrast image (black/white stripes) should score high."""
        score, details = analyze_tonal_contrast(high_contrast_gray)
        assert score >= 70, f"High contrast image scored too low: {score}"
        assert details["std_dev"] > 100  # High standard deviation

    def test_low_contrast_scores_low(self, low_contrast_gray: np.ndarray) -> None:
        """Low contrast image (uniform gray) should score low."""
        score, details = analyze_tonal_contrast(low_contrast_gray)
        assert score <= 30, f"Low contrast image scored too high: {score}"
        assert details["std_dev"] < 5  # Very low standard deviation

    def test_gradient_high_contrast(self, gradient_gray: np.ndarray) -> None:
        """Gradient image (full range 0-255) should score high on contrast."""
        score, details = analyze_tonal_contrast(gradient_gray)
        assert score >= 70, f"Full gradient scored too low: {score}"
        assert details["dynamic_range"] > 200  # Full range gradient

    def test_returns_expected_details_keys(self, high_contrast_gray: np.ndarray) -> None:
        """Should return all expected detail keys."""
        _, details = analyze_tonal_contrast(high_contrast_gray)
        expected_keys = {"std_dev", "black_ratio", "white_ratio", "dynamic_range", "light_dark_balance"}
        assert set(details.keys()) == expected_keys

    def test_score_bounded_0_100(self, high_contrast_gray: np.ndarray, low_contrast_gray: np.ndarray) -> None:
        """Score should always be between 0 and 100."""
        for gray in [high_contrast_gray, low_contrast_gray]:
            score, _ = analyze_tonal_contrast(gray)
            assert 0 <= score <= 100


class TestAnalyzeTextureDetails:
    """Tests for texture/detail analysis."""

    def test_textured_image_scores_high(self, textured_gray: np.ndarray) -> None:
        """Textured image with edges should score high."""
        score, details = analyze_texture_details(textured_gray)
        assert score >= 40, f"Textured image scored too low: {score}"
        assert details["edge_density"] > 0.01

    def test_smooth_image_scores_low(self, smooth_gray: np.ndarray) -> None:
        """Smooth uniform image should score low on texture."""
        score, details = analyze_texture_details(smooth_gray)
        assert score <= 20, f"Smooth image scored too high: {score}"
        assert details["edge_density"] < 0.01

    def test_returns_expected_details_keys(self, textured_gray: np.ndarray) -> None:
        """Should return all expected detail keys."""
        _, details = analyze_texture_details(textured_gray)
        expected_keys = {"edge_density", "canny_density", "avg_local_variance", "laplacian_variance"}
        assert set(details.keys()) == expected_keys

    def test_score_bounded_0_100(self, textured_gray: np.ndarray, smooth_gray: np.ndarray) -> None:
        """Score should always be between 0 and 100."""
        for gray in [textured_gray, smooth_gray]:
            score, _ = analyze_texture_details(gray)
            assert 0 <= score <= 100


class TestAnalyzeColorimetry:
    """Tests for colorimetry/saturation analysis."""

    def test_saturated_image_scores_low(self, saturated_bgr: np.ndarray) -> None:
        """Highly saturated (colorful) image should score low for B&W."""
        score, details = analyze_colorimetry(saturated_bgr)
        assert score <= 50, f"Saturated image scored too high: {score}"
        assert details["avg_saturation"] > 200

    def test_desaturated_image_scores_high(self, desaturated_bgr: np.ndarray) -> None:
        """Desaturated (gray) image should score high for B&W."""
        score, details = analyze_colorimetry(desaturated_bgr)
        assert score >= 80, f"Desaturated image scored too low: {score}"
        assert details["avg_saturation"] < 10

    def test_returns_expected_details_keys(self, saturated_bgr: np.ndarray) -> None:
        """Should return only avg_saturation detail key."""
        _, details = analyze_colorimetry(saturated_bgr)
        assert set(details.keys()) == {"avg_saturation"}

    def test_score_formula(self) -> None:
        """Score should be exactly 100 - (mean_sat / 255) * 100."""
        # Pure gray: saturation = 0, score should be 100
        gray_bgr = np.full((100, 100, 3), 128, dtype=np.uint8)
        score, details = analyze_colorimetry(gray_bgr)
        assert details["avg_saturation"] < 5
        assert score >= 98

    def test_score_bounded_0_100(self, saturated_bgr: np.ndarray, desaturated_bgr: np.ndarray) -> None:
        """Score should always be between 0 and 100."""
        for bgr in [saturated_bgr, desaturated_bgr]:
            score, _ = analyze_colorimetry(bgr)
            assert 0 <= score <= 100


class TestAnalyzeTonalComposition:
    """Tests for tonal composition analysis."""

    def test_center_bright_composition(self, center_bright_gray: np.ndarray) -> None:
        """Image with bright center should score reasonably."""
        score, details = analyze_tonal_composition(center_bright_gray)
        assert 0 <= score <= 100
        assert "region_luminosity_std" in details

    def test_uniform_image_low_separation(self, low_contrast_gray: np.ndarray) -> None:
        """Uniform image should have low region separation."""
        score, details = analyze_tonal_composition(low_contrast_gray)
        assert details["region_luminosity_std"] < 1  # Uniform = no separation

    def test_returns_expected_details_keys(self, center_bright_gray: np.ndarray) -> None:
        """Should return only plane separation and highlight details."""
        _, details = analyze_tonal_composition(center_bright_gray)
        assert set(details.keys()) == {"region_luminosity_std", "highlight_ratio"}

    def test_score_bounded_0_100(self, center_bright_gray: np.ndarray, low_contrast_gray: np.ndarray) -> None:
        """Score should always be between 0 and 100."""
        for gray in [center_bright_gray, low_contrast_gray]:
            score, _ = analyze_tonal_composition(gray)
            assert 0 <= score <= 100


class TestAnalyzeChannelSeparation:
    """Tests for channel separation analysis."""

    def test_high_divergence_scores_high(self) -> None:
        """Image with strongly divergent RGB channels should score high."""
        # Red on green: high divergence
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :, 2] = 255  # Red top half (BGR: R=channel 2)
        img[50:, :, 1] = 255  # Green bottom half (BGR: G=channel 1)
        score, details = analyze_channel_separation(img)
        assert score >= 60, f"High channel divergence should score >= 60, got {score}"
        assert details["mean_channel_std"] > 0

    def test_low_divergence_scores_low(self) -> None:
        """Near-monochrome image (equal RGB) should score low."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        score, details = analyze_channel_separation(img)
        assert score == 0, f"Equal RGB channels should score 0, got {score}"
        assert details["mean_channel_std"] < 1

    def test_normalization_ceiling(self) -> None:
        """Score should saturate at 100 when mean_std >= ceiling."""
        from config import ScoringConfig
        cfg = ScoringConfig(channel_sep_ceiling=10.0)
        # Max channel divergence: alternating pure R and pure B pixels
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # Blue channel maxed
        img[:, :, 2] = 255  # Red channel maxed (green = 0)
        score, details = analyze_channel_separation(img, cfg)
        assert score == 100, f"Should clamp to 100 when above ceiling"

    def test_returns_expected_details_keys(self) -> None:
        """Should return mean_channel_std detail key."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        _, details = analyze_channel_separation(img)
        assert set(details.keys()) == {"mean_channel_std"}

    def test_score_bounded_0_100(self) -> None:
        """Score should always be between 0 and 100."""
        img = np.random.default_rng(42).integers(0, 255, (100, 100, 3), dtype=np.uint8)
        score, _ = analyze_channel_separation(img)
        assert 0 <= score <= 100


class TestScorePhoto:
    """Tests for the main score_photo function."""

    def test_scores_valid_image(self, tmp_path: Path) -> None:
        """Should successfully score a valid image file."""
        img_path = tmp_path / "test_image.jpg"
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[50:150, 50:150] = 180
        img[:, :, 0] = 50
        cv2.imwrite(str(img_path), img)

        result = score_photo(img_path)

        assert isinstance(result, dict)
        assert result["filename"] == "test_image.jpg"
        assert 0 <= result["score"] <= 100
        assert "breakdown" in result
        assert "details" in result

    def test_breakdown_has_all_components(self, tmp_path: Path) -> None:
        """Breakdown should include all analysis components."""
        img_path = tmp_path / "test_image.jpg"
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        result = score_photo(img_path)
        breakdown = result["breakdown"]

        expected_keys = {"contrast", "texture", "saturation", "composition", "channel_separation"}
        assert set(breakdown.keys()) == expected_keys
        for key in expected_keys:
            assert 0 <= breakdown[key] <= 100

    def test_no_metadata_in_breakdown(self, tmp_path: Path) -> None:
        """Breakdown should not contain metadata score."""
        img_path = tmp_path / "test_image.jpg"
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        result = score_photo(img_path)
        assert "metadata" not in result["breakdown"]

    def test_no_scene_bonus_in_details(self, tmp_path: Path) -> None:
        """Details should not contain scene_bonus."""
        img_path = tmp_path / "test_image.jpg"
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        result = score_photo(img_path)
        assert "scene_bonus" not in result["details"]

    def test_invalid_file_raises_error(self, tmp_path: Path) -> None:
        """Should raise error for invalid/missing file."""
        fake_path = tmp_path / "nonexistent.jpg"
        with pytest.raises(ValueError, match="Could not read image"):
            score_photo(fake_path)

    def test_corrupted_file_raises_error(self, tmp_path: Path) -> None:
        """Should raise error for corrupted image file."""
        corrupted_path = tmp_path / "corrupted.jpg"
        corrupted_path.write_text("not an image")
        with pytest.raises(ValueError, match="Could not read image"):
            score_photo(corrupted_path)


class TestDeterminism:
    """Tests to verify scoring determinism (same input = same output)."""

    def test_same_image_same_score(self, tmp_path: Path) -> None:
        """Same image should always produce the same score."""
        img_path = tmp_path / "determinism_test.jpg"
        img = np.random.default_rng(42).integers(0, 255, (200, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        results = [score_photo(img_path) for _ in range(5)]
        scores = [r["score"] for r in results]

        assert len(set(scores)) == 1, "Scores should be identical for same image"

    def test_analysis_functions_deterministic(self) -> None:
        """Individual analysis functions should be deterministic."""
        rng = np.random.default_rng(123)
        gray = rng.integers(0, 255, (100, 100), dtype=np.uint8)
        bgr = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)

        contrast_scores = [analyze_tonal_contrast(gray)[0] for _ in range(3)]
        texture_scores = [analyze_texture_details(gray)[0] for _ in range(3)]
        color_scores = [analyze_colorimetry(bgr)[0] for _ in range(3)]
        comp_scores = [analyze_tonal_composition(gray)[0] for _ in range(3)]
        chan_sep_scores = [analyze_channel_separation(bgr)[0] for _ in range(3)]

        assert len(set(contrast_scores)) == 1
        assert len(set(texture_scores)) == 1
        assert len(set(color_scores)) == 1
        assert len(set(comp_scores)) == 1
        assert len(set(chan_sep_scores)) == 1


class TestScoreWeighting:
    """Tests to verify score weighting is applied correctly."""

    def test_final_score_is_weighted_average(self, tmp_path: Path) -> None:
        """Final score should be weighted average of the five components."""
        img_path = tmp_path / "weighted_test.jpg"
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        result = score_photo(img_path)
        breakdown = result["breakdown"]

        expected = max(0, min(100, int(round(
            0.35 * breakdown["contrast"]
            + 0.25 * breakdown["texture"]
            + 0.10 * breakdown["saturation"]
            + 0.05 * breakdown["composition"]
            + 0.25 * breakdown["channel_separation"]
        ))))

        assert abs(result["score"] - expected) <= 1, (
            f"Expected ~{expected}, got {result['score']}"
        )
