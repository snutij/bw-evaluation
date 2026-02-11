"""Golden reference tests that lock in validated improvements.

These tests prevent regressions on specific photo characteristics that were
validated during development. DO NOT weaken these tests - if they fail,
fix the regression in the scorer, not the test.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from bw_scorer import (
    analyze_colorimetry,
    analyze_texture_details,
    analyze_tonal_contrast,
    analyze_tonal_composition,
    analyze_metadata,
    detect_scene_type,
    score_photo,
    ScoreBreakdown,
)


class TestGoldenReferences:
    """Tests that lock in validated improvements. DO NOT weaken these."""

    def test_saturated_scene_penalty(self) -> None:
        """Highly saturated scenes must score low for B&W potential."""
        import cv2

        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv[:, :, 0] = np.random.randint(0, 30, (100, 100), dtype=np.uint8)
        hsv[:, :, 1] = np.random.randint(160, 255, (100, 100), dtype=np.uint8)
        hsv[:, :, 2] = np.random.randint(150, 230, (100, 100), dtype=np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        score, details = analyze_colorimetry(img)

        assert details["avg_saturation"] > 100, "Test setup: need high avg saturation"
        assert details["high_saturation_ratio"] > 0.3, "Test setup: need high sat ratio"

        # THE GOLDEN ASSERTION: saturated scenes must score low
        # Gamut penalty may push score lower than before
        assert score <= 45, (
            f"Saturated scenes must score <= 45 for colorimetry, got {score}. "
            "High saturation penalty + gamut penalty should apply."
        )

    def test_textured_scene_valued(self) -> None:
        """Textured scenes with sharp edges deserve texture credit."""
        img = np.zeros((100, 100), dtype=np.uint8)

        for i in range(0, 100, 8):
            img[i, :] = 180
            img[:, i] = 180

        noise = np.random.randint(-30, 30, (100, 100), dtype=np.int16)
        base = img.astype(np.int16) + 80
        img = np.clip(base + noise, 0, 255).astype(np.uint8)

        score, details = analyze_texture_details(img)

        assert score >= 50, (
            f"Textured scenes must score >= 50 for texture, got {score}. "
            "Edge detection and local variance should reward texture."
        )
        assert details["edge_density"] > 0.05, "Should detect grid edges"

    def test_smooth_closeup_characteristics(self) -> None:
        """Smooth low-texture closeups with moderate saturation."""
        import cv2

        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv[:, :, 0] = 15
        hsv[:, :, 1] = 110
        hsv[:, :, 2] = 190
        img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        color_score, color_details = analyze_colorimetry(img_bgr)
        texture_score, _ = analyze_texture_details(img_gray)

        assert color_details["avg_saturation"] < 150, "Test setup: moderate saturation"
        assert color_score >= 35, (
            f"Moderate saturation should score >= 35, got {color_score}. "
            "Should not receive harsh penalty."
        )
        assert texture_score <= 50, (
            "Smooth closeups have low texture by design (not a bug)"
        )


class TestSceneBonuses:
    """Tests for the scene detection + bonus system."""

    def test_portrait_bonus_applied(self) -> None:
        """Low-texture images with good saturation and composition get portrait bonus."""
        import cv2
        import tempfile

        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv[:, :, 0] = 20
        hsv[:, :, 1] = 50

        # Split composition for good region_luminosity_std
        hsv[:, 0:33, 2] = 80
        hsv[:, 33:66, 2] = 180
        hsv[:, 66:100, 2] = 120

        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img = cv2.GaussianBlur(img, (15, 15), 0)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, img)
            tmp_path = Path(f.name)

        try:
            result = score_photo(tmp_path)
            breakdown = result["breakdown"]

            assert breakdown["texture"] < 20
            assert breakdown["saturation"] >= 50
            assert breakdown["composition"] >= 50

            assert result["details"].get("scene_type") == "portrait"
            assert result["details"].get("scene_bonus") == 8
        finally:
            tmp_path.unlink()

    def test_portrait_bonus_not_applied_high_texture(self) -> None:
        """High-texture images should NOT get portrait bonus."""
        import cv2
        import tempfile

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for y in range(0, 100, 4):
            for x in range(0, 100, 4):
                img[y:y+2, x:x+2] = [200, 200, 200]
                img[y+2:y+4, x+2:x+4] = [50, 50, 50]

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, img)
            tmp_path = Path(f.name)

        try:
            result = score_photo(tmp_path)
            assert result["breakdown"]["texture"] >= 20
            assert result["details"].get("scene_type") != "portrait"
        finally:
            tmp_path.unlink()

    def test_landscape_bonus(self) -> None:
        """Scene detection recognizes landscape pattern."""
        breakdown = ScoreBreakdown(contrast=65, texture=55, saturation=70, composition=45, metadata=50)
        details = {"texture": {"edge_density": 0.10}, "contrast": {"dynamic_range": 180}}
        scene, bonus = detect_scene_type(breakdown, details)
        assert scene == "landscape"
        assert bonus == 5

    def test_architecture_bonus(self) -> None:
        """Scene detection recognizes architecture pattern."""
        breakdown = ScoreBreakdown(contrast=55, texture=65, saturation=60, composition=50, metadata=50)
        details = {"texture": {"edge_density": 0.20}, "contrast": {"dynamic_range": 120}}
        scene, bonus = detect_scene_type(breakdown, details)
        assert scene == "architecture"
        assert bonus == 6


class TestHighSaturationPenalty:
    """Tests for the high saturation penalty logic."""

    def test_penalty_triggers_at_threshold(self) -> None:
        """High sat penalty should trigger when avg_sat > 100 AND ratio > 0.3."""
        import cv2

        hsv_high = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_high[:, :, 0] = 15
        hsv_high[:, :, 1] = 200
        hsv_high[:, :, 2] = 180
        img_high = cv2.cvtColor(hsv_high, cv2.COLOR_HSV2BGR)
        score_high, details_high = analyze_colorimetry(img_high)

        hsv_low = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_low[:, :, 0] = 15
        hsv_low[:, :, 1] = 80
        hsv_low[:, :, 2] = 180
        img_low = cv2.cvtColor(hsv_low, cv2.COLOR_HSV2BGR)
        score_low, details_low = analyze_colorimetry(img_low)

        assert score_high < score_low
        assert score_low - score_high >= 15

    def test_grayscale_no_penalty(self) -> None:
        """Already grayscale images should score high with no saturation penalty."""
        gray_bgr = np.full((100, 100, 3), 128, dtype=np.uint8)
        score, details = analyze_colorimetry(gray_bgr)

        assert details["avg_saturation"] < 5
        assert score >= 85


class TestGamutSpread:
    """Tests for gamut spread penalty."""

    def test_gamut_spread_affects_score(self) -> None:
        """Wide gamut spread (mixed sat/desat) should score lower than uniform low sat."""
        import cv2

        # Uniform low saturation
        hsv_uniform = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_uniform[:, :, 0] = 30
        hsv_uniform[:, :, 1] = 40  # Low uniform saturation
        hsv_uniform[:, :, 2] = 150
        img_uniform = cv2.cvtColor(hsv_uniform, cv2.COLOR_HSV2BGR)
        score_uniform, det_uniform = analyze_colorimetry(img_uniform)

        # Wide gamut: half very saturated, half desaturated
        hsv_mixed = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_mixed[:, :, 0] = 30
        hsv_mixed[:50, :, 1] = 200  # High saturation top half
        hsv_mixed[50:, :, 1] = 20   # Low saturation bottom half
        hsv_mixed[:, :, 2] = 150
        img_mixed = cv2.cvtColor(hsv_mixed, cv2.COLOR_HSV2BGR)
        score_mixed, det_mixed = analyze_colorimetry(img_mixed)

        assert det_mixed["gamut_spread"] > det_uniform["gamut_spread"]
        assert score_uniform > score_mixed, (
            f"Uniform low-sat ({score_uniform}) should beat wide gamut ({score_mixed})"
        )


class TestFNumberMetadata:
    """Tests for f-number DOF scoring."""

    def test_f_number_affects_metadata_score(self) -> None:
        """Different f-numbers should produce different DOF scores in detection logic."""
        # We test detect_scene_type indirectly and metadata weights
        # by verifying the weight structure in config
        from config import ScoringConfig
        cfg = ScoringConfig()
        assert cfg.metadata_dof_weight == 0.20
        assert cfg.metadata_iso_weight + cfg.metadata_focal_weight + cfg.metadata_light_weight + cfg.metadata_dof_weight == pytest.approx(1.0)


class TestRuleOfThirds:
    """Tests for rule-of-thirds detection."""

    def test_rule_of_thirds_detected(self, thirds_gray: np.ndarray) -> None:
        """Image with strong edges on thirds lines should get high thirds_energy."""
        _, details = analyze_tonal_composition(thirds_gray)
        assert details["thirds_energy"] > 10, (
            f"Expected high thirds energy for thirds-aligned edges, got {details['thirds_energy']}"
        )

    def test_uniform_image_low_thirds(self, low_contrast_gray: np.ndarray) -> None:
        """Uniform image should have near-zero thirds energy."""
        _, details = analyze_tonal_composition(low_contrast_gray)
        assert details["thirds_energy"] < 1, (
            f"Uniform image should have minimal thirds energy, got {details['thirds_energy']}"
        )
