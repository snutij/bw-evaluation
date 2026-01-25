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
    score_photo,
)


class TestGoldenReferences:
    """Tests that lock in validated improvements. DO NOT weaken these."""

    def test_saturated_scene_penalty(self) -> None:
        """Highly saturated scenes must score low for B&W potential.

        Characteristics: high avg saturation (>100), high ratio of saturated
        pixels (>30%), warm/varied hues.
        """
        import cv2

        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        # Warm hues (red/orange/yellow range: 0-30 in OpenCV's 0-180 scale)
        hsv[:, :, 0] = np.random.randint(0, 30, (100, 100), dtype=np.uint8)
        # High saturation (>150 threshold triggers penalty)
        hsv[:, :, 1] = np.random.randint(160, 255, (100, 100), dtype=np.uint8)
        # Medium-high value
        hsv[:, :, 2] = np.random.randint(150, 230, (100, 100), dtype=np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        score, details = analyze_colorimetry(img)

        # Verify we created the right characteristics
        assert details["avg_saturation"] > 100, "Test setup: need high avg saturation"
        assert details["high_saturation_ratio"] > 0.3, "Test setup: need high sat ratio"

        # THE GOLDEN ASSERTION: saturated scenes must score low
        assert score <= 45, (
            f"Saturated scenes must score <= 45 for colorimetry, got {score}. "
            "High saturation penalty should apply."
        )

    def test_textured_scene_valued(self) -> None:
        """Textured scenes with sharp edges deserve texture credit.

        Characteristics: sharp edges, varied textures (grid patterns, surfaces),
        good local variance.
        """
        img = np.zeros((100, 100), dtype=np.uint8)

        # Grid pattern
        for i in range(0, 100, 8):
            img[i, :] = 180  # Horizontal lines
            img[:, i] = 180  # Vertical lines

        # Add some random texture variation
        noise = np.random.randint(-30, 30, (100, 100), dtype=np.int16)
        base = img.astype(np.int16) + 80
        img = np.clip(base + noise, 0, 255).astype(np.uint8)

        score, details = analyze_texture_details(img)

        # THE GOLDEN ASSERTION: textured scenes must score reasonably
        assert score >= 50, (
            f"Textured scenes must score >= 50 for texture, got {score}. "
            "Edge detection and local variance should reward texture."
        )
        # Should detect meaningful edges
        assert details["edge_density"] > 0.05, "Should detect grid edges"

    def test_smooth_closeup_characteristics(self) -> None:
        """Smooth low-texture closeups with moderate saturation.

        These scenes often have:
        - Low texture (smooth surfaces)
        - Warm tones (but not highly saturated)
        - Center-weighted composition

        The algorithm may undervalue these due to low texture, which is a
        known limitation (semantic content requires AI to detect).
        """
        import cv2

        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        # Warm hues - UNIFORM for smooth surface
        hsv[:, :, 0] = 15
        # Moderate saturation (~110)
        hsv[:, :, 1] = 110
        # Medium-high value
        hsv[:, :, 2] = 190
        img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Verify colorimetry doesn't over-penalize moderate saturation
        color_score, color_details = analyze_colorimetry(img_bgr)
        texture_score, _ = analyze_texture_details(img_gray)

        # Moderate saturation should not get the harsh penalty
        assert color_details["avg_saturation"] < 150, "Test setup: moderate saturation"
        assert color_score >= 40, (
            f"Moderate saturation should score >= 40, got {color_score}. "
            "Should not receive high saturation penalty."
        )

        # Document: low texture is expected for smooth surfaces
        assert texture_score <= 50, (
            "Smooth closeups have low texture by design (not a bug)"
        )


class TestPortraitBonus:
    """Tests for the portrait bonus.

    Low-texture images with smooth surfaces were consistently underrated
    because low texture penalized them. The portrait bonus compensates
    when the image has good B&W characteristics.
    """

    def test_portrait_bonus_applied(self) -> None:
        """Low-texture images with good saturation and composition get +8 bonus.

        Criteria: texture < 20, saturation >= 50, composition >= 50.
        """
        import cv2
        import tempfile
        from pathlib import Path as TmpPath

        # Create a smooth, low-saturation portrait-like image
        # Use a split composition (left dark, right bright) for good region std
        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv[:, :, 0] = 20  # Hue
        hsv[:, :, 1] = 50  # Low saturation -> good for B&W

        # Split composition: left third dark, middle bright, right medium
        # This creates good region_luminosity_std without adding texture
        hsv[:, 0:33, 2] = 80  # Left dark
        hsv[:, 33:66, 2] = 180  # Center bright (well-exposed)
        hsv[:, 66:100, 2] = 120  # Right medium

        # Apply gaussian blur to keep it smooth (no sharp edges)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img = cv2.GaussianBlur(img, (15, 15), 0)

        # Write to temp file for score_photo
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, img)
            tmp_path = TmpPath(f.name)

        try:
            result = score_photo(tmp_path)
            breakdown = result["breakdown"]

            # Verify portrait-like characteristics
            assert breakdown["texture"] < 20, f"Should be low texture, got {breakdown['texture']}"
            assert breakdown["saturation"] >= 50, f"Should have good saturation score, got {breakdown['saturation']}"
            assert breakdown["composition"] >= 50, f"Should have good composition, got {breakdown['composition']}"

            # The portrait bonus should have been applied
            # Calculate what score would be without bonus
            base_score = int(
                0.28 * breakdown["contrast"]
                + 0.22 * breakdown["texture"]
                + 0.28 * breakdown["saturation"]
                + 0.12 * breakdown["composition"]
                + 0.10 * breakdown["metadata"]
            )

            # Final score should be base + 8 (portrait bonus)
            expected_with_bonus = min(100, base_score + 8)
            assert result["score"] == expected_with_bonus, (
                f"Portrait bonus should be applied. Base: {base_score}, "
                f"Expected: {expected_with_bonus}, Got: {result['score']}"
            )
        finally:
            tmp_path.unlink()

    def test_portrait_bonus_not_applied_high_texture(self) -> None:
        """High-texture images should NOT get portrait bonus."""
        import cv2
        import tempfile
        from pathlib import Path as TmpPath

        # Create a highly textured image
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Add noise/texture pattern
        for y in range(0, 100, 4):
            for x in range(0, 100, 4):
                img[y:y+2, x:x+2] = [200, 200, 200]
                img[y+2:y+4, x+2:x+4] = [50, 50, 50]

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, img)
            tmp_path = TmpPath(f.name)

        try:
            result = score_photo(tmp_path)
            breakdown = result["breakdown"]

            # High texture should NOT trigger portrait bonus
            assert breakdown["texture"] >= 20, f"Should be high texture, got {breakdown['texture']}"

            # Score should be calculated normally without bonus
            base_score = int(
                0.28 * breakdown["contrast"]
                + 0.22 * breakdown["texture"]
                + 0.28 * breakdown["saturation"]
                + 0.12 * breakdown["composition"]
                + 0.10 * breakdown["metadata"]
            )

            assert result["score"] == base_score, (
                f"Portrait bonus should NOT be applied for high-texture images. "
                f"Base: {base_score}, Got: {result['score']}"
            )
        finally:
            tmp_path.unlink()


class TestHighSaturationPenalty:
    """Tests for the high saturation penalty logic."""

    def test_penalty_triggers_at_threshold(self) -> None:
        """High sat penalty should trigger when avg_sat > 100 AND ratio > 0.3."""
        import cv2

        # Case 1: Both conditions met -> strong penalty
        hsv_high = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_high[:, :, 0] = 15  # Hue
        hsv_high[:, :, 1] = 200  # High saturation
        hsv_high[:, :, 2] = 180  # Value
        img_high = cv2.cvtColor(hsv_high, cv2.COLOR_HSV2BGR)
        score_high, details_high = analyze_colorimetry(img_high)

        # Case 2: Low avg saturation -> weaker penalty
        hsv_low = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_low[:, :, 0] = 15
        hsv_low[:, :, 1] = 80  # Lower saturation
        hsv_low[:, :, 2] = 180
        img_low = cv2.cvtColor(hsv_low, cv2.COLOR_HSV2BGR)
        score_low, details_low = analyze_colorimetry(img_low)

        # High saturation case should score significantly lower
        assert score_high < score_low, (
            f"High saturation ({score_high}) should score lower than "
            f"moderate saturation ({score_low})"
        )
        # The difference should be substantial due to penalty
        assert score_low - score_high >= 15, (
            f"Penalty should create >= 15 point gap, got {score_low - score_high}"
        )

    def test_grayscale_no_penalty(self) -> None:
        """Already grayscale images should score high with no saturation penalty."""
        # Pure grayscale image
        gray_bgr = np.full((100, 100, 3), 128, dtype=np.uint8)

        score, details = analyze_colorimetry(gray_bgr)

        assert details["avg_saturation"] < 5, "Test setup: should be grayscale"
        assert score >= 85, (
            f"Grayscale images should score >= 85 for colorimetry, got {score}"
        )
