"""Golden reference tests that lock in validated improvements.

These tests prevent regressions on specific photo characteristics that were
validated during development. DO NOT weaken these tests - if they fail,
fix the regression in the scorer, not the test.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from bw_scorer import (
    analyze_channel_separation,
    analyze_colorimetry,
    analyze_texture_details,
)


class TestGoldenReferences:
    """Tests that lock in validated improvements. DO NOT weaken these."""

    def test_saturated_scene_scores_low(self) -> None:
        """Highly saturated scenes must score low for B&W potential."""
        import cv2

        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv[:, :, 1] = 220  # High saturation
        hsv[:, :, 2] = 180
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        score, details = analyze_colorimetry(img)

        assert details["avg_saturation"] > 150, "Test setup: need high avg saturation"
        assert score <= 40, (
            f"Saturated scenes must score <= 40 for colorimetry, got {score}. "
            "High saturation → low B&W potential."
        )

    def test_desaturated_scene_scores_high(self) -> None:
        """Near-monochrome scenes must score high for B&W potential."""
        gray_bgr = np.full((100, 100, 3), 128, dtype=np.uint8)
        score, details = analyze_colorimetry(gray_bgr)

        assert details["avg_saturation"] < 5, "Test setup: need near-zero saturation"
        assert score >= 98, (
            f"Near-monochrome scenes must score >= 98 for colorimetry, got {score}."
        )

    def test_textured_scene_valued(self) -> None:
        """Textured scenes with sharp edges deserve texture credit."""
        img = np.zeros((100, 100), dtype=np.uint8)

        for i in range(0, 100, 8):
            img[i, :] = 180
            img[:, i] = 180

        noise = np.random.default_rng(42).integers(-30, 30, (100, 100))
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
        assert color_score >= 40, (
            f"Moderate saturation should score >= 40, got {color_score}."
        )
        assert texture_score <= 50, (
            "Smooth closeups have low texture by design (not a bug)"
        )


class TestChannelSeparation:
    """Tests for channel separation scoring."""

    def test_colorful_image_high_channel_separation(self) -> None:
        """A colorful image (red on green) should have high channel separation."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :, 2] = 255  # Red top (BGR)
        img[50:, :, 1] = 255  # Green bottom
        score, _details = analyze_channel_separation(img)
        assert score >= 60, (
            f"Colorful image must score >= 60 for channel separation, got {score}. "
            "RGB divergence = creative B&W potential."
        )

    def test_grayscale_image_zero_channel_separation(self) -> None:
        """A true grayscale image should have near-zero channel separation."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        score, _details = analyze_channel_separation(img)
        assert score == 0, (
            f"Grayscale image must score 0 for channel separation, got {score}."
        )

    def test_saturated_colorful_beats_bland_desaturated(self) -> None:
        """A colorful image should outscore a bland desaturated one on channel sep."""
        # Colorful: red rose vs green background
        colorful = np.zeros((100, 100, 3), dtype=np.uint8)
        colorful[:50, :, 2] = 200  # Red
        colorful[50:, :, 1] = 200  # Green

        # Bland grey
        bland = np.full((100, 100, 3), 100, dtype=np.uint8)

        score_colorful, _ = analyze_channel_separation(colorful)
        score_bland, _ = analyze_channel_separation(bland)

        assert score_colorful > score_bland, (
            f"Colorful ({score_colorful}) should beat bland ({score_bland}) on ch.sep."
        )
