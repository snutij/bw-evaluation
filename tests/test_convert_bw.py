"""Tests for convert_bw.py"""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from convert_bw import (
    PRESETS,
    STYLE_NAMES,
    ConversionPreset,
    convert_bw,
    pick_style,
    process_photos,
)


class TestConvertBw:
    """Tests for the convert_bw function."""

    def _make_rgb_image(self) -> Image.Image:
        """Create a test RGB image with varied colors to differentiate presets."""
        rng = np.random.default_rng(42)
        arr = rng.integers(30, 220, (100, 100, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def test_output_is_grayscale(self) -> None:
        """Converted image should be single-channel (mode L)."""
        img = self._make_rgb_image()
        result = convert_bw(img, PRESETS["portrait"])
        assert result.mode == "L"
        assert result.size == img.size

    def test_different_presets_produce_different_output(self) -> None:
        """Each preset should produce a meaningfully different result."""
        img = self._make_rgb_image()
        results = {}
        for name in STYLE_NAMES:
            # Street has grain (non-deterministic), so compare without it
            preset = PRESETS[name]
            if preset.grain > 0:
                preset = ConversionPreset(
                    red=preset.red, green=preset.green, blue=preset.blue,
                    contrast=preset.contrast, shadow_lift=preset.shadow_lift,
                    grain=0, unsharp=preset.unsharp,
                )
            bw = convert_bw(img, preset)
            results[name] = np.array(bw)

        # At least all pairs should be different
        names = list(results.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                diff = np.abs(results[names[i]].astype(float) - results[names[j]].astype(float)).mean()
                assert diff > 0.5, (
                    f"{names[i]} and {names[j]} should produce different output, diff={diff:.2f}"
                )

    def test_portrait_preset_values(self) -> None:
        """Portrait preset should have documented channel mix."""
        p = PRESETS["portrait"]
        assert p.red == pytest.approx(0.35)
        assert p.green == pytest.approx(0.45)
        assert p.blue == pytest.approx(0.20)
        assert p.contrast == pytest.approx(1.15)
        assert p.shadow_lift == 10

    def test_street_has_grain(self) -> None:
        """Street preset should have grain > 0."""
        assert PRESETS["street"].grain > 0

    def test_architecture_has_unsharp(self) -> None:
        """Architecture preset should apply unsharp mask."""
        assert PRESETS["architecture"].unsharp is True


class TestPickStyle:
    """Tests for auto-style selection."""

    def test_auto_uses_scene_type(self) -> None:
        assert pick_style("landscape", None) == "landscape"
        assert pick_style("portrait", "auto") == "portrait"
        assert pick_style("architecture", "auto") == "architecture"

    def test_forced_style_overrides(self) -> None:
        assert pick_style("portrait", "high-contrast") == "high-contrast"
        assert pick_style("landscape", "street") == "street"

    def test_generic_defaults_to_portrait(self) -> None:
        assert pick_style("generic", "auto") == "portrait"
        assert pick_style(None, "auto") == "portrait"


class TestProcessPhotos:
    """Tests for the process_photos function."""

    def _setup_photos(self, tmp_path: Path, count: int = 3) -> tuple[Path, Path, list[str]]:
        """Create test photos and return (input_dir, output_dir, filenames)."""
        input_dir = tmp_path / "photos"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        filenames = []
        for i in range(count):
            name = f"test_{i}.jpg"
            img = Image.fromarray(
                np.random.default_rng(i).integers(0, 255, (50, 50, 3), dtype=np.uint8),
                mode="RGB",
            )
            img.save(input_dir / name)
            filenames.append(name)

        return input_dir, output_dir, filenames

    def test_converts_files(self, tmp_path: Path) -> None:
        """Should convert files and create output directory."""
        input_dir, output_dir, filenames = self._setup_photos(tmp_path)
        converted = process_photos(input_dir, output_dir, filenames)
        assert converted == len(filenames)
        assert output_dir.exists()

    def test_no_overwrite(self, tmp_path: Path) -> None:
        """--no-overwrite should skip existing files."""
        input_dir, output_dir, filenames = self._setup_photos(tmp_path, count=1)

        # First run
        process_photos(input_dir, output_dir, filenames)
        # Second run with no_overwrite
        converted = process_photos(input_dir, output_dir, filenames, no_overwrite=True)
        assert converted == 0

    def test_format_png(self, tmp_path: Path) -> None:
        """Output should use .png extension when format=png."""
        input_dir, output_dir, filenames = self._setup_photos(tmp_path, count=1)
        process_photos(input_dir, output_dir, filenames, fmt="png")
        stem = Path(filenames[0]).stem
        assert (output_dir / f"{stem}.png").exists()

    def test_format_original_keeps_extension(self, tmp_path: Path) -> None:
        """Original format should keep the input file's extension."""
        input_dir, output_dir, filenames = self._setup_photos(tmp_path, count=1)
        process_photos(input_dir, output_dir, filenames, fmt="original")
        # Input is .jpg, output should also be .jpg
        stem = Path(filenames[0]).stem
        assert (output_dir / f"{stem}.jpg").exists()

    def test_missing_file_skipped(self, tmp_path: Path) -> None:
        """Missing input files should be skipped gracefully."""
        input_dir = tmp_path / "photos"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        converted = process_photos(input_dir, output_dir, ["nonexistent.jpg"])
        assert converted == 0
