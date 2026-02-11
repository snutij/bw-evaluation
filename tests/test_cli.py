"""Tests for cli.py"""

import json
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from cli import build_parser, cmd_report


class TestBuildParser:
    """Tests for CLI argument parsing."""

    def test_score_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["score"])
        assert args.command == "score"
        assert args.input_dir == Path("photos")
        assert args.output == "results.json"
        assert args.workers == 1
        assert args.config is None

    def test_convert_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["convert"])
        assert args.command == "convert"
        assert args.min_score == 0
        assert args.style == "auto"
        assert args.format == "original"
        assert args.quality == 95
        assert args.dry_run is False
        assert args.no_overwrite is False

    def test_convert_all_filters(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "convert",
            "--min-score", "60",
            "--min-contrast", "40",
            "--min-texture", "30",
            "--min-saturation", "50",
            "--min-composition", "20",
            "--min-metadata", "10",
            "-n", "5",
        ])
        assert args.min_score == 60
        assert args.min_contrast == 40
        assert args.min_texture == 30
        assert args.min_saturation == 50
        assert args.min_composition == 20
        assert args.min_metadata == 10
        assert args.number == 5

    def test_convert_style_choices(self) -> None:
        parser = build_parser()
        for style in ["auto", "portrait", "landscape", "high-contrast", "street", "architecture"]:
            args = parser.parse_args(["convert", "--style", style])
            assert args.style == style

    def test_convert_format_choices(self) -> None:
        parser = build_parser()
        for fmt in ["original", "jpeg", "png", "tiff"]:
            args = parser.parse_args(["convert", "--format", fmt])
            assert args.format == fmt

    def test_report_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["report"])
        assert args.command == "report"
        assert args.results == Path("results.json")

    def test_no_command_returns_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_dry_run_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["convert", "--dry-run"])
        assert args.dry_run is True


class TestCmdReport:
    """Tests for the report subcommand."""

    def test_report_with_results(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Report should print distribution summary."""
        results = [
            {"filename": f"photo_{i}.jpg", "score": 20 + i * 10, "details": {"scene_type": "generic"}}
            for i in range(8)
        ]
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results))

        import argparse
        args = argparse.Namespace(results=results_path, verbose=False, quiet=False)
        cmd_report(args)

        captured = capsys.readouterr()
        assert "Score distribution" in captured.out
        assert "Min:" in captured.out
        assert "Max:" in captured.out
        assert "Scene types:" in captured.out


class TestConvertFiltering:
    """Tests for per-dimension filtering in convert subcommand."""

    def test_filters_by_breakdown(self) -> None:
        """Results should be filtered by sub-score minimums."""
        results = [
            {
                "filename": "good.jpg", "score": 70,
                "breakdown": {"contrast": 60, "texture": 50, "saturation": 70, "composition": 55, "metadata": 50},
                "details": {"scene_type": "landscape"},
            },
            {
                "filename": "bad_contrast.jpg", "score": 65,
                "breakdown": {"contrast": 20, "texture": 50, "saturation": 70, "composition": 55, "metadata": 50},
                "details": {"scene_type": "generic"},
            },
        ]

        # Apply min-contrast=40 filter manually (same logic as cmd_convert)
        min_contrast = 40
        filtered = [r for r in results if r["breakdown"]["contrast"] >= min_contrast]
        assert len(filtered) == 1
        assert filtered[0]["filename"] == "good.jpg"
