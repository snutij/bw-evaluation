"""Tests for report_html.py"""

import argparse
import cv2
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from report_html import generate_html_report
from cli import build_parser, cmd_report_html


def _make_results(n: int, base_score: int = 50) -> list[dict]:
    return [
        {
            "filename": f"photo_{i:02d}.jpg",
            "score": base_score + i,
            "breakdown": {
                "contrast": 40 + i,
                "texture": 30 + i,
                "saturation": 60,
                "composition": 20,
                "channel_separation": 50 + i,
            },
        }
        for i in range(n)
    ]


def _make_photo(tmp_path: Path, filename: str) -> Path:
    path = tmp_path / filename
    img = np.full((200, 300, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


class TestGenerateHtmlReport:
    def test_creates_html_file(self, tmp_path: Path) -> None:
        """generate_html_report should write a .html file."""
        _make_photo(tmp_path, "photo_00.jpg")
        results = _make_results(1)
        output = tmp_path / "report.html"
        generate_html_report(results, tmp_path, output)
        assert output.exists()

    def test_html_is_self_contained(self, tmp_path: Path) -> None:
        """Output should be a single file with no external href/src refs."""
        _make_photo(tmp_path, "photo_00.jpg")
        results = _make_results(1)
        output = tmp_path / "report.html"
        generate_html_report(results, tmp_path, output)
        html = output.read_text()
        assert "http://" not in html
        assert "https://" not in html
        assert "stylesheet" not in html  # no external CSS link

    def test_contains_score_and_filename(self, tmp_path: Path) -> None:
        """Each photo card should contain its filename and score."""
        _make_photo(tmp_path, "photo_00.jpg")
        results = _make_results(1)
        output = tmp_path / "report.html"
        generate_html_report(results, tmp_path, output)
        html = output.read_text()
        assert "photo_00.jpg" in html
        assert str(results[0]["score"]) in html

    def test_contains_base64_thumbnail(self, tmp_path: Path) -> None:
        """Should embed photo as base64 data URI."""
        _make_photo(tmp_path, "photo_00.jpg")
        results = _make_results(1)
        output = tmp_path / "report.html"
        generate_html_report(results, tmp_path, output)
        html = output.read_text()
        assert "data:image/jpeg;base64," in html

    def test_sorted_by_score_descending(self, tmp_path: Path) -> None:
        """Photos should appear in descending score order."""
        for i in range(3):
            _make_photo(tmp_path, f"photo_{i:02d}.jpg")
        results = _make_results(3)
        output = tmp_path / "report.html"
        generate_html_report(results, tmp_path, output)
        html = output.read_text()
        # photo_02 has highest score (52), photo_00 has lowest (50)
        pos_high = html.index("photo_02.jpg")
        pos_low = html.index("photo_00.jpg")
        assert pos_high < pos_low

    def test_returns_count(self, tmp_path: Path) -> None:
        """Should return the number of cards written."""
        for i in range(3):
            _make_photo(tmp_path, f"photo_{i:02d}.jpg")
        results = _make_results(3)
        output = tmp_path / "report.html"
        n = generate_html_report(results, tmp_path, output)
        assert n == 3


class TestMissingPhotoFallback:
    def test_missing_photo_renders_placeholder(self, tmp_path: Path) -> None:
        """Missing photo should show placeholder, not crash."""
        results = [{"filename": "missing.jpg", "score": 70, "breakdown": {}}]
        output = tmp_path / "report.html"
        generate_html_report(results, tmp_path, output)
        html = output.read_text()
        assert "missing.jpg" in html
        assert "not found" in html

    def test_missing_photo_no_base64(self, tmp_path: Path) -> None:
        """Placeholder should not contain a base64 data URI."""
        results = [{"filename": "ghost.jpg", "score": 30, "breakdown": {}}]
        output = tmp_path / "report.html"
        generate_html_report(results, tmp_path, output)
        html = output.read_text()
        assert "data:image/jpeg;base64," not in html


class TestMaxPhotos:
    def test_max_photos_limits_output(self, tmp_path: Path) -> None:
        """--max-photos should include only the top N by score."""
        for i in range(5):
            _make_photo(tmp_path, f"photo_{i:02d}.jpg")
        results = _make_results(5)
        output = tmp_path / "report.html"
        n = generate_html_report(results, tmp_path, output, max_photos=3)
        assert n == 3

    def test_max_photos_keeps_highest_scores(self, tmp_path: Path) -> None:
        """Top N should be the photos with the highest scores."""
        for i in range(4):
            _make_photo(tmp_path, f"photo_{i:02d}.jpg")
        results = _make_results(4)
        output = tmp_path / "report.html"
        generate_html_report(results, tmp_path, output, max_photos=2)
        html = output.read_text()
        # photo_02 (score 52) and photo_03 (score 53) should be present
        assert "photo_03.jpg" in html
        assert "photo_02.jpg" in html
        # photo_00 (score 50) should be absent
        assert "photo_00.jpg" not in html


class TestCliReportHtml:
    def test_parser_report_html_defaults(self) -> None:
        """report-html subcommand should parse with correct defaults."""
        parser = build_parser()
        args = parser.parse_args(["report-html"])
        assert args.command == "report-html"
        assert args.results == Path("results.json")
        assert args.input_dir == Path("photos")
        assert args.output == "report.html"
        assert args.max_photos is None

    def test_parser_max_photos(self) -> None:
        """--max-photos should be parsed as int."""
        parser = build_parser()
        args = parser.parse_args(["report-html", "--max-photos", "25"])
        assert args.max_photos == 25

    def test_cmd_report_html_writes_file(self, tmp_path: Path) -> None:
        """cmd_report_html should generate the output file."""
        import json
        for i in range(2):
            _make_photo(tmp_path, f"photo_{i:02d}.jpg")
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(_make_results(2)))
        output_path = tmp_path / "out.html"

        args = argparse.Namespace(
            results=results_path,
            input_dir=tmp_path,
            output=str(output_path),
            max_photos=None,
            verbose=False,
            quiet=False,
        )
        cmd_report_html(args)
        assert output_path.exists()
