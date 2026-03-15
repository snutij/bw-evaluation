"""Tests for cli.py."""

import json
from pathlib import Path

import pytest

from bw_evaluation.cli import build_parser, cmd_report


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

    def test_report_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["report"])
        assert args.command == "report"
        assert args.results == Path("results.json")

    def test_no_command_returns_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_no_convert_subcommand(self) -> None:
        """Convert subcommand should no longer exist."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["convert"])


class TestCmdReport:
    """Tests for the report subcommand."""

    def test_report_with_results(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Report should print distribution summary."""
        results = [
            {
                "filename": f"photo_{i}.jpg",
                "score": 20 + i * 10,
                "breakdown": {"channel_separation": i * 5},
            }
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
        assert "Channel separation:" in captured.out
