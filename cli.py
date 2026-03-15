"""Unified CLI for B&W photo evaluation.

Subcommands:
  score   — Analyze photos and write results.json
  report  — Print score distribution summary
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any, cast

from bw_scorer import IMAGE_EXTENSIONS, score_photos
from config import ScoringConfig
from report_html import generate_html_report

logger = logging.getLogger("bw")

# ── CLI display constants ─────────────────────────────────────────────────────
_SATURATION_MIDPOINT: int = 50  # saturation score below this is considered "high sat"
_MIN_SCORES_FOR_STDEV: int = 2  # minimum photo count required to compute stdev
_MAX_SCORE: int = 100  # score ceiling used in histogram bucket labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure_logging(*, verbose: bool = False, quiet: bool = False) -> None:
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=level,
    )


def _load_results(path: Path) -> list[dict[str, Any]]:
    """Load and return results JSON; exits with error if file is missing."""
    if not path.exists():
        logger.error("%s not found — run 'score' first.", path)
        sys.exit(1)
    with path.open() as f:
        return cast("list[dict[str, Any]]", json.load(f))


def _find_photos(directory: Path) -> list[Path]:
    if not directory.exists():
        logger.error("Directory not found: %s", directory)
        sys.exit(1)
    photos = sorted(
        f for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not photos:
        logger.error("No photos found in %s", directory)
        sys.exit(1)
    return photos


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_score(args: argparse.Namespace) -> None:
    """Score photos for B&W potential."""
    cfg = ScoringConfig.from_file(args.config) if args.config else ScoringConfig()
    photos = _find_photos(args.input_dir)

    logger.info("Analyzing %d photos…", len(photos))
    results = score_photos(photos, cfg=cfg, workers=args.workers, quiet=args.quiet)

    # Sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)

    # Print results
    for r in results:
        b = r["breakdown"]
        sat_label = "Low sat" if b["saturation"] >= _SATURATION_MIDPOINT else "High sat"
        logger.info(
            "%s: %d/100  C:%d T:%d %s:%d Comp:%d CS:%d",
            r["filename"],
            r["score"],
            b["contrast"],
            b["texture"],
            sat_label,
            b["saturation"],
            b["composition"],
            b["channel_separation"],
        )

    # Write JSON
    output = Path(args.output)
    with output.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s (%d photos)", output, len(results))

    # Quick top-5
    top = results[:5]
    if top:
        logger.info("Top %d:", len(top))
        for i, r in enumerate(top, 1):
            logger.info("  %d. %s: %d/100", i, r["filename"], r["score"])


def cmd_report_html(args: argparse.Namespace) -> None:
    """Generate self-contained HTML report with photo thumbnails."""
    results = _load_results(args.results)
    if not results:
        logger.warning("No results to report.")
        return

    photos_dir = args.input_dir
    if not photos_dir.exists():
        logger.error("Photos directory not found: %s", photos_dir)
        sys.exit(1)

    output = Path(args.output)
    n = generate_html_report(
        results,
        photos_dir=photos_dir,
        output_path=output,
        thumbnail_width=300,
        max_photos=args.max_photos,
        quiet=args.quiet,
    )
    logger.info("Report written to %s (%d photos)", output, n)


def cmd_report(args: argparse.Namespace) -> None:
    """Print score distribution summary."""
    results = _load_results(args.results)
    if not results:
        logger.warning("No results to report.")
        return

    scores = [r["score"] for r in results]
    chan_seps = [r.get("breakdown", {}).get("channel_separation", 0) for r in results]

    print(f"\n{'=' * 50}")
    print(f" Score distribution — {len(scores)} photos")
    print(f"{'=' * 50}")
    print(f"  Min:    {min(scores)}")
    print(f"  Max:    {max(scores)}")
    print(f"  Mean:   {statistics.mean(scores):.1f}")
    print(f"  Median: {statistics.median(scores):.1f}")
    if len(scores) >= _MIN_SCORES_FOR_STDEV:
        print(f"  Stdev:  {statistics.stdev(scores):.1f}")

    # Percentile buckets
    buckets = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]
    print(f"\n  {'Range':<10} {'Count':>6} {'Bar'}")
    for lo, hi in buckets:
        count = sum(1 for s in scores if lo <= s < hi)
        bar = "#" * count
        label = f"{lo}-{hi - 1}" if hi <= _MAX_SCORE else f"{lo}-100"
        print(f"  {label:<10} {count:>6}  {bar}")

    # Channel separation stats
    if chan_seps:
        print("\n  Channel separation:")
        print(f"    Mean:   {statistics.mean(chan_seps):.1f}")
        print(f"    Median: {statistics.median(chan_seps):.1f}")
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the bw CLI."""
    parser = argparse.ArgumentParser(
        prog="bw",
        description="B&W photo evaluation toolkit",
    )
    sub = parser.add_subparsers(dest="command")

    # --- score ---
    p_score = sub.add_parser("score", help="Score photos for B&W potential")
    p_score.add_argument("-i", "--input-dir", type=Path, default=Path("photos"))
    p_score.add_argument("-o", "--output", default="results.json")
    p_score.add_argument("-w", "--workers", type=int, default=1)
    p_score.add_argument(
        "--config", type=Path, default=None, help="JSON config override"
    )
    p_score.add_argument("-v", "--verbose", action="store_true")
    p_score.add_argument("-q", "--quiet", action="store_true")

    # --- report-html ---
    p_rhtml = sub.add_parser(
        "report-html", help="Generate visual HTML report with thumbnails"
    )
    p_rhtml.add_argument("-r", "--results", type=Path, default=Path("results.json"))
    p_rhtml.add_argument("-i", "--input-dir", type=Path, default=Path("photos"))
    p_rhtml.add_argument("-o", "--output", default="report.html")
    p_rhtml.add_argument(
        "--max-photos", type=int, default=None, help="Limit to top N photos"
    )
    p_rhtml.add_argument("-v", "--verbose", action="store_true")
    p_rhtml.add_argument("-q", "--quiet", action="store_true")

    # --- report ---
    p_report = sub.add_parser("report", help="Score distribution summary")
    p_report.add_argument("-r", "--results", type=Path, default=Path("results.json"))
    p_report.add_argument("-v", "--verbose", action="store_true")
    p_report.add_argument("-q", "--quiet", action="store_true")

    return parser


def main() -> None:
    """Entry point for the bw CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    _configure_logging(
        verbose=getattr(args, "verbose", False),
        quiet=getattr(args, "quiet", False),
    )

    commands = {
        "score": cmd_score,
        "report-html": cmd_report_html,
        "report": cmd_report,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
