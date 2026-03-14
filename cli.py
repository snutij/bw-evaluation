#!/usr/bin/env python3
"""
Unified CLI for B&W photo evaluation.

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

from bw_scorer import IMAGE_EXTENSIONS, ScoringConfig, score_photos

logger = logging.getLogger("bw")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=level,
    )


def _load_results(path: Path) -> list[dict]:
    if not path.exists():
        logger.error("%s not found — run 'score' first.", path)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


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
    results = score_photos(photos, cfg=cfg, workers=args.workers)

    # Sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)

    # Print results
    for r in results:
        b = r["breakdown"]
        sat_label = "Low sat" if b["saturation"] >= 50 else "High sat"
        logger.info(
            "%s: %d/100  C:%d T:%d %s:%d Comp:%d CS:%d",
            r["filename"], r["score"],
            b["contrast"], b["texture"], sat_label, b["saturation"],
            b["composition"], b["channel_separation"],
        )

    # Write JSON
    output = Path(args.output)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s (%d photos)", output, len(results))

    # Quick top-5
    top = results[:5]
    if top:
        logger.info("Top %d:", len(top))
        for i, r in enumerate(top, 1):
            logger.info("  %d. %s: %d/100", i, r["filename"], r["score"])


def cmd_report(args: argparse.Namespace) -> None:
    """Print score distribution summary."""
    results = _load_results(args.results)
    if not results:
        logger.warning("No results to report.")
        return

    scores = [r["score"] for r in results]
    chan_seps = [r.get("breakdown", {}).get("channel_separation", 0) for r in results]

    print(f"\n{'='*50}")
    print(f" Score distribution — {len(scores)} photos")
    print(f"{'='*50}")
    print(f"  Min:    {min(scores)}")
    print(f"  Max:    {max(scores)}")
    print(f"  Mean:   {statistics.mean(scores):.1f}")
    print(f"  Median: {statistics.median(scores):.1f}")
    if len(scores) >= 2:
        print(f"  Stdev:  {statistics.stdev(scores):.1f}")

    # Percentile buckets
    buckets = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]
    print(f"\n  {'Range':<10} {'Count':>6} {'Bar'}")
    for lo, hi in buckets:
        count = sum(1 for s in scores if lo <= s < hi)
        bar = "#" * count
        label = f"{lo}-{hi - 1}" if hi <= 100 else f"{lo}-100"
        print(f"  {label:<10} {count:>6}  {bar}")

    # Channel separation stats
    if chan_seps:
        print(f"\n  Channel separation:")
        print(f"    Mean:   {statistics.mean(chan_seps):.1f}")
        print(f"    Median: {statistics.median(chan_seps):.1f}")
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
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
    p_score.add_argument("--config", type=Path, default=None, help="JSON config override")
    p_score.add_argument("-v", "--verbose", action="store_true")
    p_score.add_argument("-q", "--quiet", action="store_true")

    # --- report ---
    p_report = sub.add_parser("report", help="Score distribution summary")
    p_report.add_argument("-r", "--results", type=Path, default=Path("results.json"))
    p_report.add_argument("-v", "--verbose", action="store_true")
    p_report.add_argument("-q", "--quiet", action="store_true")

    return parser


def main() -> None:
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
        "report": cmd_report,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
