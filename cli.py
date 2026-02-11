#!/usr/bin/env python3
"""
Unified CLI for B&W photo evaluation and conversion.

Subcommands:
  score   — Analyze photos and write results.json
  convert — Convert top-scoring photos to B&W
  report  — Print score distribution summary
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from pathlib import Path

from bw_scorer import IMAGE_EXTENSIONS, PhotoResult, ScoringConfig, score_photos
from convert_bw import STYLE_NAMES, process_photos

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
        scene = r["details"].get("scene_type", "")
        sat_label = "Low sat" if b["saturation"] >= 50 else "High sat"
        logger.info(
            "%s: %d/100 [%s]  C:%d T:%d %s:%d Comp:%d M:%d",
            r["filename"], r["score"], scene,
            b["contrast"], b["texture"], sat_label, b["saturation"],
            b["composition"], b["metadata"],
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


def cmd_convert(args: argparse.Namespace) -> None:
    """Convert scored photos to B&W."""
    raw_results = _load_results(args.results)

    # Apply filters
    filtered: list[dict] = []
    for r in raw_results:
        if r["score"] < args.min_score:
            continue
        b = r.get("breakdown", {})
        if b.get("contrast", 0) < args.min_contrast:
            continue
        if b.get("texture", 0) < args.min_texture:
            continue
        if b.get("saturation", 0) < args.min_saturation:
            continue
        if b.get("composition", 0) < args.min_composition:
            continue
        if b.get("metadata", 0) < args.min_metadata:
            continue
        filtered.append(r)

    filtered.sort(key=lambda r: r["score"], reverse=True)
    if args.number:
        filtered = filtered[: args.number]

    if not filtered:
        logger.warning("No photos match the given filters.")
        sys.exit(0)

    # Build scene_type lookup
    scene_types = {
        r["filename"]: r.get("details", {}).get("scene_type", "generic")
        for r in filtered
    }

    filenames = [r["filename"] for r in filtered]

    if args.dry_run:
        logger.info("Dry run — %d photos would be converted:", len(filenames))
        for r in filtered:
            scene = r.get("details", {}).get("scene_type", "generic")
            logger.info("  %s: %d/100 [%s]", r["filename"], r["score"], scene)
        return

    logger.info("Converting %d photos to B&W…", len(filenames))
    logger.info("  Input:  %s", args.input_dir)
    logger.info("  Output: %s", args.output_dir)
    logger.info("  Style:  %s", args.style)

    converted = process_photos(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        filenames=filenames,
        style=args.style,
        scene_types=scene_types,
        fmt=args.format,
        quality=args.quality,
        no_overwrite=args.no_overwrite,
    )
    logger.info("Done. %d/%d converted → %s/", converted, len(filenames), args.output_dir)


def cmd_report(args: argparse.Namespace) -> None:
    """Print score distribution summary."""
    results = _load_results(args.results)
    if not results:
        logger.warning("No results to report.")
        return

    scores = [r["score"] for r in results]
    scenes = [r.get("details", {}).get("scene_type", "unknown") for r in results]

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

    # Scene type breakdown
    scene_counts: dict[str, int] = {}
    for s in scenes:
        scene_counts[s] = scene_counts.get(s, 0) + 1
    print(f"\n  Scene types:")
    for scene, count in sorted(scene_counts.items(), key=lambda x: -x[1]):
        print(f"    {scene:<15} {count}")
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bw",
        description="B&W photo evaluation and conversion toolkit",
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

    # --- convert ---
    p_conv = sub.add_parser("convert", help="Convert photos to B&W")
    p_conv.add_argument("-n", "--number", type=int, default=None)
    p_conv.add_argument("--min-score", type=int, default=0)
    p_conv.add_argument(
        "--style",
        choices=["auto", *STYLE_NAMES],
        default="auto",
    )
    p_conv.add_argument("--format", choices=["original", "jpeg", "png", "tiff"], default="original")
    p_conv.add_argument("--quality", type=int, default=95)
    p_conv.add_argument("--dry-run", action="store_true")
    p_conv.add_argument("--no-overwrite", action="store_true")
    p_conv.add_argument("-i", "--input-dir", type=Path, default=Path("photos"))
    p_conv.add_argument("-o", "--output-dir", type=Path, default=Path("photos_bw"))
    p_conv.add_argument("-r", "--results", type=Path, default=Path("results.json"))
    # Per-dimension filters
    p_conv.add_argument("--min-contrast", type=int, default=0)
    p_conv.add_argument("--min-texture", type=int, default=0)
    p_conv.add_argument("--min-saturation", type=int, default=0)
    p_conv.add_argument("--min-composition", type=int, default=0)
    p_conv.add_argument("--min-metadata", type=int, default=0)
    p_conv.add_argument("-v", "--verbose", action="store_true")
    p_conv.add_argument("-q", "--quiet", action="store_true")

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

    # Clamp min-score
    if hasattr(args, "min_score"):
        args.min_score = max(0, min(100, args.min_score))

    commands = {
        "score": cmd_score,
        "convert": cmd_convert,
        "report": cmd_report,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
