#!/usr/bin/env python3
"""
Portrait-optimized B&W converter.

Channel mix tuned for skin tones + mild contrast boost + shadow lift.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def portrait_bw(img: Image.Image) -> Image.Image:
    """
    Convert to B&W with portrait-optimized settings.

    - Channel mix: 0.35*R + 0.45*G + 0.20*B (flattering skin tones)
    - Contrast boost: 1.15x
    - Shadow lift: preserves detail in dark areas
    """
    # Convert to numpy array
    rgb = np.array(img.convert("RGB"), dtype=np.float32)

    # Channel mixing for portrait-friendly tones
    # Higher green = smoother skin, balanced red = warm tones
    bw = (
        0.35 * rgb[:, :, 0] +  # Red
        0.45 * rgb[:, :, 1] +  # Green
        0.20 * rgb[:, :, 2]    # Blue
    )

    # Normalize to 0-255
    bw = np.clip(bw, 0, 255)

    # Contrast boost with shadow lift (S-curve approximation)
    # Lift shadows, keep midtones, slight highlight compression
    midpoint = 128
    contrast = 1.15
    shadow_lift = 10  # Lift darkest values slightly

    # Apply contrast around midpoint
    bw = (bw - midpoint) * contrast + midpoint

    # Shadow lift: smoothly raise dark values
    shadow_mask = np.clip(1 - (bw / 80), 0, 1)  # Affects values < 80
    bw = bw + shadow_lift * shadow_mask

    # Final clip
    bw = np.clip(bw, 0, 255).astype(np.uint8)

    return Image.fromarray(bw, mode="L")


def process_photos(
    input_dir: Path,
    output_dir: Path,
    filenames: list[str],
) -> None:
    """Process a list of photos."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        input_path = input_dir / filename
        if not input_path.exists():
            print(f"  SKIP: {filename} (not found)")
            continue

        try:
            img = Image.open(input_path)
            bw_img = portrait_bw(img)

            # Save as JPEG with high quality
            output_path = output_dir / filename
            bw_img.save(output_path, "JPEG", quality=95)
            print(f"  OK: {filename}")
        except Exception as e:
            print(f"  ERROR: {filename} - {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert photos to portrait-optimized B&W"
    )
    parser.add_argument(
        "-n", "--number",
        type=int,
        default=None,
        help="Convert top N photos by score (default: all)",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=0,
        help="Minimum score threshold (default: 0)",
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        default=Path("photos"),
        help="Input directory (default: photos/)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("photos_bw"),
        help="Output directory (default: photos_bw/)",
    )
    parser.add_argument(
        "-r", "--results",
        type=Path,
        default=Path("results.json"),
        help="Results file from bw_scorer.py (default: results.json)",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific files to convert (overrides results.json)",
    )

    args = parser.parse_args()

    # Get file list
    if args.files:
        filenames = args.files
    elif args.results.exists():
        with open(args.results) as f:
            results = json.load(f)

        # Filter by minimum score
        if args.min_score > 0:
            results = [r for r in results if r["score"] >= args.min_score]

        # Sort by score descending and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        if args.number:
            results = results[:args.number]

        filenames = [r["filename"] for r in results]
    else:
        print(f"Error: {args.results} not found", file=sys.stderr)
        print("Run bw_scorer.py first to generate scores.", file=sys.stderr)
        sys.exit(1)

    if not filenames:
        print("No photos to convert.", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {len(filenames)} photos to B&W...")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print()

    process_photos(args.input_dir, args.output_dir, filenames)

    print()
    print(f"Done. Output in {args.output_dir}/")


if __name__ == "__main__":
    main()
