"""Point it at a folder, get a ranked shortlist. Analysis and presentation kept apart.

`analyse_folder` does the expensive work once and returns evidence.
`render_table`, `render_json` and `render_csv` are pure functions of that
result. The split is not only for testing: a caller who wants to re-sort a
ranked folder on a different axis should not have to re-measure it.

Three presentation rules, all consequences of the package's central refusal.

The table shows two axes and a Pareto layer, never a combined score. There is
no defensible way to weigh "how little was lost" against "how much survives",
so the report puts both in front of the reader and lets them decide.

`--key` has no default that hides a weighting. `pareto_then_loss` leads with
the layer, which is the ordering that needs no weighting at all; the other two
are explicit about which axis they privilege.

The batch-relative caveat is printed on the page, not left in documentation.
The page is the only part anyone reads, and a label that looks absolute is the
one way this tool could mislead someone.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .conventions import ColorConvention
from .decolorization import SEARCH_TAU_SWEEP
from .loading import (
    PhotographMetadata,
    discover_photographs,
    load_photograph,
    unique_image_ids,
)
from .measurement import CandidacyMeasurement, measure_bw_candidacy
from .ranking import (
    MINIMUM_BATCH_SIZE,
    RANKING_KEYS,
    BatchRanking,
    rank_batch,
    total_order,
)

#: Default convention for files read off disk. Loading has already converted
#: everything to sRGB, and sRGB is gamma-encoded, so this is a statement of
#: fact about the pixels rather than a preference.
DEFAULT_CONVENTION = ColorConvention(encoding="gamma")

#: Default threshold sweep. Rankings flip with tau, which is why a sweep is
#: mandatory rather than a single value.
DEFAULT_SWEEP: tuple[float, ...] = tuple(float(tau) for tau in SEARCH_TAU_SWEEP)

_CAVEAT = (
    "Percentiles and labels are positions within THIS batch of {count}. "
    "The same photograph in different company gets different numbers. "
    "Nothing here is an absolute verdict; there is no published ground truth to calibrate one."
)


@dataclass(frozen=True)
class FolderAnalysis:
    """Everything measured about one folder, before any presentation choice."""

    folder: str
    ranking: BatchRanking
    measurements: Mapping[str, CandidacyMeasurement]
    metadata: Mapping[str, PhotographMetadata]
    skipped: tuple[tuple[str, str], ...]


def analyse_folder(
    folder: Path | str,
    *,
    recursive: bool = False,
    convention: ColorConvention = DEFAULT_CONVENTION,
    taus: tuple[float, ...] = DEFAULT_SWEEP,
) -> FolderAnalysis:
    """Decode, measure and rank every photograph in a folder.

    A file that fails to decode is recorded in `skipped` rather than aborting
    the run. A folder of a few hundred frames reliably contains one truncated
    export, and losing the whole batch to it would be the wrong trade.
    """
    root = Path(folder)
    paths = discover_photographs(root, recursive=recursive)
    identifiers = unique_image_ids(paths, root=root)

    measurements: dict[str, CandidacyMeasurement] = {}
    metadata: dict[str, PhotographMetadata] = {}
    skipped: list[tuple[str, str]] = []
    for path in paths:
        name = identifiers[path]
        try:
            photograph = load_photograph(path)
            measurements[name] = measure_bw_candidacy(
                photograph.rgb, convention=convention, taus=taus
            )
            metadata[name] = photograph.metadata
        except Exception as failure:  # noqa: BLE001 - one bad photograph must not abort a batch
            skipped.append((str(path), f"{type(failure).__name__}: {failure}"))

    if len(measurements) < MINIMUM_BATCH_SIZE:
        raise ValueError(
            f"found {len(measurements)} readable photographs in {root}, and ranking needs at "
            f"least {MINIMUM_BATCH_SIZE}. Every number this tool produces is a position within "
            "the batch, so a batch this small would describe the sample rather than the pictures."
            + (f" {len(skipped)} file(s) could not be read." if skipped else "")
        )

    return FolderAnalysis(
        folder=str(root),
        ranking=rank_batch(measurements),
        measurements=measurements,
        metadata=metadata,
        skipped=tuple(skipped),
    )


def _ordered(analysis: FolderAnalysis, key: str):
    if key not in RANKING_KEYS:
        raise ValueError(
            f"unknown ranking key {key!r}; expected one of {RANKING_KEYS}. "
            "There is deliberately no default, because the default would be a weighting."
        )
    return total_order(analysis.ranking, key=key)


def render_table(analysis: FolderAnalysis, *, key: str, top: int | None = None) -> str:
    """A readable report. Two axes, a layer, a recipe, and the reason."""
    entries = list(_ordered(analysis, key))
    total = len(entries)
    shown = entries if top is None else entries[:top]
    width = max((len(entry.image_id) for entry in entries), default=8)

    lines: list[str] = [
        f"{analysis.folder}  |  {total} photographs  |  ordered by {key}",
        "",
        # Columns spelled out rather than abbreviated to "struct". The two
        # axes are the whole point of the report, and a reader who has to
        # guess what a column means is a reader who will fall back to
        # treating the label as a score.
        (
            f"{'image'.ljust(width)}  {'loss%':>6}  {'structure%':>10}  {'layer':>5}  "
            f"{'label':<16}  {'recipe':<22}  strongest evidence"
        ),
        "-" * (width + 82),
    ]
    for entry in shown:
        measurement = analysis.measurements[entry.image_id]
        weights = ",".join(f"{value:.2f}" for value in measurement.recommended_weights)
        recipe = f"({weights}) {measurement.recommended_filter}"
        lines.append(
            f"{entry.image_id.ljust(width)}  "
            f"{entry.loss_percentile:6.2f}  {entry.structure_percentile:10.2f}  "
            f"{entry.pareto_layer:5d}  {entry.label:<16}  {recipe:<22}  {entry.dominant_evidence}"
        )
    if top is not None and total > len(shown):
        lines.append(f"... {total - len(shown)} more not shown")

    notes = [
        f"! {name}: {note}"
        for name, measurement in sorted(analysis.measurements.items())
        for note in measurement.warnings
    ]
    if notes:
        lines += ["", *notes]
    if analysis.skipped:
        lines += [
            "",
            *(f"skipped {path}: {reason}" for path, reason in analysis.skipped),
        ]
    lines += ["", _CAVEAT.format(count=total)]
    return "\n".join(lines)


def _rows(analysis: FolderAnalysis) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in sorted(analysis.ranking.images, key=lambda e: e.image_id):
        measurement = analysis.measurements[entry.image_id]
        meta = analysis.metadata[entry.image_id]
        rows.append(
            {
                "image_id": entry.image_id,
                "path": meta.path,
                "loss_percentile": entry.loss_percentile,
                "structure_percentile": entry.structure_percentile,
                "pareto_layer": entry.pareto_layer,
                "label": entry.label,
                "dominant_evidence": entry.dominant_evidence,
                "recommended_method": measurement.recommended_method,
                "recommended_weights": list(measurement.recommended_weights),
                "recommended_filter": measurement.recommended_filter,
                "camera": meta.camera,
                "lens": meta.lens,
                "iso": meta.iso,
                "shutter": meta.shutter,
                "aperture": meta.aperture,
                "focal_length": meta.focal_length,
                "captured_at": meta.captured_at,
                "colour_profile": meta.colour_profile,
                "warnings": list(measurement.warnings),
            }
        )
    return rows


def render_json(analysis: FolderAnalysis) -> str:
    """Machine-readable output, caveat included as a field rather than a comment."""
    payload = {
        "folder": analysis.folder,
        "batch_size": analysis.ranking.batch_size,
        "labels_are_batch_relative": analysis.ranking.labels_are_batch_relative,
        "caveat": _CAVEAT.format(count=analysis.ranking.batch_size),
        "skipped": [
            {"path": path, "reason": reason} for path, reason in analysis.skipped
        ],
        "images": _rows(analysis),
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def render_csv(analysis: FolderAnalysis) -> str:
    """Spreadsheet output. List-valued fields are flattened, not dropped."""
    rows = _rows(analysis)
    flattened = [
        {
            field: (
                "; ".join(str(item) for item in value)
                if isinstance(value, list)
                else value
            )
            for field, value in row.items()
        }
        for row in rows
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(flattened[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(flattened)
    return buffer.getvalue()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bw_evaluation",
        description=(
            "Rank a folder of photographs by how well they would work in black and white. "
            "Positions are relative to the folder you pass in; there are no absolute verdicts."
        ),
    )
    parser.add_argument("folder", help="folder of photographs to rank")
    parser.add_argument(
        "--recursive", action="store_true", help="include photographs in subfolders"
    )
    parser.add_argument(
        "--key",
        choices=RANKING_KEYS,
        default="pareto_then_loss",
        help=(
            "which axis leads the table. pareto_then_loss needs no weighting between the two "
            "axes and is the default for that reason"
        ),
    )
    parser.add_argument(
        "--top", type=int, default=None, help="only print the first N rows"
    )
    parser.add_argument(
        "--json", dest="json_path", default=None, help="also write JSON here"
    )
    parser.add_argument(
        "--csv", dest="csv_path", default=None, help="also write CSV here"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns an exit code rather than raising at a photographer."""
    arguments = _parser().parse_args(argv)
    try:
        analysis = analyse_folder(arguments.folder, recursive=arguments.recursive)
    except (FileNotFoundError, NotADirectoryError, ValueError) as failure:
        print(f"{failure}", file=sys.stderr)
        return 1

    print(render_table(analysis, key=arguments.key, top=arguments.top))
    if arguments.json_path:
        Path(arguments.json_path).write_text(render_json(analysis), encoding="utf-8")
        print(f"\nwrote {arguments.json_path}")
    if arguments.csv_path:
        Path(arguments.csv_path).write_text(render_csv(analysis), encoding="utf-8")
        print(f"wrote {arguments.csv_path}")
    return 0
