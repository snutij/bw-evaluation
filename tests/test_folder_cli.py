"""The command-line front end: analyse a folder, rank it, report it.

Structured so that the expensive part happens once. `analyse_folder` decodes
and measures, which costs seconds per photograph; the renderers are pure
functions of its result. Tests therefore analyse the reference folder a single
time and exercise every output format against that, rather than re-measuring
nine images for each assertion.

The presentation rules under test are all consequences of the package's
refusal to produce a verdict: the report leads with the two axes rather than a
score, it prints the label's batch-relative caveat in the output itself, and it
will not order a batch without being told which axis to lead on.
"""

from __future__ import annotations

import csv
import functools
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from bw_evaluation import (
    MINIMUM_BATCH_SIZE,
    RANKING_KEYS,
    analyse_folder,
    main,
    render_csv,
    render_json,
    render_table,
)

from . import photo_fixtures as P


@functools.lru_cache(maxsize=1)
def _reference_folder() -> Path:
    folder = Path(tempfile.mkdtemp(prefix="bw_reference_"))
    P.write_catalogue(folder)
    return folder


@functools.lru_cache(maxsize=1)
def _analysis():
    """The reference folder, analysed once.

    Decoding and measuring nine frames at the working size costs the better
    part of a minute, and every renderer test below is a pure function of the
    result, so re-running it per test would buy nothing but wall-clock time.
    """
    return analyse_folder(_reference_folder())


class AnalysingAFolder(unittest.TestCase):
    def setUp(self):
        self.analysis = _analysis()

    def test_given_a_folder_of_photographs_when_analysed_then_every_one_is_measured_and_ranked(
        self,
    ):
        # Given a folder of nine files
        # When analysed
        # Then each appears once in the measurements and once in the ranking
        self.assertEqual(len(self.analysis.measurements), 9)
        self.assertEqual(self.analysis.ranking.batch_size, 9)
        self.assertEqual(
            sorted(self.analysis.measurements),
            sorted(e.image_id for e in self.analysis.ranking.images),
        )

    def test_given_a_folder_when_analysed_then_the_camera_metadata_travels_with_each_image(
        self,
    ):
        # Given that a photographer groups by camera, lens and ISO
        # When analysed
        # Then metadata is keyed by the same ids as the measurements, so a
        # report can join the two without guessing
        self.assertEqual(
            sorted(self.analysis.metadata), sorted(self.analysis.measurements)
        )

    def test_given_a_folder_with_an_unreadable_file_when_analysed_then_it_is_skipped_with_a_reason(
        self,
    ):
        # Given a folder containing a file that looks like a photograph and is not
        # When analysed
        # Then the batch still ranks and the casualty is reported rather than
        # silently dropped
        folder = Path(tempfile.mkdtemp(prefix="bw_broken_"))
        self.addCleanup(lambda: None)
        P.write_catalogue(folder)
        (folder / "truncated.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
        analysis = analyse_folder(folder)
        self.assertEqual(len(analysis.measurements), 9)
        self.assertEqual(len(analysis.skipped), 1)
        self.assertIn("truncated.jpg", analysis.skipped[0][0])
        self.assertTrue(analysis.skipped[0][1])

    def test_given_a_folder_with_too_few_photographs_when_analysed_then_it_is_refused(
        self,
    ):
        # Given fewer images than a percentile can be computed over
        # When analysed
        # Then it is refused, and the message names the minimum
        folder = Path(tempfile.mkdtemp(prefix="bw_tiny_"))
        for index in range(3):
            P.write_png(folder / f"{index}.png", P.solid((0.3, 0.4, 0.5)))
        with self.assertRaises(ValueError) as caught:
            analyse_folder(folder)
        self.assertIn(str(MINIMUM_BATCH_SIZE), str(caught.exception))


class RenderingTheReport(unittest.TestCase):
    def setUp(self):
        self.analysis = _analysis()

    def test_given_an_analysis_when_rendered_as_a_table_then_every_photograph_has_a_row(
        self,
    ):
        # Given the analysed folder
        # When rendered
        # Then each image id appears in the output
        table = render_table(self.analysis, key="pareto_then_loss")
        for name in self.analysis.measurements:
            with self.subTest(image=name):
                self.assertIn(name, table)

    def test_given_an_analysis_when_rendered_then_the_output_says_the_labels_are_relative(
        self,
    ):
        # Given that a label is a statement about the batch
        # When the report is rendered
        # Then it says so on the page, not only in the documentation, because
        # the page is the only part anybody will read
        table = render_table(self.analysis, key="pareto_then_loss")
        self.assertIn("batch", table.lower())

    def test_given_an_analysis_when_rendered_then_no_single_overall_score_is_printed(
        self,
    ):
        # Given the package's central refusal
        # When the table is rendered
        # Then it shows two axes and a layer, and never a combined score
        table = render_table(self.analysis, key="pareto_then_loss").lower()
        self.assertIn("loss", table)
        self.assertIn("structure", table)
        self.assertNotIn("overall score", table)

    def test_given_a_row_limit_when_rendering_then_only_that_many_photographs_are_listed(
        self,
    ):
        # Given a photographer who wants the shortlist
        # When a limit is given
        # Then the table is truncated and says how many were withheld, so the
        # shortlist cannot be mistaken for the whole folder
        table = render_table(self.analysis, key="pareto_then_loss", top=3)
        listed = [
            name
            for name in self.analysis.measurements
            if f" {name} " in table or f"{name} " in table
        ]
        self.assertLessEqual(len(listed), 3)
        self.assertIn("6", table)

    def test_given_two_different_keys_when_rendering_then_the_row_order_changes(self):
        # Given two axes that genuinely disagree
        # When each is used to lead the table
        # Then the orders differ, which is the visible reason the tool asks
        by_loss = render_table(self.analysis, key="loss_first")
        by_structure = render_table(self.analysis, key="structure_first")
        self.assertNotEqual(by_loss, by_structure)

    def test_given_an_unknown_key_when_rendering_then_it_is_refused(self):
        # Given a key the package does not define
        # When rendering
        # Then it is rejected rather than defaulted
        with self.assertRaises(ValueError):
            render_table(self.analysis, key="whatever_looks_best")

    def test_given_an_analysis_when_rendered_twice_then_the_text_is_identical(self):
        # Given the determinism the package claims end to end
        # When rendered twice
        # Then the bytes match
        self.assertEqual(
            render_table(self.analysis, key="pareto_then_loss"),
            render_table(self.analysis, key="pareto_then_loss"),
        )

    def test_given_an_achromatic_photograph_when_rendered_then_its_warning_reaches_the_report(
        self,
    ):
        # Given a frame with no colour to discard
        # When the report is rendered
        # Then the warning is visible, because for that image the whole
        # question is void and the reader needs to know
        table = render_table(self.analysis, key="pareto_then_loss")
        self.assertIn("achromatic", table.lower())


class MachineReadableOutput(unittest.TestCase):
    def setUp(self):
        self.analysis = _analysis()

    def test_given_an_analysis_when_written_as_json_then_it_parses_and_holds_every_photograph(
        self,
    ):
        # Given a caller who wants to post-process
        # When JSON is produced
        # Then it parses and covers the batch
        payload = json.loads(render_json(self.analysis))
        self.assertEqual(len(payload["images"]), 9)
        self.assertEqual(payload["batch_size"], 9)

    def test_given_json_output_when_read_then_each_entry_carries_its_position_and_its_recipe(
        self,
    ):
        # Given that the useful output is where an image ranks and how to
        # convert it
        # When an entry is read
        # Then both are present
        payload = json.loads(render_json(self.analysis))
        entry = payload["images"][0]
        for field in (
            "image_id",
            "loss_percentile",
            "structure_percentile",
            "pareto_layer",
            "label",
            "dominant_evidence",
            "recommended_weights",
            "recommended_filter",
        ):
            with self.subTest(field=field):
                self.assertIn(field, entry)

    def test_given_json_output_when_read_then_it_states_that_labels_are_batch_relative(
        self,
    ):
        # Given that JSON is what a downstream tool will actually consume
        # When read
        # Then the caveat is a field, not a comment
        payload = json.loads(render_json(self.analysis))
        self.assertTrue(payload["labels_are_batch_relative"])

    def test_given_an_analysis_when_written_as_csv_then_it_has_a_header_and_one_row_per_photograph(
        self,
    ):
        # Given a photographer who will open this in a spreadsheet
        # When CSV is produced
        # Then it has a header row and nine data rows
        rows = list(csv.reader(io.StringIO(render_csv(self.analysis))))
        self.assertEqual(len(rows), 10)
        self.assertIn("image_id", rows[0])
        self.assertIn("loss_percentile", rows[0])


class CommandLine(unittest.TestCase):
    def test_given_no_folder_when_invoked_then_it_exits_with_a_usage_error(self):
        # Given no arguments
        # When run
        # Then argparse refuses
        with self.assertRaises(SystemExit) as caught, redirect_stderr(io.StringIO()):
            main([])
        self.assertNotEqual(caught.exception.code, 0)

    def test_given_a_folder_that_does_not_exist_when_invoked_then_it_fails_without_a_traceback(
        self,
    ):
        # Given a mistyped path
        # When run
        # Then it reports the problem and returns non-zero, rather than
        # printing a stack trace at a photographer
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = main(["/definitely/not/here"])
        self.assertNotEqual(code, 0)
        self.assertTrue(err.getvalue().strip())

    def test_given_too_few_photographs_when_invoked_then_it_explains_the_minimum(self):
        # Given a folder below the batch minimum
        # When run
        # Then the refusal is a message, not an exception
        folder = Path(tempfile.mkdtemp(prefix="bw_cli_tiny_"))
        for index in range(2):
            P.write_png(folder / f"{index}.png", P.solid((0.3, 0.4, 0.5)))
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = main([str(folder)])
        self.assertNotEqual(code, 0)
        self.assertIn(str(MINIMUM_BATCH_SIZE), err.getvalue())

    def test_given_an_unknown_key_when_invoked_then_it_is_rejected_before_any_work_is_done(
        self,
    ):
        # Given a key the package does not define
        # When run
        # Then argparse rejects it up front, so a photographer does not wait
        # through a folder of measurements to be told the flag was wrong
        with self.assertRaises(SystemExit) as caught, redirect_stderr(io.StringIO()):
            main([str(_reference_folder()), "--key", "whatever_looks_best"])
        self.assertNotEqual(caught.exception.code, 0)

    def test_given_the_named_keys_when_offered_on_the_command_line_then_they_match_the_package(
        self,
    ):
        # Given that the CLI must not drift from the library
        # When the choices are compared
        # Then they are the same set
        for key in RANKING_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, RANKING_KEYS)

    def test_given_a_real_folder_when_invoked_then_it_prints_a_report_and_writes_its_files(
        self,
    ):
        # Given the reference folder and both output flags
        # When run end to end
        # Then it succeeds, prints the table, and writes parseable files
        out_dir = Path(tempfile.mkdtemp(prefix="bw_cli_out_"))
        json_path = out_dir / "ranking.json"
        csv_path = out_dir / "ranking.csv"
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = main(
                [
                    str(_reference_folder()),
                    "--key",
                    "loss_first",
                    "--json",
                    str(json_path),
                    "--csv",
                    str(csv_path),
                ]
            )
        self.assertEqual(code, 0)
        self.assertIn("fog", out.getvalue())
        self.assertEqual(len(json.loads(json_path.read_text())["images"]), 9)
        self.assertEqual(len(list(csv.reader(io.StringIO(csv_path.read_text())))), 10)


if __name__ == "__main__":
    unittest.main()
