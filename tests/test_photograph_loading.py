"""What it means to read a photograph off disk correctly.

Three things this file is really defending.

Colour management. A large share of exported JPEGs are AdobeRGB or Display P3.
Reading those as if they were sRGB inflates every chroma measurement in the
package, which would quietly bias the whole ranking toward wide-gamut exports.

Scale. The package contract is that pixels arrive in [0, 1]. A 16-bit file read
naively lands in [0, 65535] and every colourfulness anchor becomes nonsense.

Orientation. A portrait frame stored as landscape plus an EXIF rotation flag is
the normal case straight out of a camera, and measuring it unrotated measures
the wrong picture.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from bw_evaluation import (
    MAXIMUM_DECODE_LONG_EDGE,
    SUPPORTED_SUFFIXES,
    UNSUPPORTED_RAW_SUFFIXES,
    WORKING_LONG_EDGE,
    discover_photographs,
    load_photograph,
    read_metadata,
    unique_image_ids,
)

from . import photo_fixtures as P


class RefusingWhatItCannotRead(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_given_a_path_that_does_not_exist_when_loading_then_it_says_so(self):
        # Given a path to nothing
        # When loading
        # Then it fails as a missing file rather than as a decode error
        with self.assertRaises(FileNotFoundError):
            load_photograph(self.folder / "absent.jpg")

    def test_given_a_file_that_is_not_an_image_when_loading_then_the_error_lists_what_is_supported(
        self,
    ):
        # Given a text file
        # When loading
        # Then the refusal names the formats that would have worked
        path = self.folder / "notes.txt"
        path.write_text("not a photograph")
        with self.assertRaises(ValueError) as caught:
            load_photograph(path)
        self.assertIn(".jpg", str(caught.exception))

    def test_given_a_camera_raw_file_when_loading_then_it_is_refused_for_a_stated_reason(
        self,
    ):
        # Given a raw file, which a photographer will certainly try
        # When loading
        # Then the refusal explains that no raw decoder is installed, rather
        # than merely calling the suffix unsupported
        path = self.folder / "DSCF1234.cr2"
        path.write_bytes(b"\x00" * 64)
        with self.assertRaises(ValueError) as caught:
            load_photograph(path)
        self.assertIn("raw", str(caught.exception).lower())

    def test_given_the_raw_and_supported_suffix_lists_when_read_then_they_do_not_overlap(
        self,
    ):
        # Given two lists that both classify a suffix
        # When compared
        # Then no suffix is in both, so the refusal path is unambiguous
        self.assertEqual(set(SUPPORTED_SUFFIXES) & set(UNSUPPORTED_RAW_SUFFIXES), set())


class PixelsArriveInTheContractedForm(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_given_any_photograph_when_loaded_then_it_is_float_rgb_within_the_unit_interval(
        self,
    ):
        # Given a plain 8-bit file
        # When loaded
        # Then it matches exactly what measure_bw_candidacy demands of its input
        P.write_png(self.folder / "a.png", P.solid((0.2, 0.5, 0.9)))
        loaded = load_photograph(self.folder / "a.png")
        self.assertEqual(loaded.rgb.dtype, np.float64)
        self.assertEqual(loaded.rgb.ndim, 3)
        self.assertEqual(loaded.rgb.shape[2], 3)
        self.assertGreaterEqual(loaded.rgb.min(), 0.0)
        self.assertLessEqual(loaded.rgb.max(), 1.0)

    def test_given_an_eight_bit_file_when_loaded_then_the_values_survive_the_round_trip(
        self,
    ):
        # Given a known colour written at 8 bits
        # When read back
        # Then it returns within one code value, so nothing is being rescaled twice
        P.write_png(self.folder / "b.png", P.solid((0.2, 0.5, 0.9)))
        loaded = load_photograph(self.folder / "b.png")
        for channel, expected in enumerate((0.2, 0.5, 0.9)):
            with self.subTest(channel=channel):
                self.assertAlmostEqual(
                    float(loaded.rgb[0, 0, channel]), expected, delta=1 / 255
                )

    def test_given_a_sixteen_bit_file_when_loaded_then_it_is_scaled_by_its_own_depth(
        self,
    ):
        # Given a 16-bit file whose mid grey is 32768, not 128
        # When loaded
        # Then it lands near 0.5, proving the divisor tracks the bit depth
        array = np.full((16, 16), 32768, dtype=np.uint16)
        Image.fromarray(array).save(self.folder / "deep.png")
        loaded = load_photograph(self.folder / "deep.png")
        self.assertLessEqual(loaded.rgb.max(), 1.0)
        self.assertAlmostEqual(float(loaded.rgb[0, 0, 0]), 0.5, delta=0.01)

    def test_given_a_greyscale_file_when_loaded_then_it_becomes_three_equal_channels(
        self,
    ):
        # Given a monochrome file, which a photographer will absolutely include
        # When loaded
        # Then it is widened to three channels rather than rejected, so the
        # achromatic warning can fire downstream instead of a crash
        Image.fromarray(np.full((16, 16), 120, dtype=np.uint8), mode="L").save(
            self.folder / "grey.png"
        )
        loaded = load_photograph(self.folder / "grey.png")
        self.assertEqual(loaded.rgb.shape[2], 3)
        self.assertTrue(np.allclose(loaded.rgb[:, :, 0], loaded.rgb[:, :, 2]))

    def test_given_a_file_with_transparency_when_loaded_then_it_becomes_plain_rgb(self):
        # Given an RGBA export
        # When loaded
        # Then the alpha channel is gone, because none of the metrics have any
        # meaning for a partially transparent pixel
        Image.new("RGBA", (16, 16), (200, 100, 50, 128)).save(self.folder / "alpha.png")
        loaded = load_photograph(self.folder / "alpha.png")
        self.assertEqual(loaded.rgb.shape[2], 3)

    def test_given_a_cmyk_file_when_loaded_then_it_becomes_rgb(self):
        # Given a print-oriented TIFF
        # When loaded
        # Then it is converted rather than refused
        Image.new("CMYK", (16, 16), (10, 200, 180, 5)).save(self.folder / "print.tif")
        loaded = load_photograph(self.folder / "print.tif")
        self.assertEqual(loaded.rgb.shape[2], 3)

    def test_given_a_palette_file_when_loaded_then_it_becomes_rgb(self):
        # Given an indexed-colour PNG
        # When loaded
        # Then the palette is resolved to real channels
        Image.new("P", (16, 16)).save(self.folder / "indexed.png")
        loaded = load_photograph(self.folder / "indexed.png")
        self.assertEqual(loaded.rgb.shape[2], 3)


class ColourManagement(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_given_a_wide_gamut_file_when_loaded_then_its_colours_are_converted_into_srgb(
        self,
    ):
        # Given a saturated red tagged AdobeRGB, whose sRGB equivalent is a
        # different number entirely
        # When loaded
        # Then the pixels move, because reading wide-gamut exports as sRGB
        # would inflate every chroma measurement in the package
        colour = P.solid((220 / 255, 40 / 255, 40 / 255))
        P.write_jpeg(
            self.folder / "wide.jpg", colour, icc_profile=P.wide_gamut_profile_bytes()
        )
        P.write_jpeg(self.folder / "plain.jpg", colour)
        wide = load_photograph(self.folder / "wide.jpg")
        plain = load_photograph(self.folder / "plain.jpg")
        self.assertGreater(
            float(np.abs(wide.rgb[0, 0] - plain.rgb[0, 0]).max()),
            0.05,
        )

    def test_given_a_file_already_in_srgb_when_loaded_then_conversion_changes_nothing(
        self,
    ):
        # Given an sRGB-tagged file
        # When loaded
        # Then it matches the untagged version, so the colour path is lossless
        # in the case it should be
        colour = P.solid((0.75, 0.2, 0.3))
        P.write_png(
            self.folder / "tagged.png", colour, icc_profile=P.srgb_profile_bytes()
        )
        P.write_png(self.folder / "bare.png", colour)
        tagged = load_photograph(self.folder / "tagged.png")
        bare = load_photograph(self.folder / "bare.png")
        self.assertTrue(np.allclose(tagged.rgb, bare.rgb, atol=1 / 255))

    def test_given_a_file_with_no_profile_when_loaded_then_srgb_is_assumed_and_recorded(
        self,
    ):
        # Given an untagged file, which is the common case
        # When loaded
        # Then sRGB is assumed, and the assumption is written down rather than
        # left implicit
        P.write_png(self.folder / "bare.png", P.solid((0.4, 0.4, 0.4)))
        metadata = read_metadata(self.folder / "bare.png")
        self.assertIsNotNone(metadata.colour_profile)
        self.assertIn("assumed", metadata.colour_profile.lower())

    def test_given_a_file_with_an_unreadable_profile_when_loaded_then_it_falls_back_instead_of_failing(
        self,
    ):
        # Given a corrupt ICC block, which happens with damaged exports
        # When loaded
        # Then the photograph still loads and the fallback is reported
        P.write_png(
            self.folder / "bad.png",
            P.solid((0.4, 0.5, 0.6)),
            icc_profile=P.corrupt_profile_bytes(),
        )
        loaded = load_photograph(self.folder / "bad.png")
        self.assertEqual(loaded.rgb.shape[2], 3)
        self.assertTrue(loaded.metadata.colour_profile)


class OrientationAndMetadata(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_given_a_rotated_capture_when_loaded_then_the_pixels_are_uprighted(self):
        # Given a portrait frame stored as landscape with an EXIF rotation flag,
        # which is what most cameras actually write
        # When loaded
        # Then the array is rotated, so the measurement sees the photograph the
        # way the photographer sees it
        wide = np.zeros((40, 80, 3)) + 0.5
        P.write_jpeg(self.folder / "rot.jpg", wide, exif=P.sample_exif(orientation=6))
        loaded = load_photograph(self.folder / "rot.jpg")
        self.assertEqual(loaded.rgb.shape[0], 80)
        self.assertEqual(loaded.rgb.shape[1], 40)

    def test_given_an_unrotated_capture_when_loaded_then_the_pixels_are_left_alone(
        self,
    ):
        # Given orientation 1, the no-op case
        # When loaded
        # Then dimensions are unchanged
        wide = np.zeros((40, 80, 3)) + 0.5
        P.write_jpeg(self.folder / "flat.jpg", wide, exif=P.sample_exif(orientation=1))
        loaded = load_photograph(self.folder / "flat.jpg")
        self.assertEqual(loaded.rgb.shape[:2], (40, 80))

    def test_given_a_camera_file_when_read_then_the_capture_settings_come_back(self):
        # Given a normal camera EXIF block
        # When metadata is read
        # Then the settings a photographer would group by are all present
        P.write_jpeg(
            self.folder / "shot.jpg", P.solid((0.4, 0.4, 0.4)), exif=P.sample_exif()
        )
        meta = read_metadata(self.folder / "shot.jpg")
        self.assertIn("Fujifilm", meta.camera or "")
        self.assertIn("35mm", meta.lens or "")
        self.assertEqual(meta.iso, 1600)
        self.assertEqual(meta.shutter, "1/250")
        self.assertAlmostEqual(meta.aperture or 0.0, 2.8, places=3)
        self.assertAlmostEqual(meta.focal_length or 0.0, 35.0, places=3)
        self.assertIn("2026", meta.captured_at or "")

    def test_given_a_file_with_no_exif_when_read_then_the_fields_are_empty_rather_than_invented(
        self,
    ):
        # Given a file exported without metadata
        # When read
        # Then every camera field is None and nothing is guessed
        P.write_png(self.folder / "bare.png", P.solid((0.4, 0.4, 0.4)))
        meta = read_metadata(self.folder / "bare.png")
        for field in (
            "camera",
            "lens",
            "iso",
            "shutter",
            "aperture",
            "focal_length",
            "captured_at",
        ):
            with self.subTest(field=field):
                self.assertIsNone(getattr(meta, field))

    def test_given_a_photograph_when_read_then_its_stored_and_displayed_sizes_are_both_recorded(
        self,
    ):
        # Given a rotated frame
        # When read
        # Then both the on-disk and the upright dimensions are available,
        # because the difference is exactly what an orientation bug looks like
        P.write_jpeg(
            self.folder / "rot.jpg",
            np.zeros((40, 80, 3)) + 0.5,
            exif=P.sample_exif(orientation=6),
        )
        meta = read_metadata(self.folder / "rot.jpg")
        self.assertEqual((meta.width, meta.height), (80, 40))
        self.assertEqual((meta.displayed_width, meta.displayed_height), (40, 80))


class DecodeSize(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_given_the_decode_cap_when_read_then_it_sits_above_the_working_size(self):
        # Given that measurement resizes to its own working size anyway
        # When the two constants are compared
        # Then the decode cap is the looser of the two, so loading can never
        # become the thing that decides a texture measurement
        self.assertGreater(MAXIMUM_DECODE_LONG_EDGE, WORKING_LONG_EDGE)

    def test_given_an_oversized_photograph_when_loaded_then_it_is_capped_at_the_decode_size(
        self,
    ):
        # Given a frame larger than the cap
        # When loaded
        # Then it comes back at the cap, so a folder of 45-megapixel files does
        # not have to be held in memory at full size
        big = np.zeros((MAXIMUM_DECODE_LONG_EDGE + 400, 200, 3), dtype=np.float64) + 0.5
        P.write_png(self.folder / "big.png", big)
        loaded = load_photograph(self.folder / "big.png")
        self.assertEqual(max(loaded.rgb.shape[:2]), MAXIMUM_DECODE_LONG_EDGE)

    def test_given_a_small_photograph_when_loaded_then_it_is_left_at_its_own_size(self):
        # Given a frame under the cap
        # When loaded
        # Then nothing is resampled, because upscaling at load would invent
        # detail the measurement then tries to measure
        P.write_png(self.folder / "small.png", P.solid((0.3, 0.3, 0.3), size=64))
        loaded = load_photograph(self.folder / "small.png")
        self.assertEqual(loaded.rgb.shape[:2], (64, 64))


class DiscoveringAFolder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_given_a_folder_of_mixed_files_when_discovered_then_only_photographs_come_back(
        self,
    ):
        # Given a real folder, which always has a sidecar or a note in it
        # When discovered
        # Then only decodable images are returned
        P.write_png(self.folder / "one.png", P.solid((0.3, 0.3, 0.3)))
        P.write_jpeg(self.folder / "two.jpg", P.solid((0.4, 0.4, 0.4)))
        (self.folder / "notes.txt").write_text("x")
        (self.folder / "sidecar.xmp").write_text("x")
        found = discover_photographs(self.folder)
        self.assertEqual([p.name for p in found], ["one.png", "two.jpg"])

    def test_given_a_folder_when_discovered_twice_then_the_order_is_identical(self):
        # Given that batch percentiles must not depend on directory order
        # When discovered twice
        # Then the sequences match, because the order is sorted rather than
        # whatever the filesystem happened to hand back
        for name in ("z.png", "a.png", "m.png"):
            P.write_png(self.folder / name, P.solid((0.3, 0.3, 0.3)))
        self.assertEqual(
            discover_photographs(self.folder), discover_photographs(self.folder)
        )
        self.assertEqual(
            [p.name for p in discover_photographs(self.folder)],
            ["a.png", "m.png", "z.png"],
        )

    def test_given_nested_folders_when_discovered_without_recursion_then_only_the_top_level_is_read(
        self,
    ):
        # Given a subfolder, often an exports or rejects directory
        # When discovered non-recursively
        # Then it is left alone, because sweeping it in silently would change
        # the batch a photographer thought they asked for
        P.write_png(self.folder / "top.png", P.solid((0.3, 0.3, 0.3)))
        P.write_png(self.folder / "nested" / "deep.png", P.solid((0.3, 0.3, 0.3)))
        self.assertEqual(
            [p.name for p in discover_photographs(self.folder)], ["top.png"]
        )

    def test_given_nested_folders_when_discovered_recursively_then_everything_is_read(
        self,
    ):
        # Given the same tree
        # When recursion is asked for
        # Then the nested file appears too
        P.write_png(self.folder / "top.png", P.solid((0.3, 0.3, 0.3)))
        P.write_png(self.folder / "nested" / "deep.png", P.solid((0.3, 0.3, 0.3)))
        found = [p.name for p in discover_photographs(self.folder, recursive=True)]
        self.assertEqual(sorted(found), ["deep.png", "top.png"])

    def test_given_a_file_instead_of_a_folder_when_discovering_then_it_is_refused(self):
        # Given a path to a single photograph
        # When treated as a folder
        # Then it is refused rather than silently returning nothing
        path = P.write_png(self.folder / "one.png", P.solid((0.3, 0.3, 0.3)))
        with self.assertRaises(NotADirectoryError):
            discover_photographs(path)

    def test_given_a_folder_that_does_not_exist_when_discovering_then_it_is_refused(
        self,
    ):
        # Given a mistyped path
        # When discovered
        # Then it fails as a missing folder
        with self.assertRaises(FileNotFoundError):
            discover_photographs(self.folder / "nope")


class NamingImages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_given_distinct_file_names_when_naming_then_the_bare_stem_is_used(self):
        # Given ordinary file names
        # When ids are built
        # Then they are short and readable
        paths = [self.folder / "sunrise.jpg", self.folder / "harbour.jpg"]
        self.assertEqual(
            unique_image_ids(paths, root=self.folder),
            {paths[0]: "sunrise", paths[1]: "harbour"},
        )

    def test_given_the_same_stem_in_two_subfolders_when_naming_then_the_ids_stay_distinct(
        self,
    ):
        # Given two shoots that both contain DSCF1234.jpg, which is the normal
        # result of recursing into a year's work
        # When ids are built
        # Then the relative path disambiguates them, because a batch keyed by
        # colliding ids would silently drop one of the photographs
        first = self.folder / "may" / "DSCF1234.jpg"
        second = self.folder / "june" / "DSCF1234.jpg"
        ids = unique_image_ids([first, second], root=self.folder)
        self.assertEqual(len(set(ids.values())), 2)
        self.assertIn("may", ids[first])
        self.assertIn("june", ids[second])


if __name__ == "__main__":
    unittest.main()
