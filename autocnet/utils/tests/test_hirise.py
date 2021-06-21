"""This module has tests for the change_detection functions."""

# This is free and unencumbered software released into the public domain.
#
# The authors of autocnet do not claim copyright on the contents of this file.
# For more details about the LICENSE terms and the AUTHORS, you will
# find files of those names at the top level of this repository.
#
# SPDX-License-Identifier: CC0-1.0

import contextlib
import unittest
from pathlib import Path
from subprocess import CalledProcessError
from unittest.mock import patch

from autocnet.utils import hirise

class TestISIS(unittest.TestCase):

    def setUp(self) -> None:
        self.resourcedir = Path("test-resources")
        red50img = self.resourcedir / "PSP_010502_2090_RED5_0.img"
        red51img = self.resourcedir / "PSP_010502_2090_RED5_1.img"

        if not all((red50img.exists(), red51img.exists())):
            self.skipTest(
                f"One or more files is missing from the {self.resourcedir} "
                "directory. Tests on real files skipped.")

    def tearDown(self):
        with contextlib.suppress(FileNotFoundError):
            for g in ("*.cub", "*.log"):
                for p in self.resourcedir.glob(g):
                    p.unlink()
            Path("print.prt").unlink()

    def test_ingest_hirise(self):
        hirise.ingest_hirise(str(self.resourcedir))
        self.assertTrue((
            self.resourcedir / "PSP_010502_2090_RED5.stitched.norm.cub"
        ).exists())

        with patch(
            "kalasiris.hi2isis",
                side_effect=CalledProcessError(returncode=1, cmd="foo")
        ):
            hirise.ingest_hirise(str(self.resourcedir))

    def test_segment_hirise(self):
        hirise.ingest_hirise(str(self.resourcedir))
        hirise.segment_hirise(str(self.resourcedir))
        base = self.resourcedir / "PSP_010502_2090_RED5.stitched.norm.cub"
        for f in (
            base.with_suffix(".1_1325.cub"),
            base.with_suffix(".725_2349.cub"),
            base.with_suffix(".1749_3373.cub"),
            base.with_suffix(".2773_4000.cub"),
        ):
            self.assertTrue(f.exists())