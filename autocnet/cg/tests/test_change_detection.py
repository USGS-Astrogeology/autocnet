"""This module has tests for the change_detection functions."""

# This is free and unencumbered software released into the public domain.
#
# The authors of autocnet do not claim copyright on the contents of this file.
# For more details about the LICENSE terms and the AUTHORS, you will
# find files of those names at the top level of this repository.
#
# SPDX-License-Identifier: CC0-1.0

import unittest
import numpy as np
import numpy.testing as npt
from autocnet.cg import change_detection as cd


class TestISIS(unittest.TestCase):

    def test_image_diff(self):
        arr1 = np.array([1.0, 2.0, -3.4028227e+38])
        arr2 = np.array([1.0, 3.0, 0])

        npt.assert_array_equal(
            np.array([0, -1.0, 0]),
            cd.image_diff(arr1, arr2)
        )

    def test_image_ratio(self):
        arr1 = np.array([1.0, 4.0, -3.4028227e+38])
        arr2 = np.array([1.0, 2.0, 0])

        npt.assert_array_equal(
            np.array([1.0, 2.0, 0]),
            cd.image_ratio(arr1, arr2)
        )