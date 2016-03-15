import os
import sys
import unittest

import numpy as np

from .. import cg

sys.path.insert(0, os.path.abspath('..'))


class TestArea(unittest.TestCase):

    def setUp(self):
        seed = np.random.RandomState(12345)
        self.pts = seed.rand(25, 2)

    def test_area_single(self):
        total_area = 1.0
        ratio = cg.convex_hull_ratio(self.pts, total_area)

        self.assertAlmostEqual(0.7566490, ratio, 5)
