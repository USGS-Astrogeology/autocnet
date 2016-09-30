import os
import sys
import unittest
sys.path.insert(0, os.path.abspath('..'))

import numpy as np

from .. import cg
from osgeo import ogr


class TestArea(unittest.TestCase):

    def setUp(self):
        seed = np.random.RandomState(12345)
        self.pts = seed.rand(25, 2)

    def test_area_single(self):
        total_area = 1.0
        ratio = cg.convex_hull_ratio(self.pts, total_area)

        self.assertAlmostEqual(0.7566490, ratio, 5)

    def test_overlap(self):
        wkt1 = "POLYGON ((0 40, 40 40, 40 0, 0 0, 0 40))"
        wkt2 = "POLYGON ((20 60, 60 60, 60 20, 20 20, 20 60))"

        poly1 = ogr.CreateGeometryFromWkt(wkt1)
        poly2 = ogr.CreateGeometryFromWkt(wkt2)

        info = cg.two_poly_overlap(poly1, poly2)

        self.assertEqual(info[1], 400)
        self.assertAlmostEqual(info[0], 14.285714285)

