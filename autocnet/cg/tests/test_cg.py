import os
import sys
import unittest
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd

from .. import cg
from osgeo import ogr
from shapely.geometry import Polygon
from unittest.mock import Mock, MagicMock
from plio.io import io_gdal

from autocnet.graph.node import Node
from autocnet.graph.network import CandidateGraph
from autocnet.graph.edge import Edge
from autocnet.utils.utils import array_to_poly


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

    def test_geom_mask(self):
        my_gdf = pd.DataFrame(columns=['x', 'y'], data=[(0, 0), (2, 2)])
        my_poly = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
        mask = cg.geom_mask(my_gdf, my_poly)
        self.assertFalse(mask[0])
        self.assertTrue(mask[1])

    def test_compute_voronoi(self):
        keypoints = pd.DataFrame({'x': (15, 18, 18, 12, 12), 'y': (6, 10, 15, 15, 10)})
        intersection = Polygon([(10, 5), (20, 5), (20, 20), (10, 20)])

        voronoi_gdf = cg.compute_voronoi(keypoints)
        self.assertAlmostEqual(voronoi_gdf.weight[0], 12.0)
        self.assertAlmostEqual(voronoi_gdf.weight[1], 13.5)
        self.assertAlmostEqual(voronoi_gdf.weight[2], 7.5)
        self.assertAlmostEqual(voronoi_gdf.weight[3], 7.5)
        self.assertAlmostEqual(voronoi_gdf.weight[4], 13.5)

        voronoi_gdf = cg.compute_voronoi(keypoints, geometry=True)
        self.assertAlmostEqual(voronoi_gdf.geometry[0].area, 12.0)
        self.assertAlmostEqual(voronoi_gdf.geometry[1].area, 13.5)
        self.assertAlmostEqual(voronoi_gdf.geometry[2].area, 7.5)
        self.assertAlmostEqual(voronoi_gdf.geometry[3].area, 7.5)
        self.assertAlmostEqual(voronoi_gdf.geometry[4].area, 13.5)

        voronoi_inter_gdf = cg.compute_voronoi(keypoints, intersection)
        self.assertAlmostEqual(voronoi_inter_gdf.weight[0], 22.5)
        self.assertAlmostEqual(voronoi_inter_gdf.weight[1], 26.25)
        self.assertAlmostEqual(voronoi_inter_gdf.weight[2], 37.5)
        self.assertAlmostEqual(voronoi_inter_gdf.weight[3], 37.5)
        self.assertAlmostEqual(voronoi_inter_gdf.weight[4], 26.25)
