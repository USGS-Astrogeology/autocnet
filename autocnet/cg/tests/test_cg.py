import os
import sys

import numpy as np
import pandas as pd

from autocnet.cg import cg
from osgeo import ogr
from shapely.geometry import Polygon
from unittest.mock import Mock, MagicMock
from plio.io import io_gdal

from autocnet.graph.node import Node
from autocnet.graph.network import CandidateGraph
from autocnet.graph.edge import Edge
from autocnet.utils.utils import array_to_poly

import pytest

@pytest.fixture
def pts():
    seed = np.random.RandomState(12345)
    return seed.rand(25, 2)

@pytest.fixture
def nspoly():
    return Polygon([(0,0),(.2,0),(.2,1), (0,1), (0,0)])

def test_area_single(pts):
    total_area = 1.0
    ratio = cg.convex_hull_ratio(pts, total_area)

    assert pytest.approx(0.7566490, 5) == ratio

def test_overlap():
    wkt1 = "POLYGON ((0 40, 40 40, 40 0, 0 0, 0 40))"
    wkt2 = "POLYGON ((20 60, 60 60, 60 20, 20 20, 20 60))"

    poly1 = ogr.CreateGeometryFromWkt(wkt1)
    poly2 = ogr.CreateGeometryFromWkt(wkt2)

    info = cg.two_poly_overlap(poly1, poly2)

    assert info[1] == 400
    assert pytest.approx(info[0]) == 14.285714285

def test_geom_mask():
    my_gdf = pd.DataFrame(columns=['x', 'y'], data=[(0, 0), (2, 2)])
    my_poly = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
    mask = cg.geom_mask(my_gdf, my_poly)
    assert mask[0] == False
    assert mask[1] == True

@pytest.fixture
def keypoints():
    return pd.DataFrame({'x': (15, 18, 18, 12, 12), 'y': (6, 10, 15, 15, 10)})

def test_voronoi_keypoints(keypoints):
    voronoi_gdf = cg.compute_voronoi(keypoints)
    for i, v in enumerate([12.0, 13.5, 7.5, 7.5, 13.5]):
        assert pytest.approx(voronoi_gdf.weight[i]) == v

def test_voronoi_keypoints_with_geom(keypoints):
    voronoi_gdf = cg.compute_voronoi(keypoints, geometry=True)
    for i, v in enumerate([12.0, 13.5, 7.5, 7.5, 13.5]):
        assert pytest.approx(voronoi_gdf.geometry.area[i]) == v

def test_voronoi_keypoint_intersection(keypoints):
    intersection = Polygon([(10, 5), (20, 5), (20, 20), (10, 20)])
    voronoi_inter_gdf = cg.compute_voronoi(keypoints, intersection)
    for i, v in enumerate([22.5, 26.25, 37.5, 37.5, 26.25]):
        assert pytest.approx(voronoi_inter_gdf.weight[i]) == v

@pytest.mark.parametrize("polygon, nexpected",[
    (Polygon([(0,0), (.2,0), (.2,1), (0,1), (0,0)]), 10),
    (Polygon([(0,0), (1,0), (1,.2), (0,.2), (0,0)]), 10),
    (Polygon([(0,0), (.2, .1), (.2,1.1), (-0.1, 1), (0,0)]), 22)
],
    ids=['vertical', 'horizontal', 'verticalskewed'])
def test_points_in_geom(polygon, nexpected):
    pts = cg.distribute_points_in_geom(polygon)
    assert len(pts) == nexpected
    
