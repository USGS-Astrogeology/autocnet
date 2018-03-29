import unittest
from unittest.mock import Mock, MagicMock

import ogr
import numpy as np
import pandas as pd
from plio.io import io_gdal
from shapely.geometry import Polygon as Poly

from autocnet.matcher import cpu_outlier_detector as od
from autocnet.examples import get_path
from autocnet.graph.network import CandidateGraph
from autocnet.utils.utils import array_to_poly

from .. import edge
from .. import node


class TestEdge(unittest.TestCase):

    def setUp(self):
        source = Mock(node.Node)
        destination = Mock(node.Node)
        self.edge = edge.Edge(source=source, destination=destination)

    def test_masks(self):
        self.assertIsInstance(self.edge.masks, pd.DataFrame)

    def test_edge_overlap(self):
        e = edge.Edge()
        e.weight = {}
        source = Mock(spec = node.Node)
        destination = Mock(spec = node.Node)
        e.destination = destination
        e.source = source
        geodata_s = Mock(spec = io_gdal.GeoDataset)
        geodata_d = Mock(spec = io_gdal.GeoDataset)
        source.geodata = geodata_s
        destination.geodata = geodata_d

        wkt1 = "POLYGON ((0 40, 40 40, 40 0, 0 0, 0 40))"
        wkt2 = "POLYGON ((20 60, 60 60, 60 20, 20 20, 20 60))"

        poly1 = ogr.CreateGeometryFromWkt(wkt1)
        poly2 = ogr.CreateGeometryFromWkt(wkt2)

        source.geodata.footprint = poly1
        destination.geodata.footprint = poly2

        e.overlap()
        self.assertEqual(e['weights']['overlap_area'], 400)
        self.assertAlmostEqual(e['weights']['overlap_percn'], 14.285714285)

    def test_coverage(self):
        adjacency = get_path('two_image_adjacency.json')
        basepath = get_path('Apollo15')
        cg = CandidateGraph.from_adjacency(adjacency, basepath=basepath)
        keypoint_df = pd.DataFrame({'x': (15, 18, 18, 12, 12), 'y': (5, 10, 15, 15, 10)})
        keypoint_matches = [[0, 0, 1, 0],
                            [0, 1, 1, 1],
                            [0, 2, 1, 2],
                            [0, 3, 1, 3],
                            [0, 4, 1, 4]]

        matches_df = pd.DataFrame(data=keypoint_matches, columns=['source_image', 'source_idx', 'destination_image', 'destination_idx'])

        e = edge.Edge()
        source_node = MagicMock(spec=node.Node())
        destination_node = MagicMock(spec=node.Node())

        source_node.get_keypoint_coordinates = MagicMock(return_value=keypoint_df)
        destination_node.get_keypoint_coordinates = MagicMock(return_value=keypoint_df)

        e.source = source_node
        e.destination = destination_node

        source_geodata = Mock(spec=io_gdal.GeoDataset)
        destination_geodata = Mock(spec=io_gdal.GeoDataset)

        e.source.geodata = source_geodata
        e.destination.geodata = destination_geodata

        source_corners = [(0, 0),
                          (20, 0),
                          (20, 20),
                          (0, 20)]

        destination_corners = [(10, 5),
                               (30, 5),
                               (30, 25),
                               (10, 25)]

        e.source.geodata.latlon_corners = source_corners
        e.destination.geodata.latlon_corners = destination_corners

        vals = {(15, 5): (15, 5), (18, 10): (18, 10), (18, 15): (18, 15), (12, 15): (12, 15), (12, 10): (12, 10)}

        def pixel_to_latlon(i, j):
            return vals[(i, j)]

        e.source.geodata.pixel_to_latlon = MagicMock(side_effect=pixel_to_latlon)
        e.destination.geodata.pixel_to_latlon = MagicMock(side_effect=pixel_to_latlon)

        e.matches = matches_df

        #self.assertRaises(AttributeError, cg.edge[0][1].coverage)
        self.assertEqual(e.coverage(), 0.3)

    def test_get_keypoints(self):
        src_keypoint_df = pd.DataFrame({'x': (0, 1, 2, 3, 4), 'y': (5, 6, 7, 8, 9),
                                        'response': (10, 11, 12, 13, 14), 'size': (15, 16, 17, 18, 19),
                                        'angle': (20, 21, 22, 23, 24), 'octave': (25, 26, 27, 28, 29),
                                        'layer': (30, 31, 32, 33, 34)})

        dst_keypoint_df = pd.DataFrame({'x': (34, 33, 32, 31, 30), 'y': (29, 28, 27, 26, 25),
                                        'response': (24, 23, 22, 21, 20), 'size': (19, 18, 17, 16, 15),
                                        'angle': (14, 13, 12, 11, 10), 'octave': (9, 8, 7, 6, 5),
                                        'layer': (4, 3, 2, 1, 0)})

        keypoint_matches = [[0, 0, 1, 4],
                            [0, 1, 1, 3],
                            [0, 2, 1, 2],
                            [0, 3, 1, 1],
                            [0, 4, 1, 0]]

        matches_df = pd.DataFrame(data=keypoint_matches, columns=['source_image', 'source_idx',
                                                                  'destination_image', 'destination_idx'])
        source_node = node.Node(node_id=0)
        destination_node = node.Node(node_id=1)
        e = edge.Edge(source_node, destination_node)

        source_node.get_keypoint_coordinates = MagicMock(return_value=src_keypoint_df)
        destination_node.get_keypoint_coordinates = MagicMock(return_value=dst_keypoint_df)

        e.clean = MagicMock(return_value=(matches_df, None))
        e.matches = matches_df

        # Test all uses for edge.get_keypoints()
        src_matched_keypts = e.get_keypoints("source")
        src_matched_keypts2 = e.get_keypoints(e.source)
        dst_matched_keypts = e.get_keypoints("destination")
        dst_matched_keypts2 = e.get_keypoints(e.destination)

        # [output df to test] [name of node] [df to test against]
        to_test = [[src_matched_keypts, "source", src_keypoint_df],
                   [src_matched_keypts2, "source", src_keypoint_df],
                   [dst_matched_keypts, "destination", dst_keypoint_df],
                   [dst_matched_keypts2, "destination", dst_keypoint_df]]

        for out_df in to_test:
            # For each row index in the appropriate column of the matches_df,
            # assert that row index exists in the function's returned df
            [self.assertIn(row_idx, out_df[0].index.values)
             for row_idx in matches_df[out_df[1] + '_idx']]
            # For each row index in the returned df
            for row_idx in out_df[0].index.values:
                # Assert that row_idx exists in the matches_df's appropriate
                # column
                self.assertIn(row_idx, matches_df[out_df[1] + '_idx'])
                # Assert that all row_idx[column] vals returned by function
                # match their counterpart in orig df
                for column in out_df[0].columns:
                    self.assertTrue(out_df[0].iloc[row_idx][column] ==
                                            out_df[2].iloc[row_idx][column])

        # Test when overlap=True
        # edge["source_mbr"] and edge["destin_mbr"] haven't been calculated
        # yet, so return val should be unmasked df of node's keypts
        s_no_overlap = e.get_keypoints(e.source, overlap=True)
        d_no_overlap = e.get_keypoints(e.destination, overlap=True)
        self.assertTrue(s_no_overlap.equals(src_keypoint_df))
        self.assertTrue(d_no_overlap.equals(dst_keypoint_df))

        # Define the MBRs
        e.overlap_latlon_coords = 0, 0

        source_node.reproject_geom = MagicMock(return_value=Poly([(1, 6), (1, 8), (3, 6), (3, 8)]))
        source_node.keypoints = src_keypoint_df

        destination_node.reproject_geom = MagicMock(return_value=Poly([(31, 26), (31, 28), (33, 26), (33, 28)]))
        destination_node.keypoints = dst_keypoint_df

        # Only keep keypt vals w/I these bounding rects
        e["source_mbr"] = (0, 2, 5, 8)
        e["destin_mbr"] = (31, 33, 26, 28)

        # Grab the keypoints on our MBR overlaps
        s_overlap = e.get_keypoints(e.source, overlap=True)
        d_overlap = e.get_keypoints(e.destination, overlap=True)

        # Assert masked src keypts coords are equal
        s_expected = pd.DataFrame({'x': (0, 1, 2), 'y': (5, 6, 7)})
        self.assertTrue(np.array_equal(s_expected['x'], s_overlap['x'].values))
        self.assertTrue(np.array_equal(s_expected['y'], s_overlap['y'].values))

        # Assert masked dst keypt coords are equal
        d_expected = pd.DataFrame({'x': (33, 32, 31), 'y': (28, 27, 26)})
        print(d_expected['x'])
        print(d_overlap['x'].values)
        self.assertTrue(np.array_equal(d_expected['x'], d_overlap['x'].values))
        self.assertTrue(np.array_equal(d_expected['y'], d_overlap['y'].values))

        # Assert type-checking in method throws proper errors
        with self.assertRaises(TypeError):
            e.get_keypoints("source", index = 456)
        with self.assertRaises(AttributeError):
            e.get_keypoints(1)
        # Check key error thrown when string arg != "source" or "destination"
        with self.assertRaises(AttributeError):
            e.get_keypoints("string")

    def test_eq(self):
        edge1 = edge.Edge()
        edge2 = edge.Edge()
        edge3 = edge.Edge()

        # Test edges w/ different keys are not equal, ones with same keys are
        edge1.__dict__["key"] = 1
        edge2.__dict__["key"] = 1
        edge3.__dict__["not_key"] = 1

        self.assertTrue(edge1 == edge2)
        self.assertFalse(edge1 == edge3)

        # Test edges with same keys, but diff df values
        edge1.__dict__["key"] = pd.DataFrame({'x': (0, 1, 2, 3, 4)})
        edge2.__dict__["key"] = pd.DataFrame({'x': (0, 1, 2, 3, 4)})
        edge3.__dict__["key"] = pd.DataFrame({'x': (0, 1, 2, 3, 5)})

        self.assertTrue(edge1 == edge2)
        self.assertFalse(edge1 == edge3)

        # Test edges with same keys, but diff np array vals
        # edge.__eq__ calls ndarray.all(), which checks that
        # all values in an array eval to true
        edge1.__dict__["key"] = np.array([True, True, True], dtype=np.bool)
        edge2.__dict__["key"] = np.array([True, True, True], dtype=np.bool)
        edge3.__dict__["key"] = np.array([True, True, False], dtype=np.bool)

        self.assertTrue(edge1 == edge2)
        self.assertFalse(edge1 == edge3)

    def test_repr(self):
        src = node.Node()
        dst = node.Node()
        masks = pd.DataFrame()

        e = edge.Edge()
        e.source = src
        e.destination = dst

        expected = """
        Source Image Index: {}
        Destination Image Index: {}
        Available Masks: {}
        """.format(src, dst, masks)

        self.assertEqual(expected, e.__repr__())

    def test_ratio_check(self):
        """
        A pretty basic test that simply tests pass through from the edge to the
        proper distance func.
        """
        # Matches is init to None
        e = edge.Edge()

        # If there are matches...
        keypoint_matches = [[0, 0, 1, 4, 5],
                            [0, 1, 1, 3, 5],
                            [0, 2, 1, 2, 5],
                            [0, 3, 1, 1, 5],
                            [0, 4, 1, 0, 5]]

        matches_df = pd.DataFrame(data=keypoint_matches, columns=['source_image', 'source_idx',
                                                                  'destination_image', 'destination_idx', 'distance'])
        e.matches = matches_df
        expected = od.distance_ratio(None, matches_df)
        e.ratio_check()
        assert expected.equals(e.masks["ratio"])

    def test_overlap_check(self):
        s = node.Node(node_id=0)
        d = node.Node(node_id=1)

        e = edge.Edge(s, d)

        src_keypoint_df = pd.DataFrame({'x': (0, 1, 2, 3, 4), 'y': (5, 6, 7, 8, 9)})
        dst_keypoint_df = pd.DataFrame({'x': (34, 33, 32, 31, 30), 'y': (29, 28, 27, 26, 25)})

        # Create keypt matches
        keypoint_matches = [[0, 0, 1, 4, 5],
                            [0, 1, 1, 3, 5],
                            [0, 2, 1, 2, 5],
                            [0, 3, 1, 1, 5],
                            [0, 4, 1, 0, 5]]

        matches_df = pd.DataFrame(data=keypoint_matches,
                                  columns=['source_image', 'source_idx',
                                           'destination_image', 'destination_idx', 'distance'])
        s.keypoints = src_keypoint_df
        d.keypoints = dst_keypoint_df
        e.matches = matches_df

        s_overlap_keypts = pd.DataFrame({'x': (0, 1), 'y': (5, 6)})
        d_overlap_keypts = pd.DataFrame({'x': (31, 30), 'y': (26, 25)}, index=[3, 4])
        expected_mask = pd.Series(data=[True, True, False, False, False])

        # Mockup of the Edge.get_keypoints() method when overlap=True
        def mock_get_keypts(node, overlap=False):
            if node == s and overlap:
                return s_overlap_keypts
            elif node == d and overlap:
                return d_overlap_keypts
            else:
                return None

        e.get_keypoints = MagicMock(side_effect=mock_get_keypts)

        # Should fail if no src & dst mbrs on edge; Warns user & mask isn't
        # populated
        e.overlap_check()
        self.assertTrue("overlap" not in e.masks)

        # Should work after MBRs are set
        e["source_mbr"] = (1, 1, 1, 1)
        e["destin_mbr"] = (1, 1, 1, 1)
        e.overlap_check()
        overlap_matches, overlap_mask = e.clean(clean_keys=['overlap'])
        self.assertTrue(expected_mask.equals(overlap_mask))
        self.assertTrue(overlap_matches.equals(e.matches[overlap_mask]))

    def test_bad_matches_type(self):
        with self.assertRaises(TypeError):
            s = node.Node(node_id=0)
            d = node.Node(node_id=1)

            e = edge.Edge(s, d)
            e.matches = ['a', 'b', 'c']