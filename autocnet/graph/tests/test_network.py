import os
import sys

import unittest

from unittest.mock import patch
from unittest.mock import PropertyMock
from osgeo import ogr

import numpy as np

from autocnet.examples import get_path

from .. import network

sys.path.insert(0, os.path.abspath('..'))


class TestCandidateGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        basepath = get_path('Apollo15')
        cls.graph = network.CandidateGraph.from_adjacency(get_path('three_image_adjacency.json'),
                                                          basepath=basepath)
        cls.disconnected_graph = network.CandidateGraph.from_adjacency(get_path('adjacency.json'))

    def test_get_name(self):
        node_number = self.graph.node_name_map['AS15-M-0297_SML.png']
        name = self.graph.get_name(node_number)
        self.assertEquals(name, 'AS15-M-0297_SML.png')

    def test_add_image(self):
        with self.assertRaises(NotImplementedError):
            self.graph.add_image()

    def test_to_json_file(self):
        self.graph.to_json_file('test_graph_to_json.json')
        self.assertTrue(os.path.exists('test_graph_to_json.json'))
        try:
            os.remove('test_graph_to_json.json')
        except:
            pass

    def test_size(self):
        graph = self.graph
        self.assertEqual(graph.size(), graph.number_of_edges())

        for u, v, e in graph.edges_iter(data=True):
            e['weight'] = 10

        self.assertEqual(graph.size('weight'), graph.number_of_edges()*10)

    def test_island_nodes(self):
        self.assertEqual(len(self.disconnected_graph.island_nodes()), 1)

    def test_apply_func_to_edges(self):
        graph = self.graph.copy()
        mst_graph = graph.minimum_spanning_tree()

        try:
            graph.apply_func_to_edges('incorrect_func')
        except AttributeError:
            pass


        mst_graph.extract_features(extractor_parameters={'nfeatures': 50})
        mst_graph.match_features()
        mst_graph.apply_func_to_edges("symmetry_check")

        # Test passing the func by signature
        mst_graph.apply_func_to_edges(graph[0][1].symmetry_check)

        self.assertFalse(graph[0][2].masks['symmetry'].all())
        self.assertFalse(graph[0][1].masks['symmetry'].all())

        try:
            self.assertTrue(graph[1][2].masks['symmetry'].all())
        except: pass

    def test_connected_subgraphs(self):
        subgraph_list = self.disconnected_graph.connected_subgraphs()
        self.assertEqual(len(subgraph_list), 2)
        islands = self.disconnected_graph.island_nodes()
        self.assertTrue(islands[0] in subgraph_list[1])

        subgraph_list = self.graph.connected_subgraphs()
        self.assertEqual(len(subgraph_list), 1)

    def test_save_load_graph(self):
        self.graph.save('test_save.cg')
        loaded = self.graph.from_graph('test_save.cg')
        self.assertEqual(self.graph.node[0].nkeypoints, loaded.node[0].nkeypoints)
        self.assertEqual(self.graph.edge[0][1], loaded.edge[0][1])

        a = self.graph.node[0].geodata.read_array()
        b = loaded.node[0].geodata.read_array()
        np.testing.assert_array_equal(a, b)

        os.remove('test_save.cg')

    def test_save_load_features(self):
        graph = self.graph.copy()
        graph.extract_features(extractor_parameters={'nfeatures': 10})
        graph.save_features('all_out.hdf')
        graph.save_features('one_out.hdf', nodes=[1])
        graph_no_features = self.graph.copy()
        graph_no_features.load_features('one_out.hdf', nodes=[1])
        self.assertEqual(graph.node[1].get_keypoints().all().all(),
                         graph_no_features.node[1].get_keypoints().all().all())

        graph_no_features.load_features('all_out.hdf')
        for n in graph.nodes():
            self.assertEqual(graph.node[n].get_keypoints().all().all(),
                             graph_no_features.node[n].get_keypoints().all().all())
        for i in ['all_out.hdf', 'one_out.hdf']:
            try:
                os.remove(i)
            except:
                pass

    def test_fromlist(self):
        mock_list = ['AS15-M-0295_SML.png', 'AS15-M-0296_SML.png', 'AS15-M-0297_SML.png',
                     'AS15-M-0298_SML.png', 'AS15-M-0299_SML.png', 'AS15-M-0300_SML.png']

        good_poly = ogr.CreateGeometryFromWkt('POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))')
        bad_poly = ogr.CreateGeometryFromWkt('POLYGON ((9999 10, 40 40, 20 40, 10 20, 30 10))')

        with patch('plio.io.io_gdal.GeoDataset.footprint', new_callable=PropertyMock) as patch_fp:
            patch_fp.return_value = good_poly
            n = network.CandidateGraph.from_filelist(mock_list, get_path('Apollo15'))
            self.assertEqual(n.number_of_nodes(), 6)
            self.assertEqual(n.number_of_edges(), 15)

            patch_fp.return_value = bad_poly
            n = network.CandidateGraph.from_filelist(mock_list, get_path('Apollo15'))
            self.assertEqual(n.number_of_nodes(), 6)
            self.assertEqual(n.number_of_edges(), 0)

        n = network.CandidateGraph.from_filelist(mock_list, get_path('Apollo15'))
        self.assertEqual(len(n.nodes()), 6)

        n = network.CandidateGraph.from_filelist(get_path('adjacency.lis'), get_path('Apollo15'))
        self.assertEqual(len(n.nodes()), 6)

    def test_subset_graph(self):
        g = self.graph
        edge_sub = g.create_edge_subgraph([(0, 2)])
        self.assertEqual(len(edge_sub.nodes()), 2)

        node_sub = g.create_node_subgraph([0, 1])
        self.assertEqual(len(node_sub), 2)

    def test_filter(self):
        def edge_func(edge):
            return edge.matches is not None and hasattr(edge, 'matches')

        graph = self.graph.copy()
        test_sub_graph = graph.create_node_subgraph([0, 1])
        test_sub_graph.extract_features(extractor_parameters={'nfeatures': 500})
        test_sub_graph.match_features(k=2)
        filtered_nodes = graph.filter_nodes(lambda node: hasattr(node, 'descriptors'))
        filtered_edges = graph.filter_edges(edge_func)

        self.assertEqual(filtered_nodes.number_of_nodes(), test_sub_graph.number_of_nodes())
        self.assertEqual(filtered_edges.number_of_edges(), test_sub_graph.number_of_edges())

    def test_subgraph_from_matches(self):
        test_sub_graph = self.graph.create_node_subgraph([0, 1])
        test_sub_graph.extract_features(extractor_parameters={'nfeatures': 500})
        test_sub_graph.match_features(k=2)

        sub_graph_from_matches = self.graph.subgraph_from_matches()

        self.assertEqual(test_sub_graph.edges(), sub_graph_from_matches.edges())

    def test_minimum_spanning_tree(self):
        test_dict = {"0": ["4", "2", "1", "3"],
                     "1": ["0", "3", "2", "6", "5"],
                     "2": ["1", "0", "3", "4", "7"],
                     "3": ["2", "0", "1", "5"],
                     "4": ["2", "0"],
                     "5": ["1", "3"],
                     "6": ["1"],
                     "7": ["2"]}

        graph = network.CandidateGraph.from_adjacency(test_dict)
        mst_graph = graph.minimum_spanning_tree()

        self.assertEqual(sorted(mst_graph.nodes()), sorted(graph.nodes()))
        self.assertEqual(len(mst_graph.edges()), len(graph.edges())-5)

    def test_triangular_cycles(self):
        cycles = self.graph.compute_triangular_cycles()
        # Node order is variable, length is not
        self.assertEqual(len(cycles), 1)

    def tearDown(self):
        pass
