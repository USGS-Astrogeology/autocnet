import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import unittest

import numpy as np

from autocnet.examples import get_path

from .. import network


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

    def test_island_nodes(self):
        self.assertEqual(len(self.disconnected_graph.island_nodes()), 1)

    def test_apply_func_to_edges(self):
        graph = self.graph.copy()
        graph.minimum_spanning_tree()

        try:
            graph.apply_func_to_edges('incorrect_func')
        except AttributeError:
            pass

        graph.extract_features(extractor_parameters={'nfeatures': 500})
        graph.match_features()
        graph.apply_func_to_edges("symmetry_check", graph_mask_keys=['mst'])

        self.assertFalse(graph[0][2].masks['symmetry'].all())
        self.assertFalse(graph[0][1].masks['symmetry'].all())
        self.assertTrue(graph[1][2].masks['symmetry'].all())

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

        a = self.graph.node[0].handle.read_array()
        b = loaded.node[0].handle.read_array()
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
            except: pass

    def test_fromlist(self):
        mock_list = ['AS15-M-0295_SML.png', 'AS15-M-0296_SML.png', 'AS15-M-0297_SML.png',
                     'AS15-M-0298_SML.png', 'AS15-M-0299_SML.png', 'AS15-M-0300_SML.png']
        n = network.CandidateGraph.from_filelist(mock_list, get_path('Apollo15'))
        self.assertEqual(len(n.nodes()), 6)

        n = network.CandidateGraph.from_filelist(get_path('adjacency.lis'), get_path('Apollo15'))
        self.assertEqual(len(n.nodes()), 6)

    def test_subset_graph(self):
        g = self.graph
        edge_sub = g.create_edge_subgraph([(0,2)])
        self.assertEqual(len(edge_sub.nodes()), 2)

        node_sub = g.create_node_subgraph([0,1])
        self.assertEqual(len(node_sub), 2)

    def test_filter(self):
        def edge_func(edge):
            return hasattr(edge, 'matches') and not edge.matches.empty

        graph = self.graph.copy()
        test_sub_graph = graph.create_node_subgraph([0, 1])
        test_sub_graph.extract_features(extractor_parameters={'nfeatures': 500})
        test_sub_graph.match_features(k=2)

        filtered_nodes = graph.filter_nodes(lambda node: hasattr(node, 'descriptors'))
        filtered_edges = graph.filter_edges(edge_func)

        self.assertEqual(filtered_nodes.number_of_nodes(), test_sub_graph.number_of_nodes())
        self.assertEqual(filtered_edges.number_of_edges(), test_sub_graph.number_of_edges())

    def tearDown(self):
        pass


class TestGraphMasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dict = {"0": ["4", "2", "1", "3"],
                         "1": ["0", "3", "2", "6", "5"],
                         "2": ["1", "0", "3", "4", "7"],
                         "3": ["2", "0", "1", "5"],
                         "4": ["2", "0"],
                         "5": ["1", "3"],
                         "6": ["1"],
                         "7": ["2"]}

        cls.graph = network.CandidateGraph.from_adjacency(cls.test_dict)
        cls.graph.minimum_spanning_tree()
        removed_edges = cls.graph.graph_masks['mst'][cls.graph.graph_masks['mst'] == False].index

        cls.mst_graph = cls.graph.copy()
        cls.mst_graph.remove_edges_from(removed_edges)

    def test_mst_output(self):
        self.assertEqual(sorted(self.mst_graph.nodes()), sorted(self.graph.nodes()))
        self.assertEqual(self.mst_graph.number_of_edges(), self.graph.number_of_edges()-5)