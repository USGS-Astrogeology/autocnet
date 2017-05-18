import os
import sys

import pytest
import unittest

from unittest.mock import patch
from unittest.mock import PropertyMock
from osgeo import ogr

import numpy as np

from autocnet.examples import get_path

from .. import network

sys.path.insert(0, os.path.abspath('..'))


@pytest.fixture()
def graph():
    basepath = get_path('Apollo15')
    return network.CandidateGraph.from_adjacency(get_path('three_image_adjacency.json'),
                                                      basepath=basepath)

@pytest.fixture()
def disconnected_graph():
    return network.CandidateGraph.from_adjacency(get_path('adjacency.json'))

def test_get_name(graph):
    node_number = graph.graph['node_name_map']['AS15-M-0297_SML.png']
    name = graph.get_name(node_number)
    assert name == 'AS15-M-0297_SML.png'

def test_size(graph):
    assert graph.size() == graph.number_of_edges()
    for u, v, e in graph.edges_iter(data=True):
        e['edge_weight'] = 10

    assert graph.size('edge_weight') == graph.number_of_edges()*10

def test_add_image(graph):
    with pytest.raises(NotImplementedError):
        graph.add_image()

def test_island_nodes(disconnected_graph):
    assert len(disconnected_graph.island_nodes()) == 1

def test_triangular_cycles(graph):
    cycles = graph.compute_triangular_cycles()
    # Node order is variable, length is not
    assert len(cycles) == 1

def test_connected_subgraphs(graph, disconnected_graph):
    subgraph_list = disconnected_graph.connected_subgraphs()
    assert len(subgraph_list) == 2

    islands = disconnected_graph.island_nodes()
    assert islands[0] in subgraph_list[1]

    subgraph_list = graph.connected_subgraphs()
    assert len(subgraph_list) == 1

def test_save_load_features(tmpdir, graph):
    # Create the graph and save the features
    graph = graph.copy()
    graph.extract_features(extractor_parameters={'nfeatures': 10})
    allout = tmpdir.join("all_out.hdf")
    oneout = tmpdir.join("one_out.hdf")

    graph.save_features(allout.strpath, format='hdf')
    graph.save_features(oneout.strpath, nodes=[1], format='hdf')

    graph_no_features = graph.copy()
    graph_no_features.load_features(allout.strpath, nodes=[1], format='hdf')
    assert graph.node[1].get_keypoints().all().all() == graph_no_features.node[1].get_keypoints().all().all()

def test_filter(graph):
    def edge_func(edge):
        return edge.matches is not None and hasattr(edge, 'matches')
    graph = graph.copy()
    test_sub_graph = graph.create_node_subgraph([0, 1])

    test_sub_graph.extract_features(extractor_parameters={'nfeatures': 25})
    test_sub_graph.match(k=2)

    filtered_nodes = graph.filter_nodes(lambda node: node.descriptors is not None)
    filtered_edges = graph.filter_edges(edge_func)

    assert filtered_nodes.number_of_nodes() == test_sub_graph.number_of_nodes()
    assert filtered_edges.number_of_edges() == test_sub_graph.number_of_edges()

def test_subset_graph(graph):
    g = graph
    edge_sub = g.create_edge_subgraph([(0, 2)])
    assert len(edge_sub.nodes()) == 2

    node_sub = g.create_node_subgraph([0, 1])
    assert len(node_sub) == 2

def test_subgraph_from_matches(graph):
    test_sub_graph = graph.create_node_subgraph([0, 1])
    test_sub_graph.extract_features(extractor_parameters={'nfeatures': 25})
    test_sub_graph.match(k=2)

    sub_graph_from_matches = graph.subgraph_from_matches()

    assert test_sub_graph.edges() == sub_graph_from_matches.edges()

def test_minimum_spanning_tree():
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

    assert sorted(mst_graph.nodes()) == sorted(graph.nodes())
    assert len(mst_graph.edges()) == len(graph.edges())-5

def test_fromlist():
    mock_list = ['AS15-M-0295_SML.png', 'AS15-M-0296_SML.png', 'AS15-M-0297_SML.png',
                 'AS15-M-0298_SML.png', 'AS15-M-0299_SML.png', 'AS15-M-0300_SML.png']

    good_poly = ogr.CreateGeometryFromWkt('POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))')
    bad_poly = ogr.CreateGeometryFromWkt('POLYGON ((9999 10, 40 40, 20 40, 10 20, 30 10))')

    with patch('plio.io.io_gdal.GeoDataset.footprint', new_callable=PropertyMock) as patch_fp:
        patch_fp.return_value = good_poly
        n = network.CandidateGraph.from_filelist(mock_list, get_path('Apollo15'))
        assert n.number_of_nodes() == 6
        assert n.number_of_edges() == 15

        patch_fp.return_value = bad_poly
        n = network.CandidateGraph.from_filelist(mock_list, get_path('Apollo15'))
        assert n.number_of_nodes() == 6
        assert n.number_of_edges() == 0

    n = network.CandidateGraph.from_filelist(mock_list, get_path('Apollo15'))
    assert len(n.nodes()) == 6

    n = network.CandidateGraph.from_filelist(get_path('adjacency.lis'), get_path('Apollo15'))
    assert len(n.nodes()) == 6

def test_apply_func_to_edges(graph):
    graph = graph.copy()
    mst_graph = graph.minimum_spanning_tree()

    try:
        graph.apply_func_to_edges('incorrect_func')
    except AttributeError:
        pass

    mst_graph.extract_features(extractor_parameters={'nfeatures': 50})
    mst_graph.match()
    mst_graph.apply_func_to_edges("symmetry_check")

    # Test passing the func by signature
    mst_graph.apply_func_to_edges(graph[0][1].symmetry_check)

    assert not graph[0][2].masks['symmetry'].all()
    assert not graph[0][1].masks['symmetry'].all()
