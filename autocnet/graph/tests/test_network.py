import copy
import os
import time
import sys

import pandas as pd
import pytest

from unittest.mock import patch, PropertyMock, MagicMock

import geopandas as gpd
import numpy as np
from osgeo import ogr
from plio.io import io_gdal

from autocnet.examples import get_path

from .. import network
from .. import edge
from .. import node

sys.path.insert(0, os.path.abspath('..'))


@pytest.fixture()
def graph():
    basepath = get_path('Apollo15')
    return network.CandidateGraph.from_adjacency(get_path('three_image_adjacency.json'),
                                                      basepath=basepath)

@pytest.fixture()
def geo_graph():
    basepath = get_path('Apollo15')
    a = 'AS15-M-0297_crop.cub'
    b = 'AS15-M-0298_crop.cub'
    c = 'AS15-M-0299_crop.cub'
    adjacency = {a:[b,c],
                 b:[a,c],
                 c:[a,b]}
    return network.CandidateGraph.from_adjacency(adjacency, basepath=basepath)

@pytest.fixture()
def disconnected_graph():
    return network.CandidateGraph.from_adjacency(get_path('adjacency.json'))

@pytest.fixture()
def candidategraph(node_a, node_b, node_c):
    # TODO: Getting this fixture from the global conf is causing deepycopy
    # to fail.  Why?
    cg = network.CandidateGraph()

    # Create a candidategraph object - we instantiate a real CandidateGraph to
    # have access of networkx functionality we do not want to test and then
    # mock all autocnet functionality to control test behavior.
    edges = [(0,1,{'data':edge.Edge(0,1)}),
             (0,2,{'data':edge.Edge(0,2)}),
             (1,2,{'data':edge.Edge(1,2)})]

    cg.add_edges_from(edges)



    match_indices = [([0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7]),
                     ([0,1,2,3,4,5,8,9], [0,1,2,3,4,5,8,9]),
                     ([0,1,2,3,4,5,8,9], [0,1,2,3,4,5,6,7])]

    matches = []
    for i, e in enumerate(edges):
        c = match_indices[i]
        source_image = np.repeat(e[0], 8)
        destin_image = np.repeat(e[1], 8)
        coords = np.zeros(8)
        data = np.vstack((source_image, c[0], destin_image, c[1],
                          coords, coords, coords, coords)).T
        matches_df = pd.DataFrame(data, columns=['source_image', 'source_idx', 'destination_image', 'destination_idx',
                                                 'source_x', 'source_y', 'destination_x', 'destination_y'])
        matches.append(matches_df)

    # Mock in autocnet methods
    cg.get_matches = MagicMock(return_value=matches)

    # Mock in the node objects onto the candidate graph
    cg.node[0]['data'] = node_a
    cg.node[1]['data'] = node_b
    cg.node[2]['data'] = node_c

    return cg

def test_get_name(graph):
    node_number = graph.graph['node_name_map']['AS15-M-0297_SML.png']
    name = graph.get_name(node_number)
    assert name == 'AS15-M-0297_SML.png'

def test_size(graph):
    assert graph.size() == graph.number_of_edges()
    for u, v, e in graph.edges.data('data'):
        e['edge_weight'] = 10

    assert graph.size('edge_weight') == graph.number_of_edges()*10

def test_serials(geo_graph):
    serials = ['APOLLO15/METRIC/1971-07-31T01:25:02.243',
               'APOLLO15/METRIC/1971-07-31T01:25:27.457',
               'APOLLO15/METRIC/1971-07-31T01:25:52.669']
    for s in serials:
        assert s in geo_graph.serials().values()

"""def test_fully_connected_components():
    G = network.CandidateGraph()
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('A', 'E'), ('A', 'F'), ('E', 'F') ])
    fc = G.compute_fully_connected_components()
    truth = [['A', 'B', 'C'], ['A', 'E', 'F']]
    sorted_fca = sorted(list(map(sorted, fc['A'])))
    assert truth == sorted_fca"""

def test_unique_fully_connected():
    G = network.CandidateGraph()
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('A', 'E'), ('A', 'F'), ('E', 'F') ])
    fc = G.compute_fully_connected_components()

def test_from_adjacency():
    basepath = get_path('Apollo15')
    a = 'AS15-M-0297_crop.cub'
    b = 'AS15-M-0298_crop.cub'
    c = 'AS15-M-0299_crop.cub'
    adjacency = {a:[b,c],
                 b:[a,c],
                 c:[a,b]}
    g =  network.CandidateGraph.from_adjacency(adjacency, basepath=basepath)
    assert len(g.nodes) == 3
    assert len(g.edges) == 3

    for s, d, e in g.edges.data('data'):
        assert isinstance(e, edge.Edge)
        assert isinstance(g.nodes[s]['data'], node.Node)

def test_add_node():
    basepath = get_path('Apollo15')
    a = 'AS15-M-0297_crop.cub'
    b = 'AS15-M-0298_crop.cub'
    c = 'AS15-M-0299_crop.cub'
    adjacency = {a:[b],
                 b:[a]}
    g = network.CandidateGraph.from_adjacency(adjacency, basepath=basepath)

    # Test without "image_name" arg (networkx parent method)
    g.add_node(2, data=node.Node(image_name=c,
                                    image_path=os.path.join(basepath, c),
                                    node_id=2))
    assert len(g.nodes) == 3
    assert g.node[2]["data"]["image_name"] == c

    # Test with "image_name" (cg method)
    g = network.CandidateGraph.from_adjacency(adjacency, basepath=basepath)
    g.add_node(image_name=c, basepath=basepath)
    assert len(g.nodes) == 3
    assert g.node[2]["data"]["image_name"] == c
    assert g.node[0].keys() == g.node[1].keys() == g.node[2].keys()

    # Test when "image_name" not found
    node_len = len(g.nodes)
    g.add_node(image_name="nonexistent.jpg")
    assert len(g.nodes) == node_len

def test_add_edge():
    basepath = get_path('Apollo15')
    a = 'AS15-M-0297_crop.cub'
    b = 'AS15-M-0298_crop.cub'
    c = 'AS15-M-0299_crop.cub'
    adjacency = {a:[b],
                 b:[a]}
    c_adj = ['AS15-M-0297_crop.cub', 'AS15-M-0298_crop.cub']
    g =  network.CandidateGraph.from_adjacency(adjacency, basepath=basepath)
    g.add_node(image_name=c, basepath=basepath, adjacency=c_adj)

    assert len(g.edges) == 3
    assert g.edges[0, 1]["data"].source == g.node[0]["data"]
    assert g.edges[0, 1]["data"].destination == g.node[1]["data"]
    assert g.edges[0, 2]["data"].source == g.node[0]["data"]
    assert g.edges[0, 2]["data"].destination == g.node[2]["data"]
    assert g.edges[1, 2]["data"].source == g.node[1]["data"]
    assert g.edges[1, 2]["data"].destination == g.node[2]["data"]
    assert g.edges[0, 1].keys() == g.edges[0, 2].keys() == g.edges[1, 2].keys()

    # Test when adj img not found
    g =  network.CandidateGraph.from_adjacency(adjacency, basepath=basepath)
    edge_len = len(g.edges)
    g.add_node(image_name=c, basepath=basepath, adjacency=["nonexistent.jpg"])
    assert len(g.edges) == edge_len

def test_equal(candidategraph):
    cg = copy.deepcopy(candidategraph)
    assert candidategraph == cg

    cg = copy.deepcopy(candidategraph)
    cg.remove_edge(0,1)
    assert candidategraph != cg

    cg = copy.deepcopy(candidategraph)
    cg.remove_node(0)
    assert candidategraph != cg

    cg = copy.deepcopy(candidategraph)
    cg.node[0]['image_name'] = 'foo'
    assert candidategraph != cg

    cg = copy.deepcopy(candidategraph)
    cg.edges[0,1]['data']['fundamental_matrix'] = np.random.random((3,3))
    assert candidategraph != cg

def test_get_matches(candidategraph):
    matches = candidategraph.get_matches()
    assert len(matches) == 3
    assert len(matches[0]) == 8
    assert isinstance(matches[0], pd.DataFrame)

def test_island_nodes(disconnected_graph):
    assert len(list(disconnected_graph.island_nodes())) == 1


def test_triangular_cycles(graph):
    cycles = graph.compute_triangular_cycles()
    # Node order is variable, length is not
    assert len(cycles) == 1


def test_connected_subgraphs(graph, disconnected_graph):
    # Calls all return generators, cast to list for positional comparison
    subgraph_list = list(disconnected_graph.connected_subgraphs())
    assert len(subgraph_list) == 2

    islands = list(disconnected_graph.island_nodes())
    assert islands[0] in subgraph_list[1]

    subgraph_list = list(graph.connected_subgraphs())
    assert len(subgraph_list) == 1


def test_filter(graph):

    test_sub_graph = graph.create_node_subgraph([0, 1])

    test_sub_graph.extract_features(extractor_parameters={'nfeatures': 25})
    test_sub_graph.match(k=2)

    filtered_nodes = graph.filter_nodes(lambda node: node.descriptors is not None)
    filtered_edges = graph.filter_edges(lambda edge: edge.matches.empty is not True)

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

    try:
        graph.apply_func_to_edges('incorrect_func')
    except AttributeError:
        pass

    graph.extract_features(extractor_parameters={'nfeatures': 50})
    graph.match()
    graph.apply_func_to_edges("symmetry_check")

    # Test passing the func by signature
    graph.apply_func_to_edges(graph[0][1]['data'].symmetry_check)
    assert not graph[0][2]['data'].masks.symmetry.all()
    #assert not mst_graph[0][1]['data'].masks.symmetry.all()


'''def test_intersection():
    # Generate the footprints for the mock nodes
    ogr_poly_list = []
    wkt0 = "MULTIPOLYGON (((2.5 7.5,7.5 7.5,7.5 12.5,2.5 12.5,2.5 7.5)))"
    ogr_poly_list.append(ogr.CreateGeometryFromWkt(wkt0))
    wkt1 = "MULTIPOLYGON (((0 10, 5 10, 5 15, 0 15, 0 10)))"
    ogr_poly_list.append(ogr.CreateGeometryFromWkt(wkt1))
    wkt2 = "MULTIPOLYGON (((5.5 5.0,10.5 5.0,10.5 10.0,5.5 10.0,5.5 5.0)))"
    ogr_poly_list.append(ogr.CreateGeometryFromWkt(wkt2))
    wkt3 = "MULTIPOLYGON (((5.5 7.5,10.5 7.5,10.5 12.5,5.5 12.5,5.5 7.5)))"
    ogr_poly_list.append(ogr.CreateGeometryFromWkt(wkt3))
    wkt4 = "MULTIPOLYGON (((8 11,13 11,13 16,8 16,8 11)))"
    ogr_poly_list.append(ogr.CreateGeometryFromWkt(wkt4))
    wkt5 = "MULTIPOLYGON (((8 14,13 14,13 19,8 19,8 14)))"
    ogr_poly_list.append(ogr.CreateGeometryFromWkt(wkt5))
    wkt6 = "MULTIPOLYGON (((11 11,16 11,16 16,11 16,11 11)))"
    ogr_poly_list.append(ogr.CreateGeometryFromWkt(wkt6))
    wkt7 = "MULTIPOLYGON (((11 14,16 14,16 19,11 19,11 14)))"
    ogr_poly_list.append(ogr.CreateGeometryFromWkt(wkt7))

    adj = {0: [1,2,3],
           1: [0],
           2: [0,3],
           3: [0,2,4],
           4: [3,5,7,6],
           5: [4,6,7],
           6: [4,5,7],
           7: [4,5,6]}

    cang = network.CandidateGraph.from_adjacency(adj)
    i = 0
    for d, node in cang.nodes.data('data'):
        #print(node, type(node), dir(node), node.geodata, type(node.geodata))
        node._geodata = MagicMock(spec=io_gdal.GeoDataset)
        node._geodata.footprint = ogr_poly_list[i]
        i += 1

    overlap, intersect_gdf = cang.compute_intersection(3)

    # Test the correct areas were found for the overlap and
    # the intersect_gdf
    assert intersect_gdf.geometry[0].area == 7.5
    assert intersect_gdf.geometry[1].area == 5
    assert intersect_gdf.geometry[2].area == 5
    assert intersect_gdf.geometry[3].area == 3.75
    assert overlap.geometry.area.values == 21.25'''


def test_set_maxsize(graph):
    maxsizes = network.MAXSIZE
    assert(graph.maxsize == maxsizes[0])
    graph.maxsize = 12
    assert(graph.maxsize == maxsizes[12])
    with pytest.raises(KeyError):
        graph.maxsize = 7

def test_update_data(graph):
   ctime = graph.graph['modifieddate']
   time.sleep(1)
   graph._update_date()
   ntime = graph.graph['modifieddate']
   assert ctime != ntime

def test_is_complete(graph):
    # Create a small incomplete graph with three nodes and two edges
    incomplete_graph = network.CandidateGraph()
    incomplete_graph.add_nodes_from([1, 2, 3])
    incomplete_graph.add_edges_from([(1, 2), (2, 3)])

    assert False == incomplete_graph.is_complete()
    assert True == graph.is_complete()

def test_get_matches(candidategraph):
    matches = candidategraph.get_matches()
    assert len(matches) == 3
    assert 'source_x' in matches[0].columns
    assert len(matches[0]) == 8

def test_apply(graph):
    def set_matches(e):
        e.matches = pd.DataFrame(['fake', 'fake', 'fake'])

    def get_matches(e):
        return e.matches

    graph.apply(set_matches)
    results = graph.apply(get_matches)

    for matches in results:
        assert len(matches) == 3

def test_tofilelist(graph):
    flist = graph.to_filelist()
    print(flist)
    truth = ['AS15-M-0297_SML.png', 'AS15-M-0298_SML.png', 'AS15-M-0299_SML.png']
    basenames = sorted([os.path.basename(i) for i in flist])
    assert truth == basenames

def test_footprints(geo_graph):
    # This is just testing the interface - should get a geodataframe back
    assert isinstance(geo_graph.footprints(), gpd.GeoDataFrame)
