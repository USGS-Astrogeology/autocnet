from collections import defaultdict, OrderedDict
import itertools
import json
import math
import os
from time import gmtime, strftime, time
import warnings

import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
from redis import StrictRedis

import shapely.affinity
import shapely.geometry
import shapely.wkt as swkt
import shapely.ops

from plio.io import io_hdf, io_json
from plio.utils import utils as io_utils
from plio.io.io_gdal import GeoDataset
from plio.io.isis_serial_number import generate_serial_number
from plio.io import io_controlnetwork as cnet

from plurmy import Slurm

from autocnet import Session, engine, config
from autocnet.cg import cg
from autocnet.graph import markov_cluster
from autocnet.graph.edge import Edge, NetworkEdge
from autocnet.graph.node import Node, NetworkNode
from autocnet.io import network as io_network
from autocnet.io.db.model import (Images, Keypoints, Matches, Cameras,
                                  Base, Overlay, Edges, Costs,
                                  Points, Measures)
from autocnet.io.db.connection import new_connection, Parent
from autocnet.vis.graph_view import plot_graph, cluster_plot
from autocnet.control import control
from autocnet.spatial.overlap import compute_overlaps_sql

#np.warnings.filterwarnings('ignore')

# The total number of pixels squared that can fit into the keys number of GB of RAM for SIFT.
MAXSIZE = {0: None,
           2: 6250,
           4: 8840,
           8: 12500,
           12: 15310}


class CandidateGraph(nx.Graph):
    """
    A NetworkX derived directed graph to store candidate overlap images.

    Attributes
    ----------

    node_counter : int
                   The number of nodes in the graph.
    node_name_map : dict
                    The mapping of image labels (i.e. file base names) to their
                    corresponding node indices

    clusters : dict
               of clusters with key as the cluster id and value as a
               list of node indices

    cn : object
         A control network object instantiated by calling generate_cnet.
    ----------
    """

    node_factory = Node
    edge_factory = Edge
    measures_keys = ['point_id', 'image_index', 'keypoint_index',
                     'edge', 'match_idx', 'x', 'y', 'x_off', 'y_off', 'corr']
    # dtypes are usful for allowing merges, otherwise they default to object
    cnet_dtypes = {
        'match_idx' : int,
        'point_id' : int,
        'image_index' : int,
        'keypoint_index' : int
    }

    def __init__(self, *args, basepath=None, node_id_map=None, overlaps=False, **kwargs):
        super(CandidateGraph, self).__init__(*args, **kwargs)

        self.graph['creationdate'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.graph['modifieddate'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.graph['node_name_map'] = {}
        self.graph['node_counter'] = 1

        self._point_id = 0
        self._measure_id = 0
        self.measure_to_point = {}
        self.controlnetwork = pd.DataFrame(columns=self.measures_keys).astype(self.cnet_dtypes)
        self.masks = pd.DataFrame()

        for i, n in self.nodes(data=True):
            if basepath:
                image_path = os.path.join(basepath, i)
            else:
                image_path = i

            if node_id_map:
                node_id = node_id_map[image_path]
            else:
                node_id = self.graph['node_counter']
                self.graph['node_counter'] += 1

            n['data'] = self.node_factory(
                image_name=i, image_path=image_path, node_id=node_id)

            self.graph['node_name_map'][i] = node_id

        # Relabel the nodes in place to use integer node ids
        nx.relabel_nodes(self, self.graph['node_name_map'], copy=False)
        for s, d, e in self.edges(data=True):
            if s > d:
                s, d = d, s
            edge = self.edge_factory(
                self.nodes[s]['data'], self.nodes[d]['data'])
            # Unidrected graph - both representation point at the same data
            self.edges[s, d]['data'] = edge
            self.edges[d, s]['data'] = edge

        if overlaps:
            self.compute_overlaps()

    def __key(self):
        # TODO: This needs to be a real self identifying key
        return 'abcde'

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        # Check the nodes
        if sorted(self.nodes()) != sorted(other.nodes()):
            return False
        for node in self.nodes:
            if not self.node[node] == other.node[node]:
                return False
        if sorted(self.edges()) != sorted(other.edges()):
            return False
        for s, d, e in self.edges.data('data'):
            if s > d:
                s, d = d, s
            if not e == other.edges[(s, d)]['data']:
                return False
        return True

    def _order_adjacency(self):  # pragma: no cover
        self.adj = OrderedDict(sorted(self.adj.items()))

    @property
    def maxsize(self):
        if not hasattr(self, '_maxsize'):
            self._maxsize = MAXSIZE[0]
        return self._maxsize

    @maxsize.setter
    def maxsize(self, value):
        if not value in MAXSIZE.keys():
            raise KeyError('Value must be in {}'.format(
                ','.join(map(str, MAXSIZE.keys()))))
        else:
            self._maxsize = MAXSIZE[value]

    @property
    def unmatched_edges(self):
        """
        Returns a list of edges (source, destination) that do not have
        entries in the matches dataframe.
        """
        unmatched = []
        for s, d, e in self.edges(data='data'):
            if len(e.matches) == 0:
                unmatched.append((s,d))

        return unmatched

    @classmethod
    def from_filelist(cls, filelist, basepath=None):
        """
        Instantiate the class using a filelist as a python list.
        An adjacency structure is calculated using the lat/lon information in the
        input images. Currently only images with this information are supported.

        Parameters
        ----------
        filelist : list
                   A list containing the files (with full paths) to construct an adjacency graph from

        Returns
        -------
        : object
          A Network graph object
        """
        if isinstance(filelist, str):
            filelist = io_utils.file_to_list(filelist)
        # TODO: Reject unsupported file formats + work with more file formats
        if basepath:
            datasets = [GeoDataset(os.path.join(basepath, f))
                        for f in filelist]
        else:
            datasets = [GeoDataset(f) for f in filelist]

        # This is brute force for now, could swap to an RTree at some point.
        adjacency_dict = {}
        valid_datasets = []

        for i in datasets:
            adjacency_dict[i.file_name] = []

            fp = i.footprint
            if fp and fp.IsValid():
                valid_datasets.append(i)
            else:
                warnings.warn(
                    'Missing or invalid geospatial data for {}'.format(i.base_name))

        # Grab the footprints and test for intersection
        for i, j in itertools.permutations(valid_datasets, 2):
            i_fp = i.footprint
            j_fp = j.footprint

            try:
                if i_fp.Intersects(j_fp):
                    adjacency_dict[i.file_name].append(j.file_name)
                    adjacency_dict[j.file_name].append(i.file_name)
            except:
                warnings.warn(
                    'Failed to calculate intersection between {} and {}'.format(i, j))
        return cls.from_adjacency(adjacency_dict)

    @classmethod
    def from_adjacency(cls, input_adjacency, node_id_map=None, basepath=None, **kwargs):
        """
        Instantiate the class using an adjacency dict or file. The input must contain relative or
        absolute paths to image files.

        Parameters
        ----------
        input_adjacency : dict or str
                          An adjacency dictionary or the name of a file containing an adjacency dictionary.

        Returns
        -------
         : object
           A Network graph object

        Examples
        --------
        >>> from autocnet.examples import get_path
        >>> inputfile = get_path('adjacency.json')
        >>> candidate_graph = CandidateGraph.from_adjacency(inputfile)
        >>> sorted(candidate_graph.nodes())
        [0, 1, 2, 3, 4, 5]
        """
        if not isinstance(input_adjacency, dict):
            input_adjacency = io_json.read_json(input_adjacency)
        return cls(input_adjacency, basepath=basepath, node_id_map=node_id_map, **kwargs)

    @classmethod
    def from_save(cls, input_file):
        return io_network.load(input_file)

    def _update_date(self):
        """
        Update the last modified date attribute.
        """
        self.graph['modifieddate'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    def get_name(self, node_index):
        """
        Get the image name for the given node.

        Parameters
        ----------
        node_index : int
                     The index of the node.

        Returns
        -------
         : str
           The name of the image attached to the given node.


        """
        return self.node[node_index]['data']['image_name']

    def get_matches(self, clean_keys=[]):
        matches = []
        for s, d, e in self.edges_iter(data=True):
            match, _ = e.clean(clean_keys=clean_keys)
            match = match[['source_image', 'source_idx',
                           'destination_image', 'destination_idx']]
            skps = e.get_keypoints('source', index=match.source_idx)
            skps.columns = ['source_x', 'source_y']
            dkps = e.get_keypoints('destination', index=match.destination_idx)
            dkps.columns = ['destination_x', 'destination_y']
            match = match.join(skps, on='source_idx')
            match = match.join(dkps, on='destination_idx')

            # TODO: This is a bandaid fix, join is creating an insane amount of duplicate points
            match = match.drop_duplicates()
            matches.append(match)

        return matches

    def add_node(self, n=None, **attr):
        """
        Adds an image node to the graph.

        Parameters
        ----------
        image_name : str
                     The file name of the node

        adjacency : str list
                    List of files names of adjacent images that correspond
                    to names in CandidateGraph.graph["node_name_map"]
        basepath : str
                    The base path to the node image file
        """

        image_name = attr.pop("image_name", None)
        adj = attr.pop("adjacency", None)
        new_node = None

        # If image name is provided, build the node from the image before
        # calling nx.add_node()
        if image_name is not None:
            if "basepath" in attr.keys():
                image_path = os.path.join(attr.pop("basepath"), image_name)
            else:
                image_path = image_name
            if not os.path.exists(image_path):
                warnings.warn("Cannot find {}".format(image_path))
                return
            n = self.graph["node_counter"]
            self.graph["node_counter"] += 1
            new_node = Node(image_name=image_name,
                            image_path=image_path,
                            node_id=n)
            self.graph["node_name_map"][new_node["image_name"]
                                        ] = new_node["node_id"]
            attr["data"] = new_node

        # Add the new node to the graph using networkx
        super(CandidateGraph, self).add_node(n, **attr)

        # Populate adjacency, if provided
        if new_node is not None and adj is not None:
            for adj_img in adj:
                if adj_img not in self.graph["node_name_map"].keys():
                    warnings.warn("{} not found in the graph".format(adj_img))
                    continue
                new_idx = new_node["node_id"]
                adj_idx = self.graph["node_name_map"][adj_img]
                self.add_edge(adj_img, new_node["image_name"])

    def add_edge(self, u, v, **attr):
        """
        Adds an edge with the given src and dst nodes to the graph

        Parameters
        ----------
        u : str
            The filename of the source image for the edge

        v : Node
            The filename of the destination image for the edge
        """
        if ("node_name_map" in self.graph.keys() and
            u in self.graph["node_name_map"].keys() and
                v in self.graph["node_name_map"].keys()):
            # Grab node ids & create edge obj
            s_id = self.graph["node_name_map"][u]
            d_id = self.graph["node_name_map"][v]
            new_edge = Edge(self.node[s_id]["data"], self.node[d_id]["data"])
            # Prepare data for networkx
            u = s_id
            v = d_id
            attr["data"] = new_edge
        # Add the new edge to the graph using networkx
        super(CandidateGraph, self).add_edge(u, v, **attr)

    def extract_features(self, band=1, *args, **kwargs):  # pragma: no cover
        """
        Extracts features from each image in the graph and uses the result to assign the
        node attributes for 'handle', 'image', 'keypoints', and 'descriptors'.
        """
        for i, node in self.nodes.data('data'):
            array = node.geodata.read_array(band=band)
            node.extract_features(array, *args, **kwargs),

    def extract_features_with_downsampling(self, downsample_amount=None, *args, **kwargs):  # pragma: no cover
        """
        Extract interest points from a downsampled array.  The array is downsampled
        by the downsample_amount keyword using the Lanconz downsample amount.  If the
        downsample keyword is not supplied, compute a downsampling constant as the
        total array size divided by the network maxsize attribute.

        Parameters
        ----------

        downsample_amount : int
                            The amount of downsampling to apply to the image
        """
        for node in self.nodes:
            if downsample_amount == None:
                total_size = node.geodata.raster_size[0] * \
                    node.geodata.raster_size[1]
                downsample_amount = math.ceil(total_size / self.maxsize**2)
            node.extract_features_with_downsampling(
                downsample_amount, *args, **kwargs)

    def extract_features_with_tiling(self, *args, **kwargs): #pragma: no cover
        """

        """
        self.apply(Node.extract_features_with_tiling, args=args, **kwargs)

    def save_features(self, out_path):
        """
        Save the features (keypoints and descriptors) for the
        specified nodes.

        Parameters
        ----------
        out_path : str
                   Location of the output file.  If the file exists,
                   features are appended.  Otherwise, the file is created.
        """
        self.apply(Node.save_features, args=(out_path,), on='node')

    def load_features(self, in_path, nodes=[], nfeatures=None, **kwargs):
        """
        Load features (keypoints and descriptors) for the
        specified nodes.

        Parameters
        ----------
        in_path : str
                  Location of the input file.

        nodes : list
                of nodes to load features for.  If empty, load features
                for all nodes
        """
        self.apply(Nodes.load_features, args=(in_path, nfeatures), on='node', **kwargs)
        for n in self.nodes:
            if node['node_id'] not in nodes:
                continue
            else:
                n.load_features(in_path, **kwargs)

    def match(self, *args, **kwargs):
        """
        For all connected edges in the graph, apply feature matching

        See Also
        ----------
        autocnet.graph.edge.Edge.match
        """
        self.apply_func_to_edges('match', *args, **kwargs)

    def decompose_and_match(self, *args, **kwargs):
        """
        For all edges in the graph, apply coupled decomposition followed by
        feature matching.

        See Also
        --------
        autocnet.graph.edge.Edge.decompose_and_match
        """
        self.apply_func_to_edges('decompose_and_match', *args, **kwargs)

    def estimate_mbrs(self, *args, **kwargs):
        """
        For each edge, estimate the overlap and compute a minimum bounding
        rectangle (mbr) in pixel space.

        See Also
        --------
        autocnet.graoh.edge.Edge.compute_mbr
        """
        self.apply_func_to_edges('estimate_mbr', *args, **kwargs)

    def compute_clusters(self, func=markov_cluster.mcl, *args, **kwargs):
        """
        Apply some graph clustering algorithm to compute a subset of the global
        graph.

        Parameters
        ----------
        func : object
               The clustering function to be applied.  Defaults to
               Markov Clustering Algorithm

        args : list
               of arguments to be passed through to the func

        kwargs : dict
                 of keyword arguments to be passed through to the func
        """
        _, self.clusters = func(self, *args, **kwargs)

    def compute_triangular_cycles(self):
        """
        Find all cycles of length 3.  This is similar
         to cycle_basis (networkX), but returns all cycles.
         As opposed to all basis cycles.

        Returns
        -------
        cycles : list
                 A list of cycles in the form [(a,b,c), (c,d,e)],
                 where letters indicate node identifiers

        Examples
        --------
        >>> g = CandidateGraph()
        >>> g.add_edges_from([(0,1), (0,2), (1,2), (0,3), (1,3), (2,3)])
        >>> sorted(g.compute_triangular_cycles())
        [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        """
        cycles = []
        for s, d in self.edges:
            for n in self.nodes:
                if(s, n) in self.edges and (d, n) in self.edges:
                    cycles.append(tuple(sorted([s, d, n])))
        return set(cycles)

    def minimum_spanning_tree(self):
        """
        Calculates the minimum spanning tree of the graph

        Returns
        -------

         : DataFrame
           boolean mask for edges in the minimum spanning tree
        """

        mst = nx.minimum_spanning_tree(self)
        return self.create_edge_subgraph(mst.edges())

    def apply_func_to_edges(self, function, nodes=[], *args, **kwargs):
        """
        Iterates over edges using an optional mask and and applies the given function.
        If func is not an attribute of Edge, raises AttributeError

        Parameters
        ----------
        function : obj
                   function to be called on every edge

        graph_mask_keys : list
                          of keys in graph_masks
        """
        return_lis = []
        if callable(function):
            function = function.__name__

        for s, d, edge in self.edges.data('data'):
            try:
                func = getattr(edge, function)
            except:
                raise AttributeError(function, ' is not an attribute of Edge')
            else:
                ret = func(*args, **kwargs)
                return_lis.append(ret)

        if any(return_lis):
            return return_lis

    def apply(self, function, on='edge', out=None, args=(), **kwargs):
        """
        Applys a function to every node or edge, returns collected return
        values.

        TODO: Merge with apply_func_to_edges?

        Parameters
        ----------
        function : callable
                   Function to apply to graph. Should accept (id, data).

        on : string
             Whether to use nodes or edges. default is 'edge'.

        out : var
              Optionally put the output in a variable rather than returning it

        args : iterable
               Some iterable of positional arguments for function.

        kwargs : dict
                 keyword args to pass into function.
        """
        options = {
            'edge': self.edges_iter,
            'edges': self.edges_iter,
            'e': self.edges_iter,
            0: self.edges_iter,
            'node': self.nodes_iter,
            'nodes': self.nodes_iter,
            'n': self.nodes_iter,
            1: self.nodes_iter
        }

        if not callable(function):
            raise TypeError('{} is not callable.'.format(function))

        res = []
        obj = 1
        # We just want to the object, not the indices, so slice appropriately
        if options[on] == self.edges_iter:
            obj = 2
        for elem in options[on](data=True):
            res.append(function(elem[obj], *args, **kwargs))

        if out:
            out = res
        else:
            return res

    def symmetry_checks(self):
        '''
        Apply a symmetry check to all edges in the graph
        '''
        self.apply_func_to_edges('symmetry_check')

    def ratio_checks(self, *args, **kwargs):
        '''
        Apply a ratio check to all edges in the graph

        See Also
        --------
        autocnet.matcher.cpu_outlier_detector.DistanceRatio.compute
        '''
        self.apply_func_to_edges('ratio_check', *args, **kwargs)

    def compute_overlaps(self, *args, **kwargs):
        '''
        Computes overlap MBRs for all edges
        '''
        self.apply_func_to_edges('compute_overlap', *args, **kwargs)

    def overlap_checks(self, *args, **kwargs):
        '''
        Apply overlap check to all edges in the graph
        '''
        self.apply_func_to_edges('overlap_check', *args, **kwargs)

    def compute_homographies(self, *args, **kwargs):
        '''
        Compute homographies for all edges using identical parameters

        See Also
        --------
        autocnet.graph.edge.Edge.compute_homography
        autocnet.matcher.cpu_outlier_detector.compute_homography
        '''
        self.apply_func_to_edges('compute_homography', *args, **kwargs)

    def compute_fundamental_matrices(self, *args, **kwargs):
        '''
        Compute fundmental matrices for all edges using identical parameters

        See Also
        --------
        autocnet.matcher.cpu_outlier_detector.compute_fundamental_matrix
        '''
        self.apply_func_to_edges('compute_fundamental_matrix', *args, **kwargs)

    def subpixel_register(self, *args, **kwargs):
        '''
        Compute subpixel offsets for all edges using identical parameters

        See Also
        --------
        autocnet.graph.edge.Edge.subpixel_register
        '''
        self.apply_func_to_edges('subpixel_register', *args, **kwargs)

    def suppress(self, *args, **kwargs):
        '''
        Apply a metric of point suppression to the graph

        See Also
        --------
        autocnet.matcher.cpu_outlier_detector.SpatialSuppression
        '''
        self.apply_func_to_edges('suppress', *args, **kwargs)

    def overlap(self):
        '''
        Compute the percentage and area coverage of two images

        See Also
        --------
        autocnet.cg.cg.two_image_overlap
        '''
        self.apply_func_to_edges('overlap')

    def to_filelist(self):
        """
        Generate a file list for the entire graph.

        Returns
        -------

        filelist : list
                   A list where each entry is a string containing the full path to an image in the graph.
        """
        filelist = []
        for i, node in self.nodes.data('data'):
            filelist.append(node['image_path'])
        return filelist

    def island_nodes(self):
        """
        Finds single nodes that are completely disconnected from the rest of the graph

        Returns
        -------
        : list
          A list of disconnected nodes, nodes of degree zero, island nodes, etc.
        """
        return nx.isolates(self)

    def connected_subgraphs(self):
        """
        Finds and returns a list of each connected subgraph of nodes. Each subgraph is a set.

        Returns
        -------

         : list
           A list of connected sub-graphs of nodes, with the largest sub-graph first. Each subgraph is a set.
        """
        return sorted(nx.connected_components(self), key=len, reverse=True)

    def serials(self):
        """
        Create a dictionary of ISIS3 compliant serial numbers for each
        node in the graph.

        Returns
        -------
        serials : dict
                  with key equal to the node id and value equal to
                  an ISIS3 compliant serial number or None
        """
        serials = {}
        for n, node in self.nodes.data('data'):
            serials[n] = generate_serial_number(node['image_path'])
        return serials

    @property
    def files(self):
        """
        Return a list of all full file PATHs in the CandidateGraph
        """
        return [node['image_path'] for _, node in self.nodes(data='data')]

    def save(self, filename):
        """
        Save the graph object to disk.
        Parameters
        ----------
        filename : str
                   The relative or absolute PATH where the network is saved
        """
        io_network.save(self, filename)

    def plot(self, ax=None, **kwargs):  # pragma: no cover
        """
        Plot the graph object

        Parameters
        ----------
        ax : object
             A MatPlotLib axes object.

        Returns
        -------
         : object
           A MatPlotLib axes object
        """
        return plot_graph(self, ax=ax, **kwargs)

    def plot_cluster(self, ax=None, **kwargs):  # pragma: no cover
        """
        Plot the graph based on the clusters generated by
        the markov clustering algorithm

        Parameters
        ----------
        ax : object
             A MatPlotLib axes object.

        Returns
        -------
        ax : object
             A MatPlotLib axes object.

        """
        return cluster_plot(self, ax, **kwargs)

    def create_node_subgraph(self, nodes):
        """
        Given a list of nodes, create a sub-graph and
        copy both the node and edge attributes to the subgraph.
        Changes to node/edge attributes are propagated back to the
        parent graph, while changes to the graph structure, i.e.,
        the topology, are not.

        Parameters
        ----------
        nodes : iterable
                An iterable (list, set, ndarray) of nodes to subset
                the graph

        Returns
        -------
        H : object
            A networkX graph object

        """
        return self.subgraph(nodes)

    def create_edge_subgraph(self, edges):
        """
        Create a subgraph using a list of edges.
        This is pulled directly from the networkx dev branch.

        Parameters
        ----------
        edges : list
                A list of edges in the form [(a,b), (c,d)] to retain
                in the subgraph

        Returns
        -------
        H : object
            A networkx subgraph object
        """
        return self.edge_subgraph(edges)

    def size(self, weight=None):
        """
        This replaces the built-in size method to properly
        support Python 3 rounding.

        Parameters
        ----------
        weight : string or None, optional (default=None)
           The edge attribute that holds the numerical value used
           as a weight.  If None, then each edge has weight 1.

        Returns
        -------
        nedges : int
            The number of edges or sum of edge weights in the graph.

        """
        if weight:
            return sum(e[weight] for s, d, e in self.edges.data('data'))
        else:
            return len(self.edges())

    def subgraph_from_matches(self):
        """
        Returns a sub-graph where all edges have matches.
        (i.e. images with no matches are removed)

        Returns
        -------
        : Object
          A networkX graph object
        """

        # get all edges that have matches
        matches = [(u, v) for u, v, edge in self.edges.data('data')
                   if not edge.matches.empty]

        return self.create_edge_subgraph(matches)

    def filter_nodes(self, func, *args, **kwargs):
        """
        Filters graph and returns a sub-graph from matches. Mimics
        python's filter() function

        Parameters
        ----------
        func : function which returns bool used to filter out nodes

        Returns
        -------
        : Object
          A networkX graph object

        """
        nodes = [node for i, node in self.nodes.data(
            'data') if func(node, *args, **kwargs)]
        return self.create_node_subgraph(nodes)

    def filter_edges(self, func, *args, **kwargs):
        """
        Filters graph and returns a sub-graph from matches. Mimics
        python's filter() function

        Parameters
        ----------
        func : function which returns bool used to filter out edges

        Returns
        -------
        : Object
          A networkX graph object
        """
        edges = [(u, v) for u, v, edge in self.edges.data(
            'data') if func(edge, *args, **kwargs)]
        return self.create_edge_subgraph(edges)

    def compute_cliques(self, node_id=None):  # pragma: no cover
        """
        Computes all maximum complete subgraphs for the given graph.
        If a node_id is given, method will return only the complete subgraphs that
        contain that node

        Parameters
        ----------
        node_id : int
                       Integer value for a given node

        Returns
        -------
        : list
          A list of lists of node ids that make up maximum complete subgraphs of the given graph
        """
        if node_id is not None:
            return list(nx.cliques_containing_node(self, nodes=node_id))
        else:
            return list(nx.find_cliques(self))

    def compute_weight(self, clean_keys, **kwargs):  # pragma: no cover
        """
        Computes a voronoi weight for each edge in a given graph.
        Can function as is, but is slightly optimized for complete subgraphs.
        ----------
        kwargs : dict
                      keyword arguments that get passed to compute_voronoi

        clean_keys : list
                     Strings used to apply masks to omit correspondences
        """

        if not self.is_connected():
            warnings.warn(
                'The given graph is not complete and may yield garbage.')

        for s, d, edge in self.edges.data('edge'):
            source_node = edge.source
            overlap, _ = self.compute_intersection(
                source_node, clean_keys=clean_keys)

            matches, _ = edge.clean(clean_keys)
            kps = edge.get_keypoints(edge.source, index=matches['source_idx'])[
                ['x', 'y']]
            reproj_geom = source_node.reproject_geom(
                overlap.geometry.values[0].__geo_interface__['coordinates'][0])
            initial_mask = cg.geom_mask(kps, reproj_geom)

            if (len(kps[initial_mask]) <= 0):
                continue

            kps['geometry'] = kps.apply(
                lambda x: shapely.geometry.Point(x['x'], x['y']), axis=1)
            kps_mask = kps['geometry'][initial_mask].apply(
                lambda x: reproj_geom.contains(x))
            voronoi_df = cg.compute_voronoi(
                kps[initial_mask][kps_mask], reproj_geom, **kwargs)

            edge['weights']['voronoi'] = voronoi_df

    def compute_unique_fully_connected_components(self, size=2):
        """
        Compute a list of all cliques with size greater than size.

        Parameters
        ----------
        size : int
               Only cliques larger than size are returned.  Default 2.

        Returns
        -------
         : list
           of lists of node ids

        Examples
        --------
        >>> G = CandidateGraph()
        >>> G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('A', 'E'), ('A', 'F'), ('E', 'F') ])
        >>> res = G.compute_unique_fully_connected_components()
        >>> sorted(map(sorted,res))
        [['A', 'B', 'C'], ['A', 'E', 'F']]
        """
        return [i for i in nx.enumerate_all_cliques(self) if len(i) > size]

    def compute_fully_connected_components(self):
        """
        For a given graph, compute all of the fully connected subgraphs with
        3+ components.

        Returns
        -------
        fc : list
             of lists of node identifiers

        Examples
        --------
        >>> G = CandidateGraph()
        >>> G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('A', 'E'), ('A', 'F'), ('E', 'F') ])
        >>> fc = G.compute_fully_connected_components()
        >>> len(fc) #A, B, C, E, A  - D is omitted because it is a singular terminal node
        5
        >>> sorted(map(sorted,fc['A']))  # Sort inner and outer lists
        [['A', 'B', 'C'], ['A', 'E', 'F']]
        """
        fully_connected = self.compute_unique_fully_connected_components()
        fc = defaultdict(list)
        for i in fully_connected:
            for j in i:
                fc[j].append(tuple(i))
        return fc

    def compute_intersection(self, source, clean_keys=[]):
        """
        Computes the intercetion of all images in a graph
        based around a given source node

        Parameters
        ----------
        source: object or int
                     Either a networkx Node object or an integer

        clean_keys : list
                           Strings used to apply masks to omit correspondences

        Returns
        -------
        intersect_gdf : dataframe
                               A geopandas dataframe of intersections for all images
                               that overlap with the source node. Also includes the common
                               overlap for all images in the source node.
        """
        if type(source) is int:
            source = self.node[source]['data']
        # May want to use a try except block here, but what error to raise?
        source_poly = swkt.loads(
            source.geodata.footprint.GetGeometryRef(0).ExportToWkt())

        source_gdf = gpd.GeoDataFrame(
            {'geometry': [source_poly], 'source_node': [source['node_id']]})

        proj_gdf = gpd.GeoDataFrame(columns=['geometry', 'proj_node'])
        proj_poly_list = []
        proj_node_list = []
        # Begin iterating through the edges in the graph that include the source node
        for s, d, edge in self.edges.data('data'):
            if s == source['node_id']:
                proj_poly = swkt.loads(
                    edge.destination.geodata.footprint.GetGeometryRef(0).ExportToWkt())
                proj_poly_list.append(proj_poly)
                proj_node_list.append(d)

            elif d == source['node_id']:
                proj_poly = swkt.loads(
                    edge.source.geodata.footprint.GetGeometryRef(0).ExportToWkt())
                proj_poly_list.append(proj_poly)
                proj_node_list.append(s)

        proj_gdf = gpd.GeoDataFrame(
            {"geometry": proj_poly_list, "proj_node": proj_node_list})
        # Overlay all geometry and find the one geometry element that overlaps all of the images
        intersect_gdf = gpd.overlay(source_gdf, proj_gdf, how='intersection')
        if len(intersect_gdf) == 0:
            raise ValueError(
                'Node ' + str(source['node_id']) + ' does not overlap with any other images in the candidate graph.')
        overlaps_mask = intersect_gdf.geometry.apply(
            lambda x: proj_gdf.geometry.contains(shapely.affinity.scale(x, .9, .9)).all())
        overlaps_all = intersect_gdf[overlaps_mask]

        # If there is no intersection polygon that overlaps all of the images, union all of the intersection
        # polygons into one large polygon that does overlap all of the images
        if len(overlaps_all) <= 0:
            new_poly = shapely.ops.unary_union(intersect_gdf.geometry)
            overlaps_all = gpd.GeoDataFrame({'source_node': source['node_id'], 'proj_node': source['node_id'],
                                             'geometry': [new_poly]})

        return overlaps_all, intersect_gdf

    def is_complete(self):
        """
        Checks if the graph is a complete graph
        """
        nneighbors = len(self) - 1
        for n in self.nodes:
            if self.degree(n) != nneighbors:
                return False
        return True

    def footprints(self):
        geoms = []
        names = []
        for i, node in self.nodes.data('data'):
            geoms.append(node.footprint)
            names.append(node['image_name'])

        return gpd.GeoDataFrame(names, geometry=geoms)

    def identify_potential_overlaps(self, **kwargs):
        cc = control.identify_potential_overlaps(
            self, self.controlnetwork, **kwargs)
        return cc

    def to_isis(self, outname, *args, **kwargs):
        serials = self.serials()
        files = self.files()
        self.controlnetwork.to_isis(outname, serials, files, *args, **kwargs)

    def nodes_iter(self, data=False):
        for i, n in self.nodes.data('data'):
            if data:
                yield i, n
            else:
                yield i

    def edges_iter(self, data=False):
        for s, d, e in self.edges.data('data'):
            if data:
                yield s, d, e
            else:
                yield s, d

    def generate_control_network(self, clean_keys=[], mask=None):
        """
        Generates a fresh control network from edge matches.

        parameters
        ----------
        clean_keys : list
                     A list of clean keys, same that would be used to filter edges

        mask


        """
        def add_measure(lis, key, edge, match_idx, fields, point_id=None):
            """
            Create a new measure that is coincident to a given point.  This method does not
            create the point if is missing.  When a measure is added to the graph, an associated
            row is added to the measures dataframe.

            Parameters
            ----------
            key : hashable
                      Some hashable id.  In the case of an autocnet graph object the
                      id should be in the form (image_id, match_id)

            point_id : hashable
                       The point to link the node to.  This is most likely an integer, but
                       any hashable should work.
            """
            if key in self.measure_to_point.keys():
                return
            if point_id == None:
                point_id = self._point_id
            self.measure_to_point[key] = point_id
            # The node_id is a composite key (image_id, correspondence_id), so just grab the image
            image_id = int(key[0])
            match_id = int(key[1])
            lis.append([point_id, image_id, match_id, edge, int(match_idx), *fields, 0, 0, np.inf])
            self._measure_id += 1

        # TODO: get rid of these wack variables
        self.measure_to_point = {}
        self._measure_id = 0
        self.point_id = 0

        matches = self.get_matches(clean_keys)
        cnet_lis = []
        for match in matches:
            for row in match.to_records():
                edge = (row.source_image, row.destination_image)
                source_key = (row.source_image, row.destination_image, row.source_idx)
                source_fields = [row.source_x, row.source_y]
                destin_key = (row.destination_image, row.source_image, row.destination_idx)
                destin_fields = [row.destination_x, row.destination_y]
                if self.measure_to_point.get(source_key, None) is not None:
                    tempid = self.measure_to_point[source_key]
                    add_measure(cnet_lis, destin_key, edge, row[0],
                                    destin_fields, point_id=tempid)
                elif self.measure_to_point.get(destin_key, None) is not None:
                    tempid = self.measure_to_point[destin_key]
                    add_measure(cnet_lis, source_key, edge, row[0],
                                    source_fields, point_id=tempid)
                else:
                    add_measure(cnet_lis, source_key, edge, row[0],  source_fields)
                    add_measure(cnet_lis, destin_key, edge, row[0],  destin_fields)
                    self._point_id += 1

        self.controlnetwork = pd.DataFrame(cnet_lis, columns=self.measures_keys)
        self.controlnetwork.index.name = 'measure_id'

    def remove_measure(self, idx):
        self.controlnetwork = self.controlnetwork.drop(
            self.controlnetwork.index[idx])
        for r in idx:
            self.measure_to_point.pop(r, None)

    def validate_points(self):
        """
        Ensure that all control points currently in the nework are valid.
        Criteria for validity:
        * Singularity: A control point can have one and only one measure from any image
        Returns
        -------
        : pd.Series
        """

        def func(g):
            # One and only one measure constraint
            if g.image_index.duplicated().any():
                return True
            else: return False

        return self.controlnetwork.groupby('point_id').apply(func)

    def clean_singles(self):
        """
        Take the `controlnetwork` dataframe and return only those points with
        at least two measures.  This is automatically called before writing
        as functions such as subpixel matching can result in orphaned measures.
        """
        return self.controlnetwork.groupby('point_id').apply(lambda g: g if len(g) > 1 else None)

    def to_isis(self, outname, serials, olist, *args, **kwargs):  # pragma: no cover
        """
        Write the control network out to the ISIS3 control network format.
        """

        if self.validate_points().any() == True:
            warnings.warn(
                'Control Network is not ISIS3 compliant.  Please run the validate_points method on the control network.')
            return

        # Apply the subpixel shift
        self.controlnetwork.x += self.controlnetwork.x_off
        self.controlnetwork.y += self.controlnetwork.y_off

        to_isis(outname + '.net', self.controlnetwork.query('valid == True'),
                serials, *args, **kwargs)
        write_filelist(olist, outname + '.lis')

        # Back out the subpixel shift
        self.controlnetwork.x -= self.controlnetwork.x_off
        self.controlnetwork.y -= self.controlnetwork.y_off

    def to_bal(self):
        """
        Write the control network out to the Bundle Adjustment in the Large
        (BAL) file format.  For more information see:
        http://grail.cs.washington.edu/projects/bal/
        """
        pass

class NetworkCandidateGraph(CandidateGraph):
    node_factory = NetworkNode
    edge_factory = NetworkEdge

    def __init__(self, *args, **kwargs):
        super(NetworkCandidateGraph, self).__init__(*args, **kwargs)
        if config.get('redis', None):
            self._setup_queues()
        # Job metadata
        self.job_status = defaultdict(dict)

        for i, d in self.nodes(data='data'):
            d.parent = self
        for s, d, e in self.edges(data='data'):
            e.parent = self

        # Execute the computation to compute overlapping geometries
        self._execute_sql(compute_overlaps_sql)

        # Setup the redis queues
        redis = config.get('redis')
        if redis:
            self.processing_queue = redis['processing_queue']

    def _setup_queues(self):
        """
        Setup a 2 queue redis connection for pushing and pulling work/results
        """
        conf = config['redis']

        self.redis_queue = StrictRedis(host=conf['host'],
                                       port=conf['port'],
                                       db=0)

    def empty_queues(self):
        """
        Delete all messages from the redis queue. This a convenience method.
        The `redis_queue` object is a redis-py StrictRedis object with API
        documented at: https://redis-py.readthedocs.io/en/latest/#redis.StrictRedis
        """
        return self.redis_queue.flushall()

    def _execute_sql(self, sql):
        """
        Execute a raw SQL string in the database currently specified
        by the AutoCNet config file.

        Use this method with caution as you can easily do things like
        truncate a table.

        Parameters
        ----------
        sql : str
              The SQL string to be passed to the DB engine and executed.
        """
        conn = engine.connect()
        conn.execute(sql)
        conn.close()

    def apply(self, function, on='edge', args=(), walltime='01:00:00', **kwargs):
        """
        A mirror of the apply function from the standard CandidateGraph object. This implementation
        dispatches the job to the cluster as an independent operation instead of applying an arbitrary function
        locally.

        This methods returns the number of jobs submitted. The job status is then asynchronously
        updated as the jobs complete.

        Parameters
        ----------

        function : obj
                   The function to apply

        on : str
             {'edge', 'edges', 'e', 0} for an edge
             {'node', 'nodes', 'n' 1} for a node

        args : tuple
               Of additional arguments to pass to the apply function

        walltime : str
                   in the format Hour:Minute:Second, 00:00:00
        """

        options = {
            'edge' : self.edges,
            'edges' : self.edges,
            'e' : self.edges,
            0 : self.edges,
            'node' : self.nodes,
            'nodes' : self.nodes,
            'n' : self.nodes,
            1 : self.nodes
        }

        # Determine which obj will be called
        onobj = options[on]

        res = []

        for job_counter, elem in enumerate(onobj.data('data')):
            # Determine if we are working with an edge or a node
            if len(elem) > 2:
                id = (elem[2].source['node_id'],
                      elem[2].destination['node_id'])
                image_path = (elem[2].source['image_path'],
                              elem[2].destination['image_path'])
            else:
                id = (elem[0])
                image_path = elem[1]['image_path']

            msg = {'id':id,
                    'func':function,
                    'args':args,
                    'kwargs':kwargs,
                    'walltime':walltime,
                    'image_path':image_path,
                    'param_step':1}

            self.redis_queue.rpush(self.processing_queue, json.dumps(msg))

        # SLURM is 1 based, while enumerate is 0 based
        job_counter += 1

        # Submit the jobs
        submitter = Slurm('acn_submit',
                     mem_per_cpu=config['cluster']['processing_memory'],
                     time=walltime,
                     partition=config['cluster']['queue'],
                     output=config['cluster']['cluster_log_dir']+'/slurm-%A_%a.out')
        submitter.submit(array='1-{}'.format(job_counter))
        return job_counter

    def generic_callback(self, msg):
        """
        This method manages the responses from the jobs and updates
        the status on this object. The msg is in a standard, parseable
        format.
        """
        id = msg['id']
        if isinstance(id, (int, float, str)):
            # Working with a node
            obj = self.nodes[id]['data']
        else:
            obj = self.edges[id]['data']
            # Working with an edge

        func = msg['func']
        obj.job_status[func]['success'] = msg['success']

        # If the job was successful, no need to resubmit
        if msg['success'] == True:
            return

    def generate_vrts(self, **kwargs):
        """
        For the nodes in the graph, genreate a GDAL compliant vrt file.
        This is just a dispatcher to the knoten generate_vrt file.
        """
        for _, n in self.nodes(data='data'):
            n.generate_vrt(**kwargs)

    def to_isis(self, path, flistpath=None,sql = """
SELECT points.id, measures.serial, points.pointtype, measures.sample, measures.line, measures.measuretype,
measures.imageid
FROM measures INNER JOIN points ON measures.pointid = points.id
WHERE points.active = True AND measures.active=TRUE AND measures.jigreject=FALSE;
"""):
        """
        Given a set of points/measures in an autocnet database, generate an ISIS
        compliant control network.

        Parameters
        ----------
        path : str
               The full path to the output network.

        flistpath : str
                    (Optional) the path to the output filelist. By default
                    the outout filelist path is genrated programatically
                    as the provided path with the extension replaced with .lis.
                    For example, out.net would have an associated out.lis file.

        sql : str
              The sql query to execute in the database.

        """

        df = pd.read_sql(sql, engine)
        df.rename(columns={'imageid':'image_index','id':'point_id',
                           'sample':'x', 'line':'y'}, inplace=True)
        if flistpath is None:
            flistpath = os.path.splitext(path)[0] + '.lis'

        cnet.to_isis(path, df, self.serials())
        cnet.write_filelist(self.files, path=flistpath)

    @staticmethod
    def update_from_jigsaw(session, path):
        """
        Updates the measures table in the database with data from
        a jigsaw bundle adjust

        Parameters
        ----------
        path : str
               Full path to a bundle adjusted isis control network
        """
        # Ingest isis control net as a df and do some massaging
        data = cnet.from_isis(path)
        data['jigsawFullRejected'] = data['pointJigsawRejected'] | data['jigsawRejected']
        data_to_update = data[['id', 'serialnumber', 'jigsawFullRejected', 'sampleResidual', 'lineResidual', 'samplesigma', 'linesigma', 'adjustedCovar', 'apriorisample', 'aprioriline']]
        data_to_update = data_to_update.rename(columns = {'serialnumber': 'serial', 'jigsawFullRejected': 'jigreject', 'sampleResidual': 'sampler', 'lineResidual': 'liner', 'adjustedCovar': 'covar'})
        data_to_update['covar'] = data_to_update['covar'].apply(lambda row : list(row))
        data_to_update['id'] = data_to_update['id'].apply(lambda row : int(row))

        # Generate a temp table, update the real table, then drop the temp table
        data_to_update.to_sql('temp_measures', engine, if_exists='replace', index_label='serialnumber', index = False)

        sql = """
        UPDATE measures AS f
        SET jigreject = t.jigreject, sampler = t.sampler, liner = t.liner, samplesigma = t.samplesigma, linesigma = t.linesigma, apriorisample = t.apriorisample, aprioriline = t.aprioriline
        FROM temp_measures AS t
        WHERE f.serial = t.serial AND f.pointid = t.id;

        DROP TABLE temp_measures;
        """

        session.execute(sql)
        session.commit()

    @classmethod
    def from_filelist(cls, filelist):
        """
        Parse a filelist to add nodes to the database. Using the
        information in the database, then instantiate a complete,
        NCG.

        Parameters
        ----------
        filelist : list, str
                   If a list, this is a list of paths. If a str, this is
                   a path to a file containing a list of image paths
                   that is newline ("\\n") delimited.

        Returns
        -------
        ncg : object
              A network candidate graph object
        """
        if isinstance(filelist, list):
            pass
        elif os.path.exists(filelist):
            filelist = io_utils.file_to_list(filelist)
        else:
            warning.warn('Unable to parse the passed filelist')

        for f in filelist:
            # Create the nodes in the graph. Really, this is creating the
            # images in the DB
            image_name = os.path.basename(f)
            NetworkNode(image_path=f, image_name=image_name)
        
        return cls.from_database()
        
    @classmethod
    def from_database(cls, query_string='SELECT * FROM public.images'):
        """
        This is a constructor that takes the results from an arbitrary query string,
        uses those as a subquery into a standard polygon overlap query and
        returns a NetworkCandidateGraph object.  By default, an images
        in the Image table will be used in the outer query.

        Parameters
        ----------
        query_string : str
                       A valid SQL select statement that targets the Images table

        Usage
        -----
        Here, we provide usage examples for a few, potentially common use cases.

        ## Spatial Query
        This example selects those images that intersect a given bounding polygon.  The polygon is
        specified as a Well Known Text LINESTRING with the first and last points being the same.
        The query says, select the footprint_latlon (the bounding polygons in the database) that
        intersect the user provided polygon (the LINESTRING) in the given spatial reference system
        (SRID), 949900.

        "SELECT * FROM Images WHERE ST_INTERSECTS(footprint_latlon, ST_Polygon(ST_GeomFromText('LINESTRING(159 10, 159 11, 160 11, 160 10, 159 10)'),949900)) = TRUE"
from_database
        ## Select from a specific orbit
        This example selects those images that are from a particular orbit. In this case,
        the regex string pulls all P##_* orbits and creates a graph from them. This method
        does not guarantee that the graph is fully connected.

        "SELECT * FROM Images WHERE (split_part(path, '/', 6) ~ 'P[0-9]+_.+') = True"

        """
        composite_query = """WITH
	i as ({})
SELECT i1.id as i1_id,i1.path as i1_path, i2.id as i2_id, i2.path as i2_path
FROM
	i as i1, i as i2
WHERE ST_INTERSECTS(i1.footprint_latlon, i2.footprint_latlon) = TRUE
AND i1.id < i2.id""".format(query_string)

        session = Session()
        res = session.execute(composite_query)

        adjacency = defaultdict(list)
        adjacency_lookup = {}
        for r in res:
            sid, spath, did, dpath = r

            adjacency_lookup[spath] = sid
            adjacency_lookup[dpath] = did
            if spath != dpath:
                adjacency[spath].append(dpath)
        session.close()
        # Add nodes that do not overlap any images
        obj = cls(adjacency, node_id_map=adjacency_lookup, config=config)

        return obj
