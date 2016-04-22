import itertools
import math
import os
import warnings

import dill as pickle
import networkx as nx
import numpy as np
import pandas as pd

from autocnet.control.control import C
from autocnet.fileio import io_hdf
from autocnet.fileio import io_json
from autocnet.fileio import io_utils
from autocnet.fileio.io_gdal import GeoDataset
from autocnet.graph import markov_cluster
from autocnet.graph.edge import Edge
from autocnet.graph.node import Node
from autocnet.matcher.matcher import FlannMatcher
from autocnet.vis.graph_view import plot_graph


class CandidateGraph(nx.Graph):
    """
    A NetworkX derived directed graph to store candidate overlap images.

    Parameters
    ----------

    Attributes
    node_counter : int
                   The number of nodes in the graph. 
    node_name_map : dict
                    The mapping of image labels (i.e. file base names) to their
                    corresponding node indices

    clusters : dict
               of clusters with key as the cluster id and value as a
               list of node indices
    ----------
    """
    edge_attr_dict_factory = Edge

    def __init__(self, *args, basepath=None, **kwargs):
        super(CandidateGraph, self).__init__(*args, **kwargs)
        self.node_counter = 0
        node_labels = {}
        self.node_name_map = {}

        for node_name in self.nodes():
            image_name = os.path.basename(node_name)
            image_path = node_name
            # Replace the default attr dict with a Node object
            self.node[node_name] = Node(image_name, image_path, self.node_counter)

            # fill the dictionary used for relabelling nodes with relative path keys
            node_labels[node_name] = self.node_counter
            # fill the dictionary used for mapping base name to node index
            self.node_name_map[self.node[node_name].image_name] = self.node_counter
            self.node_counter += 1

        nx.relabel_nodes(self, node_labels, copy=False)

        for s, d in self.edges():
            if s > d:
                s, d = d, s
            e = self.edge[s][d]
            e.source = self.node[s]
            e.destination = self.node[d]
            #del self.adj[d][s]

        # Add the Edge class as a edge data structure
        #for s, d, edge in self.edges_iter(data=True):
            #self.edge[s][d] = Edge(self.node[s], self.node[d])

    @classmethod
    def from_graph(cls, graph):
        """
        Return a graph object from a pickled file
        Parameters
        ----------
        graph : str
                PATH to the graph object

        Returns
        -------
        graph : object
                CandidateGraph object
        """
        with open(graph, 'rb') as f:
            graph = pickle.load(f)
        return graph

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
            datasets = [GeoDataset(os.path.join(basepath, f)) for f in filelist]
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
                warnings.warn('Missing or invalid geospatial data for {}'.format(i.base_name))

        # Grab the footprints and test for intersection
        for i, j in itertools.permutations(valid_datasets, 2):
            i_fp = i.footprint
            j_fp = j.footprint

            try:
                if i_fp.Intersects(j_fp):
                    adjacency_dict[i.file_name].append(j.file_name)
                    adjacency_dict[j.file_name].append(i.file_name)
            except:
                warnings.warn('Failed to calculated intersection between {} and {}'.format(i, j))

        return cls(adjacency_dict)


    @classmethod
    def from_adjacency(cls, input_adjacency, basepath=None):
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
        >>> candidate_graph = network.CandidateGraph.from_adjacency(inputfile)
        """
        if not isinstance(input_adjacency, dict):
            input_adjacency = io_json.read_json(input_adjacency)
            if basepath is not None:
                for k, v in input_adjacency.items():
                    input_adjacency[k] = [os.path.join(basepath, i) for i in v]
                    input_adjacency[os.path.join(basepath, k)] = input_adjacency.pop(k)
        return cls(input_adjacency)

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
        return self.node[node_index].image_name

    def add_image(self, *args, **kwargs):
        """
        Adds an image node to the graph.

        Parameters
        ----------

        """

        raise NotImplementedError

    def extract_features(self, method='orb', extractor_parameters={}):
        """
        Extracts features from each image in the graph and uses the result to assign the
        node attributes for 'handle', 'image', 'keypoints', and 'descriptors'.

        Parameters
        ----------
        method : {'orb', 'sift', 'fast'}
                 The descriptor method to be used

        extractor_parameters : dict
                               A dictionary containing OpenCV SIFT parameters names and values.

        downsampling : int
                       The divisor to image_size to down sample the input image.
        """
        for i, node in self.nodes_iter(data=True):
            image = node.get_array()
            node.extract_features(image, method=method,
                                  extractor_parameters=extractor_parameters)

    def save_features(self, out_path, nodes=[]):
        """

        Save the features (keypoints and descriptors) for the
        specified nodes.

        Parameters
        ----------
        out_path : str
                   Location of the output file.  If the file exists,
                   features are appended.  Otherwise, the file is created.

        nodes : list
                of nodes to save features for.  If empty, save for all nodes
        """

        if os.path.exists(out_path):
            mode = 'a'
        else:
            mode = 'w'

        hdf = io_hdf.HDFDataset(out_path, mode=mode)

        # Cleaner way to do this?
        if nodes:
            for i, n in self.subgraph(nodes).nodes_iter(data=True):
                n.save_features(hdf)
        else:
            for i, n in self.nodes_iter(data=True):
                n.save_features(hdf)

        hdf = None

    def load_features(self, in_path, nodes=[], nfeatures=None):
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
        hdf = io_hdf.HDFDataset(in_path, 'r')

        if nodes:
            for i, n in self.subgraph(nodes).nodes_iter(data=True):
                n.load_features(hdf)
        else:
            for i, n in self.nodes_iter(data=True):
                n.load_features(hdf)

        hdf = None

    def match_features(self, k=None):
        """
        For all connected edges in the graph, apply feature matching

        Parameters
        ----------
        k : int
            The number of matches to find per feature.
        """
        # Instantiate a single flann matcher to be resused for all nodes

        self._fl = FlannMatcher()
        for i, node in self.nodes_iter(data=True):

            # Grab the descriptors
            if not hasattr(node, 'descriptors'):
                raise AttributeError('Descriptors must be extracted before matching can occur.')
            descriptors = node.descriptors
            # Load the neighbors of the current node into the FLANN matcher
            neighbors = self.neighbors(i)

            # if node has no neighbors, skip
            if not neighbors:
                continue

            for n in neighbors:
                neighbor_descriptors = self.node[n].descriptors
                self._fl.add(neighbor_descriptors, n)
            self._fl.train()

            if k is None:
                k = (self.degree(i) * 2)

            # Query and then empty the FLANN matcher for the next node
            matches = self._fl.query(descriptors, i, k=k)
            self.add_matches(matches)

            self._fl.clear()

    def add_matches(self, matches):
        """
        Adds match data to a node and attributes the data to the
        appropriate edges, e.g. if A-B have a match, edge A-B is attributed
        with the pandas dataframe.

        Parameters
        ----------
        matches : dataframe
                  The pandas dataframe containing the matches
        """
        edges = self.edges()
        source_groups = matches.groupby('source_image')
        for i, source_group in source_groups:
            for j, dest_group in source_group.groupby('destination_image'):
                destination_key = int(dest_group['destination_image'].values[0])
                source_key = int(dest_group['source_image'].values[0])
                if (source_key, destination_key) in edges:
                    edge = self.edge[source_key][destination_key]
                else:
                    edge = self.edge[destination_key][source_key]
                    dest_group.rename(columns={'source_image': 'destination_image',
                                               'source_idx': 'destination_idx',
                                               'destination_image': 'source_image',
                                               'destination_idx': 'source_idx'},
                                      inplace=False)
                if hasattr(edge, 'matches'):
                    df = edge.matches
                    edge.matches = df.append(dest_group, ignore_index=True)
                else:
                    edge.matches = dest_group

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

    def apply_func_to_edges(self, function, *args, **kwargs):
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
        if not isinstance(function, str):
            function = function.__name__

        for s, d, edge in self.edges_iter(data=True):
            try:
                func = getattr(edge, function)
            except:
                raise AttributeError(function, ' is not an attribute of Edge')
            else:
                func(*args, **kwargs)

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

    def to_filelist(self):
        """
        Generate a file list for the entire graph.

        Returns
        -------

        filelist : list
                   A list where each entry is a string containing the full path to an image in the graph.
        """
        filelist = []
        for i, node in self.nodes_iter(data=True):
            filelist.append(node.image_path)
        return filelist

    def to_cnet(self, clean_keys=[], isis_serials=False):
        """
        Generate a control network (C) object from a graph

        Parameters
        ----------

        clean_keys : list
             of strings identifying the masking arrays to use, e.g. ratio, symmetry

        isis_serials : bool
                       Replace the node ID (nid) values with an ISIS
                       serial number. Default False

        Returns
        -------

        merged_cnet : C
                      A control network object
        """

        def _validate_cnet(cnet):
            """
            Once the control network is aggregated from graph edges,
            ensure that a given correspondence in a given image does
            not match multiple correspondences in a different image.

            Parameters
            ----------
            cnet : C
                   control network object

            Returns
            -------
             : C
               the cleaned control network
            """
            mask = np.zeros(len(cnet), dtype=bool)
            counter = 0
            for i, group in cnet.groupby('pid'):
                group_size = len(group)
                if len(group) != len(group['nid'].unique()):
                    mask[counter: counter + group_size] = False
                else:
                    mask[counter: counter + group_size] = True
                counter += group_size

            return cnet[mask]

        merged_cnet = None

        for source, destination, edge in self.edges_iter(data=True):
            matches = edge.matches

            # Merge all of the masks
            if clean_keys:
                matches, mask = edge._clean(clean_keys)

            subpixel = False
            point_type = 2
            if 'subpixel' in clean_keys:
                subpixel = True
                point_type = 3

            kp1 = self.node[source].get_keypoints()
            kp2 = self.node[destination].get_keypoints()
            pt_idx = 0
            values = []
            for i, (idx, row) in enumerate(matches.iterrows()):
                # Composite matching key (node_id, point_id)
                m1_pid = int(row['source_idx'])
                m2_pid = int(row['destination_idx'])
                m1 = (source, int(row['source_idx']))
                m2 = (destination, int(row['destination_idx']))


                values.append([kp1.loc[m1_pid]['x'],
                               kp1.loc[m1_pid]['y'],
                               m1,
                               pt_idx,
                               source,
                               idx,
                               point_type])

                if subpixel:
                    kp2x = kp2.loc[m2_pid]['x'] + row['x_offset']
                    kp2y = kp2.loc[m2_pid]['y'] + row['y_offset']
                else:
                    kp2x = kp2.loc[m2_pid]['x']
                    kp2y = kp2.loc[m2_pid]['y']

                values.append([kp2x,
                               kp2y,
                               m2,
                               pt_idx,
                               destination,
                               idx,
                               point_type])
                pt_idx += 1

            columns = ['x', 'y', 'idx', 'pid', 'nid', 'mid', 'point_type']

            cnet = C(values, columns=columns)

            if merged_cnet is None:
                merged_cnet = cnet.copy(deep=True)
            else:
                pid_offset = merged_cnet['pid'].max() + 1  # Get the current max point index
                cnet[['pid']] += pid_offset

                # Inner merge on the dataframe identifies common points
                common = pd.merge(merged_cnet, cnet, how='inner', on='idx', left_index=True, suffixes=['_r',
                                                                                                      '_l'])

                # Iterate over the points to be merged and merge them in.
                for i, r in common.iterrows():
                    new_pid = r['pid_r']
                    update_pid = r['pid_l']
                    cnet.loc[cnet['pid'] == update_pid, ['pid']] = new_pid  # Update the point ids

                # Perform the concat
                merged_cnet = pd.concat([merged_cnet, cnet])
                merged_cnet.drop_duplicates(['idx', 'pid'], keep='first', inplace=True)

        # Final validation to remove any correspondence with multiple correspondences in the same image
        merged_cnet = _validate_cnet(merged_cnet)

        # If the user wants ISIS serial numbers, replace the nid with the serial.
        if isis_serials is True:
            nid_to_serial = {}
            for i, node in self.nodes_iter(data=True):
                nid_to_serial[i] = node.isis_serial
            merged_cnet.replace({'nid': nid_to_serial}, inplace=True)

        return merged_cnet

    def to_json_file(self, outputfile):
        """
        Write the edge structure to a JSON adjacency list

        Parameters
        ----------

        outputfile : str
                     PATH where the JSON will be written
        """
        adjacency_dict = {}
        for n in self.nodes():
            adjacency_dict[n] = self.neighbors(n)
        io_json.write_json(adjacency_dict, outputfile)

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

    def save(self, filename):
        """
        Save the graph object to disk.
        Parameters
        ----------
        filename : str
                   The relative or absolute PATH where the network is saved
        """
        for i, node in self.nodes_iter(data=True):
            # Close the file handle because pickle doesn't handle SwigPyObjects
            node._handle = None

        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def plot(self, ax=None, **kwargs): # pragma: no cover
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
        return plot_graph(self, ax=ax,  **kwargs)

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
        H = self.__class__()
        adj = self.adj
        # Filter out edges that don't correspond to nodes in the graph.
        edges = ((u, v) for u, v in edges if u in adj and v in adj[u])
        for u, v in edges:
            # Copy the node attributes if they haven't been copied
            # already.
            if u not in H.node:
                H.node[u] = self.node[u]
            if v not in H.node:
                H.node[v] = self.node[v]
            # Create an entry in the adjacency dictionary for the
            # nodes u and v if they don't exist yet.
            if u not in H.adj:
                H.adj[u] = H.adjlist_dict_factory()
            if v not in H.adj:
                H.adj[v] = H.adjlist_dict_factory()
            # Copy the edge attributes.
            H.edge[u][v] = self.edge[u][v]
            #H.edge[v][u] = self.edge[v][u]
        H.graph = self.graph
        return H

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
        s = sum(self.degree(weight=weight).values()) / 2
        if weight is None:
            return math.ceil(s)
        else:
            return s

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
        bunch = set(self.nbunch_iter(nodes))
        # create new graph and copy subgraph into it
        H = self.__class__()
        # copy node and attribute dictionaries
        for n in bunch:
            H.node[n] = self.node[n]
        # namespace shortcuts for speed
        H_adj = H.adj
        self_adj = self.adj
        for i in H.node:
            adj_nodes = set(self.adj[i].keys()).intersection(bunch)
            H.adj[i] = {}
            for j, edge in self.adj[i].items():
                if j in adj_nodes:
                    H.adj[i][j] = edge

        H.graph = self.graph
        return H

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
        matches = [(u, v) for u, v, edge in self.edges_iter(data=True)
                   if hasattr(edge, 'matches') and
                   not edge.matches.empty]

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
        nodes = [n for n, d in self.nodes_iter(data=True) if func(d, *args, **kwargs)]
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
        edges = [(u, v) for u, v, edge in self.edges_iter(data=True) if func(edge, *args, **kwargs)]
        return self.create_edge_subgraph(edges)
