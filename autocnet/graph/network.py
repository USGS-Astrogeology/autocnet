import os

import networkx as nx
import pandas as pd
import cv2
from pysal.cg.shapes import Polygon
import numpy as np

from scipy.misc import bytescale

from autocnet.control.control import C
from autocnet.fileio import io_json
from autocnet.fileio.io_gdal import GeoDataset
from autocnet.matcher import feature_extractor as fe
from autocnet.matcher import outlier_detector as od
from autocnet.matcher import subpixel as sp
from autocnet.cg.cg import convex_hull_ratio, overlapping_polygon_area
from autocnet.vis.graph_view import plot_node, plot_edge


class Edge(object):
    """
    Attributes
    ----------
    source : hashable
             The source node
    destination : hashable
                  The destination node
    masks : set
            A list of the available masking arrays
    """

    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self._masks = set()
        self._mask_arrays = {}
        self._homography = None
        self._subpixel_offsets = None

    @property
    def masks(self):
        return self._masks

    @masks.setter
    def masks(self, v):
        self._masks.add(v[0])
        self._mask_arrays[v[0]] = v[1]

    @property
    def homography(self):
        return self._homography

    @homography.setter
    def homography(self, v):
        self._homography = v

    @property
    def subpixel_offsets(self):
       return self._subpixel_offsets

    @subpixel_offsets.setter
    def subpixel_offsets(self, v):
        self._subpixel_offsets = v

    def symmetry_check(self):
        if hasattr(self, 'matches'):
            mask = od.mirroring_test(self.matches)
            self.masks = ('symmetry', mask)
        else:
            raise AttributeError('No matches have been computed for this edge.')

    def ratio_check(self, ratio=0.8):
        if hasattr(self, 'matches'):
            mask = od.distance_ratio(self.matches, ratio=ratio)
            self.masks = ('ratio', mask)
        else:
            raise AttributeError('No matches have been computed for this edge.')

    def compute_homography(self, outlier_algorithm=cv2.RANSAC, clean_keys=[]):
        """
        For each edge in the (sub) graph, compute the homography
        Parameters
        ----------
        outlier_algorithm : object
                            An openCV outlier detections algorithm, e.g. cv2.RANSAC

        clean_keys : list
                     of string keys to masking arrays
                     (created by calling outlier detection)
        Returns
        -------
        transformation_matrix : ndarray
                                The 3x3 transformation matrix

        mask : ndarray
               Boolean array of the outliers
        """

        if hasattr(self, 'matches'):
            matches = self.matches
        else:
            raise AttributeError('Matches have not been computed for this edge')

        if clean_keys:
            mask = np.prod([self._mask_arrays[i] for i in clean_keys], axis=0, dtype=np.bool)
            matches = matches[mask]
            full_mask = np.where(mask == True)

        s_keypoints = self.source.keypoints.iloc[matches['source_idx'].values]
        d_keypoints = self.destination.keypoints.iloc[matches['destination_idx'].values]

        transformation_matrix, ransac_mask = od.compute_homography(s_keypoints[['x', 'y']].values,
                                                                   d_keypoints[['x', 'y']].values)

        ransac_mask = ransac_mask.ravel()
        # Convert the truncated RANSAC mask back into a full length mask
        if clean_keys:
            mask[full_mask] = ransac_mask
        else:
            mask = ransac_mask
        self.masks = ('ransac', mask)
        self.homography = transformation_matrix

    def compute_subpixel_offset(self, clean_keys=[], threshold=0.8, upsampling=10,
                                 template_size=9, search_size=39):
        """
        For the entire graph, compute the subpixel offsets using pattern-matching and add the result
        as an attribute to each edge of the graph.

        Parameters
        ----------
        clean_keys : list
             of string keys to masking arrays
             (created by calling outlier detection)

        threshold : float
                    On the range [-1, 1].  Values less than or equal to
                    this threshold are masked and can be considered
                    outliers

        upsampling : int
                     The multiplier to the template and search shapes to upsample
                     for subpixel accuracy

        template_size : int
                        The size of the template in pixels, must be odd

        search_size : int
                      The size of the search
        """

        matches = self.matches

        full_offsets = np.zeros((len(matches), 3))

        # Build up a composite mask from all of the user specified masks
        if clean_keys:
            mask = np.prod([self._mask_arrays[i] for i in clean_keys], axis=0, dtype=np.bool)
            matches = matches[mask]
            full_mask = np.where(mask == True)

        # Preallocate the numpy array to avoid appending and type conversion
        edge_offsets = np.empty((len(matches),3))

        # for each edge, calculate this for each keypoint pair
        for i, (idx, row) in enumerate(matches.iterrows()):

            s_idx = int(row['source_idx'])
            d_idx = int(row['destination_idx'])

            s_keypoint = self.source.keypoints.iloc[s_idx][['x', 'y']].values
            d_keypoint = self.destination.keypoints.iloc[d_idx][['x', 'y']].values

            # Get the template and search windows
            s_template = sp.clip_roi(self.source.handle, s_keypoint, template_size)
            d_search = sp.clip_roi(self.destination.handle, d_keypoint, search_size)

            edge_offsets[i] = sp.subpixel_offset(s_template, d_search, upsampling=upsampling)

        # Compute the mask for correlations less than the threshold
        threshold_mask = edge_offsets[edge_offsets[:, -1] >= threshold]

        # Convert the truncated mask back into a full length mask
        if clean_keys:
            mask[full_mask] = threshold_mask
            full_offsets[full_mask] = edge_offsets
        else:
            mask = threshold_mask

        self.subpixel_offsets = pd.DataFrame(full_offsets, columns=['x_offset',
                                                                             'y_offset',
                                                                             'correlation'])
        self.masks = ('subpixel', mask)

    def coverage_ratio(self, clean_keys=[]):
        """
        Compute the ratio $area_{convexhull} / area_{imageoverlap}$.

        Returns
        -------
        ratio : float
                The ratio $area_{convexhull} / area_{imageoverlap}$
        """
        if self.homography is None:
            raise AttributeError('A homography has not been computed. Unable to determine image overlap.')

        matches = self.matches
        # Build up a composite mask from all of the user specified masks
        if clean_keys:
            mask = np.prod([self._mask_arrays[i] for i in clean_keys], axis=0, dtype=np.bool)
            matches = matches[mask]

        d_idx = matches['destination_idx'].values
        keypoints = self.destination.keypoints.iloc[d_idx][['x', 'y']].values
        if len(keypoints) < 3:
            raise ValueError('Convex hull computation requires at least 3 measures.')

        source_geom, proj_geom, ideal_area = self.compute_homography_overlap()

        ratio = convex_hull_ratio(keypoints, ideal_area)
        return ratio

    def compute_homography_overlap(self):
        """
        Using the homography, estimate the overlapping area
        between images on the edge

        Returns
        -------
        source_geom : object
                      PySAL Polygon object of the source pixel bounding box

        projected_geom : object
                         PySAL Polygon object of the destination geom projected
                         into the source reference system using the current
                         homography

        area : float
               The estimated area
        """

        source_geom = self.source.handle.pixel_polygon
        destination_geom = self.destination.handle.pixel_polygon

        # Project using the homography
        vertices_to_project = destination_geom.vertices
        for i, v in enumerate(vertices_to_project):
            vertices_to_project[i] = tuple(np.array([v[0], v[1], 1]).dot(self.homography)[:2])
        projected_geom = Polygon(vertices_to_project)

        # Estimate the overlapping area
        area = overlapping_polygon_area([source_geom, projected_geom])

        return source_geom, projected_geom, area

    def plot(self, ax=None, clean_keys=[], **kwargs):
        return plot_edge(self, ax=ax, clean_keys=clean_keys, **kwargs)

    def update(self, *args):
        # Added for NetworkX
        pass


class Node(object):
    """
    Attributes
    ----------

    image_name : str
                 Name of the image, with extension
    image_path : str
                 Relative or absolute PATH to the image
    handle : object
             File handle to the object
    keypoints : dataframe
                With columns, x, y, and response
    nkeypoints : int
                 The number of keypoints found for this image
    descriptors : ndarray
                  32-bit array of feature descriptors returned by OpenCV
    masks : set
            A list of the available masking arrays
    """

    def __init__(self, image_name, image_path):
        self.image_name = image_name
        self.image_path = image_path
        self._masks = set()
        self._mask_arrays = {}

    @property
    def handle(self):
        if not getattr(self, '_handle', None):
            self._handle = GeoDataset(self.image_path)
        return self._handle

    @property
    def nkeypoints(self):
        if hasattr(self, '_nkeypoints'):
            return self._nkeypoints
        else:
            return 0

    @nkeypoints.setter
    def nkeypoints(self, v):
        self._nkeypoints = v

    @property
    def masks(self):
        return self._masks

    @masks.setter
    def masks(self, v):
        self._masks.add(v[0])
        self._mask_arrays[v[0]] = v[1]

    def get_array(self, band=1):
        """
        Get a band as a 32-bit numpy array

        Parameters
        ----------
        band : int
               The band to read, default 1
        """

        array = self.handle.read_array(band=band)
        return bytescale(array)

    def extract_features(self, array, **kwargs):
        """
        Extract features for the node

        Parameters
        ----------
        array : ndarray

        kwargs : dict
                 KWargs passed to autocnet.feature_extractor.extract_features

        """
        keypoint_objs, descriptors = fe.extract_features(array, **kwargs)
        keypoints = np.empty((len(keypoint_objs), 4),dtype=np.float32)
        for i, kpt in enumerate(keypoint_objs):
            keypoints[i] = kpt.pt[0], kpt.pt[1], kpt.response, kpt.size  # y, x
        self.keypoints = pd.DataFrame(keypoints, columns=['x', 'y', 'response', 'size'])
        self._nkeypoints = len(self.keypoints)
        self.descriptors = descriptors.astype(np.float32)

    def anms(self, nfeatures=100, robust=0.9):
        mask = od.adaptive_non_max_suppression(self.keypoints,nfeatures,robust)
        self.masks = ('anms', mask)

    def coverage_ratio(self, clean_keys=[]):
        """
        Compute the ratio $area_{convexhull} / area_{total}$

        Returns
        -------
        ratio : float
                The ratio of convex hull area to total area.
        """
        ideal_area = self.handle.pixel_area
        if not hasattr(self, 'keypoints'):
            raise AttributeError('Keypoints must be extracted already, they have not been.')

        if clean_keys:
            mask = np.prod([self._mask_arrays[i] for i in clean_keys], axis=0, dtype=np.bool)
            keypoints = self.keypoints[mask]

        keypoints = self.keypoints[['x', 'y']].values

        ratio = convex_hull_ratio(keypoints, ideal_area)
        return ratio

    def plot(self, clean_keys=[], **kwargs):
        return plot_node(self, clean_keys=clean_keys, **kwargs)

    def update(self, *args):
        # Empty pass method to get NetworkX to accept a non-dict
        pass


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
                    corresponding node indices.

    ----------
    """

    def __init__(self,*args, basepath=None, **kwargs):
        super(CandidateGraph, self).__init__(*args, **kwargs)
        self.node_counter = 0
        node_labels = {}
        self.node_name_map = {}

        # the node_name is the relative path for the image
        for node_name, node in self.nodes_iter(data=True):
            image_name = os.path.basename(node_name)
            image_path = node_name

            # Replace the default node dict with an object
            self.node[node_name] = Node(image_name, image_path)

            # fill the dictionary used for relabelling nodes with relative path keys
            node_labels[node_name] = self.node_counter
            # fill the dictionary used for mapping base name to node index
            self.node_name_map[self.node[node_name].image_name] = self.node_counter
            self.node_counter += 1

        nx.relabel_nodes(self, node_labels, copy=False)

        # Add the Edge class as a edge data structure
        for s, d, edge in self.edges_iter(data=True):
            self.edge[s][d] = Edge(self.node[s], self.node[d])

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
                    input_adjacency[k] =  [os.path.join(basepath, i) for i in v]
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

    def get_node(self, node_name):
        """
        Get the node with the given name.

        Parameters
        ----------
        node_name : str
                    The name of the node.
        
        Returns
        -------
         : object
           The node with the given image name.


        """
        return self.node[self.node_name_map[node_name]]

    def get_keypoints(self, nodekey):
        """
        Get the list of keypoints for the given node.
        
        Parameters
        ----------
        nodeIndex : int or string
                    The key for the node, by index or name.
        
        Returns
        -------
         : list
           The list of keypoints for the given node.
        
        """
        try:
            return self.get_node(nodekey).keypoints
        except:
            return self.node[nodekey].keypoints

    def add_image(self, *args, **kwargs):
        """
        Adds an image node to the graph.

        Parameters
        ----------

        """

        raise NotImplementedError
        self.add_node(self.node_counter, *args, **kwargs)
        #self.node_labels[self.node[self.node_counter]['image_name']] = self.node_counter
        self.node_counter += 1

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

    def add_matches(self, matches):
        """
        Adds match data to a node and attributes the data to the
        appropriate edges, e.g. if A-B have a match, edge A-B is attributes
        with the pandas dataframe.

        Parameters
        ----------
        source_node : str
                      The identifier for the node

        matches : dataframe
                  The pandas dataframe containing the matches
        """
        source_groups = matches.groupby('source_image')
        for i, source_group in source_groups:
            for j, dest_group in source_group.groupby('destination_image'):
                source_key = dest_group['source_image'].values[0]
                destination_key = dest_group['destination_image'].values[0]
                try:
                    edge = self.edge[source_key][destination_key]
                except: # pragma: no cover
                    edge = self.edge[destination_key][source_key]

                if hasattr(edge, 'matches'):
                    df = edge.matches
                    edge.matches = pd.concat([df, dest_group])
                else:
                    edge.matches = dest_group

    def symmetry_checks(self):
        """
        Perform a symmetry check on all edges in the graph
        """
        for s, d, edge in self.edges_iter(data=True):
            edge.symmetry_check()

    def ratio_checks(self, ratio=0.8):
        """
        Perform a ratio check on all edges in the graph
        """
        for s, d, edge in self.edges_iter(data=True):
            edge.ratio_check(ratio=ratio)

    def compute_homographies(self, outlier_algorithm=cv2.RANSAC, clean_keys=[]):
        """
        Compute homographies for all edges using identical parameters

        Parameters
        ----------
        outlier_algorithm : object
                            Function to apply for outlier detection

        clean_keys : list
                     Of keys in the mask dict

        """

        for s, d, edge in self.edges_iter(data=True):
            edge.compute_homography(outlier_algorithm=outlier_algorithm,
                                    clean_keys=clean_keys)

    def compute_subpixel_offsets(self, clean_keys=[], threshold=0.8, upsampling=10,
                                 template_size=9, search_size=27):
         """
         Compute subpixel offsets for all edges using identical parameters
         """
         for s, d, edge in self.edges_iter(data=True):
             edge.compute_subpixel_offset(clean_keys=clean_keys, threshold=threshold,
                                          upsampling=upsampling, template_size=template_size,
                                          search_size=search_size)

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

    def to_cnet(self, clean_keys=[]):
        """
        Generate a control network (C) object from a graph

        Parameters
        ----------
        clean_keys : list
             of strings identifying the masking arrays to use, e.g. ratio, symmetry

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
                mask = np.prod([edge._mask_arrays[i] for i in clean_keys], axis=0, dtype=np.bool)
                matches = matches[mask]

            if 'subpixel' in clean_keys:
                offsets = edge.subpixel_offsets

            kp1 = self.node[source].keypoints
            kp2 = self.node[destination].keypoints
            pt_idx = 0
            values = []
            for i, (idx, row) in enumerate(matches.iterrows()):
                # Composite matching key (node_id, point_id)
                m1_pid = int(row['source_idx'])
                m2_pid = int(row['destination_idx'])
                m1 = (source, int(row['source_idx']))
                m2 = (destination, int(row['destination_idx']))



                values.append([kp1.iloc[m1_pid]['x'],
                               kp1.iloc[m1_pid]['y'],
                               m1,
                               pt_idx,
                               source])

                kp2x = kp2.iloc[m2_pid]['x']
                kp2y = kp2.iloc[m2_pid]['y']

                if 'subpixel' in clean_keys:
                    kp2x += offsets['x_offset'].values[i]
                    kp2y += offsets['y_offset'].values[i]
                values.append([kp2x,
                               kp2y,
                               m2,
                               pt_idx,
                               destination])
                pt_idx += 1

            columns = ['x', 'y', 'idx', 'pid', 'nid']

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
        return merged_cnet

    def to_json_file(self, outputfile):
        """
        Write the edge structure to a JSON adjacency list

        Parameters
        ==========

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
