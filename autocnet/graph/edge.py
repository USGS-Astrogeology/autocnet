from functools import wraps, singledispatch
import warnings
from collections import MutableMapping

import numpy as np
import pandas as pd
import networkx as nx

from scipy.spatial.distance import cdist

import autocnet
from autocnet.graph.node import Node
from autocnet.utils import utils
from autocnet.matcher import cpu_outlier_detector as od
from autocnet.matcher import suppression_funcs as spf
from autocnet.matcher import subpixel as sp
from autocnet.transformation import fundamental_matrix as fm
from autocnet.transformation import homography as hm
from autocnet.vis.graph_view import plot_edge, plot_node, plot_edge_decomposition
from autocnet.cg import cg

from plio.io.io_gdal import GeoDataset


class Edge(dict, MutableMapping):
    """
    Attributes
    ----------
    source : hashable
             The source node

    destination : hashable
                  The destination node
    masks : set
            A list of the available masking arrays

    weights : dict
             Dictionary with two keys overlap_area, and overlap_percn
             overlap_area returns the area overlaped by both images
             overlap_percn retuns the total percentage of overlap
    """

    def __init__(self, source=None, destination=None):
        self.source = source
        self.destination = destination
        self['homography'] = None
        self['fundamental_matrix'] = None
        self.masks = pd.DataFrame()
        self.subpixel_matches = pd.DataFrame()
        self._matches = pd.DataFrame()
        self['weights'] = {}
        
        self['source_mbr'] = None
        self['destin_mbr'] = None
        self['overlap_latlon_coords'] = None

    def __repr__(self):
        return """
        Source Image Index: {}
        Destination Image Index: {}
        Available Masks: {}
        """.format(self.source, self.destination, self.masks)

    def __eq__(self, other):
        return utils.compare_dicts(self.__dict__, other.__dict__) *\
               utils.compare_dicts(self, other)

    @property
    def matches(self):
        if not hasattr(self, '_matches'):
            self._matches = pd.DataFrame()
        return self._matches

    @matches.setter
    def matches(self, value):
        if isinstance(value, pd.DataFrame):
            self._matches = value
        else:
            raise(TypeError)
            
    def match(self, k=2, **kwargs):

        """
        Given two sets of descriptors, utilize a FLANN (Approximate Nearest
        Neighbor KDTree) matcher to find the k nearest matches.  Nearness is
        the euclidean distance between descriptors.

        The matches are then added as an attribute to the edge object.

        Parameters
        ----------
        k : int
            The number of neighbors to find
        """
        # Reset the edge masks because matching is happening (again)
        self.masks = pd.DataFrame()
        kwargs['aidx'] = self.get_keypoints('source', overlap=True).index
        kwargs['bidx'] = self.get_keypoints('destination', overlap=True).index
        Edge._match(self, k=k, **kwargs)

    @staticmethod
    def _match(edge, k=2, **kwargs):
        """
        Patches the static cpu_matcher.match(edge) or cuda_match.match(edge)
        into the member method Edge.match()

        Parameters
        ----------
        edge : Edge
               The edge object to compute matches for; Edge.match() calls this
               with self
        k : int
            The number of neighbors to find
        """
        pass

    def decompose(self):
        """
        Apply coupled decomposition to the images and
        match identified sub-images
        """
        pass

    def decompose_and_match(*args, **kwargs):
        pass

    def overlap_check(self):
        """Creates a mask for matches on the overlap"""
        if not (self["source_mbr"] and self["destin_mbr"]):
            warnings.warn(
                "Cannot use overlap constraint, minimum bounding rectangles"
                " have not been computed for one or more Nodes")
            return
        # Get overlapping keypts
        s_idx = self.get_keypoints(self.source, overlap=True).index
        d_idx = self.get_keypoints(self.destination, overlap=True).index
        # Create a mask from matches whose rows have both source idx &
        # dest idx in the overlapping keypts
        mask = pd.Series(False, index=self.matches.index)
        mask.loc[(self.matches["source_idx"].isin(s_idx)) &
                 (self.matches["destination_idx"].isin(d_idx))] = True
        self.masks['overlap'] = mask

    def symmetry_check(self):
        self.masks['symmetry'] = od.mirroring_test(self.matches)

    def ratio_check(self, clean_keys=[], maskname='ratio', **kwargs):
        matches, mask = self.clean(clean_keys)
        self.masks[maskname] = self._ratio_check(self, matches, **kwargs)

    @staticmethod
    def _ratio_check(edge, matches, **kwargs):
        pass
        #return.masks[maskname] = od.distance_ratio(matches, **kwargs)

    def compute_fundamental_matrix(self, clean_keys=[], maskname='fundamental', **kwargs):
        """
        Estimate the fundamental matrix (F) using the correspondences tagged to this
        edge.


        Parameters
        ----------
        clean_keys : list
                     Of strings used to apply masks to omit correspondences

        method : {linear, nonlinear}
                 Method to use to compute F.  Linear is significantly faster at
                 the cost of reduced accuracy.

        See Also
        --------
        autocnet.transformation.transformations.FundamentalMatrix

        """
        matches, mask = self.clean(clean_keys)

        # TODO: Homogeneous is horribly inefficient here, use Numpy array notation
        s_keypoints = self.get_keypoints('source', index=matches['source_idx'])
        d_keypoints = self.get_keypoints('destination', index=matches['destination_idx'])


        # Replace the index with the matches index.
        s_keypoints.index = matches.index
        d_keypoints.index = matches.index
        
        self['fundamental_matrix'], fmask = fm.compute_fundamental_matrix(s_keypoints, d_keypoints, **kwargs)

        if isinstance(self['fundamental_matrix'], np.ndarray):
            # Convert the truncated RANSAC mask back into a full length mask
            mask[mask] = fmask

            # Set the initial state of the fundamental mask in the masks
            self.masks[maskname] = mask

    @utils.methodispatch
    def get_keypoints(self, node, index=None, homogeneous=False, overlap=False):
        if not hasattr(index, '__iter__') and index is not None:
            raise TypeError
        keypts = node.get_keypoint_coordinates(index=index, homogeneous=homogeneous)
        # If we only want keypoints in the overlap
        if overlap:
            if self.source == node:
                mbr = self['source_mbr']
            else:
                mbr = self['destin_mbr']
            # Can't use overlap if we haven't computed MBRs
            if mbr is None:
                return keypts
            return keypts.query('x >= {} and x <= {} and y >= {} and y <= {}'.format(*mbr))
        return keypts

    @get_keypoints.register(str)
    def _(self, node, index=None, homogeneous=False, overlap=False):
        if not hasattr(index, '__iter__') and index is not None:
            raise TypeError
        node = node.lower()
        node = getattr(self, node)
        return self.get_keypoints(node, index=index, homogeneous=homogeneous, overlap=overlap)

    def compute_fundamental_error(self, clean_keys=[]):
        """
        Given a fundamental matrix, compute the reprojective error between
        a two sets of keypoints.

        Parameters
        ----------
        clean_keys : list
                     of string keys to masking arrays
                     (created by calling outlier detection)

        Returns
        -------
        error : pd.Series
                of reprojective error indexed to the matches data frame
        """
        if self['fundamental_matrix'] is None:
            warnings.warn('No fundamental matrix has been compute for this edge.')
        matches, masks = self.clean(clean_keys)

        source_kps = self.source.get_keypoint_coordinates(index=matches['source_idx'])
        destination_kps = self.destination.get_keypoint_coordinates(index=matches['destination_idx'])

        error = fm.compute_fundamental_error(self['fundamental_matrix'], source_kps, destination_kps)

        error = pd.Series(error, index=matches.index)
        return error

    def compute_homography(self, method='ransac', clean_keys=[], pid=None, maskname='homography', **kwargs):
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
        matches, mask = self.clean(clean_keys)

        s_keypoints = self.source.get_keypoint_coordinates(index=matches['source_idx'])
        d_keypoints = self.destination.get_keypoint_coordinates(index=matches['destination_idx'])

        self['homography'], hmask = hm.compute_homography(s_keypoints.values, d_keypoints.values)

        # Convert the truncated RANSAC mask back into a full length mask
        mask[mask] = hmask
        self.masks['homography'] = mask

    def subpixel_register(self, clean_keys=[], threshold=0.8,
                          template_size=19, search_size=53, max_x_shift=1.0,
                          max_y_shift=1.0, tiled=False, **kwargs):
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

        max_x_shift : float
                      The maximum (positive) value that a pixel can shift in the x direction
                      without being considered an outlier

        max_y_shift : float
                      The maximum (positive) value that a pixel can shift in the y direction
                      without being considered an outlier
        """
        for column, default in {'x_offset': 0, 'y_offset': 0, 'correlation': 0, 'reference': -1}.items():
            if column not in self.subpixel_matches.columns:
                self.subpixel_matches[column] = default

        # Build up a composite mask from all of the user specified masks
        matches, mask = self.clean(clean_keys)

        # Grab the full images, or handles
        if tiled is True:
            s_img = self.source.geodata
            d_img = self.destination.geodata
        else:
            s_img = self.source.geodata.read_array()
            d_img = self.destination.geodata.read_array()

        source_image = (matches.iloc[0]['source_image'])

        pts = []
        # for each edge, calculate this for each keypoint pair
        for i, (idx, row) in enumerate(matches.iterrows()):
            s_idx = int(row['source_idx'])
            d_idx = int(row['destination_idx'])

            s_keypoint = self.source.get_keypoint_coordinates(s_idx)
            d_keypoint = self.destination.get_keypoint_coordinates(d_idx)

            # Get the template and search window
            s_template = sp.clip_roi(s_img, s_keypoint, template_size)
            d_search = sp.clip_roi(d_img, d_keypoint, search_size)
            if 0 in s_template.shape or 0 in d_search.shape:
                continue
            try:
                (x_offset, y_offset, strength),ref = sp.subpixel_offset(s_template, d_search, **kwargs)
                self.subpixel_matches.loc[idx, ('x_offset', 'y_offset', 'correlation', 'reference')]= [x_offset, y_offset, strength, source_image]
                pts.append([s_template, d_search, ref, x_offset, y_offset])
            except:
                warnings.warn('Template-Search size mismatch, failing for this correspondence point.')

        # Compute the mask for correlations less than the threshold
        threshold_mask = self.subpixel_matches['correlation'] >= threshold

        # Compute the mask for the point shifts that are too large
        query_string = 'x_offset <= -{0} or x_offset >= {0} or y_offset <= -{1} or y_offset >= {1}'.format(max_x_shift,max_y_shift)
        sp_shift_outliers = self.subpixel_matches.query(query_string)
        shift_mask = pd.Series(True, index=self.subpixel_matches.index)
        shift_mask.loc[sp_shift_outliers.index] = False

        # Generate the composite mask and write the masks to the mask data structure
        mask = threshold_mask & shift_mask
        self.masks['shift'] = shift_mask
        self.masks['threshold'] = threshold_mask
        self.masks['subpixel'] = mask
        return pts

    def suppress(self, suppression_func=spf.correlation, clean_keys=[], maskname='suppression', **kwargs):
        """
        Apply a disc based suppression algorithm to get a good spatial
        distribution of high quality points, where the user defines some
        function to be used as the quality metric.

        Parameters
        ----------
        suppression_func : object
                           A function that returns a scalar value to be used
                           as the strength of a given row in the matches data
                           frame.

        suppression_args : tuple
                           Arguments to be passed on to the suppression function

        clean_keys : list
                     of mask keys to be used to reduce the total size
                     of the matches dataframe.
        """
        if not isinstance(self.matches, pd.DataFrame):
            raise AttributeError('This edge does not yet have any matches computed.')

        matches, mask = self.clean(clean_keys)
        domain = self.source.geodata.raster_size

        # Massage the dataframe into the correct structure
        coords = self.source.get_keypoint_coordinates()
        merged = matches.merge(coords, left_on=['source_idx'], right_index=True)
        merged['strength'] = merged.apply(suppression_func, axis=1, args=([self]))

        smask, k = od.spatial_suppression(merged, domain, **kwargs)

        mask[mask] = smask
        self.masks[maskname] = mask

    def plot_source(self, ax=None, clean_keys=[], **kwargs):  # pragma: no cover
        matches, mask = self.clean(clean_keys=clean_keys)
        indices = pd.Index(matches['source_idx'].values)
        return plot_node(self.source, index_mask=indices, **kwargs)

    def plot_destination(self, ax=None, clean_keys=[], **kwargs):  # pragma: no cover
        matches, mask = self.clean(clean_keys=clean_keys)
        indices = pd.Index(matches['destination_idx'].values)
        return plot_node(self.destination, index_mask=indices, **kwargs)

    def plot(self, ax=None, clean_keys=[], node=None, **kwargs):  # pragma: no cover
        dest_keys = [0, '0', 'destination', 'd', 'dest']
        source_keys = [1, '1', 'source', 's']

        # If node is not none, plot a single node
        if node in source_keys:
            return self.plot_source(self, clean_keys=clean_keys, **kwargs)

        elif node in dest_keys:
            return self.plot_destination(self, clean_keys=clean_keys, **kwargs)

        # Else, plot the whole edge
        return plot_edge(self, ax=ax, clean_keys=clean_keys, **kwargs)

    def plot_decomposition(self, *args, **kwargs): #pragma: no cover
        return plot_edge_decomposition(self, *args, **kwargs)

    def clean(self, clean_keys):
        """
        Given a list of clean keys compute the mask of valid
        matches

        Parameters
        ----------
        clean_keys : list
                     of columns names (clean keys)

        Returns
        -------
        matches : dataframe
                  A masked view of the matches dataframe

        mask : series
               A boolean series to inflate back to the full match set
        """
        if clean_keys:
            mask = self.masks[clean_keys].all(axis=1)
        else:
            mask = pd.Series(True, self.matches.index)

        return self.matches[mask], mask

    def overlap(self):
        """
        Acts on an edge and returns the overlap area and percentage of overlap
        between the two images on the edge. Data is returned to the
        weights dictionary
        """
        poly1 = self.source.geodata.footprint
        poly2 = self.destination.geodata.footprint

        overlapinfo = cg.two_poly_overlap(poly1, poly2)

        self['weights']['overlap_area'] = overlapinfo[1]
        self['weights']['overlap_percn'] = overlapinfo[0]

    def coverage(self, clean_keys = []):
        """
        Acts on the edge given either the source node
        or the destination node and returns the percentage
        of overlap covered by the keypoints. Data for the
        overlap is gathered from the source node of the edge
        resulting in a maximum area difference of 2% when compared
        to the destination.

        Returns
        -------
        total_overlap_percentage : float
                                   returns the overlap area
                                   covered by the keypoints
        """
        matches, mask = self.clean(clean_keys)
        source_array = self.source.get_keypoint_coordinates(index=matches['source_idx']).values

        source_coords = self.source.geodata.latlon_corners
        destination_coords = self.destination.geodata.latlon_corners

        convex_hull = cg.convex_hull(source_array)

        convex_points = [self.source.geodata.pixel_to_latlon(row[0], row[1]) for row in convex_hull.points[convex_hull.vertices]]
        convex_coords = [(x, y) for x, y in convex_points]

        source_poly = utils.array_to_poly(source_coords)
        destination_poly = utils.array_to_poly(destination_coords)
        convex_poly = utils.array_to_poly(convex_coords)

        intersection_area = cg.get_area(source_poly, destination_poly)

        total_overlap_coverage = (convex_poly.GetArea()/intersection_area)

        return total_overlap_coverage

    def compute_weights(self, clean_keys, **kwargs):
        """
        Computes a voronoi diagram for the overlap between two images
        then gets the area of each polygon resulting in a voronoi weight.
        These weights are then appended to the matches dataframe.

        Parameters
        ----------
        clean_keys : list
                     Of strings used to apply masks to omit correspondences

        """
        if not isinstance(self.matches, pd.DataFrame):
            raise AttributeError('Matches have not been computed for this edge')
        voronoi = cg.vor(self, clean_keys, **kwargs)
        self.matches = pd.concat([self.matches, voronoi[1]['vor_weights']], axis=1)

    def compute_overlap(self, buffer_dist=0, **kwargs):
        """
        Estimate a source and destination minimum bounding rectangle, in
        pixel space.
        """
        if not isinstance(self.source.geodata, GeoDataset):
            smbr = None
            dmbr = None
        else:
            try:
                self['overlap_latlon_coords'], smbr, dmbr = self.source.geodata.compute_overlap(self.destination.geodata, **kwargs)
                smbr = list(smbr)
                dmbr = list(dmbr)
                for i in range(4):
                    if i % 2:
                        buf = buffer_dist
                    else:
                        buf = -buffer_dist
                    smbr[i] += buf
                    dmbr[i] += buf

            except:
                smbr = self.source.geodata.xy_extent
                dmbr = self.source.geodata.xy_extent
                warnings.warn("Overlap between {} and {} could not be "
                                "computed.  Using the full image extents".format(self.source['image_name'],
                                                      self.destination['image_name']))
                smbr = [smbr[0][0], smbr[1][0], smbr[0][1], smbr[1][1]]
                dmbr = [dmbr[0][0], dmbr[1][0], dmbr[0][1], dmbr[1][1]]
        self['source_mbr'] = smbr
        self['destin_mbr'] = dmbr

    def get_matches(self, clean_keys=[]): # pragma: no cover
        if self.matches.empty:
            return pd.DataFrame()

        match, _ = self.clean(clean_keys=clean_keys)
        match = match[['source_image', 'source_idx',
                       'destination_image', 'destination_idx']]
        skps = self.get_keypoints('source', index=match.source_idx)
        skps.columns = ['source_x', 'source_y']
        dkps = self.get_keypoints('destination', index=match.destination_idx)
        dkps.columns = ['destination_x', 'destination_y']
        match = match.join(skps, on='source_idx')
        match = match.join(dkps, on='destination_idx')
        return match
