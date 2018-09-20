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
from autocnet.matcher import cpu_ring_matcher
from autocnet.transformation import fundamental_matrix as fm
from autocnet.transformation import homography as hm
from autocnet.transformation import spatial
from autocnet.vis.graph_view import plot_edge, plot_node, plot_edge_decomposition, plot_matches
from autocnet.cg import cg

from plio.io.io_gdal import GeoDataset
from csmapi import csmapi

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
    def masks(self):
        if not hasattr(self, '_masks'):
            self._masks = pd.DataFrame()
        return self._masks

    @masks.setter
    def masks(self, value):
        if isinstance(value, pd.DataFrame):
            self._masks = value
        else:
            raise(TypeError)

    @property
    def matches(self):
        if not hasattr(self, '_matches'):
            self._matches = pd.DataFrame()
        return self._matches
    
    @matches.setter
    def matches(self, value):
        if isinstance(value, pd.DataFrame):
            self._matches = value
            # Ensure that the costs df remains in sync with the matches df
            if not self.costs.index.equals(value.index):
                self.costs = pd.DataFrame(index=value.index)
        else:
            raise(TypeError)
    
    @property
    def costs(self):
        if not hasattr(self, '_costs'):
            self._costs = pd.DataFrame(index=self.matches.index)
        return self._costs

    @costs.setter
    def costs(self, value):
        if isinstance(value, pd.DataFrame):
            self._costs = value
        else:
            raise(TypeError)

    @property
    def ring(self):
        if not hasattr(self, '_ring'):
            self._ring = None
        return self._ring

    @ring.setter
    def ring(self, val):
        self._ring = val

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

    def ring_match(self, *args, **kwargs):
        ref_kps =  self.source.keypoints
        ref_desc = self.source.descriptors
        tar_kps = self.destination.keypoints
        tar_desc = self.destination.descriptors

        if not 'xm' in ref_kps.columns:
            warnings.warn('To ring match body centered coordinates (xm, ym, zm) must be in the keypoints')
            return
        ref_feats = ref_kps[['x', 'y', 'xm', 'ym', 'zm']].values
        tar_feats = tar_kps[['x', 'y', 'xm', 'ym', 'zm']].values

        _, _, pidx, ring = cpu_ring_matcher.ring_match(ref_feats, tar_feats,
                                                           ref_desc, tar_desc,
                                                           *args, **kwargs)

        if pidx is None:
            return
        self.ring = ring
        pidx = cpu_ring_matcher.check_pidx_duplicates(pidx)

        #Set the columns of the matches df
        matches = np.empty((pidx.shape[0], 4))
        matches[:,0] = self.source['node_id']
        matches[:,1] = ref_kps.index[pidx[:,0]].values
        matches[:,2] = self.destination['node_id']
        matches[:,3] = tar_kps.index[pidx[:,1]].values

        matches = pd.DataFrame(matches, columns=['source',
                                                 'source_idx',
                                                 'destination',
                                                 'destination_idx']).astype(np.float32)
        
        matches = matches.drop_duplicates()

        self.matches = matches

    def add_coordinates_to_matches(self):
        """
        Add source and destination x/y columns to the matches dataframe. This
        will add to the overall memory needed to store matches, but makes
        access to x,y easier as a join on the keypoints is not requires.
        """
        skps = self.get_keypoints(self.source, index=self.matches.source_idx)
        skps.reindex(self.matches['source_idx'])
        self.matches['source_x'] = skps.values[:,0]
        self.matches['source_y'] = skps.values[:,1]
        dkps = self.get_keypoints(self.destination, index=self.matches.destination_idx)
        dkps.reindex(self.matches['destination_idx'])
        self.matches['destination_x'] = dkps.values[:,0]
        self.matches['destination_y'] = dkps.values[:,1]
        
    def project_matches(self, semimajor, semiminor, on='source', srid=None):
        """
        Project matches.
        """
        try:
            coords = self.matches[['{}_y'.format(on),'{}_x'.format(on)]].values
        except:
            self.add_coordinates_to_matches()
            coords = self.matches[['{}_y'.format(on),'{}_x'.format(on)]].values

        node = getattr(self, on)
        camera = getattr(node, 'camera')
        if camera is None:
            warnings.warn('Unable to project matches without a sensor model.')
            return
        
        matches = self.matches
        
        gnd = np.empty((len(coords), 3))
        # Project the points to the surface and reproject into latlon space
        for i in range(gnd.shape[0]):
            ic = csmapi.ImageCoord(coords[i][0], coords[i][1])
            ground = camera.imageToGround(ic, 0)
            gnd[i] = [ground.x, ground.y, ground.z]
        lon, lat, alt = spatial.reproject(gnd.T, semimajor, semiminor,
                                    'geocent', 'latlon')
        if srid:
            geoms = []
            for coord in zip(lon, lat, alt):
                geoms.append('SRID={};POINTZ({} {} {})'.format(srid, coord[0],
                                                                     coord[1],
                                                                     coord[2]))
            matches['geom'] = geoms
        
        matches['lat'] = lat
        matches['lon'] = lon
        self.matches = matches

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

    @utils.methodispatch
    def get_keypoints(self, node, index=None, homogeneous=False, overlap=False):
        if not hasattr(index, '__iter__') and index is not None:
            raise TypeError
        keypts = node.get_keypoint_coordinates(index=index, homogeneous=homogeneous)
        # If the index is passed, the results are returned sorted. The index is not
        # necessarily sorted, so 'unsort' so that the return order matches the passed
        # order
        if index is not None:
            keypts = keypts.reindex(index)
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
        _, mask = self.clean(clean_keys)
        s_keypoints, d_keypoints = self.get_match_coordinates(clean_keys=clean_keys)
        self.fundamental_matrix, fmask = fm.compute_fundamental_matrix(s_keypoints, d_keypoints, **kwargs)
        

        if isinstance(self.fundamental_matrix, np.ndarray):
            # Convert the truncated RANSAC mask back into a full length mask
            mask[mask] = fmask

            # Set the initial state of the fundamental mask in the masks
            self.masks[maskname] = mask

    def compute_fundamental_error(self, method='equality', clean_keys=[]):
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
        if self.fundamental_matrix is None:
            warnings.warn('No fundamental matrix has been compute for this edge.')

        matches, mask = self.clean(clean_keys)
        s_keypoints, d_keypoints = self.get_match_coordinates(clean_keys=clean_keys)
        if method == 'equality':
            error = fm.compute_fundamental_error(self.fundamental_matrix, s_keypoints, d_keypoints)
        elif method == 'projection':
            error = fm.compute_reprojection_error(self.fundamental_matrix, s_keypoints, d_keypoints)

        self.costs.loc[mask, 'fundamental_{}'.format(method)] = error

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

    def subpixel_register(self, method='phase', clean_keys=[],
                          template_size=251, search_size=251, **kwargs):
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
                        The size of the template in pixels, must be odd. If using phase, 
                        only the template size is used.

        search_size : int
                      The size of the search area. When method='template', this size should
                      be >= the template size

        """
        # Build up a composite mask from all of the user specified masks
        matches, mask = self.clean(clean_keys)

        # Get the img handles
        s_img = self.source.geodata
        d_img = self.destination.geodata

        # Determine which algorithm is going ot be used.
        if method == 'phase':
            func = sp.iterative_phase
            nstrengths = 2
        elif method == 'template':
            func = sp.subpixel_template
            nstrengths = 1
        shifts_x, shifts_y, strengths, new_x, new_y = sp._prep_subpixel(len(matches), nstrengths)

        # for each edge, calculate this for each keypoint pair
        for i, (idx, row) in enumerate(matches.iterrows()):
            s_idx = int(row['source_idx'])
            d_idx = int(row['destination_idx'])

            if 'source_x' in row.index:
                sx = row.source_x
                sy = row.source_y
            else:
                s_keypoint = self.source.get_keypoint_coordinates([s_idx])
                sx = s_keypoint.x
                sy = s_keypoint.y
    
            if 'destination_x' in row.index:
                dx = row.destination_x
                dy = row.destination_y
            else:
                d_keypoint = self.destination.get_keypoint_coordinates([d_idx])
                dx = d_keypoint.x
                dy = d_keypoint.y

            if method == 'phase':
                res = sp.iterative_phase(sx, sy, dx, dy, s_img, d_img, size=template_size, **kwargs)
                if res[0]:
                    new_x[i] = res[0]
                    new_y[i] = res[1]
                    strengths[i] = res[2]
            elif method == 'template':
                new_x[i], new_y[i], strengths[i] = sp.subpixel_template(sx, sy, dx, dy, s_img, d_img,
                                                                     search_size=search_size, 
                                                                     template_size=template_size, **kwargs)

            # Capture the shifts
            shifts_x[i] = new_x[i] - dx
            shifts_y[i] = new_y[i] - dy

        self.matches.loc[mask, 'shift_x'] = shifts_x
        self.matches.loc[mask, 'shift_y'] = shifts_y
        self.matches.loc[mask, 'destination_x'] = new_x
        self.matches.loc[mask, 'destination_y'] = new_y

        if method == 'phase':
            self.costs.loc[mask, 'phase_diff'] = strengths[:,0]
            self.costs.loc[mask, 'rmse'] = strengths[:,1]
        elif method == 'template':
            self.costs.loc[mask, 'correlation'] = strengths[:,0]
 

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
        rs = self.source.geodata.raster_size
        domain = [0, 0, rs[0], rs[1]]
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

    def plot_matches(self, clean_keys=[], **kwargs):  # pragme: no cover
        matches, mask = self.clean(clean_keys=clean_keys)
        sourcegd = self.source.geodata
        destingd = self.destination.geodata
        return plot_matches(matches, sourcegd, destingd, **kwargs)

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

        m = mask[mask==True]
        return self.matches.loc[m.index], mask

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

    def get_match_coordinates(self, clean_keys=[]):
        matches = self.get_matches(clean_keys=clean_keys)
        skps = matches[['source_x', 'source_y']].astype(np.float)
        dkps = matches[['destination_x', 'destination_y']].astype(np.float)

        return skps, dkps

    def get_matches(self, clean_keys=[]): # pragma: no cover
        if self.matches.empty:
            return pd.DataFrame()
        self.add_coordinates_to_matches()
        matches, _ = self.clean(clean_keys=clean_keys)
        skps = matches[['source_x', 'source_y']]
        dkps = matches[['destination_x', 'destination_y']]
        return matches
