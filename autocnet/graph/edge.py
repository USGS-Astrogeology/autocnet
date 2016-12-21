import warnings
from collections import MutableMapping

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import autocnet
from autocnet.utils import utils
from autocnet.matcher import outlier_detector as od
from autocnet.matcher import suppression_funcs as spf
from autocnet.matcher import subpixel as sp
from autocnet.matcher.cpu_matcher import FlannMatcher
from autocnet.transformation.decompose import coupled_decomposition
from autocnet.transformation.transformations import FundamentalMatrix, Homography
from autocnet.vis.graph_view import plot_edge, plot_node, plot_edge_decomposition
from autocnet.cg import cg


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

    provenance : dict
                 With key equal to an autoincrementing integer and value
                 equal to a dict of parameters used to generate this
                 realization.

    weight : dict
             Dictionary with two keys overlap_area, and overlap_percn
             overlap_area returns the area overlaped by both images
             overlap_percn retuns the total percentage of overlap
    """

    def __init__(self, source=None, destination=None):
        self.source = source
        self.destination = destination

        self.homography = None
        self.fundamental_matrix = None
        self.matches = None
        self._subpixel_offsets = None

        self.weight = {}

        self._observers = set()

    def __repr__(self):
        return """
        Source Image Index: {}
        Destination Image Index: {}
        Available Masks: {}
        """.format(self.source, self.destination, self.masks)

    @property
    def masks(self):
        mask_lookup = {'fundamental': 'fundamental_matrix',
                       'ratio': 'distance_ratio'}
        if not hasattr(self, '_masks'):
            if self.matches is not None:
                self._masks = pd.DataFrame(True, columns=['symmetry'],
                                           index=self.matches.index)
            else:
                self._masks = pd.DataFrame()
        # If the mask is coming form another object that tracks
        # state, dynamically draw the mask from the object.
        for c in self._masks.columns:
            if c in mask_lookup:
                try:
                    truncated_mask = getattr(self, mask_lookup[c]).mask
                    self._masks[c] = False
                    self._masks[c].iloc[truncated_mask.index] = truncated_mask
                except Exception:
                    #TODO: Get rid of state
                    pass
        return self._masks

    @masks.setter
    def masks(self, v):
        column_name = v[0]
        boolean_mask = v[1]
        self.masks[column_name] = boolean_mask

    def decompose_and_match(self, k=2, maxiteration=3, size=18, buf_dist=3,**kwargs):
        """
        Similar to match, this method first decomposed the image into
        $4^{maxiteration}$ subimages and applys matching between each sub-image.

        This method is potential slower than the standard match due to the
        overhead in matching, but can be significantly more accurate.  The
        increase in accuracy is a function of the total image size.  Suggested
        values for maxiteration are provided below.

        Parameters
        ----------
        k : int
            The number of neighbors to find

        method : {'coupled', 'whole'}
                 whether to utilize coupled decomposition
                 or match the whole image

        maxiteration : int
                       When using coupled decomposition, the number of recursive
                       divisions to apply.  The total number of resultant
                       sub-images will be 4 ** maxiteration.

                       Approximate Guidelines

                       +----------------------+--------------+
                       | Number of megapixels | maxiteration |
                       +======================+==============+
                       | m < 10               |1 - 2         |
                       | 10 < m < 30          | 3            |
                       | 30 < m < 100         | 4            |
                       | 100 < m < 1000       | 5            |
                       | m > 1000             | 6            |
                       +----------------------+--------------+

        size : int
               When using coupled decomposition, the total number of points
               to check in each sub-image to try and find a match.
               Selection of this number is a balance between seeking a
               representative mid-point and computational cost.

        buf_dist : int
                   When using coupled decomposition, the distance from the edge of
                   the (sub)image a point must be in order to be used as a
                   partioning point.  The smaller the distance, the more likely
                   percision errors can results in erroneous partitions.
        """

        def mono_matches(a, b, aidx=None, bidx=None):
            """
            Apply the FLANN match_features

            Parameters
            ----------
            a : object
                A node object

            b : object
                A node object

            aidx : iterable
                   An index for the descriptors to subset

            bidx : iterable
                   An index for the descriptors to subset
            """
            # Subset if requested
            if aidx is not None:
                ad = a.descriptors[aidx]
            else:
                ad = a.descriptors

            if bidx is not None:
                bd = b.descriptors[bidx]
            else:
                bd = b.descriptors

            # Load, train, and match
            fl.add(ad, a.node_id, index=aidx)
            fl.train()
            matches = fl.query(bd, b.node_id, k, index=bidx)
            self._add_matches(matches)
            fl.clear()

        def func(group):
            ratio = 0.8
            res = [False] * len(group)
            if len(res) == 1:
                return [single]
            if group.iloc[0] < group.iloc[1] * ratio:
                res[0] = True
            return res

        # Grab the original image arrays
        sdata = self.source.get_array()
        ddata = self.destination.get_array()

        ssize = sdata.shape
        dsize = ddata.shape

        # Grab all the available candidate keypoints
        skp = self.source.get_keypoints()
        dkp = self.destination.get_keypoints()

        # Set up the membership arrays
        self.smembership = np.zeros(sdata.shape, dtype=np.int16)
        self.dmembership = np.zeros(ddata.shape, dtype=np.int16)
        self.smembership[:] = -1
        self.dmembership[:] = -1
        pcounter = 0

        # FLANN Matcher
        fl= FlannMatcher()

        for k in range(maxiteration):
            partitions = np.unique(self.smembership)
            for p in partitions:
                sy_part, sx_part = np.where(self.smembership == p)
                dy_part, dx_part = np.where(self.dmembership == p)

                # Get the source extent
                minsy = np.min(sy_part)
                maxsy = np.max(sy_part) + 1
                minsx = np.min(sx_part)
                maxsx = np.max(sx_part) + 1

                # Get the destination extent
                mindy = np.min(dy_part)
                maxdy = np.max(dy_part) + 1
                mindx = np.min(dx_part)
                maxdx = np.max(dx_part) + 1

                # Clip the sub image from the full images
                asub = sdata[minsy:maxsy, minsx:maxsx]
                bsub = ddata[mindy:maxdy, mindx:maxdx]

                # Utilize the FLANN matcher to find a match to approximate a center
                fl.add(self.destination.descriptors, self.destination.node_id)
                fl.train()

                scounter = 0
                decompose = False
                while True:
                    sub_skp = skp.query('x >= {} and x <= {} and y >= {} and y <= {}'.format(minsx, maxsx, minsy, maxsy))
                    # Check the size to ensure a valid return
                    if len(sub_skp) == 0:
                        break # No valid keypoints in this (sub)image
                    if size > len(sub_skp):
                        size = len(sub_skp)
                    candidate_idx = np.random.choice(sub_skp.index, size=size, replace=False)
                    candidates = self.source.descriptors[candidate_idx]
                    matches = fl.query(candidates, self.source.node_id, k=3, index=candidate_idx)

                    # Apply Lowe's ratio test to try to find a 'good' starting point
                    mask = matches.groupby('source_idx')['distance'].transform(func).astype('bool')
                    candidate_matches = matches[mask]
                    match_idx = candidate_matches['source_idx']

                    # Extract those matches that pass the ratio check
                    sub_skp = skp.iloc[match_idx]

                    # Check that valid points remain
                    if len(sub_skp) == 0:
                        break

                    # Locate the candidate closest to the middle of all of the matches
                    smx, smy = sub_skp[['x', 'y']].mean()
                    mid = np.array([[smx, smy]])
                    dists = cdist(mid, sub_skp[['x', 'y']])
                    closest = sub_skp.iloc[np.argmin(dists)]
                    closest_idx = closest.name
                    soriginx, soriginy = closest[['x', 'y']]

                    # Grab the corresponding point in the destination
                    q = candidate_matches.query('source_idx == {}'.format(closest.name))
                    dest_idx = q['destination_idx'].iat[0]
                    doriginx = dkp.at[dest_idx, 'x']
                    doriginy = dkp.at[dest_idx, 'y']

                    if mindy + buf_dist <= doriginy <= maxdy - buf_dist\
                     and mindx + 3 <= doriginx <= maxdx - 3:
                        # Point is good to split on
                        decompose = True
                        break
                    else:
                        scounter += 1
                        if scounter >= maxiteration:
                            break

                # Clear the Flann matcher for reuse
                fl.clear()

                # Check that the identified match falls within the (sub)image
                # This catches most bad matches that have passed the ratio check
                if not (buf_dist <= doriginx - mindx <= bsub.shape[1] - buf_dist) or not\
                       (buf_dist <= doriginy - mindy <= bsub.shape[0] - buf_dist):
                       decompose = False

                if decompose:
                    # Apply coupled decomposition, shifting the origin to the sub-image
                    s_submembership, d_submembership = coupled_decomposition(asub, bsub,
                                                                         sorigin=(soriginx - minsx, soriginy - minsy),
                                                                         dorigin=(doriginx - mindx, doriginy - mindy),
                                                                         **kwargs)

                    # Shift the returned membership counters to a set of unique numbers
                    s_submembership += pcounter
                    d_submembership += pcounter

                    # And assign membership
                    self.smembership[minsy:maxsy,
                                minsx:maxsx] = s_submembership
                    self.dmembership[mindy:maxdy,
                                mindx:maxdx] = d_submembership
                    pcounter += 4

        # Now match the decomposed segments to one another
        for p in np.unique(self.smembership):
            sy_part, sx_part = np.where(self.smembership == p)
            dy_part, dx_part = np.where(self.dmembership == p)

            # Get the source extent
            minsy = np.min(sy_part)
            maxsy = np.max(sy_part) + 1
            minsx = np.min(sx_part)
            maxsx = np.max(sx_part) + 1

            # Get the destination extent
            mindy = np.min(dy_part)
            maxdy = np.max(dy_part) + 1
            mindx = np.min(dx_part)
            maxdx = np.max(dx_part) + 1

            # Get the indices of the candidate keypoints within those regions / variables are pulled before decomp.
            sidx = skp.query('x >= {} and x <= {} and y >= {} and y <= {}'.format(minsx, maxsx, minsy, maxsy)).index
            didx = dkp.query('x >= {} and x <= {} and y >= {} and y <= {}'.format(mindx, maxdx, mindy, maxdy)).index
            # If the candidates < k, OpenCV throws an error
            if len(sidx) >= k and len(didx) >=k:
                mono_matches(self.source, self.destination, sidx, didx)
                mono_matches(self.destination, self.source, didx, sidx)

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
        pass

    def cuda_match(self, ratio=0.8, **kwargs):
        """
        Apply a composite CUDA matcher and ratio check.  If this method is used,
        no additional ratio check is necessary and no symmetry check is required.
        The ratio check is embedded on the cuda side and returned as an
        ambiguity value.  In testing symmetry is not required as it is expensive
        without significant gain in accuracy when using this implementation.
        """
        if not autocnet.cudasift:
            warnings.warn('CudaSift is not available, please use the standard matcher.')
        s_siftdata = autocnet.cs.PySiftData.from_data_frame(self.source.get_keypoints(), self.source.descriptors)
        d_siftdata = autocnet.cs.PySiftData.from_data_frame(self.destination.get_keypoints(), self.destination.descriptors)

        autocnet.cs.PyMatchSiftData(s_siftdata, d_siftdata)
        matches, _ = s_siftdata.to_data_frame()
        source = np.empty(len(matches))
        source[:] = self.source.node_id
        destination = np.empty(len(matches))
        destination[:] = self.destination.node_id


        df = pd.concat([pd.Series(source), pd.Series(matches.index),
                        pd.Series(destination), matches.match,
                        matches.score, matches.ambiguity], axis=1)
        df.columns = ['source_image', 'source_idx', 'destination_image',
                        'destination_idx', 'score', 'ambiguity']
        print(df)
        self.matches = df
    def _add_matches(self, matches):
        """
        Given a dataframe of matches, either append to an existing
        matches edge attribute or initially populate said attribute.

        Parameters
        ----------
        matches : dataframe
                  A dataframe of matches
        """
        if self.matches is None:
            self.matches = matches
        else:
            df = self.matches
            self.matches = df.append(matches,
                                     ignore_index=True,
                                     verify_integrity=True)

    def symmetry_check(self):
        if hasattr(self, 'matches'):
            mask = od.mirroring_test(self.matches)
            self.masks = ('symmetry', mask)
        else:
            raise AttributeError('No matches have been computed for this edge.')

    def ratio_check(self, clean_keys=[], **kwargs):
        if hasattr(self, 'matches'):

            matches, mask = self.clean(clean_keys)

            self.distance_ratio = od.DistanceRatio(matches)
            self.distance_ratio.compute(mask=mask, **kwargs)

            # Setup to be notified
            self.distance_ratio._notify_subscribers(self.distance_ratio)

            self.masks = ('ratio', self.distance_ratio.mask)
        else:
            raise AttributeError('No matches have been computed for this edge.')

    def compute_fundamental_matrix(self, clean_keys=[], **kwargs):
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
        if not hasattr(self, 'matches'):
            raise AttributeError('Matches have not been computed for this edge')
            return
        matches, mask = self.clean(clean_keys)

        # TODO: Homogeneous is horribly inefficient here, use Numpy array notation
        s_keypoints = self.source.get_keypoint_coordinates(index=matches['source_idx'],
                                                                 homogeneous=True)
        d_keypoints = self.destination.get_keypoint_coordinates(index=matches['destination_idx'],
                                                                homogeneous=True)


        # Replace the index with the matches index.
        s_keypoints.index = matches.index
        d_keypoints.index = matches.index

        self.fundamental_matrix = FundamentalMatrix(np.zeros((3,3)), index=matches.index)
        self.fundamental_matrix.compute(s_keypoints, d_keypoints, **kwargs)

        # Convert the truncated RANSAC mask back into a full length mask
        mask[mask] = self.fundamental_matrix.mask

        # Set the initial state of the fundamental mask in the masks
        self.masks = ('fundamental', mask)

    def refine_fundamental_matrix_matches(self, **kwargs): # pragma: no cover
        """
        Given an estimated fundamental matrix, refine the correspondences based
        on the reprojective error.

        See Also
        --------
        autocnet.transformation.transformations.FundamentalMatrix.refine_matches
        """
        if not hasattr(self, 'fundamental_matrix'):
            raise AttributeError('No fundamental matrix exists for this edge.')
            return

        self.fundamental_matrix.refine_matches(**kwargs)

    def compute_homography(self, method='ransac', clean_keys=[], pid=None, **kwargs):
        """
        For each edge in the (sub) graph, compute H, the homography.

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

        matches, mask = self.clean(clean_keys)

        s_keypoints = self.source.get_keypoint_coordinates(index=matches['source_idx'])
        d_keypoints = self.destination.get_keypoint_coordinates(index=matches['destination_idx'])

        self.homography = Homography(np.zeros((3,3)), index=self.masks.index)
        self.homography.compute(s_keypoints.values,
                                d_keypoints.values)

        # Convert the truncated RANSAC mask back into a full length mask
        mask[mask] = self.homography.mask
        self.masks = ('ransac', mask)

        # Finalize the array to get custom attrs to propagate
        self.homography.__array_finalize__(self.homography)

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
        matches = self.matches
        for column, default in {'x_offset': 0, 'y_offset': 0, 'correlation': 0, 'reference': -1}.items():
            if column not in self.matches.columns:
                self.matches[column] = default

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

        # for each edge, calculate this for each keypoint pair
        for i, (idx, row) in enumerate(matches.iterrows()):
            s_idx = int(row['source_idx'])
            d_idx = int(row['destination_idx'])

            s_keypoint = self.source.get_keypoint_coordinates(s_idx)
            d_keypoint = self.destination.get_keypoint_coordinates(d_idx)

            # Get the template and search window
            s_template = sp.clip_roi(s_img, s_keypoint, template_size)
            d_search = sp.clip_roi(d_img, d_keypoint, search_size)
            try:
                x_offset, y_offset, strength = sp.subpixel_offset(s_template, d_search, **kwargs)
                self.matches.loc[idx, ('x_offset', 'y_offset',
                                       'correlation', 'reference')] = [x_offset, y_offset, strength, source_image]
            except:
                warnings.warn('Template-Search size mismatch, failing for this correspondence point.')

        # Compute the mask for correlations less than the threshold
        threshold_mask = self.matches['correlation'] >= threshold

        # Compute the mask for the point shifts that are too large
        query_string = 'x_offset <= -{0} or x_offset >= {0} or y_offset <= -{1} or y_offset >= {1}'.format(max_x_shift,
                                                                                                           max_y_shift)
        sp_shift_outliers = self.matches.query(query_string)
        shift_mask = pd.Series(True, index=self.matches.index)
        shift_mask.loc[sp_shift_outliers.index] = False

        # Generate the composite mask and write the masks to the mask data structure
        mask = threshold_mask & shift_mask
        self.masks = ('shift', shift_mask)
        self.masks = ('threshold', threshold_mask)
        self.masks = ('subpixel', mask)

    def suppress(self, suppression_func=spf.correlation, clean_keys=[], **kwargs):
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
        if not hasattr(self, 'matches'):
            raise AttributeError('This edge does not yet have any matches computed.')

        matches, mask = self.clean(clean_keys)
        domain = self.source.geodata.raster_size

        # Massage the dataframe into the correct structure
        coords = self.source.get_keypoint_coordinates()
        merged = matches.merge(coords, left_on=['source_idx'], right_index=True)
        merged['strength'] = merged.apply(suppression_func, axis=1, args=([self]))

        if not hasattr(self, 'suppression'):
            # Instantiate the suppression object and suppress matches
            self.suppression = od.SpatialSuppression(merged, domain, **kwargs)
            self.suppression.suppress()
        else:
            for k, v in kwargs.items():
                if hasattr(self.suppression, k):
                    setattr(self.suppression, k, v)
            self.suppression.suppress()

        mask[mask] = self.suppression.mask
        self.masks = ('suppression', mask)

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

    def clean(self, clean_keys, pid=None):
        """
        Given a list of clean keys and a provenance id compute the
        mask of valid matches

        Parameters
        ----------
        clean_keys : list
                     of columns names (clean keys)
        pid : int
              The provenance id of the parameter set to be cleaned.
              Defaults to the last run.

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
        weight dictionary
        """
        poly1 = self.source.geodata.footprint
        poly2 = self.destination.geodata.footprint

        overlapinfo = cg.two_poly_overlap(poly1, poly2)

        self.weight['overlap_area'] = overlapinfo[1]
        self.weight['overlap_percn'] = overlapinfo[0]

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
        if self.matches is None:
            raise AttributeError('Edge needs to have features extracted and matched')
            return
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
