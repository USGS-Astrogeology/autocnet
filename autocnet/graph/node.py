from collections import defaultdict, MutableMapping
import os
import warnings

import numpy as np
import pandas as pd
from plio.io.io_gdal import GeoDataset
from plio.io import io_hdf
from plio.io.isis_serial_number import generate_serial_number
from scipy.misc import bytescale

from autocnet.cg import cg
from autocnet.control.control import Correspondence, Point

from autocnet.matcher.add_depth import deepen_correspondences
from autocnet.matcher import feature_extractor as fe
from autocnet.matcher import outlier_detector as od
from autocnet.matcher import suppression_funcs as spf
from autocnet.cg.cg import convex_hull_ratio

from autocnet.vis.graph_view import plot_node
from autocnet.utils import utils


class Node(dict, MutableMapping):
    """
    This class represents a node in a graph and is synonymous with an
    image.  The node (image) stores PATH information, an accessor to the
    on-disk data set, and correspondences information that references the image.


    Attributes
    ----------
    image_name : str
                 Name of the image, with extension

    image_path : str
                 Relative or absolute PATH to the image

    geodata : object
             File handle to the object

    keypoints : dataframe
                With columns, x, y, and response

    nkeypoints : int
                 The number of keypoints found for this image

    descriptors : ndarray
                  32-bit array of feature descriptors returned by OpenCV

    masks : set
            A list of the available masking arrays

    isis_serial : str
                  If the input images have PVL headers, generate an
                  ISIS compatible serial number
    """

    def __init__(self, image_name=None, image_path=None, node_id=None):
        self.image_name = image_name
        self.image_path = image_path
        self.node_id = node_id
        self._mask_arrays = {}
        self.point_to_correspondence = defaultdict(set)
        self.point_to_correspondence_df = None

    def __repr__(self):
        return """
        NodeID: {}
        Image Name: {}
        Image PATH: {}
        Number Keypoints: {}
        Available Masks : {}
        Type: {}
        """.format(self.node_id, self.image_name, self.image_path,
                   self.nkeypoints, self.masks, self.__class__)

    @property
    def geodata(self):
        if not getattr(self, '_geodata', None):
            self._geodata = GeoDataset(self.image_path)
        return self._geodata

    @property
    def masks(self):
        mask_lookup = {'suppression': 'suppression'}

        if not hasattr(self, '_keypoints'):
            warnings.warn('Keypoints have not been extracted')
            return

        if not hasattr(self, '_masks'):
            self._masks = pd.DataFrame(index=self._keypoints.index)

        # If the mask is coming form another object that tracks
        # state, dynamically draw the mask from the object.
        for c in self._masks.columns:
            if c in mask_lookup:
                self._masks[c] = getattr(self, mask_lookup[c]).mask
        return self._masks

    @masks.setter
    def masks(self, v):
        column_name = v[0]
        boolean_mask = v[1]
        self.masks[column_name] = boolean_mask

    @property
    def isis_serial(self):
        """
        Generate an ISIS compatible serial number using the data file
        associated with this node.  This assumes that the data file
        has a PVL header.
        """
        if not hasattr(self, '_isis_serial'):
            try:
                self._isis_serial = generate_serial_number(self.image_path)
            except:
                self._isis_serial = None
        return self._isis_serial

    @property
    def nkeypoints(self):
        if hasattr(self, '_keypoints'):
            return len(self._keypoints)
        else:
            return 0

    def coverage(self):
        """
        Determines the area of keypoint coverage
        using the unprojected image, resulting
        in a rough estimation of the percentage area
        being covered.

        Returns
        -------
        coverage_area :  float
                        Area covered by the generated
                        keypoints
        """

        points = self.get_keypoint_coordinates()
        hull = cg.convex_hull(points)
        hull_area = hull.volume

        max_x = self.geodata.raster_size[0]
        max_y = self.geodata.raster_size[1]

        total_area = max_x * max_y

        self.coverage_area = (hull_area/total_area)*100

        return self.coverage_area

    def get_array(self, band=1):
        """
        Get a band as a 32-bit numpy array

        Parameters
        ----------
        band : int
               The band to read, default 1
        """

        array = self.geodata.read_array(band=band)
        return bytescale(array)

    def get_keypoints(self, index=None):
        """
        Return the keypoints for the node.  If index is passed, return
        the appropriate subset.

        Parameters
        ----------
        index : iterable
                indices for of the keypoints to return

        Returns
        -------
         : dataframe
           A pandas dataframe of keypoints

        """
        if hasattr(self, '_keypoints'):
            if index is not None:
                return self._keypoints.loc[index]
            else:
                return self._keypoints

    def get_keypoint_coordinates(self, index=None, homogeneous=False):
        """
        Return the coordinates of the keypoints without any ancillary data

        Parameters
        ----------
        index : iterable
                indices for of the keypoints to return

        homogeneous : bool
                      If True, return homogeneous coordinates in the form
                      [x, y, 1]. Default: False

        Returns
        -------
         : dataframe
           A pandas dataframe of keypoint coordinates
        """
        keypoints = self.get_keypoints(index=index)[['x', 'y']]

        if homogeneous:
            keypoints['homogeneous'] = 1

        return keypoints

    def extract_features(self, array, **kwargs):
        """
        Extract features for the node

        Parameters
        ----------
        array : ndarray

        kwargs : dict
                 kwargs passed to autocnet.feature_extractor.extract_features

        """
        self._keypoints, self.descriptors = fe.extract_features(array, **kwargs)

    def load_features(self, in_path):
        """
        Load keypoints and descriptors for the given image
        from a HDF file.

        Parameters
        ----------
        in_path : str or object
                  PATH to the hdf file or a HDFDataset object handle
        """
        if isinstance(in_path, str):
            hdf = io_hdf.HDFDataset(in_path, mode='r')
        else:
            hdf = in_path

        self.descriptors = hdf['{}/descriptors'.format(self.image_name)][:]
        raw_kps = hdf['{}/keypoints'.format(self.image_name)][:]
        index = raw_kps['index']
        clean_kps = utils.remove_field_name(raw_kps, 'index')
        columns = clean_kps.dtype.names

        allkps = pd.DataFrame(data=clean_kps, columns=columns, index=index)

        self._keypoints = allkps.sort_values(by='response', ascending=False)

        if isinstance(in_path, str):
            hdf = None

    def save_features(self, out_path):
        """
        Save the extracted keypoints and descriptors to
        the given HDF5 file.

        Parameters
        ----------
        out_path : str or object
                   PATH to the hdf file or a HDFDataset object handle
        """

        if not hasattr(self, '_keypoints'):
            warnings.warn('Node {} has not had features extracted.'.format(i))
            return

        # If the out_path is a string, access the HDF5 file
        if isinstance(out_path, str):
            if os.path.exists(out_path):
                mode = 'a'
            else:
                mode = 'w'
            hdf = io_hdf.HDFDataset(out_path, mode=mode)
        else:
            hdf = out_path

        try:
            hdf.create_dataset('{}/descriptors'.format(self.image_name),
                               data=self.descriptors,
                               compression=io_hdf.DEFAULT_COMPRESSION,
                               compression_opts=io_hdf.DEFAULT_COMPRESSION_VALUE)
            hdf.create_dataset('{}/keypoints'.format(self.image_name),
                               data=hdf.df_to_sarray(self._keypoints.reset_index()),
                               compression=io_hdf.DEFAULT_COMPRESSION,
                               compression_opts=io_hdf.DEFAULT_COMPRESSION_VALUE)
        except:
            warnings.warn('Descriptors for the node {} are already stored'.format(self.image_name))

        # If the out_path is a string, assume this method is being called as a singleton
        # and close the hdf file gracefully.  If an object, let the instantiator of the
        # object close the file
        if isinstance(out_path, str):
            hdf = None

    def group_correspondences(self, cg, *args, clean_keys=['fundamental'], deepen=False, **kwargs):
        """

        Parameters
        ----------
        cg : object
             The graph object this node is a member of

        deepen : bool
                 If True, attempt to punch matches through to all incident edges.  Default: False
        """
        node = self.node_id
        # Get the edges incident to the current node
        incident_edges = set(cg.edges(node)).intersection(set(cg.edges()))

        # If this node is free floating, ignore it.
        if not incident_edges:
             # TODO: Add dangling correspondences to control network anyway.  Subgraphs handle this segmentation if req.
            return

        # Grab all the incident edge matches and concatenate into a group match set.
        # All share the same source node
        edge_matches = []
        for e in incident_edges:
            edge = cg[e[0]][e[1]]
            matches, mask = edge._clean(clean_keys=clean_keys)
            # Add a depth mask that initially mirrors the fundamental mask
            edge_matches.append(matches)
        d = pd.concat(edge_matches)

        # Counter for point identifiers
        pid = 0

        # Iterate through all of the correspondences and attempt to add additional correspondences using
        # the epipolar constraint
        for idx, g in d.groupby('source_idx'):
            # Pull the source index to be used as the search
            source_idx = g['source_idx'].values[0]

            # Add the point object onto the node
            point = Point(pid)

            covered_edges = list(map(tuple, g[['source_image', 'destination_image']].values))
            # The reference edge that we are deepening with
            ab = cg.edge[covered_edges[0][0]][covered_edges[0][1]]

            # Get the coordinates of the search correspondence
            ab_keypoints = ab.source.get_keypoint_coordinates(index=g['source_idx'])
            ab_x = None

            for j, (r_idx, r) in enumerate(g.iterrows()):
                kp = ab_keypoints.iloc[j].values

                # Homogenize the coord used for epipolar projection
                if ab_x is None:
                    ab_x = np.array([kp[0], kp[1], 1.])

                kpd = ab.destination.get_keypoint_coordinates(index=g['destination_idx']).values[0]
                # Add the existing source and destination correspondences
                self.point_to_correspondence[point].add((r['source_image'],
                                                                  Correspondence(r['source_idx'],
                                                                                 kp[0],
                                                                                 kp[1],
                                                                                 serial=self.isis_serial)))
                self.point_to_correspondence[point].add((r['destination_image'],
                                                                  Correspondence(r['destination_idx'],
                                                                                 kpd[0],
                                                                                 kpd[1],
                                                                                 serial=cg.node[r['destination_image']].isis_serial)))

            # If the user wants to punch correspondences through
            if deepen:
                search_edges = incident_edges.difference(set(covered_edges))
                for search_edge in search_edges:
                    bc = cg.edge[search_edge[0]][search_edge[1]]
                    coords, idx = deepen_correspondences(ab_x, bc, source_idx)

                    if coords is not None:
                        cg.node[node].point_to_correspondence[point].add((search_edge[1],
                                                                          Correspondence(idx,
                                                                                         coords[0],
                                                                                         coords[1],
                                                                                         serial=cg.node[search_edge[1]].isis_serial)))

            pid += 1

        # Convert the dict to a dataframe
        data = []
        for k, measures in self.point_to_correspondence.items():
            for image_id, m in measures:
                data.append((k.point_id, k.point_type, m.serial, m.measure_type, m.x, m.y, image_id))

        columns = ['point_id', 'point_type', 'serialnumber', 'measure_type', 'x', 'y', 'node_id']
        self.point_to_correspondence_df = pd.DataFrame(data, columns=columns)

    def suppress(self, func=spf.response, **kwargs):
        if not hasattr(self, '_keypoints'):
            raise AttributeError('No keypoints extracted for this node.')

        domain = self.handle.raster_size
        self._keypoints['strength'] = self._keypoints.apply(func, axis=1)

        if not hasattr(self, 'suppression'):
            # Instantiate a suppression object and suppress keypoints
            self.suppression = od.SpatialSuppression(self._keypoints, domain, **kwargs)
            self.suppression.suppress()
        else:
            # Update the suppression object attributes and process
            for k, v in kwargs.items():
                if hasattr(self.suppression, k):
                    setattr(self.suppression, k, v)
            self.suppression.suppress()

        self.masks = ('suppression', self.suppression.mask)

    def coverage_ratio(self, clean_keys=[]):
        """
        Compute the ratio $area_{convexhull} / area_{total}$

        Returns
        -------
        ratio : float
                The ratio of convex hull area to total area.
        """
        ideal_area = self.geodata.pixel_area
        if not hasattr(self, '_keypoints'):
            raise AttributeError('Keypoints must be extracted already, they have not been.')

        matches, mask = self._clean(clean_keys)
        keypoints = self._keypoints[mask][['x', 'y']].values

        ratio = convex_hull_ratio(keypoints, ideal_area)
        return ratio

    def plot(self, clean_keys=[], **kwargs):  # pragma: no cover
        return plot_node(self, clean_keys=clean_keys, **kwargs)

    def _clean(self, clean_keys):
        """
        Given a list of clean keys compute the
        mask of valid matches

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
        if not hasattr(self, '_keypoints'):
            raise AttributeError('Keypoints have not been extracted for this node.')
        panel = self.masks
        mask = panel[clean_keys].all(axis=1)
        matches = self._keypoints[mask]
        return matches, mask

