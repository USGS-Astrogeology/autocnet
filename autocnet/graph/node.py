from collections import defaultdict, MutableMapping
import itertools
import os
import warnings

import numpy as np
import pandas as pd
from plio.io.io_gdal import GeoDataset
from plio.io.isis_serial_number import generate_serial_number
from scipy.misc import bytescale, imresize
from shapely.geometry import Polygon
from shapely import wkt

from autocnet.cg import cg

from autocnet.io import keypoints as io_keypoints

from autocnet.matcher.add_depth import deepen_correspondences
from autocnet.matcher import cpu_extractor as fe
from autocnet.matcher import cpu_outlier_detector as od
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
        self['image_name'] = image_name
        self['image_path'] = image_path
        self['node_id'] = node_id
        self['hash'] = image_name
        self.descriptors = None
        self.keypoints = pd.DataFrame()
        self.masks = pd.DataFrame()

    def __repr__(self):
        return """
        NodeID: {}
        Image Name: {}
        Image PATH: {}
        Number Keypoints: {}
        Available Masks : {}
        Type: {}
        """.format(self['node_id'], self['image_name'], self['image_path'],
                   self.nkeypoints, self.masks, self.__class__)

    def __hash__(self): #pragma: no cover
        return hash(repr(self))

    def __gt__(self, other):
        myid = self['node_id']
        oid = other['node_id']
        return myid > oid

    def __ge__(self, other):
        myid = self['node_id']
        oid = other['node_id']
        return myid >= oid

    def __lt__(self, other):
        myid = self['node_id']
        oid = other['node_id']
        return myid < oid

    def __le__(self, other):
        myid = self['node_id']
        oid = other['node_id']
        return myid <= oid

    def __str__(self):
        return str(self['node_id'])

    def __eq__(self, other):
        return utils.compare_dicts(self.__dict__, other.__dict__) *\
               utils.compare_dicts(self, other)


    @property
    def geodata(self):
        if not getattr(self, '_geodata', None) and self['image_path'] is not None:
            self._geodata = GeoDataset(self['image_path'])
            return self._geodata
        if hasattr(self, '_geodata'):
            return self._geodata
        else:
            return None

    @property
    def footprint(self):
        if not getattr(self, '_footprint', None):
            try:
                self._footprint = wkt.loads(self.geodata.footprint.GetGeometryRef(0).ExportToWkt())
            except:
                return None
        return self._footprint

    @property
    def isis_serial(self):
        """
        Generate an ISIS compatible serial number using the data file
        associated with this node.  This assumes that the data file
        has a PVL header.
        """
        if not hasattr(self, '_isis_serial'):
            try:
                self._isis_serial = generate_serial_number(self['image_path'])
            except:
                self._isis_serial = None
        return self._isis_serial

    @property
    def nkeypoints(self):
        return len(self.keypoints)

    def coverage(self):
        """
        Determines the area of keypoint coverage
        using the unprojected image, resulting
        in a rough estimation of the percentage area
        being covered.

        Returns
        -------
        coverage_area :  float
                         percentage area covered by the generated
                         keypoints
        """

        points = self.get_keypoint_coordinates()
        hull = cg.convex_hull(points)
        hull_area = hull.volume

        max_x = self.geodata.raster_size[0]
        max_y = self.geodata.raster_size[1]

        total_area = max_x * max_y

        return hull_area / total_area

    def get_byte_array(self, band=1):
        """
        Get a band as a 32-bit numpy array

        Parameters
        ----------
        band : int
               The band to read, default 1
        """

        array = self.geodata.read_array(band=band)
        return bytescale(array)

    def get_array(self, band=1, **kwargs):
        """
        Get a band as a 32-bit numpy array

        Parameters
        ----------
        band : int
               The band to read, default 1
        """

        array = self.geodata.read_array(band=band, **kwargs)
        return array

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
        if index is not None:
            return self.keypoints.loc[index]
        else:
            return self.keypoints

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
        if index is None:
            keypoints = self.keypoints[['x', 'y']]
        else:
            keypoints = self.keypoints.loc[index][['x', 'y']]

        if homogeneous:
            keypoints['homogeneous'] = 1

        return keypoints

    def get_raw_keypoint_coordinates(self, index=slice(None)):
        """
        The performance of get_keypoint_coordinates can be slow
        due to the ability for fancier indexing.  This method
        returns coordinates using numpy array accessors.

        Parameters
        ----------
        index : iterable
                positional indices to return from the global keypoints dataframe
        """
        return self.keypoints.values[index,:2]

    @staticmethod
    def _extract_features(array, *args, **kwargs):  # pragma: no cover
        """
        Extract features for the node

        Parameters
        ----------
        array : ndarray

        kwargs : dict
                 kwargs passed to autocnet.cpu_extractor.extract_features

        """
        pass

    def extract_features(self, array, xystart=[], *args, **kwargs):
        arraysize = array.shape[0] * array.shape[1]

        try:
            maxsize = self.maxsize[0] * self.maxsize[1]
        except:
            maxsize = np.inf

        if arraysize > maxsize:
            warnings.warn('Node: {}. Maximum feature extraction array size is {}.  Maximum array size is {}. Please use tiling or downsampling.'.format(self['node_id'], maxsize, arraysize))

        keypoints, descriptors = Node._extract_features(array, *args, **kwargs)
        count = len(self.keypoints)

        if xystart:
            keypoints['x'] += xystart[0]
            keypoints['y'] += xystart[1]

        self.keypoints = pd.concat((self.keypoints, keypoints))
        descriptor_mask = self.keypoints.duplicated()[count:]
        number_new = descriptor_mask.sum()

        # Removed duplicated and re-index the merged keypoints
        self.keypoints.drop_duplicates(inplace=True)
        self.keypoints.reset_index(inplace=True, drop=True)

        if self.descriptors is not None:
            self.descriptors = np.concatenate((self.descriptors, descriptors[~descriptor_mask]))
        else:
            self.descriptors = descriptors

    def extract_features_from_overlaps(self, overlaps=[], downsampling=False, tiling=False, *args, **kwargs):
        # iterate through the overlaps
        # check for downsampling or tiling and dispatch as needed to that func
        # that should then dispatch to the extract features func
        pass

    def extract_features_with_downsampling(self, downsample_amount,
                                           array_read_args={},
                                           interp='lanczos', *args, **kwargs):
        """
        Extract interest points for the this node (image) by first downsampling,
        then applying the extractor, and then upsampling the results backin to
        true image space.

        Parameters
        ----------
        downsample_amount : int
                            The amount to downsample by
        """
        array_size = self.geodata.raster_size
        total_size = array_size[0] * array_size[1]
        shape = (int(array_size[0] / downsample_amount),
                 int(array_size[1] / downsample_amount))
        array = imresize(self.geodata.read_array(**array_read_args), shape, interp=interp)
        self.extract_features(array, *args, **kwargs)
        self.keypoints['x'] *= downsample_amount
        self.keypoints['y'] *= downsample_amount

    def extract_features_with_tiling(self, tilesize=1000, overlap=500, *args, **kwargs):
        array_size = self.geodata.raster_size
        stepsize = tilesize - overlap
        if stepsize < 0:
            raise ValueError('Overlap can not be greater than tilesize.')
        # Compute the tiles
        if tilesize >= array_size[1]:
            ytiles = [(0, array_size[1])]
        else:
            ystarts = range(0, array_size[1], stepsize)
            ystops = range(tilesize, array_size[1], stepsize)
            ytiles = list(zip(ystarts, ystops))
            ytiles.append((ytiles[-1][0] + stepsize, array_size[1]))

        if tilesize >= array_size[0]:
            xtiles = [(0, array_size[0])]
        else:
            xstarts = range(0, array_size[0], stepsize)
            xstops = range(tilesize, array_size[0], stepsize)
            xtiles = list(zip(xstarts, xstops))
            xtiles.append((xtiles[-1][0] + stepsize, array_size[0]))
        tiles = itertools.product(xtiles, ytiles)

        for tile in tiles:
            # xstart, ystart, xcount, ycount
            xstart = tile[0][0]
            ystart = tile[1][0]
            xstop = tile[0][1]
            ystop = tile[1][1]
            pixels = [xstart, ystart,
                      xstop - xstart,
                      ystop - ystart]

            array = self.geodata.read_array(pixels=pixels)
            xystart = [xstart, ystart]
            self.extract_features(array, xystart, *args, **kwargs)

    def load_features(self, in_path, format='npy'):
        """
        Load keypoints and descriptors for the given image
        from a HDF file.

        Parameters
        ----------
        in_path : str or object
                  PATH to the hdf file or a HDFDataset object handle

        format : {'npy', 'hdf'}
        """
        if format == 'npy':
            keypoints, descriptors = io_keypoints.from_npy(in_path)
        elif format == 'hdf':
            keypoints, descriptors = io_keypoints.from_hdf(in_path,
                                                           key=self['image_name'])

        self.keypoints = keypoints
        self.descriptors = descriptors

    def save_features(self, out_path):
        """
        Save the extracted keypoints and descriptors to
        the given file.  By default, the .npz files are saved
        along side the image, e.g. in the same folder as the image.

        Parameters
        ----------
        out_path : str or object
                   PATH to the directory for output and base file name
        """
        if self.keypoints.empty:
            warnings.warn('Node {} has not had features extracted.'.format(self['node_id']))
            return

        io_keypoints.to_npy(self.keypoints, self.descriptors,
                            out_path + '_{}.npz'.format(self['node_id']))

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
        if self.keypoints.empty:
            raise AttributeError('Keypoints have not been extracted for this node.')
        panel = self.masks
        mask = panel[clean_keys].all(axis=1)
        matches = self.keypoints[mask]
        return matches, mask

    def reproject_geom(self, coords):   # pragma: no cover
        """
        Reprojects a set of latlon coordinates into pixel space using the nodes
        geodata. These are then returned as a shapely polygon

        Parameters
        ----------
        coords : ndarray
                      (n, 2) array of latlon coordinates

        Returns
        ----------
        : object
          A shapely polygon object made using the reprojected coordinates
        """
        reproj = []

        for x, y in coords:
            reproj.append(self.geodata.latlon_to_pixel(y, x))
        return Polygon(reproj)
