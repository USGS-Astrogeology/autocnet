from collections import defaultdict, MutableMapping
import itertools
import os
import warnings

from csmapi import csmapi
import numpy as np
import pandas as pd

from plio.io.io_gdal import GeoDataset
from plio.io.isis_serial_number import generate_serial_number
from skimage.transform import resize
import shapely
from knoten.csm import generate_latlon_footprint, generate_vrt, create_camera, generate_boundary

from autocnet import Session, engine, config
from autocnet.matcher import cpu_extractor as fe
from autocnet.matcher import cpu_outlier_detector as od
from autocnet.cg import cg
from autocnet.io.db.model import Images, Keypoints, Matches, Cameras,  Base, Overlay, Edges, Costs, Points, Measures
from autocnet.io.db.connection import Parent
from autocnet.io import keypoints as io_keypoints
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
        self.masks = pd.DataFrame()

    @property
    def camera(self):
        if not hasattr(self, '_camera'):
            self._camera = None
        return self._camera

    @camera.setter
    def camera(self, camera):
        self._camera = camera

    @property
    def descriptors(self):
        if not hasattr(self, '_descriptors'):
            self._descriptors = None
        return self._descriptors

    @descriptors.setter
    def descriptors(self, desc):
        self._descriptors = desc

    @property
    def keypoints(self):
        if not hasattr(self, '_keypoints'):
            self._keypoints = pd.DataFrame()
        return self._keypoints

    @keypoints.setter
    def keypoints(self, kps):
        self._keypoints = kps

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
        return hash(self['node_id'])

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
        return self['node_id'] == other

    @classmethod
    def create(cls, image_name, node_id, basepath=None):
        try:
            image_name = os.path.basename(image_name)
        except: pass  # Use the input name even if not a valid PATH
        if basepath is not None:
            image_path = os.path.join(basepath, image_name)
        else:
            image_path = image_name
        return cls(image_name, image_path, node_id)

    @property
    def geodata(self):
        if not getattr(self, '_geodata', None) and self['image_path'] is not None:
            try:
                self._geodata = GeoDataset(self['image_path'])
                return self._geodata
            except:
                return self['node_id']
        if hasattr(self, '_geodata'):
            return self._geodata
        else:
            return None

    @property
    def footprint(self):
        if not getattr(self, '_footprint', None):
            try:
                self._footprint = shapely.wkt.loads(self.geodata.footprint.GetGeometryRef(0).ExportToWkt())
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
        try:
            return len(self.keypoints)
        except:
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
        return utils.bytescale(array)

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
            keypoints = keypoints.assign(homogeneous = 1)
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

    def extract_features(self, array, xystart=[], camera=None, *args, **kwargs):

        new_keypoints, new_descriptors = Node._extract_features(array, *args, **kwargs)
        count = len(self.keypoints)

        # If this is a tile, push the keypoints to the correct start xy
        if xystart:
            new_keypoints['x'] += xystart[0]
            new_keypoints['y'] += xystart[1]

        concat_kps = pd.concat((self.keypoints, new_keypoints))
        concat_kps.reset_index(inplace=True, drop=True)
        concat_kps.drop_duplicates(inplace=True)
        # Removed duplicated and re-index the merged keypoints

        # Update the descriptors to be the same size as the keypoints, maintaining alignment
        if self.descriptors is not None:
            concat = np.concatenate((self.descriptors, new_descriptors))
        else:
            concat = new_descriptors
        new_descriptors = concat[concat_kps.index.values]

        self.descriptors = new_descriptors
        self.keypoints = concat_kps.reset_index(drop=True)

        lkps = len(self.keypoints)

        assert lkps == len(self.descriptors)

        if lkps > 0:
            return True

    def extract_features_from_overlaps(self, overlaps=[], downsampling=False, tiling=False, *args, **kwargs):
        # iterate through the overlaps
        # check for downsampling or tiling and dispatch as needed to that func
        # that should then dispatch to the extract features func
        pass

    def extract_features_with_downsampling(self, downsample_amount,
                                           array_read_args={}, *args, **kwargs):
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
        array = resize(self.geodata.read_array(**array_read_args), shape, preserve_range=True)
        self.extract_features(array, *args, **kwargs)

        self.keypoints['x'] *= downsample_amount
        self.keypoints['y'] *= downsample_amount

        if len(self.keypoints) > 0:
            return True

    def extract_features_with_tiling(self, tilesize=1000, overlap=500, *args, **kwargs):
        array_size = self.geodata.raster_size
        slices = utils.tile(array_size, tilesize=tilesize, overlap=overlap)
        for s in slices:
            xystart = [s[0], s[1]]
            array = self.geodata.read_array(pixels=s)
            self.extract_features(array, xystart, *args, **kwargs)

        if len(self.keypoints) > 0:
            return True

    def project_keypoints(self):
        if self.camera is None:
            # Without a camera, it is not possible to project
            warnings.warn('Unable to project points, no camera available.')
            return False
        # Project the sift keypoints to the ground
        def func(row, args):
            camera = args[0]
            imagecoord = csmapi.ImageCoord(float(row[1]), float(row[0]))
            # An elevation at the ellipsoid is plenty accurate for this work
            gnd = getattr(camera, 'imageToGround')(imagecoord, 0)
            return [gnd.x, gnd.y, gnd.z]
        feats = self.keypoints[['x', 'y']].values
        gnd = np.apply_along_axis(func, 1, feats, args=(self.camera, ))
        gnd = pd.DataFrame(gnd, columns=['xm', 'ym', 'zm'], index=self.keypoints.index)
        self.keypoints = pd.concat([self.keypoints, gnd], axis=1)

        return True

    def load_features(self, in_path, format='npy', **kwargs):
        """
        Load keypoints and descriptors for the given image
        from a HDF file.

        Parameters
        ----------
        in_path : str or object
                  PATH to the hdf file or a HDFDataset object handle

        format : {'npy', 'hdf'}
                 The format that the features are stored in.  Default: npy.
        """
        if format == 'npy':
            keypoints, descriptors = io_keypoints.from_npy(in_path)
        elif format == 'hdf':
            keypoints, descriptors = io_keypoints.from_hdf(in_path, **kwargs)

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
        return shapely.geometryPolygon(reproj)

class NetworkNode(Node):
    def __init__(self, *args, parent=None, **kwargs):
        super(NetworkNode, self).__init__(*args, **kwargs)
        # If this is the first time that the image is seen, add it to the DB
        if parent is None:
            self.parent = Parent(config)
        else:
            self.parent = parent

        # Create a session to work in
        session = Session()

        # For now, just use the PATH to determine if the node/image is in the DB
        res = session.query(Images).filter(Images.path == kwargs['image_path']).first()
        session.close()
        if res is None:
            kpspath = io_keypoints.create_output_path(self.geodata.file_name)

            # Create the keypoints entry
            kps = Keypoints(path=kpspath, nkeypoints=0)
            cam = self.create_camera()
            try:
                fp = self.footprint
            except Exception as e:
                warnings.warn('Unable to generate image footprint.\n{}'.format(e))
                fp = None
            # Create the image
            i = Images(name=kwargs['image_name'],
                       path=kwargs['image_path'],
                       footprint_latlon=fp,
                       keypoints=kps,
                       cameras=cam,
                       serial=self.isis_serial)
            session = Session()
            session.add(i)
            session.commit()
            session.close()
        self.job_status = defaultdict(dict)

    def _from_db(self, table_obj, key='image_id'):
        """
        Generic database query to pull the row associated with this node
        from an arbitrary table. We assume that the row id matches the node_id.

        Parameters
        ----------
        table_obj : object
                    The declared table class (from db.model)

        key : str
              The name of the column to compare this object's node_id with. For
              most tables this will be the default, 'image_id' because 'image_id'
              is the foreign key in the DB. For the Images table (the parent table),
              the key is simply 'id'.
        """
        if 'node_id' not in self.keys():
            return
        session = Session()
        res = session.query(table_obj).filter(getattr(table_obj,key) == self['node_id']).first()
        session.close()
        return res

    @property
    def keypoint_file(self):
        res = self._from_db(Keypoints)
        if res is None:
            return
        return res.path

    @property
    def keypoints(self):
        try:
            return io_keypoints.from_hdf(self.keypoint_file, descriptors=False)
        except:
            return pd.DataFrame()

    @keypoints.setter
    def keypoints(self, kps):
        session = Session()
        io_keypoints.to_hdf(self.keypoint_file, keypoints=kps)
        res = session.query(Keypoints).filter(getattr(Keypoints,'image_id') == self['node_id']).first()

        if res is None:
            _ = self.keypoint_file
            res = self._from_db(Keypoints)
        res.nkeypoints = len(kps)
        session.commit()
        session.close()

    @property
    def descriptors(self):
        try:
            return io_keypoints.from_hdf(self.keypoint_file, keypoints=False)
        except:
            return

    @descriptors.setter
    def descriptors(self, desc):
        if isinstance(desc, np.array):
            io_keypoints.to_hdf(self.keypoint_file, descriptors=desc)

    @property
    def nkeypoints(self):
        """
        Get the number of keypoints from the database
        """
        res = self._from_db(Keypoints)
        nkps = res.nkeypoints
        return nkps

    def create_camera(self):
        # Create the camera entry
        import pvl
        import requests
        import json

        label = pvl.dumps(self.geodata.metadata).decode()
        url = config['pfeffernusse']['url']
        response = requests.post(url, json={'label':label})
        response = response.json()
        model_name = response.get('name_model', None)
        if model_name is None:
            return
        isdpath = os.path.splitext(self['image_path'])[0] + '.json'
        try:
            with open(isdpath, 'w') as f:
                json.dump(response, f)
        except Exception as e:
            warnings.warn('Failed to write JSON ISD for image {}.\n{}'.format(self['image_path'], e))
        isd = csmapi.Isd(self['image_path'])
        plugin = csmapi.Plugin.findPlugin('UsgsAstroPluginCSM')
        self._camera = plugin.constructModelFromISD(isd, model_name)
        serialized_camera = self._camera.getModelState()

        cam = Cameras(camera=serialized_camera, image_id=self['node_id'])
        return cam

    @property
    def camera(self):
        """
        Get the camera object from the database.
        """
        # TODO: This should use knoten once it is stable.
        import csmapi
        if not getattr(self, '_camera', None):
            res = self._from_db(Cameras)
            plugin = csmapi.Plugin.findPlugin('UsgsAstroPluginCSM')
            if res is not None:
                self._camera = plugin.constructModelFromState(res.camera)
            else:
                self.create_camera()
        return self._camera

    @property
    def footprint(self):
        session = Session()
        res = session.query(Images).filter(Images.id == self['node_id']).first()
        session.close()
        # not in database, create footprint
        if res is None:
            # get ISIS footprint if possible
            if utils.find_in_dict(self.geodata.metadata, "Polygon"):
                footprint_latlon =  shapely.wkt.loads(self.geodata.footprint.ExportToWkt())
                return footprint_latlon
            # Get CSM footprint
            else:
                boundary = generate_boundary(self.geodata.raster_size[::-1])  # yx to xy
                try:
                    geodata = GeoDataset(config['spatial']['dem'])
                except Exception as e:
                    warnings.warn('Unable to get the Geodata from dem.\n{}'.format(e))
                    geodata = 0.0

                footprint_latlon = generate_latlon_footprint(self.camera, boundary, dem=geodata)
                footprint_latlon.FlattenTo2D()
                return footprint_latlon
        else:
            # in database, return footprint
            footprint_latlon = res.footprint_latlon
            return footprint_latlon

    @property
    def points(self):
        session = Session()
        pids = session.query(Measures.pointid).filter(Measures.imageid == self['node_id']).all()
        res = session.query(Points).filter(Points.id.in_(pids)).all()
        session.close()
        return res

    @property
    def measures(self):
        session = Session()
        res = session.query(Measures).filter(Measures.imageid == self['node_id']).all()
        session.close()
        return res

    def generate_vrt(self, **kwargs):
        """
        Using the image footprint, generate a VRT to that is usable inside
        of a GIS (QGIS or ArcGIS) to visualize a warped version of this image.
        """
        outpath = config['directories']['vrt_dir']
        generate_vrt.warped_vrt(self.camera, self.geodata.raster_size,
                                self.geodata.file_name, outpath=outpath)
