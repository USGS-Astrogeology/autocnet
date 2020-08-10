from math import modf, floor
import numpy as np

from pysis.exceptions import ProcessError

from skimage.feature import register_translation

from skimage import transform as tf
from plio.io.io_gdal import GeoDataset

from matplotlib import pyplot as plt

from autocnet.matcher.naive_template import pattern_match, pattern_match_autoreg
from autocnet.matcher import ciratefi
from autocnet.io.db.model import Measures, Points, Images, JsonEncoder
from autocnet import spatial
from autocnet.utils.utils import bytescale


class Roi():
    """
    Region of interest (ROI) object that is a sub-image taken from
    a larger image or array. This object supports transformations
    between the image coordinate space and the ROI coordinate
    space.

    Attributes
    ----------

    x : float
        The x coordinate in image space

    y : float
        The y coordinate in image space

    size_x : int
             1/2 the total ROI width in pixels

    size_y : int
             1/2 the total ROI height in pixels

    left_x : int
             The left pixel coordinate in image space

    right_x : int
              The right pixel coordinage in image space

    top_y : int
            The top image coordinate in image space

    bottom_y : int
               The bottom image coordinate in imge space
    """
    def __init__(self, geodataset, x, y, size_x=200, size_y=200):
        self.geodataset = geodataset

        self.x = x
        self.y = y
        self.size_x = size_x
        self.size_y = size_y

    @property
    def x(self):
        return self._x + self.axr

    @x.setter
    def x(self, x):
        self.axr, self._x = modf(x)

    @property
    def y(self):
        return self._y + self.ayr

    @y.setter
    def y(self, y):
        self.ayr, self._y = modf(y)

    @property
    def image_extent(self):
        """
        In full image space, this method computes the valid
        pixel indices that can be extracted.
        """
        try:
            # Geodataset object
            raster_size = self.geodataset.raster_size
        except:
            # Numpy array in y,x form
            raster_size = self.geodataset.shape[::-1]

        # what is the extent that can actually be extracted?
        left_x = self._x - self.size_x
        right_x = self._x + self.size_x
        top_y = self._y - self.size_y
        bottom_y = self.y + self.size_y

        if self._x - self.size_x < 0:
            left_x = 0
        if self._y - self.size_y < 0:
            top_y = 0
        if self._x + self.size_x > raster_size[0]:
            right_x = raster_size[0]
        if self._y + self.size_y > raster_size[1]:
            bottom_y = raster_size[1]

        return list(map(int, [left_x, right_x, top_y, bottom_y]))

    def geotransform(self, other):
        """

        """
        other_size = other.raster_size

        # specifically not putting this in a try/except, this should never fail
        mlat, mlon = spatial.pixel_to_latlon(self.geodataset.file_name, sample=self.x, line=self.y)
        centery, centerx = spatial.latlon_to_pixel(other.file_name, lon=mlon, lat=mlat)

        startx, stopx, starty, stopy = self.image_extent
        base_corners = [(startx, starty),
                        (startx, stopy),
                        (stopx, stopy),
                        (stopx, starty)]

        dst_corners = []
        for x,y in base_corners:
            try:
                lat, lon = spatial.pixel_to_latlon(self.geodataset.file_name, sample=x, line=y)
                line, sample = spatial.latlon_to_pixel(other.file_name, lon=lon, lat=lat)
                dst_corners.append((line, sample))
            except ProcessError as e:
                if 'Requested position does not project in camera model' in e.stderr:
                    print(f'Skip geom_match; Region of interest corner located at ({lon}, {lat}) does not project to image {other.base_name}')
                    return None, None


        base_gcps = np.array([*base_corners])
        base_gcps[:,0] -= startx
        base_gcps[:,1] -= starty

        dst_gcps = np.array([*dst_corners])
        startx = dst_gcps[:,0].min()
        starty = dst_gcps[:,1].min()
        stopx = dst_gcps[:,-1].max()
        stopy = dst_gcps[:,1].max()
        dst_gcps[:,0] -= startx
        dst_gcps[:,1] -= starty

        affine = tf.estimate_transform('affine', np.array([*base_gcps]), np.array([*dst_gcps]))
        otherRoi = Roi(other, centerx, centery, int(max(dst_gcps[:,0])//2), int(max(dst_gcps[:,1])//2))
        return otherRoi, affine


    def clip(self, dtype=None):
        pixels = self.image_extent
        if isinstance(self.geodataset, np.ndarray):
            array = self.geodataset[pixels[2]:pixels[3]+1,
                                         pixels[0]:pixels[1]+1]
        else:
            # Have to reformat to [xstart, ystart, xnumberpixels, ynumberpixels]
            pixels = [pixels[0], pixels[2], pixels[1]-pixels[0], pixels[3]-pixels[2]]
            array = self.geodataset.read_array(pixels=pixels, dtype=dtype)

        return array

    def transform(self, x, y):
        """
        Convert arbitrary coordinates from the ROI coordinate system
        to the full image coordinate system.
        """
        pass
