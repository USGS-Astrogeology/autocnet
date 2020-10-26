from math import modf, floor
import numpy as np


class Roi():
    """
    Region of interest (ROI) object that is a sub-image taken from
    a larger image or array. This object supports transformations
    between the image coordinate space and the ROI coordinate
    space.

    Attributes
    ----------
    data : ndarray/object
           An ndarray or an object with a raster_size attribute

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
    def __init__(self, data, x, y, size_x=200, size_y=200, dtype=None, ndv=None):
        self.data = data
        self.ndv = ndv
        self.x = x
        self.y = y
        self.dtype = dtype
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
    def ndv(self):
        """
        The no data value of the ROI. Used by the is_valid
        property to determine if the ROI contains any null
        pixels.
        """
        if hasattr(self.data, 'no_data_value'):
            self._ndv = self.data.no_data_value
        return self._ndv

    @ndv.setter
    def ndv(self, ndv):
        self._ndv = ndv

    @property
    def size_x(self):
        return self._size_x

    @size_x.setter
    def size_x(self, size_x):
        if not isinstance(size_x, int):
            raise TypeError(f'size_x must be type integer, not {type(size_x)}')
        self._size_x = size_x

    @property
    def size_y(self):
        return self._size_y

    @size_y.setter
    def size_y(self, size_y):
        if not isinstance(size_y, int):
            raise TypeError(f'size_y must be type integer, not {type(size_y)}')
        self._size_y = size_y

    @property
    def image_extent(self):
        """
        In full image space, this method computes the valid
        pixel indices that can be extracted.
        """
        try:
            # Geodataset object
            raster_size = self.data.raster_size
        except:
            # Numpy array in y,x form
            raster_size = self.data.shape[::-1]

        # what is the extent that can actually be extracted?
        left_x = self._x - self.size_x
        right_x = self._x + self.size_x
        top_y = self._y - self.size_y
        bottom_y = self._y + self.size_y

        if self._x - self.size_x < 0:
            left_x = 0
        if self._y - self.size_y < 0:
            top_y = 0
        if self._x + self.size_x > raster_size[0]:
            right_x = raster_size[0]
        if self._y + self.size_y > raster_size[1]:
            bottom_y = raster_size[1]

        return list(map(int, [left_x, right_x, top_y, bottom_y]))

    @property
    def center(self):
        ie = self.image_extent
        return (ie[1] - ie[0])/2, (ie[3]-ie[2])/2

    @property
    def is_valid(self):
        """
        True if all elements in the clipped ROI are valid, i.e.,
        no null pixels (as defined by the no data value (ndv)) are
        present.
        """
        return self.ndv not in self.array

    @property
    def array(self):
        """
        The clipped array associated with this ROI.
        """
        pixels = self.image_extent
        if isinstance(self.data, np.ndarray):
             return self.data[pixels[2]:pixels[3]+1,pixels[0]:pixels[1]+1]
        else:
            # Have to reformat to [xstart, ystart, xnumberpixels, ynumberpixels]
            pixels = [pixels[0], pixels[2], pixels[1]-pixels[0]+1, pixels[3]-pixels[2]+1]
            return self.data.read_array(pixels=pixels, dtype=self.dtype)

    def clip(self, dtype=None):
        """
        Compatibility function that makes a call to the array property.

        Warning: The dtype passed in via this function resets the dtype attribute of this
        instance.

        Parameters
        ----------
        dtype : str
                The datatype to be used when reading the ROI information if the read
                occurs through the data object using the read_array method. When using
                this object when the data are a numpy array the dtype has not effect.

        Returns
        -------
         : ndarray
           The array attribute of this object.
        """
        self.dtype = dtype
        return self.array
