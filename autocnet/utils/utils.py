import itertools
import json

from functools import reduce, singledispatch, update_wrapper

import numpy as np
import pandas as pd
import networkx as nx

from osgeo import ogr

def tile(array_size, tilesize=1000, overlap=500):
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

    slices = []
    for tile in tiles:
        # xstart, ystart, xcount, ycount
        xstart = tile[0][0]
        ystart = tile[1][0]
        xstop = tile[0][1]
        ystop = tile[1][1]
        pixels = [xstart, ystart,
                  xstop - xstart,
                  ystop - ystart]
        slices.append(pixels)
    return slices

def compare_dicts(d, o):
    """
    Given two dictionaries, compare them with support for np.ndarray and
    pd.DataFrame objects

    Parameters
    ----------
    d : dict
        first dict to compare

    o : dict
        second dict to compare

    Examples
    --------
    >>> d = {'a':0}
    >>> o = {'a':0}
    >>> compare_dicts(d, o)
    True
    >>> d['a'] = 1
    >>> compare_dicts(d,o)
    False
    >>> d['a'] = np.arange(3)
    >>> o['a'] = np.arange(3)
    >>> compare_dicts(d,o)
    True
    """
    for k in o.keys():
        if k not in d.keys():
            return False
    for k, v in d.items():
        if v is None and o[k] is not None:
            return False
        if isinstance(v, pd.DataFrame):
            if not v.equals(o[k]):
                return False
        elif isinstance(v, np.ndarray):
            if not np.allclose(v, o[k]):
                return False
        else:
            if k == '_geodata':
                continue
            if not v == o[k]:
                return False
    return True

def crossform(a):
    """
    Return the cross form, e.g. a in the cross product of a b.
    Parameters
    ----------
    a : ndarray
        (3,) vector

    Returns
    -------
    a : ndarray
        (3,3)
    """
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])


def normalize_vector(line):
    """
    Normalize a standard form line

    Parameters
    ----------
    line : ndarray
           Standard form of a line (Ax + By + C = 0)

    Returns
    -------
    line : ndarray
           The normalized line

    Examples
    --------
    >>> x = np.array([3, 1, 2])
    >>> nv = normalize_vector(x)
    >>> print(np.round(nv, 6))  # For doc test float percision
    [0.801784 0.267261 0.534522]
    """
    if isinstance(line, pd.DataFrame):
        line = line.values
    n = np.sqrt((line[0]**2 + line[1]**2 + line[2]**2))
    return line / abs(n)

def getnearest(iterable, value):
    """
    Given an iterable, get the index nearest to the input value

    Parameters
    ----------
    iterable : iterable
               An iterable to search

    value : int, float
            The value to search for

    Returns
    -------
        : int
          The index into the list
    """
    return min(enumerate(iterable), key=lambda i: abs(i[1] - value))


def checkbandnumbers(bands, checkbands):
    """
    Given a list of input bands, check that the passed
    tuple contains those bands.

    In case of THEMIS, we check for band 9 as band 9 is the temperature
    band required to derive thermal temperature.  We also check for band 10
    which is required for TES atmosphere calculations.

    Parameters
    ----------
    bands : tuple
            of bands in the input image
    checkbands : list
                 of bands to check against

    Returns
    -------
     : bool
       True if the bands are present, else False
    """
    for c in checkbands:
        if c not in bands:
            return False
    return True


def checkdeplaid(incidence):
    """
    Given an incidence angle, select the appropriate deplaid method.

    Parameters
    ----------
    incidence : float
                incidence angle extracted from the campt results.

    """
    if incidence >= 95 and incidence <= 180:
        return 'night'
    elif incidence >=90 and incidence < 95:
        return 'night'
    elif incidence >= 85 and incidence < 90:
        return 'day'
    elif incidence >= 0 and incidence < 85:
        return 'day'
    else:
        return False


def checkmonotonic(iterable, piecewise=False):
    """
    Check if a given iterable is monotonically increasing.

    Parameters
    ----------
    iterable : iterable
                Any Python iterable object

    piecewise : boolean
                If false, return a boolean for the entire iterable,
                else return a list with elementwise monotinicy checks

    Returns
    -------
    monotonic : bool/list
                A boolean list of all True if monotonic, or including
                an inflection point
    """
    monotonic = [True] + [x < y for x, y in zip(iterable, iterable[1:])]
    if piecewise is True:
        return monotonic
    else:
        return all(monotonic)


def find_in_dict(obj, key):
    """
    Recursively find an entry in a dictionary

    Parameters
    ----------
    obj : dict
          The dictionary to search
    key : str
          The key to find in the dictionary

    Returns
    -------
    item : obj
           The value from the dictionary
    """
    if key in obj:
        return obj[key]
    for k, v in obj.items():
        if isinstance(v,dict):
            item = find_in_dict(v, key)
            if item is not None:
                return item


def find_nested_in_dict(data, key_list):
    """
    Traverse a list of keys into a dict.

    Parameters
    ----------
    data : dict
           The dictionary to be traversed
    key_list: list
              The list of keys to be travered.  Keys are
              traversed in the order they are entered in
              the list

    Returns
    -------
    value : object
            The value in the dict
    """
    return reduce(lambda d, k: d[k], key_list, data)


def make_homogeneous(points):
    """
    Convert a set of points (n x dim array) to
        homogeneous coordinates.

    Parameters
    ----------
    points : arraylike
             n x m array of points, where n is the number
             of points.

    Returns
    -------
     : arraylike
       n x m + 1 array of homogeneous points
    """
    homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    if isinstance(points, pd.DataFrame):
        columns = points.columns.values.tolist() + ['z']
        homogeneous = pd.DataFrame(homogeneous, index=points.index,
                                    columns=columns)
    return homogeneous



def remove_field_name(a, name):
    """
    Given a numpy structured array, remove a column and return
    a copy of the remainder of the array

    Parameters
    ----------
    a : ndarray
        Numpy structured array

    name : str
           of the index (column) to be removed

    Returns
    -------
    b : ndarray
        Numpy structured array with the 'name' column removed
    """
    names = list(a.dtype.names)
    if name in names:
        names.remove(name)
    b = a[names]
    return b


def calculate_slope(x1, x2):
    """
    Calculates the 2-dimensional slope between the points in two dataframes each containing two columns ['x', 'y']
    The slope is calculated from x1 to x2.

    Parameters
    ----------
    x1 : dataframe
         Each row is a point with columns ['x', 'y']
    x2 : dataframe
        Each row is a point with columns ['x', 'y']

    Returns
    -------
    : dataframe
      A dataframe with the slope between the points in x1 and x2 for each row.
    """


    sl = False
    if isinstance(x1, pd.DataFrame):
        index = x1.index
        sl = True
        x1 = x1.values
    if isinstance(x2, pd.DataFrame):
        x2 = x2.values
    slopes = (x2[:,1] - x1[:,1])/(x2[:,0] - x1[:,0])

    if sl:
        slopes = pd.Series(slopes, index=index)
    return slopes


def cartesian(arrays, out=None):

    """
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    from scikit-learn
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out


def array_to_poly(array):
    """
    Generate a geojson geom
    Parameters
    ----------
    array : array-like
            2-D array of size (n, 2) of x, y coordinates

    Returns
    -------
    geom : GeoJson
           geojson containing the necessary data to construct
           a poly gon
    """
    array = np.asarray(array)
    size = np.shape(array)
    if size[1] != 2:
        raise ValueError('Array is not the proper size.')
        return
    geom_array = np.append(array, [array[0]], axis = 0).tolist()
    geom = {"type": "Polygon", "coordinates": [geom_array]}
    poly = ogr.CreateGeometryFromJson(json.dumps(geom))
    return poly


def methodispatch(func):
    """
    New dispatch decorator that looks at the second argument to
    avoid self

    Parameters
    ----------
    func : Object
        Function object to be dispatched

    Returns
    wrapper : Object
        Wrapped function call chosen by the dispatcher
    ----------

    """
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, dispatcher)
    return wrapper


def decorate_class(cls, decorator, exclude=[], *args, **kwargs): # pragma: no cover
    """
    Decorates a class with a give docorator. Returns a subclass with
    dectorations applied

    Parameters
    ----------
    cls : Class
          A class to be decorated

    decorator : callable
                callable to wrap cls's methods with

    exclude : list
              list of method names to exclude from being decorated

    args, kwargs : list, dict
                   Parameters to pass into decorator
    """
    if not callable(decorator):
        raise Exception('Decorator must be callable.')

    def decorate(cls):
        attributes = cls.__dict__.keys()
        for attr in attributes: # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                name = getattr(cls, attr).__name__
                if name[0] == '_' or name in exclude:
                    continue
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    # return decorated copy (i.e. a subclass with decorations)
    return decorate(type('cls_copy', cls.__bases__, dict(cls.__dict__)))

def create_decorator(dec, **namespace):
    """
    Create a decorator function using arbirary params. The objects passed in
    can be used in the body. Originally designed with the idea of automatically
    updating one object after the decorated object was modified.
    """

    def decorator(func, *args, **kwargs):
        def wrapper(*args, **kwarg):
            for key in namespace.keys():
                locals()[key] = namespace[key]
            ret = func(*args, **kwargs)
            exec(dec.__code__, locals(), globals())
            if ret:
                return ret
        return wrapper
    return decorator

def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    This is pulled directly from scipy.misc as they are deprecating bytescale.

    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)