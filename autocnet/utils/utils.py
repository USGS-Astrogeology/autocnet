import json

from functools import reduce

import numpy as np
import pandas as pd

from osgeo import ogr

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
    >>> x = np.random.random((3,3))
    >>> normalize_vector(x)
    array([[ 0.88280225,  0.4697448 ,  0.11460811],
       [ 0.26090555,  0.96536433,  0.91648305],
       [ 0.58271501,  0.81267657,  0.30796395]])
    """
    if isinstance(line, pd.DataFrame):
        line = line.values
    try:
        n = np.sqrt(line[:, 0]**2 + line[:, 1]**2).reshape(-1, 1)
    except:
        n = np.sqrt(line[0]**2 + line[1]**2)
    line = line / n
    return line


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
    points : ndarray
             n x m array of points, where n is the number
             of points.

    Returns
    -------
     : ndarray
       n x m + 1 array of homogeneous points
    """
    return np.hstack((points, np.ones((points.shape[0], 1))))


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

