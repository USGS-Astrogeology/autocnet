import importlib
import itertools
import json

from functools import reduce, singledispatch, update_wrapper

import numpy as np
import pandas as pd
import networkx as nx

from osgeo import ogr

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from scipy.spatial import Delaunay

from shapely import geometry
from shapely.geometry import MultiPoint
from shapely.ops import cascaded_union, polygonize

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

def import_func(func):
    """
    Imports a function from the autocnet package.

    Parameters
    ----------
    func : str
           import path. For example, to import the place_points_in_overlap function,
           this func can be called with: 'spatial.overlap.place_points_in_overlap'

    Returns
    -------
    func : obj
           The function object for use.
    """
    if not func[0] == '.':
        # Since this intentionally forces the package to be autocnet
        # need the import path relative to the package name. Convenience
        # for the caller to add the '.' so they don't get a cryptic
        # ModuleImportError.
        func = f'.{func}'

    module, func = func.rsplit('.', 1)
    module = importlib.import_module(module, package='autocnet')
    func = getattr(module, func)
    return func


def compute_depression(input_dem, scale_factor=1, curvature_percentile=75):
    """
    Compute depressions and return a new image with larges depressions filled in. 
    
    Parameters
    ----------
    
    input_dem : np.array, rd.rdarray
                2d array of elevation DNs, a DEM
    
    scale_factor : float
                   Value to scale the erotion of planform curvatures by
                   
    curvature_percentile : float 
                           what percentile of the curvature to keep, lower values
                           results in bigger blobs 
                   
    
    Returns
    -------
    dem : rd.rdarray
          Dem with filled depressions
    
    mask : np.array
           Change mask, true on pixels that have been changed 
    
    
    """
    if isinstance(input_dem, np.ndarray):
        dem = rd.rdarray(input_dem.copy(), no_data=0)
    elif isinstance(input_dem, rd.rdarray):
        # take ownership of the reference
        dem = input_dem.copy()

    # create filled DEM
    demfilled = rd.FillDepressions(dem, epsilon=True, in_place=False, topology="D8")
    
    # Mask out filled areas
    mask = np.abs(dem-demfilled)
    thresh = np.percentile(mask, 95)
    mask[mask <= thresh] = False
    mask[mask > thresh] = True
    
    curvatures = rd.TerrainAttribute(dem, attrib='planform_curvature')
    curvatures = (curvatures - np.min(curvatures))/np.ptp(curvatures) 
    curvatures[curvatures < np.percentile(curvatures, curvature_percentile)] = 0
    curvatures[mask.astype(bool)] = 0
    
    demfilled -= curvatures * scale_factor
    
    mask = (curvatures+mask).astype(bool)
    
    # Get 3rd nn distance 
    coords = np.argwhere(mask)
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='kd_tree').fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    eps = np.percentile(dists, 95)
    
    # Cluster
    db = DBSCAN(eps=eps, min_samples=3).fit(coords)
    labels = db.labels_
    unique, counts = np.unique(labels, return_counts=True)
    
    # First count are outliers, ignore
    counts = counts[1:]
    unique = unique[1:]
    
    index = np.argwhere(counts == counts.max())
    group = unique[index][0][0]
    cluster = coords[labels == group]
    
    # mask out depression
    dmask = np.full(dem.shape, False)
    dmask[[*cluster.T]] = True
    
    dem[dmask] = 0
    demfilled[~dmask] = 0
    dem = dem+demfilled

    return dem, dmask


def rasterize_polygon(shape, vertices, dtype=bool):
    """
    Simple tool to convert poly into a boolean numpy array.
    
    source: https://stackoverflow.com/questions/37117878/generating-a-filled-polygon-inside-a-numpy-array
    
    Parameters
    ----------
    
    shape : tuple 
            size of the array in (y,x) format
    
    vertices : np.array, list
               array of vertices in [[x0, y0], [x1, y1]...] format
    
    dtype : type
            datatype of output mask
    
    Returns
    -------
    
    mask : np.array
           mask with filled polygon set to true
    
    """
    def check(p1, p2, base_array):
        idxs = np.indices(base_array.shape) # Create 3D array of indices

        p1 = p1.astype(float)
        p2 = p2.astype(float)

        # Calculate max column idx for each row idx based on interpolated line between two points
        if p1[0] == p2[0]:
            max_col_idx = (idxs[0] - p1[0]) * idxs.shape[1]
            sign = np.sign(p2[1] - p1[1])
        else:
            max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
            sign = np.sign(p2[0] - p1[0])
            
        return idxs[1] * sign <= max_col_idx * sign

    base_array = np.zeros(shape, dtype=dtype)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k-1], vertices[k], base_array)], axis=0)
    
    print(fill.any())
    # Set all values inside polygon to one
    base_array[fill] = 1
    return base_array


def generate_dem(alpha=1.0, size=800, scales=[160,80,32,16,8,4,2,1], scale_factor=5):
    """
    Produces a random DEM
    
    Parameters
    ----------
    
    alpha : float 
            Controls height variation. Lower number makes a shallower and noisier DEM, 
            higher values create smoother DEM with large peaks and valleys. 
            Reccommended range = (0, 1.5]
    
    size : int
           size of DEM, output DEM is in the shape of (size, size)
    
    scale_factor : float 
                   Scalar to multiply the slope degradation by, higher values = more erosion.
                   Recommended to increase proportionately with alpha
                   (higher alphas mean you might want higher scale_factor)
    
    Returns 
    -------
    
    dem : np.array 
          DEM array in the shape (size, size)
    
    """
    
    topo=np.zeros((2,2))+random.rand(2,2)*(200/(2.**alpha))

    for k in range(len(scales)):
        nn = size/scales[k]
        topo = scipy.misc.imresize(topo, (int(nn), int(nn)), "cubic", mode="F")
        topo = topo + random.rand(int(nn), int(nn))*(200/(nn**alpha))
    
    topo = rd.rdarray(topo, no_data=0)
    
    curvatures = rd.TerrainAttribute(topo, attrib='slope_riserun')
    curvatures = (curvatures - np.min(curvatures))/np.ptp(curvatures) * scale_factor
    return topo - curvatures


def hillshade(img, azi=255, min_slope=20, max_slope=100, min_bright=0, grayscale=False):
    """
    hillshade a DEM, based on IDL code by Colin Dundas 
    
    Parameters
    ----------
    
    img : np.array
          DEM to hillshade
    
    azi : float 
          Sun azimuth 
    
    min_slope : float 
                minimum slope value 
    
    max_slope : float 
                maximum slope value 
    
    min_bright : float 
                 minimum brightness 
    
    grayscale : bool 
                whether or not to produce grayscale image 
    
    
    Returns
    -------
    
    dem : np.array 
          hillshaded DEM 
    
    """
    dem = np.array(np.flip(bytescale(img), axis = 0), dtype=int)
    emax = np.max(dem)
    emin = np.min(dem)

    indices = np.linspace(0, 255, 256) / 25.5

    red_array = [0,25,50,101,153,204,255,255,255,255,255,255]
    red_index = np.arange(len(red_array))
    red_vec = np.interp(indices, red_index, red_array)

    green_array = [42,101,153,204,237,255,255,238,204,153,102,42]
    green_index = np.arange(len(green_array))
    green_vec = np.interp(indices, green_index, green_array)

    blue_array = [255,255,255,255,255,255,204,153,101,50,25,0]
    blue_index = np.arange(len(blue_array))
    blue_vec = np.interp(indices, blue_index, blue_array)

    zz = (255.0/(emax-emin))*(dem-emin)
    zz = zz.astype(int)

    nx = (np.roll(dem, 1, axis = 1) - dem)
    ny = (np.roll(dem, 1, axis = 0) - dem)
    sz = np.shape(nx)
    nz = np.ones(sz)
    nl = np.sqrt(np.power(nx, 2.0) + np.power(ny, 2.0) + np.power(nz, 2.0))
    nx = nx/nl
    ny = ny/nl
    nz = nz/nl

    math.cos(math.radians(1))
    azi_rad = math.radians(azi)
    alt_rad = math.radians(alt)
    lx = math.sin(azi_rad)*math.cos(alt_rad)
    ly = math.cos(azi_rad)*math.cos(alt_rad)
    lz = math.sin(alt_rad)

    dprod = nx*lx + ny*ly + nz*lz

    if min_slope is not None:
        min_dprod = math.cos(math.radians(max_slope + 90.0 - alt))
    else:
        min_dprod = np.min(dprod)

    if max_slope is not None:
        max_dprod = math.cos(math.radians(90.0 - alt - max_slope))
    else:
        max_dprod = np.max(dprod)

    bright = ((dprod - min_dprod) + min_bright)/((max_dprod - min_dprod) + min_bright)

    if grayscale:
        qq=(255*bright)
    else:
        qq = red_vec[zz]*bright

    if grayscale:
        rr = (255*bright)
    else:
        rr = green_vec[zz]*bright

    if grayscale:
        ss=(255*bright)
    else:
        ss = blue_vec[zz]*bright

    arrforout = np.dstack((qq, rr ,ss))
    arrforout = np.flip(arrforout.astype(int), axis = 0)
    arrfotout = bytescale(arrforout)
    arrforout.shape
    return arrforout
