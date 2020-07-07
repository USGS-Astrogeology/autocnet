import numpy as np
from matplotlib.path import Path
from shapely.geometry import Point, MultiPoint
import geopandas as gpd
import cv2
from sklearn.cluster import  OPTICS
from plio.io.io_gdal import GeoDataset
from scipy.spatial import cKDTree
from skimage.feature import blob_log, blob_doh
from math import sqrt

from autocnet.utils.utils import bytescale
from autocnet.matcher.cpu_extractor import extract_features

def image_diff(arr1, arr2):
     arr1 = arr1.astype("float32")
     arr2 = arr2.astype("float32")
     arr1[arr1 == 0] = np.nan
     arr2[arr2 == 0] = np.nan

     diff = arr1-arr2
     diff[np.isnan(diff)] = 0

     return diff


def okubogar_detector(image1, image2, nbins=50, extractor_method="orb", image_func=image_diff,
                      extractor_kwargs={"nfeatures": 2000, "scaleFactor": 1.1, "nlevels": 1}):
     """
     Simple change detection algorithm which produces an overlay image of change hotspots
     (i.e. a 2d histogram image of detected change density).

     Largely based on a method created by Chris Okubo and Brendon Bogar. Histogram step
     was added for readability.



     image1
           \
             image subtraction/ratio -> feature_extraction -> feature_histogram
           /
     image2

     TODO: Paper/abstract might exist, cite

     Parameters
     ----------

     image1 : np.array, plio.GeoDataset
             Image representing the "before" state of the ROI, can be a 2D numpy array or plio GeoDataset

     image2 : np.array, plio.GeoDataset
             Image representing the "after" state of the ROI, can be a 2D numpy array or plio GeoDataset

     image_func : callable
                  Function used to create a derived image from image1 and image2, which in turn is
                  the input for the feature extractor. The first two arguments are 2d numpy arrays, image1 and image2,
                  and must return a 2d numpy array. Default function returns a difference image.

                  Example func:

                  >>> from numpy import random
                  >>> image1, image2 = random.random((50,50)), random.random((50,50))
                  >>> def ratio(image1, image2):
                  >>>   # do stuff with image1 and image2
                  >>>   new_image = image1/image2
                  >>>   return new_image # must return a single variable, a 2D numpy array
                  >>> results = okubogar_detector(image1, image2, image_func=ratio)

                  Or, alternatively:

                  >>> from numpy import random
                  >>> image1, image2 = random.random((50,50)), random.random((50,50))
                  >>> results = okubogar_detector(image1, image2, image_func=lambda im1, im2: im1/im2)

     nbins : int
            number of bins to use in the 2d histogram

     extractor_method : {'orb', 'sift', 'fast', 'surf', 'vl_sift'}
               The detector method to be used.  Note that vl_sift requires that
               vlfeat and cyvlfeat dependencies be installed.

     extractor_kwargs : dict
                        A dictionary containing OpenCV SIFT parameters names and values.

     See Also
     --------

     feature extractor: autocnet.matcher.cpu_extractor.extract_features

     """
     if isinstance(image1, GeoDataset):
         image1 = image1.read_array()

     if isinstance(image2, GeoDataset):
         image2 = image2.read_array()

     image1[image1 == image1.min()] = 0
     image2[image2 == image2.min()] = 0
     arr1 = bytescale(image1)
     arr2 = bytescale(image2)

     bdiff = image_func(arr1, arr2)

     keys, descriptors = extract_features(bdiff, extractor_method, extractor_parameters=extractor_kwargs)
     x,y = keys["x"], keys["y"]

     points = [Point(xval, yval) for xval,yval in zip(x,y)]

     heatmap, xedges, yedges = np.histogram2d(y, x, bins=nbins, range=[[0, bdiff.shape[0]], [0, bdiff.shape[1]]])
     heatmap = cv2.resize(heatmap, dsize=(bdiff.shape[1], bdiff.shape[0]), interpolation=cv2.INTER_NEAREST)

     return points, heatmap, bdiff


def okbm_detector(image1, image2, nbins=50, extractor_method="orb",  image_func=image_diff,
                 extractor_kwargs={"nfeatures": 2000, "scaleFactor": 1.1, "nlevels": 1},
                 cluster_params={"min_samples": 10, "max_eps": 10, "eps": .5, "xi":.5}):
     """
     okobubogar modified detector, experimental feature based change detection algorithmthat expands on okobubogar to allow for
     programmatic change detection. Returns detected feature changes as weighted polygons.


     Parameters
     ----------

     image1 : GeoDataset
             Image representing the "before" state of the ROI, can be a 2D numpy array or plio GeoDataset

     image2 : GeoDataset
             Image representing the "after" state of the ROI, can be a 2D numpy array or plio GeoDataset

     image_func : callable
                  Function used to create a derived image from image1 and image2, which in turn is
                  the input for the feature extractor. The first two arguments are 2d numpy arrays, image1 and image2,
                  and must return a 2d numpy array. Default function returns a difference image.

                  Example func:

                  >>> from numpy import random
                  >>> image1, image2 = random.random((50,50)), random.random((50,50))
                  >>> def ratio(image1, image2):
                  >>>   # do stuff with image1 and image2
                  >>>   new_image = image1/image2
                  >>>   return new_image # must return a single variable, a 2D numpy array
                  >>> results = okbm_detector(image1, image2, image_func=ratio)

                  Or, alternatively:

                  >>> from numpy import random
                  >>> image1, image2 = random.random((50,50)), random.random((50,50))
                  >>> results = okbm_detector(image1, image2, image_func=lambda im1, im2: im1/im2)

     nbins : int
            number of bins to use in the 2d histogram

     extractor_method : {'orb', 'sift', 'fast', 'surf', 'vl_sift'}
               The detector method to be used.  Note that vl_sift requires that
               vlfeat and cyvlfeat dependencies be installed.

     extractor_kwargs : dict
                        A dictionary containing OpenCV SIFT parameters names and values.

     cluster_params : dict
                      A dictionary containing sklearn.cluster.OPTICS parameters

     """

     if isinstance(image1, GeoDataset):
         image1 = image1.read_array()

     if isinstance(image2, GeoDataset):
         image2 = image2.read_array()

     image1[image1 == image1.min()] = 0
     image2[image2 == image2.min()] = 0
     arr1 = bytescale(image1)
     arr2 = bytescale(image2)

     bdiff = image_func(arr1, arr2)

     keys, descriptors = extract_features(bdiff, extractor_method, extractor_parameters=extractor_kwargs)
     x,y = keys["x"], keys["y"]

     points = [Point(xval, yval) for xval,yval in zip(x,y)]

     optics = OPTICS(**cluster_params).fit(list(zip(x,y)))

     classes = gpd.GeoDataFrame(columns=["label", "point"], geometry="point")
     classes["label"] = optics.labels_
     classes["point"] = points
     class_groups = classes.groupby("label").groups

     polys = []
     weights = []

     # array of x,y pairs
     xv, yv = np.mgrid[0:bdiff.shape[1], 0:bdiff.shape[0]]

     for label, indices in class_groups.items():
         if label == -1:
             continue

         points = classes.loc[indices]["point"]
         poly = MultiPoint(points.__array__()).convex_hull
         xmin, ymin, xmax, ymax = np.asarray(poly.bounds).astype("uint64")
         xv, yv = np.mgrid[xmin:xmax, ymin:ymax]
         xv = xv.flatten()
         yv = yv.flatten()

         points = np.vstack((xv,yv)).T.astype("uint64")

         mask = Path(np.asarray(poly.exterior.xy).T.astype("uint64")).contains_points(points).reshape(int(ymax-ymin), int(xmax-xmin))
         weight = bdiff[ymin:ymax,xmin:xmax].mean()

         polys.append(poly)
         weights.append(weight)

     return polys, weights, bdiff


def blob_detector(image1, image2, image_func=image_diff, max_sigma=30, num_sigma=10,
                  threshold=.075, n_neighbors=3, dist_upper_bound=5, angle_tolerance=10,
                  sub_solar_azimuth=None):
     """

     Parameters
     ----------

     image1 : GeoDataset
             Image representing the "before" state of the ROI, can be a 2D numpy array or plio GeoDataset

     image2 : GeoDataset
             Image representing the "after" state of the ROI, can be a 2D numpy array or plio GeoDataset

     image_func : callable
                  Function used to create a derived image from image1 and image2, which in turn is
                  the input for the feature extractor. The first two arguments are 2d numpy arrays, image1 and image2,
                  and must return a 2d numpy array. Default function returns a difference image.

                  Example func:

                  >>> from numpy import random
                  >>> image1, image2 = random.random((50,50)), random.random((50,50))
                  >>> def ratio(image1, image2):
                  >>>   # do stuff with image1 and image2
                  >>>   new_image = image1/image2
                  >>>   return new_image # must return a single variable, a 2D numpy array
                  >>> results = okbm_detector(image1, image2, image_func=ratio)

                  Or, alternatively:

                  >>> from numpy import random
                  >>> image1, image2 = random.random((50,50)), random.random((50,50))
                  >>> results = okbm_detector(image1, image2, image_func=lambda im1, im2: im1/im2)

     max_sigma : scalar or sequence of scalars
                 The maximum standard deviation for Gaussian kernel. Keep this
                 high to detect larger blobs. The standard deviations of the
                 Gaussian filter are given for each axis as a sequence, or as a
                 single number, in which case it is equal for all axes.

     n_sigma : int
               The number of intermediate values of standard deviations to
               consider.

     threshold : float
                 The absolute lower bound for scale space maxima.
                 Local maxima smaller than thresh are ignored.
                 Reduce this to detect blobs with less intensities.

     n_neighbors : int
                   Number of closest neighbors (blobs) to search.

     dist_upper_bound : int
                        The maximum distance between blobs to be considered
                        neighbors.

     angle_tolerance : int
                       The mismatch tolerance between the subsolar azimuth and
                       the angle between the direction vector w.r.t. the x axis.
                       For example, a subsolar azimuth of 85 degrees would
                       require an angle tolerance of 5 in order to consider
                       blobs with a 90 degree angle as candidates.

     sub_solar_azimuth : scalar or 2d np.array
                         Per-pixel subsolar azimuth or a single subsolar azimuth
                         value to be used for the entire image.
     """

     if isinstance(image1, GeoDataset):
         image1 = image1.read_array()

     if isinstance(image2, GeoDataset):
         image2 = image2.read_array()

     image1[image1 == image1.min()] = 0
     image2[image2 == image2.min()] = 0
     arr1 = bytescale(image1)
     arr2 = bytescale(image2)

     bdiff = image_func(arr1, arr2)

     # Laplacian of Gaussian only finds light blobs on a dark image.  In order to
     #  find dark blobs on a light image, we invert.
     inv = 255-bdiff

     # Laplacian of Gaussian of diff image (light on dark)
     blobs_log = blob_log(bdiff, max_sigma=30, num_sigma=10, threshold=.075)
     # Laplacian of Gaussian on diff image (inverse -- dark on light)
     blobs_log_inv = blob_log(inv, max_sigma=30, num_sigma=10, threshold=.075)
     # Compute radii in the 3rd column.  Radii are appx equal to sqrt2 * sigma
     blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
     blobs_log_inv[:, 2] = blobs_log_inv[:, 2] * sqrt(2)

     # Create a KDTree to facilitate nearest neighbor search
     tree = cKDTree(blobs_log)
     # Query the kdtree to find neighboring points
     _, idx_log = tree.query(blobs_log_inv, k=n_neighbors,
                                    distance_upper_bound=dist_upper_bound)

     # Points that have at least one neighbor within threshold distance.
     close_points = blobs_log_inv[[x[0] < len(blobs_log) for x in idx_log]]

     # Indices of nearest neighbors
     neighbors_idx = [j for j in [i[i!=len(blobs_log)]for i in idx_log] if j.size > 0]

     def is_azimuth_colinear(pt1, pt2, subsolar_azimuth, tolerance):
         """ Returns true if pt1, pt2, and subsolar azimuth are colinear within
             some tolerance.
         """
         x, y = (pt2[1]-pt1[1], pt2[0]-pt1[0])
         # Find angle of vector w.r.t. x axis
         angle = (math.atan2(y, x) * 180 / math.pi)%360
         return -tolerance <= subsolar_azimuth - angle <= tolerance

     type1_change = []
     type2_change = []
     for idx, pt1 in enumerate(close_points):
         for pt2 in neighbors[idx]:
             try:
                 azimuth = sub_solar_azimuth[pt1]
             except IndexError:
                 azimuth = sub_solar_azimuth
             if is_azimuth_colinear(pt1, pt2, azimuth, angle_tolerance):
                 type1_change.append([pt1,pt2])
             elif is_azimuth_colinear(pt2, pt1, azimuth, angle_tolerance):
                 type2_change.append([pt1,pt2])
     type1_change = np.array(type1_change)
     type2_change = np.array(type2_change)

     return (type1_change, type2_change)
