import numpy as np
from matplotlib.path import Path
from shapely.geometry import Point, MultiPoint
import geopandas as gpd
import cv2
from sklearn.cluster import  OPTICS
from plio.io.io_gdal import GeoDataset
from scipy.spatial import cKDTree
from skimage.feature import blob_log, blob_doh
from math import sqrt, atan2, pi

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


def image_diff_sq(arr1, arr2):
     return image_diff(arr1, arr2)**2


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


def blob_detector(image1, image2, sub_solar_azimuth, image_func=image_diff_sq,
                  subtractive=False,  min_sigma=.45, max_sigma=30, num_sigma=10,
                  threshold=.25, overlap=.5, log_scale=False, exclude_border=False,
                  n_neighbors=3, dist_upper_bound=5, angle_tolerance=3):
     """
     Blob based change detection.

     Creates a difference image and uses Laplacian of Gaussian (LoG) blob
     detection to find light / dark areas.  Creates a KDTree to find neighboring
     light / dark blobs, then filters based on colinearity of the light/dark pair
     with subsolar azimuth.

     Based on the method described in https://doi.org/10.1016/j.pss.2019.104733

     Parameters
     ----------

     image1 : GeoDataset
             Image representing the "before" state of the ROI, can be a 2D numpy array or plio GeoDataset

     image2 : GeoDataset
             Image representing the "after" state of the ROI, can be a 2D numpy array or plio GeoDataset


     sub_solar_azimuth : scalar or 2d np.array
                         Per-pixel subsolar azimuth or a single subsolar azimuth
                         value to be used for the entire image.

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

     subtractive : Boolean
                   Find subtractive features instead of additive features.  In other
                   words, find locations in which a feature "used to be present"
                   but has since moved.

     min_sigma : scalar or sequence of scalars
                 The minimum standard deviation for Gaussian kernel. Keep this
                 low to detect smaller blobs. The standard deviations of the
                 Gaussian filter are given for each axis as a sequence, or as a
                 single number, in which case it is equal for all axes.


     max_sigma : scalar or sequence of scalars
                 The maximum standard deviation for Gaussian kernel. Keep this
                 high to detect larger blobs. The standard deviations of the
                 Gaussian filter are given for each axis as a sequence, or as a
                 single number, in which case it is equal for all axes.

     num_sigma : int
                 The number of intermediate values of standard deviations to
                 consider between min_sigma and max_sigma.

     threshold : float
                 The absolute lower bound for scale space maxima.
                 Local maxima smaller than thresh are ignored.
                 Reduce this to detect blobs with less intensities.

     overlap : float
               A value between 0 and 1. If the area of two blobs overlaps by a
               fraction greater than threshold, the smaller blob is eliminated.

     log_scale : bool
                 If set intermediate values of standard deviations are
                 interpolated using a logarithmic scale to the base 10. If not,
                 linear interpolation is used.

     exclude_border: tuple of ints, int, or False
                 If tuple of ints, the length of the tuple must match the input
                 array’s dimensionality. Each element of the tuple will exclude
                 peaks from within exclude_border-pixels of the border of the
                 image along that dimension. If nonzero int, exclude_border
                 excludes peaks from within exclude_border-pixels of the border
                 of the image. If zero or False, peaks are identified regardless
                 of their distance from the border.

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

     Returns
     -------

     changes : np.ndarray
               A numpy array containing candidate change points in the form (y,x,radius)

     bdiff : np.ndarray
             A numpy array containing the image upon which the change detection
             algorithm operates, i.e. the image resulting from image_func.

     """

     def is_azimuth_colinear(pt1, pt2, subsolar_azimuth, tolerance, subtractive=False):
         """ Returns true if pt1, pt2, and subsolar azimuth are colinear within
             some tolerance.
         """
         x, y = (pt2[1]-pt1[1], pt2[0]-pt1[0])
         # Find angle of vector w.r.t. x axis
         angle = (atan2(y, x) * 180 / pi)%360
         # If finding subtractive changes, invert the angle.
         if subtractive:
             angle = (angle+180)%360
         return -tolerance <= subsolar_azimuth - angle <= tolerance

     if isinstance(image1, GeoDataset):
         image1 = image1.read_array()

     if isinstance(image2, GeoDataset):
         image2 = image2.read_array()

     bdiff = image_func(image1,image2)
     bdiff = bytescale(bdiff)

     # Laplacian of Gaussian only finds light blobs on a dark image.  In order to
     #  find dark blobs on a light image, we invert.
     inv = bdiff.max()-bdiff

     # Laplacian of Gaussian of diff image (light on dark)
     blobs_log = blob_log(bdiff, min_sigma=min_sigma, max_sigma=max_sigma,
                          num_sigma=num_sigma, threshold=threshold, overlap=overlap,
                          log_scale=log_scale, exclude_border=exclude_border)
     # Laplacian of Gaussian on diff image (inverse -- dark on light)
     blobs_log_inv = blob_log(inv, min_sigma=min_sigma, max_sigma=max_sigma,
                              num_sigma=num_sigma, threshold=threshold, overlap=overlap,
                              log_scale=log_scale, exclude_border=exclude_border)
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

     # Nearest neighbors
     neighbors = [blobs_log[j] for j in [i[i!=len(blobs_log)]for i in idx_log] if j.size > 0]

     changes = []
     for idx, pt1 in enumerate(close_points):
         for pt2 in neighbors[idx]:
             try:
                 azimuth = sub_solar_azimuth[int(pt1[0]), int(pt1[1])]
             except IndexError as e:
                 azimuth = sub_solar_azimuth
             if is_azimuth_colinear(pt1, pt2, azimuth, angle_tolerance, subtractive):
                 changes.append([pt1,pt2])
     changes = np.array(changes)

     return changes, bdiff
