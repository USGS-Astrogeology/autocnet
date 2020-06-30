import numpy as np
from matplotlib.path import Path

from shapely.geometry import Point, MultiPoint
import geopandas as gpd

import cv2
from sklearn.cluster import  OPTICS

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

     image1 : GeoDataset
             Image representing the "before" state of the ROI

     image2 : GeoDataset
             Image representing the "after" state of the ROI

     image_func : callable
              Function use to create a derived image from image1 and image2, which in turn is
              the input for the feature extractor. Default function create a difference image.

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

     arr1 = image1.read_array()
     arr2 = image2.read_array()
     arr1[arr1 == arr1.min()] = 0
     arr2[arr2 == arr2.min()] = 0
     arr1 = bytescale(arr1)
     arr2 = bytescale(arr2)

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
             Image representing the "before" state of the ROI

     image2 : GeoDataset
             Image representing the "after" state of the ROI

     image_func : callable
              Function use to create a derived image from image1 and image2, which in turn is
              the input for the feature extractor. Default function create a difference image.

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

     arr1 = image1.read_array()
     arr2 = image2.read_array()
     arr1[arr1 == arr1.min()] = 0
     arr2[arr2 == arr2.min()] = 0
     arr1 = bytescale(arr1)
     arr2 = bytescale(arr2)

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


