
import numpy as np
from matplotlib.path import Path
from shapely.geometry import Point, MultiPoint
import geopandas as gpd

import cv2
from sklearn.cluster import  OPTICS

from autocnet.utils.utils import bytescale
from autocnet.matcher.cpu_extractor import extract_features

def image_diff(arr1, arr2):
    diff = arr1-arr2
    diff[np.isnan(diff)] = 0

    bdiff = bytescale(diff)
    return bdiff


def okubogar_detector(image1, image2, nbins=50, extractor_method="orb", extractor_kwargs={"nfeatures": 2000, "scaleFactor": 1.1, "nlevels": 1}, image_func=image_diff):
    arr1 = image1.read_array()
    arr2 = image2.read_array()
    arr1[arr1 == arr1.min()] = np.nan
    arr2[arr2 == arr2.min()] = np.nan

    bdiff = image_func(arr1, arr2)

    keys, descriptors = extract_features(bdiff, extractor_method, extractor_parameters=extractor_kwargs)
    x,y = keys["x"], keys["y"]

    points = [Point(xval, yval) for xval,yval in zip(x,y)]

    optics = OPTICS(min_samples=10, max_eps=20,  eps=.3, p=2, xi=.5).fit(list(zip(x,y)))

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

    return polys, weights
