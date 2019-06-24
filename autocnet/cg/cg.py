from math import isclose
import warnings

import pandas as pd
import numpy as np
import networkx as nx
import geopandas as gpd
import ogr

from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi
import shapely.geometry
from shapely.geometry import Polygon, Point
from shapely.affinity import scale

from autocnet.utils import utils


def convex_hull_ratio(points, ideal_area):
    """

    Parameters
    ----------
    points : ndarray
             (n, 2) array of point coordinates

    ideal_area : float
                 The total area that could be covered

    Returns
    -------
    ratio : float
            The ratio convex hull volume / ideal_area

    """
    hull = ConvexHull(points)
    return hull.volume / ideal_area


def convex_hull(points):

    """

    Parameters
    ----------
    points : ndarray
             (n, 2) array of point coordinates

    Returns
    -------
    hull : 2-D convex hull
            Provides a convex hull that is used
            to determine coverage

    """

    if isinstance(points, pd.DataFrame) :
        points = pd.DataFrame.as_matrix(points)

    hull = ConvexHull(points)
    return hull


def geom_mask(keypoints, geom): # ADDED
    """
    Masks any points that are outside of the bounds of the given
    geometry.

    Parameters
    ----------
    keypoints : dataframe
                      A pandas dataframe of points to mask

    geom : object
                Shapely geometry object to use as a mask
    """

    def _in_mbr(r, mbr):
        if (mbr[0] <= r.x <= mbr[2]) and (mbr[1] <= r.y <= mbr[3]):
            return True
        else:
            return False

    mbr = geom.bounds
    initial_mask = keypoints.apply(_in_mbr, axis=1, args=(mbr,))

    return initial_mask


def two_poly_overlap(poly1, poly2):
    """

    Parameters
    ----------
    poly1 : ogr polygon
            Any polygon that shares some kind of overlap
            with poly2

    poly2 : ogr polygon
            Any polygon that shares some kind of overlap
            with poly1

    Returns
    -------
    overlap_percn : float
                    The percentage of image overlap

    overlap_area : float
                   The total area of overalap

    """
    overlap_area_polygon = poly2.Intersection(poly1)
    overlap_area = overlap_area_polygon.GetArea()
    area1 = poly1.GetArea()
    area2 = poly2.GetArea()

    overlap_percn = (overlap_area / (area1 + area2 - overlap_area)) * 100
    return overlap_percn, overlap_area, overlap_area_polygon


def get_area(poly1, poly2):
    """

    Parameters
    ----------
    poly1 : ogr polygon
            General ogr polygon

    poly2 : ogr polygon
            General ogr polygon

    Returns
    -------
    intersection_area : float
                        returns the intersection area
                        of two polygons

    """
    intersection_area = poly1.Intersection(poly2).GetArea()
    return intersection_area


def compute_voronoi(keypoints, intersection=None, geometry=False, s=30): # ADDED
        """
        Creates a voronoi diagram for all edges in a graph, and assigns a given
        weight to each edge. This is based around voronoi polygons generated
        by scipy's voronoi method, to determine if an image has significant coverage.

        Parameters
        ----------
        graph : object
               A networkx graph object

        clean_keys : list
                     Of strings used to apply masks to omit correspondences

        s : int
            Offset for the corners of the image
        """
        vor_keypoints = []

        keypoints.apply(lambda x: vor_keypoints.append((x['x'], x['y'])), axis = 1)

        if intersection is None:
            keypoint_bounds = Polygon(vor_keypoints).bounds
            intersection = shapely.geometry.box(keypoint_bounds[0], keypoint_bounds[1],
                                                    keypoint_bounds[2], keypoint_bounds[3])

        scaled_coords = np.array(scale(intersection, s, s).exterior.coords)

        vor_keypoints = np.vstack((vor_keypoints, scaled_coords))
        vor = Voronoi(vor_keypoints)
        # Might move the code below to its own method depending on feedback
        if geometry:
            voronoi_df = gpd.GeoDataFrame(data = keypoints, columns=['x', 'y', 'weight', 'geometry'])
        else:
            voronoi_df = gpd.GeoDataFrame(data = keypoints, columns=['x', 'y', 'weight'])

        i = 0
        vor_points = np.asarray(vor.points)
        for region in vor.regions:
            region_point = vor_points[np.argwhere(vor.point_region==i)]

            if not -1 in region:
                polygon_points = [vor.vertices[i] for i in region]

                if len(polygon_points) != 0:
                    polygon = Polygon(polygon_points)

                    intersection_poly = polygon.intersection(intersection)

                    voronoi_df.loc[(voronoi_df["x"] == region_point[0][0][0]) &
                                   (voronoi_df["y"] == region_point[0][0][1]),
                                   'weight'] = intersection_poly.area
                    if geometry:
                        voronoi_df.loc[(voronoi_df["x"] == region_point[0][0][0]) &
                                       (voronoi_df["y"] == region_point[0][0][1]),
                                       'geometry'] = intersection_poly
            i += 1

        return voronoi_df

def single_centroid(geom):
    """
    For a geom, return the centroid

    Parameters
    ----------
    geom : shapely.geom object

    Returns
    -------

     : list
            in the form [(x,y)]
    """
    x, y = geom.centroid.xy
    return [(x[0],y[0])]

def nearest(pt, search):
    """
    Fine the index of nearest (Euclidean) point in a list
    of points.

    Parameters
    ----------
    pt : ndarray
         (2,1) array
    search : ndarray
             (n,2) array of points to search within. The
             returned index is the closet point in this set
             to the search

    Returns
    -------
     : int
       The index to the nearest point.
    """
    return np.argmin(np.sum((search - pt)**2, axis=1))

def create_points_along_line(p1, p2, npts):
    """
    Compute a set of nodes equally spaced between
    two points, not including the end points.

    Parameters
    ----------
    p1 : iterable
         in the form (x,y)
    
    p2 : iterable
         in the form(x,y)

    npts : int
           The number of nodes to be returned

    Returns
    -------
     : ndarray
       (n,2) array of nodes
    """
    # npts +2 since the endpoints are included in linspace
    # but this func clips them
    return np.linspace(p1, p2, npts+2)[1:-1]

def xy_in_polygon(x,y, geom):
    """
    Returns true is an x,y pair is contained within
    the geom. 

    Parameters
    ----------
    x : Number
        The x coordinate
    
    y : Number
        The y coordinate

    Returns
    -------
     : bool
       True if the point is contained within the geom.
    """
    return geom.contains(Point(x, y))

def distribute_points(geom, nspts, ewpts):
    """
    This is a decision tree that attempts to perform a
    very simplistic approximation of the shape
    of the geometry and then place some number of
    north/south and east/west points into the geometry.

    Parameters
    ----------
    geom : shapely.geom
           A shapely geometry object

    nspts : int
            The number of points to attempt to place
            in the N/S (up/down) direction

    ewpts : int
            The number of points to attempt to place
            in the E/W (right/left) direction

    Returns
    -------
    valid : list
            of point coordinates in the form [(x1,y1), (x2,y2), ..., (xn, yn)]
    """
    geom_coords = np.column_stack(geom.exterior.xy)

    coords = np.array(list(zip(*geom.envelope.exterior.xy))[:-1])

    ll = coords[0]
    lr = coords[1]
    ur = coords[2]
    ul = coords[3]

    # Find the points nearest the ul and ur
    ul_actual = geom_coords[nearest(ul, geom_coords)]
    ur_actual = geom_coords[nearest(ur, geom_coords)]
    newtop = create_points_along_line(ul_actual, ur_actual, ewpts)

    # Find the points nearest the ll and lr
    ll_actual = geom_coords[nearest(ll, geom_coords)]
    lr_actual = geom_coords[nearest(lr, geom_coords)]
    newbot = create_points_along_line(ll_actual, lr_actual, ewpts)

    points = []
    for i in range(len(newtop)):
        top = newtop[i]
        bot = newbot[i]
        
        line_of_points = create_points_along_line(top, bot, nspts)
        points.append(line_of_points)

    points = np.vstack(points)
    # Perform a spatial intersection check to eject points that are not valid
    valid = [p for p in points if xy_in_polygon(p[0], p[1], geom)]
    return valid

def distribute_points_in_geom(geom, 
                              nspts_func=lambda x: int(round(x,1)*10), 
                              ewpts_func=lambda x: int(round(x,1)*5)):
    """
    Given a geometry, attempt a basic classification of the shape.
    RIght now, this simply attempts to determine if the bounding box
    is generally N/S or generally E/W trending. Once the determination
    is made, the algorithm places points in the geometry and returns
    a list of valid (intersecting) points.

    The kwargs for this algorithm take a function that expects a number
    as an input and returns an integer number of points to place. The 
    input number is the distance between the top/bottom or left/right
    sides of the geometry. 

    This algorithm does not know anything about the units being used
    so the caller is responsible for acocunting for units (if appropriate)
    in the passed funcs.

    Parameters
    ----------
    geom : shapely.geom object
           The geometry object

    nspts_func : obj
                 Function taking a Number and returning an int
    
    ewpts_func : obj
                 Function taking a Number and returning an int

    Returns
    -------
    valid : list
            of valid points in the form (x,y) or (lon,lat)

    """
    coords = list(zip(*geom.envelope.exterior.xy))
    short = np.inf
    lng = -np.inf
    shortid = 0
    longid = 0
    for i, p in enumerate(coords[:-1]):
        d = np.sqrt((coords[i+1][0] - p[0])**2+(coords[i+1][1]-p[1])**2)
        if d < short:
            short = d
            shortid = i
        if d > lng:
            lng = d
            longid = i
    ratio = short/lng
    ns = False
    ew = False
    valid = []
    # The polygons should be encoded with a lower left origin in counter-clockwise direction.
    # Therefore, if the 'bottom' is the short edge it should be id 0 and modulo 2 == 0.
    if shortid % 2 == 0:
        # Also if the geom is a perfect square
        ns = True
    elif longid % 2 == 0:
        ew = True
    # Decision Tree
    if ratio < 0.16 and geom.area < 0.01:
        # Class: Slivers - ignore.
        return
    elif geom.area <= 0.004 and ratio >= 0.25:
        # Single point at the centroid
        valid = single_centroid(geom)
    elif ns==True:
        # Class, north/south poly, multi-point
        nspts = nspts_func(lng)
        ewpts = ewpts_func(short)
        if nspts == 1 and ewpts == 1:
            valid = single_centroid(geom)
        else:
            valid = distribute_points(geom, nspts, ewpts)
    elif ew == True:
        # Since this is an LS, we should place these diagonally from the 'lower left' to the 'upper right'
        nspts = ewpts_func(short)
        ewpts = nspts_func(lng)
        if nspts == 1 and ewpts == 1:
            valid = single_centroid(geom)
        else:
            valid = distribute_points(geom, nspts, ewpts)

    return valid
