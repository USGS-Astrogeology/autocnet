import pandas as pd

from scipy.spatial import ConvexHull


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
    a_o = poly2.Intersection(poly1).GetArea()
    area1 = poly1.GetArea()
    area2 = poly2.GetArea()

    overlap_area = a_o
    overlap_percn = (a_o / (area1 + area2 - a_o)) * 100
    return overlap_percn, overlap_area


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
