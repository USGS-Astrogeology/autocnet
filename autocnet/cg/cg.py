import json
import ogr
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


def overlapping_polygon_area(polys):
    """

    Parameters
    ----------
    polys : list
            of polygon object with a __geo_interface__

    Returns
    -------
    area : float
           The area of the intersecting polygons
    """
    geom = json.dumps(polys[0].__geo_interface__)
    intersection = ogr.CreateGeometryFromJson(geom)
    for p in polys[1:]:
        geom = json.dumps(p.__geo_interface__)
        geom = ogr.CreateGeometryFromJson(geom)
        intersection = intersection.Intersection(geom)
    area = intersection.GetArea()
    return area
