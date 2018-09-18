import warnings
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from autocnet.matcher import subpixel as sp

from plio.io.io_controlnetwork import to_isis, write_filelist


def identify_potential_overlaps(cg, cn, overlap=True):
    """
    Identify those points that could have additional measures

    Parameters
    ----------
    overlap : boolean
              If True, apply aprint(g)n additional point in polygon check, where
              the polygon is the footprint intersection between images and
              the point is a keypoint projected into lat/lon space.  Note
              that the projection can be inaccurate if the method used
              estimates the transformation.

    Returns
    -------
    candidate_cliques : DataFrame
                        with the index as the point id (in the data attribute)
                        and the value as an iterable of image ids to search
                        for a new point.
    """

    fc = cg.compute_fully_connected_components()

    candidate_cliques = []
    geoms = []
    idx = []
    for i, p in cn.groupby('point_id'):
        # Which images are covered already.  This finds any connected cycles that
        #  a node is in (this can be more than one - an hourglass network for example)
        # Extract the fully connected subgraph for each covered image in order to
        #  identify which subgraph the measure is in
        covered = p['image_index']
        candidate_cycles = [fc[c] for c in covered]
        cycle = [i for i in candidate_cycles if candidate_cycles.count(i) > 1]
        cycle_to_punch = cycle[0][0]

        # Using the cycles to punch, which images could also be covered?
        uncovered = tuple(set(cycle_to_punch).difference(set(covered)))

        # All candidates are covered, skip this point
        if not uncovered:
            continue

        # Determine whether a 'real' lat/lon are to be used and reproject
        if overlap:
            row = p.iloc[0]
            lat, lon = cg.node[row.image_index]['data'].geodata.pixel_to_latlon(
                row.x, row.y)
        else:
            lat, lon = 0, 0

        # Build the data for the geodataframe - can the index be cleaner?
        geoms.append(Point(lon, lat))
        candidate_cliques.append([uncovered, cycle_to_punch])
        idx.append(i)

    candidate_cliques = gpd.GeoDataFrame(candidate_cliques, index=idx,
                                         columns=['candidates', 'subgraph'], geometry=geoms)

    def overlaps(group):
        """
        Take a group, find the subgraph, compute the intersection of footprints
        and apply a group point in polygon check. This is an optimization where
        n-points are intersected with the poly at once (as opposed to the
        single iteration approach.)
        """
        cycle_to_punch = group.subgraph.iloc[0]
        subgraph = cg.create_node_subgraph(cycle_to_punch)
        union, _ = subgraph.compute_intersection(
            cycle_to_punch[0])  # .query('overlaps_all == True')
        intersection = group.intersects(union.unary_union)
        return intersection

    # If the overlap check is going to be used, apply it.
    if overlap:
        candidate_cliques['overlap'] = False
        for i, g in candidate_cliques.groupby('candidates'):
            intersection = overlaps(g)
            candidate_cliques.loc[intersection.index, 'overlap'] = intersection
        return candidate_cliques.query('overlap == True')['candidates']
    else:
        return candidate_cliques.candidates
