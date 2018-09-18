from unittest.mock import MagicMock
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

import pytest
from autocnet.control import control

def test_identify_potential_overlaps(controlnetwork, candidategraph):
    res = control.identify_potential_overlaps(candidategraph,
                                              controlnetwork,
                                              overlap=False)

    assert res.equals(pd.Series([(2,), (2,),
                                 (1,), (1,),
                                 (0,), (0,)],
                                 index=[6,7,8,9,10,11]))

def test_potential_overlap(controlnetwork, candidategraph):
    # Patch in an intersection check so that all points intersect all geoms
    candidategraph.create_node_subgraph = MagicMock(return_value=candidategraph)
    coords = [(-1., -1.), (-1., 1.), (1., 1.), (1., -1.), (-1., -1.)]
    poly = gpd.GeoSeries(Polygon(coords))
    candidategraph.compute_intersection = MagicMock(return_value=(poly, 0))
    res = control.identify_potential_overlaps(candidategraph,
                                              controlnetwork,
                                              overlap=True)

    assert res.equals(pd.Series([(2,), (2,),
                                 (1,), (1,),
                                 (0,), (0,)],
                                 index=[6,7,8,9,10,11]))

"""
def test_fromcandidategraph(candidategraph, controlnetwork_data):#, controlnetwork):
    matches = candidategraph.get_matches()
    cn = control.ControlNetwork.from_candidategraph(matches)
    assert cn.data[['point_id', 'image_index']].equals(controlnetwork_data[['point_id', 'image_index']])

def test_add_measure():
    cn = control.ControlNetwork()

    # Add the point 0 from image 0
    key = (0,0)
    cn.add_measure(key, (0,1), 3, [1,1])
    assert key in cn.measure_to_point.keys()
    assert cn.measure_to_point[key] == 0

    # Add the point 1 from image 2
    key = (1,2)
    cn.add_measure(key, (0,1), 2, [1,1])
    assert key in cn.measure_to_point.keys()

    # Add another measure associated with point (0,0)
    #  Key is the source and this methods is called to add the destination
    key = (2,1)
    cn.add_measure(key, (0,2), 3, [1,1], point_id = 0)
    assert key in cn.measure_to_point.keys()
    assert cn.measure_to_point[key] == 0

def test_validate_points(controlnetwork):
    assert not controlnetwork.validate_points().any()

def test_bad_validate_points(bad_controlnetwork):
    assert bad_controlnetwork.validate_points().iloc[0] == True
    assert not bad_controlnetwork.validate_points().iloc[1:].all()




"""