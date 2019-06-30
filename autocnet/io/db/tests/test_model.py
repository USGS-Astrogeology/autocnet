from datetime import datetime
import json

import numpy as np
import pandas as pd
import psycopg2
import pytest
import sqlalchemy
from unittest.mock import MagicMock, patch

from autocnet.io.db import model

from autocnet.graph.network import NetworkCandidateGraph

from shapely.geometry import MultiPolygon, Polygon, Point

@pytest.fixture
def tables():
    from autocnet import engine
    return engine.table_names()

def test_keypoints_exists(tables):
    assert model.Keypoints.__tablename__ in tables

def test_edges_exists(tables):
    assert model.Edges.__tablename__ in tables

def test_costs_exists(tables):
    assert model.Costs.__tablename__ in tables

def test_matches_exists(tables):
    assert model.Matches.__tablename__ in tables

def test_cameras_exists(tables):
    assert model.Cameras.__tablename__ in tables

def test_measures_exists(tables):
    assert model.Measures.__tablename__ in tables

def test_create_camera_without_image(session):
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        model.Cameras.create(**{'image_id':1})

def test_create_camera(session):
    #with pytest.raises(sqlalchemy.exc.IntegrityError):
    c = model.Cameras.create(**{'id':1})
    with session() as session:
        res = session.query(model.Cameras).one()
        assert 1 == res.id

def test_create_camera_unique_constraint(session):
    data = {'image_id':1}
    model.Images.create(**{'id': 1})    
    model.Cameras.create(**data)
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        model.Cameras.create(**data)

def test_images_exists(tables):
    assert model.Images.__tablename__ in tables

@pytest.mark.parametrize('data', [
    {'id':1},
    {'name':'foo',
     'path':'/neither/here/nor/there'},
    ])
def test_create_images(session, data):
    model.Images.create(**data)
    with session() as session:
        resp = session.query(model.Images).first()
        for k, v in data.items():
            assert getattr(resp, k) == v

@pytest.mark.parametrize('data', [
    {'serial':'foo'}
])
def test_create_images_constrained(session, data):
    """
    Test that the images unique constraint is being observed.
    """
    with session() as session:
        model.Images.create(**data)
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            model.Images.create(**data)

def test_overlay_exists(tables):
    assert model.Overlay.__tablename__ in tables

@pytest.mark.parametrize('data', [
    {'id':1},
    {'id':1, 'intersections':[1,2,3]},
    {'id':1, 'intersections':[1,2,3],
     'geom':Polygon([(0,0), (1,0), (1,1), (0,1), (0,0)])}

])
def test_create_overlay(session, data):
    model.Overlay.create(**data)
    key = data['id']
    with session() as session:
        resp = session.query(model.Overlay).filter(model.Overlay.id == key).one()
        for k, v in data.items():
                assert getattr(resp, k) == v

def test_points_exists(tables):
    assert model.Points.__tablename__ in tables

@pytest.mark.parametrize("data", [
    {'id':1, 'pointtype':2},
    {'id':2, 'pointtype':2, 'identifier':'123abc'},
    {'id':1, 'pointtype':3, 'apriori':Point(0,0,0)},
    {'id':2, 'pointtype':3, 'adjusted':Point(0,0,0)},
    {'id':1, 'pointtype':2, 'adjusted':Point(1,1,1), 'active':True}
])
def test_create_point(session, data):
    model.Points.create(**data)
    key = data['id']
    with session() as session:
        resp = session.query(model.Points).filter(model.Points.id == key).one()
        for k, v in data.items():
            assert getattr(resp, k) == v

@pytest.mark.parametrize("data, expected", [
    ({'pointtype':3, 'adjusted':Point(0,-1,0)}, Point(-90, 0)),
    ({'pointtype':3}, None)
])
def test_create_point_geom(session, data, expected):
    model.Points.create(**data)
    with session() as session:
        resp = session.query(model.Points).first()
        assert resp.geom == expected

@pytest.mark.parametrize("data, new_adjusted, expected", [
    ({'pointtype':3, 'adjusted':Point(0,-1,0)}, None, None),
    ({'pointtype':3, 'adjusted':Point(0,-1,0)}, Point(0,1,0), Point(90, 0)),
    ({'pointtype':3}, Point(0,-1,0), Point(-90, 0))
])
def test_update_point_geom(session, data, new_adjusted, expected):
    model.Points.create(**data)
    with session() as sess:
        p = sess.query(model.Points).one()
        p.adjusted = new_adjusted
    with session() as sess:    
        resp = sess.query(model.Points).one()
        assert resp.geom == expected

def test_measures_exists(tables):
    assert model.Measures.__tablename__ in tables

@pytest.mark.parametrize("data, serialized", [
    ({'foo':np.arange(5)}, {"foo": [0, 1, 2, 3, 4]}),
    ({'foo':np.int64(1)}, {"foo": 1}),
    ({'foo':b'bar'}, {"foo": "bar"}),
    ({'foo':set(['a', 'b', 'c'])}, {"foo": ["a", "b", "c"]}),
    ({'foo':Point(0,0)}, {"foo": 'POINT (0 0)'}),
    ({'foo':datetime(1982, 9, 8)}, {"foo": '1982-09-08 00:00:00'})

])
def test_json_encoder(data, serialized):
    res = json.dumps(data, cls=model.JsonEncoder)
    res = json.loads(res)
    if isinstance(res['foo'], list):
        res['foo'] = sorted(res['foo'])
    assert res == serialized

@pytest.mark.parametrize("measure_data, point_data, image_data", [({'id': 1, 'pointid': 1, 'imageid': 1, 'serial': 'ISISSERIAL', 'measuretype': 3, 'sample': 0, 'line': 0},
                                                                   {'id':1, 'pointtype':2},
                                                                   {'id':1, 'serial': 'ISISSERIAL'})])
@patch('plio.io.io_controlnetwork.from_isis', return_value = pd.DataFrame.from_dict({'id': [1],
                                                                                     'serialnumber': ['ISISSERIAL'],
                                                                                     'pointJigsawRejected': [False],
                                                                                     'jigsawRejected': [False],
                                                                                     'sampleResidual': [0.1],
                                                                                     'lineResidual': [0.1],
                                                                                     'samplesigma': [0],
                                                                                     'linesigma': [0],
                                                                                     'adjustedCovar': [[]],
                                                                                     'apriorisample': [0],
                                                                                     'aprioriline': [0]}))
def test_jigsaw_append(mockFunc, session, measure_data, point_data, image_data):
    model.Images.create(**image_data)
    model.Points.create(**point_data)
    model.Measures.create(**measure_data)
    with session() as sess:
        resp = sess.query(model.Measures).first()
        assert resp.liner == None
        assert resp.sampler == None

    # Intentionally 2 sessions as this approximates the workflow
    with session() as sess:
        NetworkCandidateGraph.update_from_jigsaw('/Some/Path/To/An/ISISNetwork.cnet')
        resp = sess.query(model.Measures).filter(model.Measures.id == 1).first()
        assert resp.liner == 0.1
        assert resp.sampler == 0.1

def test_null_footprint(session):
    model.Images.create(footprint_latlon=None, serial = 'serial')
    with session() as session:
        i = session.query(model.Images).first()
        assert i.footprint_latlon is None

def test_broken_bad_geom(session):
    # An irreperablly damaged poly
    geom = MultiPolygon([Polygon([(0,0), (1,1), (1,2), (1,1), (0,0)])])
    model.Images.create(footprint_latlon=geom, serial = 'serial')
    with session() as session:
        resp = session.query(model.Images).one()
        assert resp.active == False

def test_fix_bad_geom(session):
    geom = MultiPolygon([Polygon([(0,0), (0,1), (1,1), (0,1), (1,1), (1,0), (0,0) ])])
    model.Images.create(footprint_latlon=geom, serial = 'serial' )
    with session() as session:
        resp = session.query(model.Images).one()
        assert resp.active == True
        assert resp.footprint_latlon == MultiPolygon([Polygon([(0,0), (0,1), (1,1), (1,0), (0,0) ])])
