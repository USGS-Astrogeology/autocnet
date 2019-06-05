import pytest
import sqlalchemy

from autocnet.io.db import model
from autocnet import Session, engine

@pytest.fixture
def tables():
    return engine.table_names()

@pytest.fixture
def session(tables, request):
    session = Session()

    def cleanup():
        session.rollback()  # Necessary because some tests intentionally fail
        for t in reversed(tables):
            session.execute(f'TRUNCATE TABLE {t} CASCADE')
            # Reset the autoincrementing
            if t in ['Images', 'Cameras', 'Matches']:
                session.execute(f'ALTER SEQUENCE {t}_id_seq RESTART WITH 1')
        session.commit()

    request.addfinalizer(cleanup)
    
    return session

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

def test_create_camera_without_image(session):
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        model.Cameras.create(session, **{'image_id':1})

def test_create_camera(session):
    #with pytest.raises(sqlalchemy.exc.IntegrityError):
    c = model.Cameras.create(session)
    res = session.query(model.Cameras).first()
    assert c.id == res.id

def test_images_exists(tables):
    assert model.Images.__tablename__ in tables

@pytest.mark.parametrize('data', [
    {'id':1},
    {'name':'foo',
     'path':'/neither/here/nor/there'},
    ])
def test_create_images(session, data):
    i = model.Images.create(session, **data)
    resp = session.query(model.Images).filter(model.Images.id==i.id).first()
    assert i == resp

@pytest.mark.parametrize('data', [
    {'id':1},
    {'serial':'foo'}
])
def test_create_images_constrined(session, data):
    """
    Test that the images unique constraint is being observed.
    """
    model.Images.create(session, **data)
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        model.Images.create(session, **data)

def test_overlay_exists(tables):
    assert model.Overlay.__tablename__ in tables

def test_points_exists(tables):
    assert model.Points.__tablename__ in tables

def test_measures_exists(tables):
    assert model.Measures.__tablename__ in tables
