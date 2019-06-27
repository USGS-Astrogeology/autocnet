import pytest

import pandas as pd

from autocnet.io.db import model
from autocnet import Session, engine
from autocnet.graph.network import NetworkCandidateGraph

from unittest.mock import patch, PropertyMock, MagicMock

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
            if t in ['Images', 'Cameras', 'Matches', 'Measures']:
                session.execute(f'ALTER SEQUENCE {t}_id_seq RESTART WITH 1')
        session.commit()

    request.addfinalizer(cleanup)

    return session

@pytest.fixture()
def cnet():
    return pd.DataFrame.from_dict({
            'id' : [1],
            'pointtype' : 2,
            'serialnumber' : ['BRUH'],
            'jigsawRejected' : [False],
            'sampleResidual' : [0.1],
            'pointingore' : [False],
            'pointjigsawRejected': [False],
            'lineResidual' : [0.1],
            'linesigma' : [0],
            'samplesigma': [0],
            'adjustedCovar' : [[]],
            'apriorisample' : [0],
            'aprioriline' : [0],
            'line' : [1],
            'sample' : [2],
            'ignore': [False],
            'adjustedX' : [0],
            'adjustedY' : [0],
            'adjustedZ' : [0],
            'aprioriX' : [0],
            'aprioriY' : [0],
            'aprioriZ' : [0],
            'measuretype' : [1]
            })

def test_creation():
    ncg = NetworkCandidateGraph()


@pytest.mark.parametrize("image_data, expected", [({'id':1, 'serial': 'BRUH'}, 1)])
def test_place_points_from_cnet(session, cnet, image_data, expected):
    model.Images.create(session, **image_data)
    ncg = NetworkCandidateGraph.from_database()

    ncg.place_points_from_cnet(cnet)

    resp = session.query(model.Points)
    assert len(resp.all()) == expected

