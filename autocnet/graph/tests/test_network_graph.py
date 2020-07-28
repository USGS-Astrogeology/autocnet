import os
import pytest
import sys

import pandas as pd

from autocnet.io.db import model
from autocnet.graph.network import NetworkCandidateGraph

from unittest.mock import patch, PropertyMock, MagicMock

if sys.platform.startswith("darwin"):
    pytest.skip("skipping DB tests for MacOS", allow_module_level=True)

@pytest.fixture()
def cnet():
    return pd.DataFrame.from_dict({
            'id' : [1],
            'pointType' : 2,
            'serialnumber' : ['BRUH'],
            'measureJigsawRejected': [False],
            'sampleResidual' : [0.1],
            'pointIgnore' : [False],
            'pointJigsawRejected': [False],
            'lineResidual' : [0.1],
            'linesigma' : [0],
            'samplesigma': [0],
            'adjustedCovar' : [[]],
            'apriorisample' : [0],
            'aprioriline' : [0],
            'line' : [1],
            'sample' : [2],
            'measureIgnore': [False],
            'adjustedX' : [0],
            'adjustedY' : [0],
            'adjustedZ' : [0],
            'aprioriX' : [0],
            'aprioriY' : [0],
            'aprioriZ' : [0],
            'measureType' : [1]
            })

@pytest.mark.parametrize("image_data, expected_npoints", [({'id':1, 'serial': 'BRUH'}, 1)])
def test_place_points_from_cnet(session, cnet, image_data, expected_npoints, ncg):
    session = ncg.Session()
    model.Images.create(session, **image_data)

    ncg.place_points_from_cnet(cnet)

    resp = session.query(model.Points)
    assert len(resp.all()) == expected_npoints
    assert len(resp.all()) == cnet.shape[0]
    session.close()

def test_to_isis(db_controlnetwork, ncg, node_a, node_b, tmpdir):
    ncg.add_edge(0,1)
    ncg.nodes[0]['data'] = node_a
    ncg.nodes[1]['data'] = node_b

    outpath = tmpdir.join('outnet.net')
    ncg.to_isis(outpath)

    assert os.path.exists(outpath)
