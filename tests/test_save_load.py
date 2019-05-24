from autocnet.examples import get_path
from autocnet.graph.network import CandidateGraph
from autocnet.io.network import load

import numpy as np

import pandas as pd

def test_save_project(tmpdir, candidategraph):
    path = tmpdir.join('prject.proj')
    candidategraph.save(path.strpath)
    candidategraph2 = load(path.strpath)
    assert candidategraph == candidategraph2

def test_save_features(tmpdir, candidategraph):
    path = tmpdir.join('features')
    candidategraph.save_features(path.strpath)

    d = np.load(path.strpath + '_1.npz')
    np.testing.assert_array_equal(d['descriptors'],
                                         candidategraph.node[1]['data'].descriptors)
