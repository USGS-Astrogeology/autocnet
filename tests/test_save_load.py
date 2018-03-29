from autocnet.examples import get_path
from autocnet.graph.network import CandidateGraph
from autocnet.io.network import load

import numpy as np

def test_save_project(tmpdir, candidategraph):
    path = tmpdir.join('prject.proj')
    candidategraph.save(path.strpath)
    candidategraph2 = load(path.strpath)

    # Prints just help with debugging because this is probably the most 
    # fragile stuff in the code base.
    for i,n in candidategraph.nodes.data('data'):
        print('Node {}: {}'.format(i,n == candidategraph2.node[i]['data']))

    for s,d,e in candidategraph.edges.data('data'):
        print(type(candidategraph2.edges[s,d]), candidategraph2.edges[s,d].keys())
        print('Edge {}: {}'.format((s,d), e == candidategraph2.edges[s,d]['data']))
        e1 = candidategraph2.edges[s,d]['data']
    assert candidategraph == candidategraph2

def test_save_features(tmpdir, candidategraph):
    path = tmpdir.join('features')
    candidategraph.save_features(path.strpath)

    d = np.load(path.strpath + '_0.npz')
    np.testing.assert_array_equal(d['descriptors'],
                                         candidategraph.node[0]['data'].descriptors)
