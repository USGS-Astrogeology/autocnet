from autocnet.examples import get_path
from autocnet.graph.network import CandidateGraph
from autocnet.io.network import load

def test_save_project(tmpdir):
    path = tmpdir.join('prject.proj')
    #Point to the adjacency Graph
    adjacency = get_path('three_image_adjacency.json')
    basepath = get_path('Apollo15')
    cg = CandidateGraph.from_adjacency(adjacency, basepath=basepath)

    #Apply SIFT to extract features
    cg.extract_features(method='sift', extractor_parameters={'nfeatures':500})

    #Match
    cg.match()

    cg.symmetry_checks()
    cg.ratio_checks()
    cg.compute_fundamental_matrices(clean_keys=['ratio', 'symmetry'], method='ransac')

    cg.save(path.strpath)
    cg2 = load(path.strpath)

    assert cg == cg2
