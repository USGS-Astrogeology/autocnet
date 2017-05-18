import cudasift as cs
import numpy as np
import pandas as pd


def decompose_and_match(self, maxiterations=3, ratio=0.9, buf_dist=3, **kwargs):
    sdata = self.source.get_array()
    ddata = self.destination.get_array()

    ret = cs.PyDecomposeAndMatch(sdata, ddata, maxiterations=maxiterations,
                                 ratio=ratio, buf_dist=buf_dist, **kwargs)

    self.smembership = ret[0]
    self.dmembership = ret[1]
    self.source._keypoints = skps = ret[3]
    self.source.descriptors = ret[4]
    self.destination._keypoints = dkps = ret[5]
    self.destination.descriptors = ddesc = ret[6]

    # Parse the matches by decomposed sections into a global matches dataframe
    raw_matches = ret[2]

    columns = ['source_image', 'source_idx', 'destination_image',
               'destination_idx', 'score', 'ambiguity']

    m = np.empty((len(raw_matches), len(columns)))
    m[:,0] = self.source['node_id']
    m[:,1] = raw_matches.index
    m[:,2] = self.destination['node_id']
    m[:,3] = raw_matches['match']
    m[:,4] = raw_matches['score']
    m[:,5] = raw_matches['ambiguity']

    self.matches = pd.DataFrame(m, columns=columns)

    # Set the ratio mask
    self.masks = ('ratio', self.matches['ambiguity'] <= ratio)
