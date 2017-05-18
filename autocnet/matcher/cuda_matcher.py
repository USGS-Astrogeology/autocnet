import warnings

import cudasift as cs
import numpy as np
import pandas as pd

def match(self, ratio=0.8, **kwargs):
    """
    Apply a composite CUDA matcher and ratio check.  If this method is used,
    no additional ratio check is necessary and no symmetry check is required.
    The ratio check is embedded on the cuda side and returned as an
    ambiguity value.  In testing symmetry is not required as it is expensive
    without significant gain in accuracy when using this implementation.
    """
    s_siftdata = cs.PySiftData.from_data_frame(self.source.get_keypoints(),
					       self.source.descriptors)
    d_siftdata = cs.PySiftData.from_data_frame(self.destination.get_keypoints(),
					       self.destination.descriptors)

    cs.PyMatchSiftData(s_siftdata, d_siftdata)
    matches, _ = s_siftdata.to_data_frame()
    source = np.empty(len(matches))
    source[:] = self.source['node_id']
    destination = np.empty(len(matches))
    destination[:] = self.destination['node_id']


    df = pd.concat([pd.Series(source), pd.Series(matches.index),
		    pd.Series(destination), matches.match,
		    matches.score, matches.ambiguity], axis=1)
    df.columns = ['source_image', 'source_idx', 'destination_image',
		    'destination_idx', 'score', 'ambiguity']

    # Set the matches and set the 'ratio' (ambiguity) mask
    self.matches = df
    self.masks = ('ratio', df['ambiguity'] <= ratio)
