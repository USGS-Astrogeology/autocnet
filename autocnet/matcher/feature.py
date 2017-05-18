import warnings

import cv2
import numpy as np
import pandas as pd

FLANN_INDEX_KDTREE = 1  # Algorithm to set centers,
DEFAULT_FLANN_PARAMETERS = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)


class FlannMatcher(object):
    """
    A wrapper to the OpenCV Flann based matcher class that adds
    metadata tracking attributes and methods.  This takes arbitrary
    descriptors and so should be available for use with any
    descriptor data stored as an ndarray.

    Attributes
    ----------
    image_indices : dict
                    with key equal to the train image index (returned by the DMatch object),
                    e.g. an integer array index
                    and value equal to the image identifier, e.g. the name

    image_index_counter : int
                          The current number of images loaded into the matcher
    """

    def __init__(self, flann_parameters=DEFAULT_FLANN_PARAMETERS):
        self._flann_matcher = cv2.FlannBasedMatcher(flann_parameters, {})
        self.nid_lookup = {}
        self.search_idx = {}
        self.node_counter = 0

    def add(self, descriptor, nid, index=None):
        """
        Add a set of descriptors to the matcher and add the image
        index key to the image_indices attribute

        Parameters
        ----------
        descriptor : ndarray
                     The descriptor to be added

        nid : int
              The node ids
        """
        self._flann_matcher.add([descriptor])
        self.nid_lookup[self.node_counter] = nid
        self.node_counter += 1
        if index is not None:
            self.search_idx = dict((i, j) for i, j in enumerate(index))
        else:
            self.search_idx = dict((i,i) for i in range(len(descriptor)))

    def clear(self):
        """
        Remove all nodes from the tree and resets
        all counters
        """
        self._flann_matcher.clear()
        self.nid_lookup = {}
        self.node_counter = 0
        self.search_idx = {}

    def train(self):
        """
        Using the descriptors, generate the KDTree
        """
        self._flann_matcher.train()

    def query(self, descriptor, query_image, k=3, index=None):
        """

        Parameters
        ----------
        descriptor : ndarray
                     The query descriptor to search for

        query_image : hashable
                      Key of the query image

        k : int
            The number of nearest neighbors to search for

        index : iterable
                An iterable of observation indices to utilize for
                the input descriptors

        Returns
        -------
        matched : dataframe
                  containing matched points with columns containing:
                  matched image name, query index, train index, and
                  descriptor distance
        """

        matches = self._flann_matcher.knnMatch(descriptor, k=k)
        matched = []
        for i, m in enumerate(matches):
            for j in m:
                if index is not None:
                    qid = index[i]
                else:
                    qid = j.queryIdx
                source = query_image
                destination = self.nid_lookup[j.imgIdx]
                if source < destination:
                    matched.append((query_image,
                                    qid,
                                    destination,
                                    self.search_idx[j.trainIdx],
                                    j.distance))
                elif source > destination:
                    matched.append((destination,
                                    self.search_idx[j.trainIdx],
                                    query_image,
                                    qid,
                                    j.distance))
                else:
                    warnings.warn('Likely self neighbor in query!')
        return pd.DataFrame(matched, columns=['source_image', 'source_idx',
                                              'destination_image', 'destination_idx',
                                              'distance']).astype(np.float32)

def cudamatcher():
    pass
