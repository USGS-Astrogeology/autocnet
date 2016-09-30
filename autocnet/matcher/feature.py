import warnings

import cv2
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
        self.node_counter = 0

    def add(self, descriptor, nid):
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

    def clear(self):
        """
        Remove all nodes from the tree and resets
        all counters
        """
        self._flann_matcher.clear()
        self.nid_lookup = {}
        self.node_counter = 0

    def train(self):
        """
        Using the descriptors, generate the KDTree
        """
        self._flann_matcher.train()

    def query(self, descriptor, query_image, k=3):
        """

        Parameters
        ----------
        descriptor : ndarray
                     The query descriptor to search for

        query_image : hashable
                      Key of the query image

        k : int
            The number of nearest neighbors to search for

        Returns
        -------
        matched : dataframe
                  containing matched points with columns containing:
                  matched image name, query index, train index, and
                  descriptor distance
        """

        matches = self._flann_matcher.knnMatch(descriptor, k=k)
        matched = []
        for m in matches:
            for i in m:
                source = query_image
                destination = self.nid_lookup[i.imgIdx]
                if source < destination:
                    matched.append((query_image,
                                    i.queryIdx,
                                    destination,
                                    i.trainIdx,
                                    i.distance))
                elif source > destination:
                    matched.append((destination,
                                    i.trainIdx,
                                    query_image,
                                    i.queryIdx,
                                    i.distance))
                else:
                    warnings.warn('Likely self neighbor in query!')
        return pd.DataFrame(matched, columns=['source_image', 'source_idx',
                                              'destination_image', 'destination_idx',
                                              'distance'])
