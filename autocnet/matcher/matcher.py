import warnings

import cv2
import pandas as pd

from scipy.ndimage.interpolation import zoom

FLANN_INDEX_KDTREE = 1  # Algorithm to set centers,
DEFAULT_FLANN_PARAMETERS = dict(algorithm=FLANN_INDEX_KDTREE,
                                trees=3)


def pattern_match(template, image, upsampling=16, func=cv2.TM_CCOEFF_NORMED, error_check=False):
    """
    Call an arbitrary pattern matcher

    Parameters
    ----------
    template : ndarray
               The input search template used to 'query' the destination
               image

    image : ndarray
            The image or sub-image to be searched

    upsampling : int
                 The multiplier to upsample the template and image.

    func : object
           The function to be used to perform the template based matching
           Options: {cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED, cv2.TM_SQDIFF_NORMED}
           In testing the first two options perform significantly better with Apollo data.

    error_check : bool
                  If True, also apply a different matcher and test that the values
                  are not too divergent.  Default, False.

    Returns
    -------

    x : float
        The x offset

    y : float
        The y offset

    strength : float
               The strength of the correlation in the range [-1, 1].
    """

    different = {cv2.TM_SQDIFF_NORMED: cv2.TM_CCOEFF_NORMED,
                 cv2.TM_CCORR_NORMED: cv2.TM_SQDIFF_NORMED,
                 cv2.TM_CCOEFF_NORMED: cv2.TM_SQDIFF_NORMED}

    if upsampling < 1:
        raise ValueError

    u_template = zoom(template, upsampling, order=1)
    u_image = zoom(image, upsampling, order=1)

    result = cv2.matchTemplate(u_image, u_template, method=func)
    min_corr, max_corr, min_loc, max_loc = cv2.minMaxLoc(result)
    if func == cv2.TM_SQDIFF or func == cv2.TM_SQDIFF_NORMED:
        x, y = (min_loc[0], min_loc[1])
    else:
        x, y = (max_loc[0], max_loc[1])

    # Compute the idealized shift (image center)
    ideal_y = u_image.shape[0] / 2
    ideal_x = u_image.shape[1] / 2

    # Compute the shift from template upper left to template center
    y += (u_template.shape[0] / 2)
    x += (u_template.shape[1] / 2)

    x = (ideal_x - x) / upsampling
    y = (ideal_y - y) / upsampling
    return x, y, max_corr


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
