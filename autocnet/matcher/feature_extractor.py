import warnings

import autocnet
import cv2
import numpy as np
import pandas as pd
from scipy.misc import bytescale

try:
    import cyvlfeat as vl
    vlfeat = True
except Exception:  # pragma: no cover
    vlfeat = False
    pass


def extract_features(array, method='orb', extractor_parameters={}):
    """
    This method finds and extracts features from an image using the given dictionary of keyword arguments.
    The input image is represented as NumPy array and the output features are represented as keypoint IDs
    with corresponding descriptors.

    Parameters
    ----------
    array : ndarray
            a NumPy array that represents an image

    method : {'orb', 'sift', 'fast', 'surf', 'vl_sift'}
              The detector method to be used.  Note that vl_sift requires that
              vlfeat and cyvlfeat dependencies be installed.

    extractor_parameters : dict
                           A dictionary containing OpenCV SIFT parameters names and values.

    Returns
    -------
    keypoints : DataFrame
                data frame of coordinates ('x', 'y', 'size', 'angle', and other available information)

    descriptors : ndarray
                  Of descriptors
    """
    detectors = {'fast': cv2.FastFeatureDetector_create,
                 'sift': cv2.xfeatures2d.SIFT_create,
                 'surf': cv2.xfeatures2d.SURF_create,
                 'orb': cv2.ORB_create}

    if method == 'vlfeat' and vlfeat != True:
        raise ImportError('VLFeat is not available.  Please install vlfeat or use a different extractor.')

    if  method == 'vlfeat':
        keypoint_objs, descriptors  = vl.sift.sift(array,
                                                   compute_descriptor=True,
                                                   float_descriptors=True)
        # Swap columns for value style access, vl_feat returns y, x
        keypoint_objs[:, 0], keypoint_objs[:, 1] = keypoint_objs[:, 1], keypoint_objs[:, 0].copy()
        keypoints = pd.DataFrame(keypoint_objs, columns=['x', 'y', 'size', 'angle'])
    else:
        # OpenCV requires the input images to be 8-bit
        if not array.dtype == 'int8':
            array = bytescale(array)
        detector = detectors[method](**extractor_parameters)
        keypoint_objs, descriptors = detector.detectAndCompute(array, None)

        keypoints = np.empty((len(keypoint_objs), 7), dtype=np.float32)
        for i, kpt in enumerate(keypoint_objs):
            octave = kpt.octave & 8
            layer = (kpt.octave >> 8) & 255
            if octave < 128:
                octave = octave
            else:
                octave = (-128 | octave)
            keypoints[i] = kpt.pt[0], kpt.pt[1], kpt.response, kpt.size, kpt.angle, octave, layer  # y, x
        keypoints = pd.DataFrame(keypoints, columns=['x', 'y', 'response', 'size',
                                                     'angle', 'octave', 'layer'])

        if descriptors.dtype != np.float32:
            descriptors = descriptors.astype(np.float32)

    return keypoints, descriptors
