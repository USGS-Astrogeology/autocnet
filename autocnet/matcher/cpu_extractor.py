import warnings

from cv2 import ORB_create, FastFeatureDetector_create
import numpy as np
import pandas as pd
from autocnet.utils.utils import bytescale

try:
    import cyvlfeat as vl
    vlfeat = True
except Exception:  # pragma: no cover
    vlfeat = False
    pass

try:
    from cv2 import xfeatures2d
    SIFT = xfeatures2d.SIFT_create
    SURF = xfeatures2d.SURF_create
except Exception:  # pragma: no cover
    SIFT = None
    SURF = None
    pass


def extract_features(array, extractor_method='sift', extractor_parameters={}):
    """
    This method finds and extracts features from an image using the given dictionary of keyword arguments.
    The input image is represented as NumPy array and the output features are represented as keypoint IDs
    with corresponding descriptors.

    Parameters
    ----------
    array : ndarray
            a NumPy array that represents an image

    extractor_method : {'orb', 'sift', 'fast', 'surf', 'vl_sift'}
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
    detectors = {'fast': FastFeatureDetector_create,
                 'sift': SIFT,
                 'surf': SURF,
                 'orb' : ORB_create}

    if extractor_method == 'vlfeat' and vlfeat != True:
        raise ImportError('VLFeat is not available.  Please install vlfeat or use a different extractor.')

    if  extractor_method == 'vlfeat':
        keypoint_objs, descriptors  = vl.sift.sift(array,
                                                   compute_descriptor=True,
                                                   float_descriptors=True,
                                                   **extractor_parameters)
        # Swap columns for value style access, vl_feat returns y, x
        keypoint_objs[:, 0], keypoint_objs[:, 1] = keypoint_objs[:, 1], keypoint_objs[:, 0].copy()
        keypoints = pd.DataFrame(keypoint_objs, columns=['x', 'y', 'size', 'angle'])
    else:
        # OpenCV requires the input images to be 8-bit
        if not array.dtype == 'int8':
            array = bytescale(array)
        detector = detectors[extractor_method](**extractor_parameters)
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

        if hasattr(descriptors, 'dtype') and descriptors.dtype != np.float32:
            descriptors = descriptors.astype(np.float32)

    return keypoints, descriptors

def extract_most_interesting(image, extractor_method='orb', extractor_parameters={'nfeatures':10}):
    """
    Given an image, extract the most interesting feature. Interesting is defined
    as the feature descriptor that has the maximum variance. By default, this func
    finds 10 features in the image and then selects the best.

    Parameters
    ----------
    image : ndarray
            of DN values

    extractor_method : str
                       Any valid, autocnet extractor. Default (orb)

    exctractor_parameters : dict
                            of extractor parameters passed through to the feature extractor

    Returns
    -------
     : pd.series
       The keypoints row with the higest variance. The row has 'x' and 'y' columns to
       get the location.
    """
    kps, desc = extract_features(image,
                                 extractor_method=extractor_method,
                                 extractor_parameters=extractor_parameters)

    if len(kps) == 0:
        return None
    # Naively assume that the maximum variance is the most unique feature
    vari = np.var(desc, axis=1)
    return kps.iloc[np.argmax(vari)]
