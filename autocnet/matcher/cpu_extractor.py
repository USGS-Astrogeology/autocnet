import warnings

from cv2 import ORB_create, FastFeatureDetector_create
import numpy as np
import pandas as pd

from autocnet.utils.utils import bytescale
from autocnet.transformation import roi

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

        if descriptors.dtype != np.float32:
            descriptors = descriptors.astype(np.float32)

    return keypoints, descriptors


def extract_most_interesting(image, size=5, n=1, extractor_method='orb', extractor_parameters={'nfeatures': 15, 'edgeThreshold': 2}):
    """
    Given an image, extract the most interesting feature. Interesting is defined
    as the feature descriptor that has the maximum variance. By default, this func
    finds 10 features in the image and then selects the best.

    Parameters
    ----------
    image : ndarray
            of DN values

    n : int
        Number of keypoints to return in increasing score value (1 mean return one keypoint with highest score)

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
    score_func = lambda r: np.var(roi.Roi(image, r.x, r.y, size, size).clip())

    kps, desc = extract_features(image,
                                  extractor_method=extractor_method,
                                  extractor_parameters=extractor_parameters)

    kps['score'] = kps.apply(score_func, axis=1)
    kps = kps.sort_values(by=['score'], ascending=False).iloc[0:n]

    return kps


def find_common_feature(roi1, roi2, thresh=5, n=10, extractor_parameters={'nfeatures': 15, 'edgeThreshold': 1, 'scaleFactor':1.2}):
    """
    Find a single feature that is similar enough between the two images. Essentially, feature extraction and
    basic matching between two regions of interst that are projected on eachother.

    Parameters
    ----------

    roi1 : np.array
           array object of image 1 used as the base
    roi2 : np.array
           array object of image 2, projected to match roi1
    thresh : float
             distance threshold, point pairs below this threshold are rejected.
    n : int
        nuber of candidate points to attempt to extract
    geom : bool
           If true, runs roi2 through the projection step. Otherwise, uses roi1 as is. Default is True

    Returns
    -------

    : pd.Series
      Single DF row containing the passsing point and associated metadata

    """
    p1 = extract_most_interesting(roi1, n=n, extractor_method='orb', extractor_parameters=extractor_parameters)
    p2 = extract_most_interesting(roi2, n=n, extractor_method='orb', extractor_parameters=extractor_parameters)

    if n == 1:
        dist = np.linalg.norm([p1.x-p2.x, p1.y-p2.y])
        return p1 if dist < thresh else None

    # sometimes, the extractor fails to exract enough points from either one image or the other,
    # so throw out the extra features to make them parrallel
    n_matches = min(len(p1), len(p2))

    # Drop index, enabling parallel subtraction
    p1 = p1.iloc[:n_matches].reset_index(drop=True)
    p2 = p2.iloc[:n_matches].reset_index(drop=True)

    dist = np.linalg.norm(list(zip(p1.x-p2.x, p1.y-p2.y)), axis=1)
    p1['dist'] = dist

    # return the lower dist in the thresh
    p1 = p1.sort_values(by=['dist'], ascending=True).iloc[0]
    if p1['dist'] < thresh:
        return p1


