import cv2


def extract_features(array, method='orb', extractor_parameters=None):
    """
    This method finds and extracts features from an image using the given dictionary of keyword arguments.
    The input image is represented as NumPy array and the output features are represented as keypoint IDs
    with corresponding descriptors.

    Parameters
    ----------
    array : ndarray
            a NumPy array that represents an image

    detector : {'orb', 'sift', 'fast', 'surf'}
              The detector method to be used

    extractor_parameters : dict
                           A dictionary containing OpenCV SIFT parameters names and values.

    Returns
    -------
    : tuple
      in the form ([list of OpenCV KeyPoints], [NumPy array of descriptors as geometric vectors])
    """

    detectors = {'fast': cv2.FastFeatureDetector_create,
                 'sift': cv2.xfeatures2d.SIFT_create,
                 'surf': cv2.xfeatures2d.SURF_create,
                 'orb': cv2.ORB_create}

    detector = detectors[method](**extractor_parameters)
    return detector.detectAndCompute(array, None)
