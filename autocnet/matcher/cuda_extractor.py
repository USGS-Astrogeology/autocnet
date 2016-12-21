import warnings

import cudasift as cs

def extract_features(array, nfeatures=None):
    """
    A generic wrapper to the PyCudaSift wrapper.  We suggest allowing
    nfeatures to default due to the behavior of PyCudaSift and the
    spatial distribution of the keypoints.

    Parameters
    ----------

    array : ndarray
            (m,n) input array

    nfeatures : int
                The total number of features to extract.

    Returns
    -------
    keypoints : dataframe
                Pandas DataFrame

    descriptors : ndarray
                  (n,128) array of SIFT descriptors
    """

    if not nfeatures:
        nfeatures = int(max(array.shape) / 1.75)
    else:
        warnings.warn('NFeatures specified with the CudaSift implementation.  Please ensure the distribution of keypoints is what you expect.')

    siftdata = cs.PySiftData(nfeatures)
    cs.ExtractKeypoints(array, siftdata)
    keypoints, descriptors = siftdata.to_data_frame()
    keypoints = keypoints[['xpos', 'ypos', 'scale', 'sharpness', 'edgeness', 'orientation', 'score', 'ambiguity']]
    # Set the columns that have unfilled values to zero to avoid confusion
    keypoints['score'] = 0.0
    keypoints['ambiguity'] = 0.0

    return keypoints, descriptors
