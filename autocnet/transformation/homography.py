import numpy as np
import pandas as pd

from autocnet.utils.utils import make_homogeneous

try:
    import cv2
    cv2_avail = True
except:
    cv_avail = False

def compute_error(H, x, x1):
    """
    Give this homography, compute the planar reprojection error
    between points a and b.  x and x1 can be n, 2 or n,3 homogeneous
    coordinates.  If only x, y coordinates, assume that z=1, e.g. x, y, 1 for
    all coordiantes.

    Parameters
    ----------

    x : ndarray
        n,2 array of x,y coordinates

    x1 : ndarray
        n,2 array of x,y coordinates

    Returns
    -------

    df : dataframe
         With columns for x_residual, y_residual, rmse, and
         error contribution.  The dataframe also has cumulative
         x, t, and total RMS statistics accessible via
         df.x_rms, df.y_rms, and df.total_rms, respectively.
    """
    if x.shape[1] == 2:
        x = make_homogeneous(x)
    if x1.shape[1] == 2:
        x1 = make_homogeneous(x1)

    z = np.empty(x.shape)
    # ToDo: Vectorize for performance
    for i, j in enumerate(x):
        z[i] = H.dot(j)
        z[i] /= z[i][-1]

    data = np.empty((x.shape[0], 4))

    data[:, 0] = x_res = x1[:, 0] - z[:, 0]
    data[:, 1] = y_res = x1[:, 1] - z[:, 1]

    if data[:,:1].all() == 0:
        data[:] = 0.0
        total_rms = x_rms = y_rms = 0
    else:
        data[:, 2] = rms = np.sqrt(x_res**2 + y_res**2)
        total_rms = np.sqrt(np.mean(x_res**2 + y_res**2))
        x_rms = np.sqrt(np.mean(x_res**2))
        y_rms = np.sqrt(np.mean(y_res**2))
        data[:, 3] = rms / total_rms

    df = pd.DataFrame(data,
                      columns=['x_residuals',
                               'y_residuals',
                               'rmse',
                               'error_contribution'])
    df.total_rms = total_rms
    df.x_rms = x_rms
    df.y_rms = y_rms
    return df

def compute_homography(x1, x2, method='ransac', reproj_threshold=2.0):
    """
    Compute a planar homography given two sets of keypoints


    Parameters
    ----------
    x1 : ndarray
          (n, 2) of coordinates from the source image

    x2 : ndarray
          (n, 2) of coordinates from the destination image

    method : {'ransac', 'lmeds', 'normal'}
             The openCV algorithm to use for outlier detection

    reproj_threshold : float
                       The maximum distances in pixels a reprojected points
                       can be from the epipolar line to be considered an inlier

    Returns
    -------
    H : ndarray
        (3,3) homography

    mask : ndarray
           Boolean mask for those values omitted from the homography.
    """

    if method == 'ransac':
        method_ = cv2.RANSAC
    elif method == 'lmeds':
        method_ = cv2.LMEDS
    elif method == 'normal':
        method_ = 0  # Normal method
    else:
        raise ValueError("Unknown outlier detection method.  Choices are: 'ransac', 'lmeds', or 'normal'.")
    H, mask = cv2.findHomography(x1, x2, method_, reproj_threshold)
    if mask is not None:
        mask = mask.astype(bool)

    return H, mask
