import warnings
import numpy as np
import pandas as pd
from scipy import optimize
from autocnet.camera import camera
from autocnet.camera import utils as camera_utils
from autocnet.utils.utils import make_homogeneous, normalize_vector

try:
    import cv2
    cv2_avail = True
except:  # pragma: no cover
    cv_avail = False


def compute_reprojection_error(F, x, x1):
    """
    Given a set of matches and a known fundamental matrix,
    compute distance between match points and the associated
    epipolar lines.

    The distance between a point and the associated epipolar
    line is computed as: $d = \frac{\lvert ax_{0} + by_{0} + c \rvert}{\sqrt{a^{2} + b^{2}}}$.

    Parameters
    ----------
    F : ndarray
        (3,3) Fundamental matrix

    x : arraylike
        (n,2) or (n,3) array of homogeneous coordinates

    x1 : arraylike
        (n,2) or (n,3) array of homogeneous coordinates with the same
        length as argument x

    Returns
    -------
    F_error : ndarray
              n,1 vector of reprojection errors
    """

    if x.shape[1] != 3:
        x = make_homogeneous(x)
    if x1.shape[1] != 3:
        x1 = make_homogeneous(x1)

    # Normalize the vector
    l_norms = normalize_vector(x.dot(F.T))
    F_error = np.abs(np.sum(l_norms * x1, axis=1))

    return F_error

def compute_fundamental_error(F, x, x1):
    """
    Compute the fundamental error using the idealized error metric.

    Ideal error is defined by $x^{\intercal}Fx = 0$,
    where $x$ are all matchpoints in a given image and
    $x^{\intercal}F$ defines the standard form of the
    epipolar line in the second image.

    This method assumes that x and x1 are ordered such that x[0]
    correspondes to x1[0].

    Parameters
    ----------
    F : ndarray
        (3,3) Fundamental matrix

    x : arraylike
        (n,2) or (n,3) array of homogeneous coordinates

    x1 : arraylike
        (n,2) or (n,3) array of homogeneous coordinates with the same
        length as argument x

    Returns
    -------
    F_error : ndarray
              n,1 vector of reprojection errors
    """

    # TODO: Can this be vectorized for performance?
    if x.shape[1] != 3:
        x = make_homogeneous(x)
    if x1.shape[1] != 3:
        x1 = make_homogeneous(x1)

    if isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(x1, pd.DataFrame):
        x1 = x1.values

    err = np.empty(len(x))
    for i in range(len(x)):
        err[i] = x1[i].T.dot(F).dot(x[i])
    return err

def update_fundamental_mask(F, x1, x2, threshold=1.0, index=None, method='reprojection'):
    """
    Given a Fundamental matrix and two sets of points, compute the
    reprojection error between x1 and x2.  A mask is returned with all
    repojection errors greater than the error set to false.

    Parameters
    ----------
    F : ndarray
        (3,3) Fundamental matrix

    x1 : arraylike
         (n,2) or (n,3) array of homogeneous coordinates

    x2 : arraylike
         (n,2) or (n,3) array of homogeneous coordinates

    threshold : float
                The new upper limit for error.  If using
                reprojection this is measured in pixels (the default).  If
                using fundamental, the idealized error is 0.  Values +- 0.05
                should be good.

    index : ndarray
            Optional index for mapping between reprojective error
            and an associated dataframe (e.g., an indexed matches dataframe).

    Returns
    -------
    mask : dataframe

    """
    if method == 'reprojection':
        error = compute_reprojection_error(F, x1, x2)
    elif method == 'fundamental':
        error = compute_fundamental_error(F, x1, x2)
    else:
        warnings.warn('Unknown error method.  Options are "reprojection" or "fundamental".')
    mask = pd.DataFrame(np.abs(error) <= threshold, index=index, columns=['fundamental'])
    if index != None:
        mask.index = index

    return mask

def enforce_singularity_constraint(F):
    """
    The fundamental matrix should be rank 2.  In instances when it is not,
    the singularity constraint should be enforced.  This is forces epipolar lines
    to be conincident.

    Parameters
    ----------
    F : ndarray
        (3,3) Fundamental Matrix

    Returns
    -------
    F : ndarray
        (3,3) Singular Fundamental Matrix

    References
    ----------
    .. [Hartley2003]

    """
    if np.linalg.matrix_rank(F) != 2:
        u, d, vt = np.linalg.svd(F)
        F = u.dot(np.diag([d[0], d[1], 0])).dot(vt)

    return F

def compute_fundamental_matrix(kp1, kp2, method='mle', reproj_threshold=2.0,
                               confidence=0.99):
    """
    Given two arrays of keypoints compute the fundamental matrix.  This function
    accepts two dataframe of keypoints that have

    Parameters
    ----------
    kp1 : arraylike
          (n, 2) of coordinates from the source image

    kp2 : ndarray
          (n, 2) of coordinates from the destination image


    method : {'ransac', 'lmeds', 'normal', '8point'}
              The openCV algorithm to use for outlier detection

    reproj_threshold : float
                       The maximum distances in pixels a reprojected points
                       can be from the epipolar line to be considered an inlier

    confidence : float
                 [0, 1] that the estimated matrix is correct

    Notes
    -----
    While the method is user definable, if the number of input points
    is < 7, normal outlier detection is automatically used, if 7 > n > 15,
    least medians is used, and if 7 > 15, ransac can be used.
    """
    if method == 'mle':
        # Grab an initial estimate using RANSAC, then apply MLE
        method_ = cv2.FM_RANSAC
    elif method == 'ransac':
        method_ = cv2.FM_RANSAC
    elif method == 'lmeds':
        method_ = cv2.FM_LMEDS
    elif method == 'normal':
        method_ = cv2.FM_7POINT
    elif method == '8point':
        method_ = cv2.FM_8POINT
    else:
        raise ValueError("Unknown estimation method. Choices are: 'lme', 'ransac', 'lmeds', '8point', or 'normal'.")

    # OpenCV wants arrays
    F, mask = cv2.findFundamentalMat(np.asarray(kp1),
                                     np.asarray(kp2),
                                     method_,
                                     param1=reproj_threshold,
                                     param2=confidence)


    if F.shape != (3,3):
        warnings.warn('F computation fell back to 7-point algorithm, not setting F.')
        return None, None
    # Ensure that the singularity constraint is met
    F = enforce_singularity_constraint(F)

    try:
        mask = mask.astype(bool).ravel()  # Enforce dimensionality
    except:
        return  # pragma: no cover

    if method == 'mle':
        # Now apply the gold standard algorithm to refine F

        if kp1.shape[1] != 3:
            kp1 = make_homogeneous(kp1)
        if kp2.shape[1] != 3:
            kp2 = make_homogeneous(kp2)

        # Generate an idealized and to be updated camera model
        p1 = camera.estimated_camera_from_f(F)
        p = camera.idealized_camera()

        # Grab the points used to estimate F
        pt = kp1.loc[mask].T
        pt1 = kp2.loc[mask].T

        if pt.shape[1] < 9 or pt1.shape[1] < 9:
            warnings.warn("Unable to apply MLE.  Not enough correspondences.  Returning with a RANSAC computed F matrix.")
            return F, mask

        # Apply Levenber-Marquardt to perform a non-linear lst. squares fit
        #  to minimize triangulation error (this is a local bundle)
        result = optimize.least_squares(camera.projection_error, p1.ravel(),
                                        args=(p, pt, pt1),
                                        method='lm')

        gold_standard_p = result.x.reshape(3, 4) # SciPy Lst. Sq. requires a vector, camera is 3x4
        optimality = result.optimality
        gold_standard_f = camera_utils.crossform(gold_standard_p[:,3]).dot(gold_standard_p[:,:3])

        F = gold_standard_f

        mask = update_fundamental_mask(F, kp1, kp2,
                                       threshold=reproj_threshold).values


    return F, mask
