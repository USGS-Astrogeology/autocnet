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

def compute_epipolar_lines(F, x, index=None):
    """
    Given a fundamental matrix and a set of homogeneous points

    Parameters
    ----------
    F : ndarray
        of shape (3,3) that represents the fundamental matrix

    x : ndarray
        of shape (n, 3) of homogeneous coordinates

    Returns
    -------
    lines : ndarray
            of shape (n,3) of epipolar lines in standard form
    """
    if isinstance(x, pd.DataFrame):
        x = x.values

    if not x.shape[1] == 3:
        raise ValueError('The input points must be homogenous with shape (n,3)')

    # Compute the unnormalized epipolar lines
    lines = np.inner(F, x)

    # Normalize the lines
    nu = lines[0] ** 2 + lines[1] ** 2
    try:
        nu = 1 / np.sqrt(nu)
    except:
        nu = 1
    lines *= nu

    lines = lines.T

    if index is not None:
        lines = pd.DataFrame(lines, columns=['a', 'b', 'c'], index=index)

    # Inner transposes the result, so transpose back into the 3 column form
    return lines

def epipolar_distance(lines, pts):
    """
    Given a set of epipolar lines and a set of points, compute the euclidean
    distance between each point and the corresponding epipolar line

    Parameters
    ----------
    lines : ndarray
            of shape (n,3) of epipolar lines in standard form

    pts : ndarray
          of shape (n, 3) of homogeneous coordinates
    """
    num = np.abs(lines[:,0] * pts[:,0] + lines[:,1] * pts[:,1] + lines[:,2])
    denom = np.sqrt(lines[:,0] ** 2 + lines[:,1] ** 2)
    return num / denom

def compute_reprojection_error(F, x, x1, index=None):
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

    if not x.shape[1] == 3 or not x1.shape[1] == 3:
        raise ValueError('The input points must be homogenous with shape (n,3)')

    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.values
    if isinstance(x1, (pd.Series, pd.DataFrame)):
        x1 = x1.values

    # Compute the epipolar lines
    lines1 = compute_epipolar_lines(F,x)
    lines2 = compute_epipolar_lines(F.T, x1)

    # Compute the euclidean distance from the pt to the line
    d1 = epipolar_distance(lines2, x)
    d2 = epipolar_distance(lines1, x1)

    # Grab the max err from either reprojection
    err = np.max(np.column_stack((d1,d2)), axis=1)

    if index is not None:
        err = pd.Series(err, index=index)

    return err

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
    if index is not None:
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
                               confidence=0.99, mle_reproj_threshold=0.5):
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

    Returns
    -------
    F : ndarray
        A 3x3 fundamental matrix

    mask : pd.Series
           A boolean mask identifying those points that are valid.

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

    if len(kp1) == 0 or len(kp2) == 0:
        warnings.warn("F-matix computation failed. One of the keypoint args is empty. kp1:{}, kp2:{}.".format(len(kp1), len(kp2)))
        return None, None

    # OpenCV wants arrays
    try: # OpenCV < 3.4.1
        F, mask = cv2.findFundamentalMat(np.asarray(kp1),
                                         np.asarray(kp2),
                                         method_,
                                         param1=reproj_threshold,
                                         param2=confidence)
    except: # OpenCV >= 3.4.1
        F, mask = cv2.findFundamentalMat(np.asarray(kp1),
                                         np.asarray(kp2),
                                         method_,
                                         ransacReprojThreshold=reproj_threshold,
                                         confidence=confidence)
    if F is None:
        warnings.warn("F computation failed with no result. Returning None.")
        return None, None
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
        p1 = camera.camera_from_f(F)
        p = camera.idealized_camera()
        if kp1[mask].shape[0] <=12 or kp2[mask].shape[0] <=12:
            warnings.warn("Unable to apply MLE.  Not enough correspondences.  Returning with a RANSAC computed F matrix.")
            return F, mask

        # Apply Levenber-Marquardt to perform a non-linear lst. squares fit
        #  to minimize triangulation error (this is a local bundle)
        result = optimize.least_squares(camera.projection_error, p1.ravel(),
                                        args=(p, kp1[mask].T, kp2[mask].T),
                                        method='lm')

        gold_standard_p = result.x.reshape(3, 4) # SciPy Lst. Sq. requires a vector, camera is 3x4
        optimality = result.optimality
        gold_standard_f = camera_utils.crossform(gold_standard_p[:,3]).dot(gold_standard_p[:,:3])

        F = gold_standard_f

        mask = update_fundamental_mask(F, kp1, kp2,
                                       threshold=mle_reproj_threshold).values

    return F, mask
