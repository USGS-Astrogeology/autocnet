import numpy as np
import cv2


def compute_epipoles(f):
    """
    Compute the epipole and epipolar prime

    Parameters
    ----------
    f : ndarray
        (3,3) fundamental matrix or autocnet Fundamental Matrix object

    Returns
    -------
    e : ndarray
        (3,1) epipole

    e1 : ndarray
         (3,3) epipolar prime matrix
    """
    u, _, _ = np.linalg.svd(f)
    e = u[:, -1]
    e1 = np.array([[0, -e[2], e[1]],
                   [e[2], 0, -e[0]],
                   [-e[1], e[0], 0]])

    return e, e1


def idealized_camera():
    """
    Create an idealized camera transformation matrix

    Returns
    -------
     : ndarray
       (3,4) with diagonal 1
    """
    return np.eye(3, 4)


def estimated_camera_from_f(f):
    """
    Estimate a camera matrix using a fundamental matrix.


    Parameters
    ----------
    f : ndarray
        (3,3) fundamental matrix or autocnet Fundamental Matrix object

    Returns
    -------
    p1 : ndarray
         Estimated camera matrix
    """

    e, e1 = compute_epipoles(f)
    p1 = np.empty((3, 4))
    p1[:, :3] = e1.dot(f)
    p1[:, 3] = e

    return p1


def triangulate(pt, pt1, p, p1):
    """
    Given two sets of homogeneous coordinates and two camera matrices,
    triangulate the 3D coordinates.  The image correspondences are
    assumed to be implicitly ordered.

    References
    ----------
    .. [Hartley2003]

    Parameters
    ----------
    pt : ndarray
         (n, 3) array of homogeneous correspondences

    pt1 : ndarray
          (n, 3) array of homogeneous correspondences

    p : ndarray
        (3, 4) camera matrix

    p1 : ndarray
         (3, 4) camera matrix

    Returns
    -------
    coords : ndarray
             (4, n) projection matrix

    """
    pt = np.asarray(pt)
    pt1 = np.asarray(pt1)

    # Transpose for the openCV call if needed
    if pt.shape[0] != 3:
        pt = pt.T
    if pt1.shape[0] != 3:
        pt1 = pt1.T

    X = cv2.triangulatePoints(p, p1, pt[:2], pt1[:2])

    # Homogenize
    X /= X[3]

    return X


def projection_error(p1, p, pt, pt1):
    """
    Based on Hartley and Zisserman p.285 this function triangulates
    image correspondences and computes the reprojection error
    by back-projecting the points into the image.

    References
    ----------
    .. [Hartley2003]

    Parameters
    -----------
    p1 : ndarray
         (3,4) camera matrix

    p : ndarray
        (3,4) idealized camera matrix in the form np.eye(3,4)

    pt : dataframe or ndarray
         of homogeneous coordinates in the form (x_{i}, y_{i}, 1)

    pt1 : dataframe or ndarray
          of homogeneous coordinates in the form (x_{i}, y_{i}, 1)

    Returns
    -------
    residuals : ndarray
                (n, 1) residuals for each correspondence

    cumulative_error : float
                       sum of the residuals


    """
    if p1.shape != (3,4):
        p1 = p1.reshape(3,4)

    # Triangulate the correspondences
    xw_est = triangulate(pt, pt1, p, p1)

    # Back project and homogenize
    xhat = np.dot(p, xw_est)
    xhat /= xhat[2]
    x2hat = np.dot(p1, xw_est)
    x2hat /= x2hat[2]

    # Compute residuals
    dist = (pt.T - xhat)**2 + (pt1.T - x2hat)**2
    residuals = np.sum(dist, axis=0)
    reproj_error = np.sum(dist)

    return residuals, reproj_error
