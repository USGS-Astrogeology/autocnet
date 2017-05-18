import numpy as np
from autocnet.camera.utils import crossform
try:
    import cv2
except:
    cv2 = None

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
    e1 = crossform(e)

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
    #if cv2:
    X = cv2.triangulatePoints(p, p1, pt[:2], pt1[:2])
    X /= X[3] # Homogenize
    return X
    """
    # Stubbed in for a ticket addressing making OpenCV an optional dependency
    else:
        npts = len(pt)
        a = np.zeros((4, 4))
        coords = np.empty((npts, 4))
        coords[:] = 1
        for i in range(npts):
            # Compute AX = 0
            a[0] = pt[i][0] * p[2] - p[0]
            a[1] = pt[i][1] * p[2] - p[1]
            a[2] = pt1[i][0] * p1[2] - p1[0]
            a[3] = pt1[i][1] * p1[2] - p1[1]
            # v.T is a least squares solution that minimizes the error residual
            u, s, vh = np.linalg.svd(a)
            v = vh.T
            coords[i] = v[:,3] / (v[:,3][-1])
        return coords.T
    """
def projection_error(p1, p, pt, pt1):
    """
    Based on Hartley and Zisserman p.285 this function triangulates
    image correspondences and computes the reprojection error
    by back-projecting the points into the image.

    This is the classic cost function (minimization problem) into
    the gold standard method for fundamental matrix estimation.

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
    reproj_error : ndarray
                   (n, 1) vector of reprojection errors


    """
    # SciPy least squares solver needs a vector, so reshape back to a 3x4 c
    # camera matrix at each iteration

    if p1.shape != (3,4):
        p1 = p1.reshape(3,4)

    # Triangulate the correspondences
    xhat = triangulate(pt, pt1, p, p1)
    xhat1 = xhat[:3] / xhat[2]
    xhat2 = p1.dot(xhat)
    xhat2 /= xhat2[2]

    # Compute error
    cost = (pt - xhat1)**2 + (pt1 - xhat2)**2
    cost = np.sqrt(np.sum(cost, axis=0))

    return cost
