import numpy as np
from scipy.stats import pearsonr

RADIAL_SIZE = 720
RADIAL_STEP = 2 * np.pi / RADIAL_SIZE
THETAS = np.round(np.arange(0, 2 * np.pi, RADIAL_STEP), 5)

def cart2polar(x, y):
    theta = np.arctan2(y, x)
    return theta

def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

def reproject_image_into_polar(data, origin=None):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the theta coords will be
    x, y = index_coords(data, origin=origin)
    theta = cart2polar(x, y)
    # -180 to 180 conversion to 0 to 360
    theta[theta < 0] += 2 * np.pi
    return theta

def coupled_decomposition(sdata, ddata, sorigin=(), dorigin=(), M=4, sub_skp=None):
    """

    Apply coupled decomposition to two 2d images.

    sdata : ndarray
            (n,m) array of values to decompose

    ddata : ndarray
            (j,k) array of values to decompose

    sorigin : tuple
              in the form (x,y)

    dorigin : tuple
              in the form (x,y)

    """

    soriginx, soriginy = sorigin
    doriginx, doriginy = dorigin

    # Create membership arrays for each input image
    smembership = np.ones(sdata.shape)
    dmembership = np.ones(ddata.shape)

    # Project the image into a polar coordinate system centered on p_{1}
    stheta = reproject_image_into_polar(sdata, origin=(int(soriginx), int(soriginy)))
    dtheta = reproject_image_into_polar(ddata, origin=(int(doriginx), int(doriginy)))

    # Compute the mean profiles for each radial slice
    smean = np.empty(RADIAL_SIZE)
    dmean = np.empty(RADIAL_SIZE)
    for i, t in enumerate(THETAS):
        # The way this method words, it is possible to get nan values in some of the steps as this is discrete
        smean[i] = np.mean(sdata[(t <= stheta) & (stheta <= t + RADIAL_STEP)])
        dmean[i] = np.mean(ddata[(t <= dtheta) & (dtheta <= t + RADIAL_STEP)])

    # Rotate the second image around the origin and compute the correlation coeff. for each 0.5 degree rotation.
    maxp = -1
    maxidx = 0
    for j in range(RADIAL_SIZE):
        dsearch = np.concatenate((dmean[j:], dmean[:j]))
        r, p = pearsonr(smean, dsearch)
        if r >= maxp:
            maxp = r
            maxidx = j

    # Maximum correlation (theta) defines the angle of rotation for the destination image
    theta = THETAS[maxidx]

    if theta <= np.pi:
        lam = theta
    else:
        lam = 2 * np.pi - theta

    # Classify the sub-images based on the decomposition size (M) and theta
    breaks = np.linspace(0, 2 * np.pi, M + 1)
    for i, t in enumerate(breaks[:-1]):
        smembership[(t <= stheta) & ( stheta <= breaks[i+1])] = i

    for i, t in enumerate(breaks[:-1]):
        # Handle the boundary crossers
        start_theta = t + theta
        stop_theta = breaks[i + 1] + theta

        if stop_theta > 2 * np.pi:
            stop_theta -= 2 * np.pi
        if start_theta > 2 * np.pi:
            start_theta -= 2 * np.pi

        if start_theta > stop_theta:
            # Handles the case where theta is a negative rotation
            dmembership[(start_theta <= dtheta) & (dtheta <= 2 * np.pi)] = i
            dmembership[(0 <= dtheta) * dtheta <= stop_theta + lam] = i
            dmembership[(start_theta <= dtheta) & (dtheta <= stop_theta)] = i
        else:
            # Handles the standard case without boundary crossers
            dmembership[(start_theta <= dtheta) & (dtheta <= stop_theta)] = i

    return smembership, dmembership
