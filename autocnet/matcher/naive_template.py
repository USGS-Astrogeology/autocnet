import cv2
from scipy.ndimage.interpolation import zoom


def pattern_match(template, image, upsampling=16, func=cv2.TM_CCOEFF_NORMED, error_check=False):
    """
    Call an arbitrary pattern matcher

    Parameters
    ----------
    template : ndarray
               The input search template used to 'query' the destination
               image

    image : ndarray
            The image or sub-image to be searched

    upsampling : int
                 The multiplier to upsample the template and image.

    func : object
           The function to be used to perform the template based matching
           Options: {cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED, cv2.TM_SQDIFF_NORMED}
           In testing the first two options perform significantly better with Apollo data.

    error_check : bool
                  If True, also apply a different matcher and test that the values
                  are not too divergent.  Default, False.

    Returns
    -------

    x : float
        The x offset

    y : float
        The y offset

    strength : float
               The strength of the correlation in the range [-1, 1].
    """

    different = {cv2.TM_SQDIFF_NORMED: cv2.TM_CCOEFF_NORMED,
                 cv2.TM_CCORR_NORMED: cv2.TM_SQDIFF_NORMED,
                 cv2.TM_CCOEFF_NORMED: cv2.TM_SQDIFF_NORMED}

    if upsampling < 1:
        raise ValueError

    u_template = zoom(template, upsampling, order=3)
    u_image = zoom(image, upsampling, order=3)

    result = cv2.matchTemplate(u_image, u_template, method=func)
    min_corr, max_corr, min_loc, max_loc = cv2.minMaxLoc(result)
    if func == cv2.TM_SQDIFF or func == cv2.TM_SQDIFF_NORMED:
        x, y = (min_loc[0], min_loc[1])
    else:
        x, y = (max_loc[0], max_loc[1])

    # Compute the idealized shift (image center)
    ideal_y = u_image.shape[0] / 2
    ideal_x = u_image.shape[1] / 2

    # Compute the shift from template upper left to template center
    y += (u_template.shape[0] / 2)
    x += (u_template.shape[1] / 2)

    x = (ideal_x - x) / upsampling
    y = (ideal_y - y) / upsampling
    return x, y, max_corr

