import numpy as np

from autocnet.matcher import naive_template
from autocnet.matcher import ciratefi


# TODO: look into KeyPoint.size and perhaps use to determine an appropriately-sized search/template.


def clip_roi(img, center, img_size):
    """
    Given an input image, clip a square region of interest
    centered on some pixel at some size.

    Parameters
    ----------
    img : ndarray or object
          The input image to be clipped or an object
          with a read_array method that takes a pixels
          argument in the form [xstart, ystart, xstop, ystop]

    center : tuple
             (y,x) coordinates to center the roi

    img_size : int
               Odd, total image size

    Returns
    -------
    clipped_img : ndarray
                  The clipped image
    """
    if img_size % 2 == 0:
        raise ValueError('Image size must be odd.')

    i = int((img_size - 1) / 2)

    x, y = map(int, center)

    y_start = y - i
    x_start = x - i
    x_stop = (x + i) - x_start
    y_stop = (y + i) - y_start

    if isinstance(img, np.ndarray):
        clipped_img = img[y_start:y_start + y_stop + 1,
                          x_start:x_start + x_stop + 1]
    else:
        clipped_img = img.read_array(pixels=[x_start, y_start,
                                             x_stop + 1, y_stop + 1])
    return clipped_img


def subpixel_offset(template, search, **kwargs):
    """
    Uses a pattern-matcher on subsets of two images determined from the passed-in keypoints and optional sizes to
    compute an x and y offset from the search keypoint to the template keypoint and an associated strength.

    Parameters
    ----------
    template : ndarray
               The template used to search

    search : ndarray
             The search image

    Returns
    -------
    x_offset : float
               Shift in the x-dimension

    y_offset : float
               Shift in the y-dimension

    strength : float
               Strength of the correspondence in the range [-1, 1]
    """

    if 'method' in kwargs.keys():
        method = kwargs['method']
        kwargs.pop('method', None)
    else:
        method = 'naive'

    functions = { 'naive' : naive_template.pattern_match,
                  'ciratefi' : ciratefi.ciratefi}

    x_offset, y_offset, strength = functions[method](template, search, **kwargs)
    return x_offset, y_offset, strength
