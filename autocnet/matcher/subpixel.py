from math import modf, floor
import numpy as np

from skimage.feature import register_translation

from autocnet.matcher import naive_template
from autocnet.matcher import ciratefi

import geopandas as gpd
import pandas as pd

# TODO: look into KeyPoint.size and perhaps use to determine an appropriately-sized search/template.
def _prep_subpixel(nmatches, nstrengths=2):
    """
    Setup the data strutures to return for subpixel matching.

    Parameters
    ----------
    nmatches : int
                The number of pixels to be subpixel matches

    nstrengths : int
                    The number of 'strength' values to be returned
                    by the subpixel matching method.

    Returns
    -------
    shifts_x : ndarray
               (nmatches, 1) to store the x_shift parameter

    shifts_y : ndarray
               (nmatches, 1) to store the y_shift parameter

    strengths : ndarray
                (nmatches, nstrengths) to store the strengths for each point

    new_x : ndarray
            (nmatches, 1) to store the updated x coordinates

    new_y : ndarray
            (nmatches, 1) to store the updated y coordinates
    """
    # Setup to store output to append to dataframes
    shifts_x = np.empty(nmatches)
    shifts_x[:] = np.nan
    shifts_y = np.empty(nmatches)
    shifts_y[:] = np.nan
    strengths = np.empty((nmatches, nstrengths))
    strengths[:] = np.nan

    new_x = np.empty(nmatches)
    new_y = np.empty(nmatches)

    return shifts_x, shifts_y, strengths, new_x, new_y

def clip_roi(img, center_x, center_y, size_x=200, size_y=200):
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
             (x,y) coordinates to center the roi

    img_size : int
               Odd, total image size

    Returns
    -------
    clipped_img : ndarray
                  The clipped image
    """

    try:
        raster_size = img.raster_size
    except:
        # x,y form
        raster_size = img.shape[::-1]
    axr, ax = modf(center_x)
    ayr, ay = modf(center_y)

    if ax + size_x > raster_size[0]:
        size_x = floor(raster_size[0] - center_x)
    if ax - size_x < 0:
        size_x = int(ax)
    if ay + size_y > raster_size[1]:
        size_y = floor(raster_size[1] - center_y)
    if ay - size_y < 0:
        size_y = int(ay)

    # Read from the upper left origin
    pixels=(int(ax-size_x), int(ay-size_y), size_x * 2, size_y * 2)
    if isinstance(img, np.ndarray):
        subarray = img[pixels[1]:pixels[1] + pixels[3] + 1, pixels[0]:pixels[0] + pixels[2] + 1]
    else:
        try:
            subarray = img.read_array(pixels=pixels)
        except:
            return None, 0, 0
    return subarray, axr, ayr

def subpixel_phase(template, search, **kwargs):
    """
    Apply the spectral domain matcher to a search and template image. To
    shift the images, the x_shift and y_shift, need to be subtracted from
    the center of the search image. It may also be necessary to apply the
    fractional pixel adjustment as well (if for example the center of the
    search is not an integer); this function do not manage shifting.

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

    strength : tuple
               With the RMSE error and absolute difference in phase
    """
    if not template.shape == search.shape:
        raise ValueError('Both the template and search images must be the same shape.')
    (y_shift, x_shift), error, diffphase = register_translation(search, template, **kwargs)
    return x_shift, y_shift, (error, diffphase)

def subpixel_template(sx, sy, dx, dy, s_img, d_img, search_size=251, template_size=51, **kwargs):
    """
    Uses a pattern-matcher on subsets of two images determined from the passed-in keypoints and optional sizes to
    compute an x and y offset from the search keypoint to the template keypoint and an associated strength.

    Parameters
    ----------
    sx : numeric
         The x position of the center of the template to be matched to
    sy : numeric
         The y position of the center of the template to be matched to
    dx : numeric
         The x position of the center of the search to be matched from
    dy : numeric
         The y position of the center of the search to be matched to
    s_img : object
            A plio geodata object from which the template is extracted
    d_img : object
            A plio geodata object from which the search is extracted
    search_size : int
                  An odd integer for the size of the search image
    template_size : int
                    A odd integer for the size of the template that is iterated
                    over the search images

    Returns
    -------
    x_offset : float
               Shift in the x-dimension

    y_offset : float
               Shift in the y-dimension

    strength : float
               Strength of the correspondence in the range [-1, 1]
    """

    template, _, _ = clip_roi(d_img, dx, dy,
                              size_x=template_size, size_y=template_size)
    search, dxr, dyr = clip_roi(s_img, sx, sy,
                                size_x=search_size, size_y=search_size)
    if template is None or search is None:
        return None, None, None
    if 'method' in kwargs.keys():
        method = kwargs['method']
        kwargs.pop('method', None)
    else:
        method = 'naive'

    functions = { 'naive' : naive_template.pattern_match,
                  'ciratefi' : ciratefi.ciratefi}
    x_offset, y_offset, strength = functions[method](template, search, **kwargs)
    dx += (x_offset + dxr)
    dy += (y_offset + dyr)
    return dx, dy, strength

def iterative_phase(sx, sy, dx, dy, s_img, d_img, size=251, reduction=11, convergence_threshold=1.0, max_dist=50, **kwargs):
    """
    Iteratively apply a subpixel phase matcher to source (s_img) amd destination (d_img)
    images. The size parameter is used to set the initial search space. The algorithm
    is recursively applied to reduce the total search space by reduction until the convergence criteria
    are met. Convergence is defined as the point at which the computed shifts (x_shift,y_shift) are
    less than the convergence_threshold. In instances where the size is reducted to 1 pixel the
    algorithm terminates and returns None.

    Parameters
    ----------
    sx : numeric
         The x position of the center of the template to be matched to
    sy : numeric
         The y position of the center of the template to be matched to
    dx : numeric
         The x position of the center of the search to be matched from
    dy : numeric
         The y position of the center of the search to be matched to
    s_img : object
            A plio geodata object from which the template is extracted
    d_img : object
            A plio geodata object from which the search is extracted
    size : int
           One half of the total size of the template, so a 251 default results in a 502 pixel search space
    reduction : int
                With each recursive call to this func, the size is reduced by this amount
    convergence_threshold : float
                            The value under which the result can shift in the x and y directions to force a break

    Returns
    -------
    dx : float
         The new x value for the match in the destination (d) image
    dy : float
         The new y value for the match in the destination (d) image
    metrics : tuple
              A tuple of metrics. In the case of the phase matcher this are difference
              and RMSE in the phase dimension.

    See Also
    --------
    subpixel_phase : the function that applies a single iteration of the phase matcher
    """

    # get initial destination location
    dsample = dx
    dline = dy
    while True:
        s_template, _, _ = clip_roi(s_img, sx, sy,
                                   size_x=size, size_y=size)
        d_search, dxr, dyr = clip_roi(d_img, dx, dy,
                                 size_x=size, size_y=size)

        if (s_template is None) or (d_search is None):
            return None, None, None
        if s_template.shape != d_search.shape:
            s_size = s_template.shape
            d_size = d_search.shape
            updated_size = int(min(s_size + d_size) / 2)
            # Since the image is smaller than the requested size, set the size to
            # the current maximum image size and reduce from there on potential
            # future iterations.
            size = updated_size
            s_template, _, _ = clip_roi(s_template, sx, sy,
                                 size_x=updated_size, size_y=updated_size)
            d_search, dxr, dyr = clip_roi(d_search, dx, dy,
                                size_x=updated_size, size_y=updated_size)
            if (s_template is None) or (d_search is None):
                return None, None, None

        # Apply the phase matcher
        try:
            shift_x, shift_y, metrics = subpixel_phase(s_template, d_search,**kwargs)
        except:
            return None, None, None
        # Apply the shift to d_search and compute the new correspondence location
        dx += (shift_x + dxr)
        dy += (shift_y + dyr)

        # Break if the solution has converged
        size -= reduction
        dist = np.linalg.norm([dsample-dx, dline-dy])
        if size <1:
            return None, None, None
        if abs(shift_x) <= convergence_threshold and\
           abs(shift_y) <= convergence_threshold and\
           abs(dist) <= max_dist:
            break
    return dx, dy, metrics

