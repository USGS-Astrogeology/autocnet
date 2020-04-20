import json
from math import modf, floor
import numpy as np

from skimage.feature import register_translation

from skimage import transform as tf

from matplotlib import pyplot as plt

from plio.io.io_gdal import GeoDataset
from pysis.exceptions import ProcessError

from autocnet.matcher.naive_template import pattern_match, pattern_match_autoreg
from autocnet.matcher import ciratefi
from autocnet.io.db.model import Measures, Points, Images, JsonEncoder
from autocnet.graph.node import NetworkNode
from autocnet.transformation import roi
from autocnet import spatial
from autocnet.utils.utils import bytescale

import geopandas as gpd
import pandas as pd

import pvl

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
    shifts_x = np.zeros(nmatches)
    shifts_y = np.zeros(nmatches)
    strengths = np.zeros((nmatches, nstrengths))

    new_x = np.empty(nmatches)
    new_y = np.empty(nmatches)

    return shifts_x, shifts_y, strengths, new_x, new_y

def check_image_size(imagesize):
    """
    Given an x,y tuple, ensure that the values
    are odd. Used by the subpixel template to also ensure
    that the template size is the one requested and not 2x
    the template size.

    Parameters
    ----------
    imagesize : tuple
                in the form (size_x, size_y)
    """
    if isinstance(imagesize, int):
        imagesize = (imagesize, imagesize)

    x = imagesize[0]
    y = imagesize[1]

    if x % 2 == 0:
        x += 1
    if y % 2 == 0:
        y += 1
    x = floor(x/2)
    y = floor(y/2)
    return x,y

def clip_roi(img, center_x, center_y, size_x=200, size_y=200, dtype="uint64"):
    """
    Given an input image, clip a square region of interest
    centered on some pixel at some size.

    Parameters
    ----------
    img : ndarray or object
          The input image to be clipped or an object
          with a read_array method that takes a pixels
          argument in the form [xstart, ystart, xstop, ystop]

    center_x : Numeric
               The x coordinate to the center of the roi

    center_y : Numeric
               The y coordinate to the center of the roi

    img_size : int
               1/2 of the total image size. This value is the
               number of pixels grabbed from each side of the center

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
        size_y =floor(raster_size[1] - center_y)
    if ay - size_y < 0:
        size_y = int(ay)

    # Read from the upper left origin
    pixels = [ax-size_x, ay-size_y, size_x*2, size_y*2]
    pixels = list(map(int, pixels))  #
    if isinstance(img, np.ndarray):
        subarray = img[pixels[1]:pixels[1] + pixels[3] + 1, pixels[0]:pixels[0] + pixels[2] + 1]
    else:
        try:
            subarray = img.read_array(pixels=pixels, dtype=dtype)
        except:
            return None, 0, 0
    return subarray, axr, ayr


def subpixel_phase(sx, sy, dx, dy,
                   s_img, d_img,
                   image_size=(251, 251),
                   **kwargs):
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
    image_size = check_image_size(image_size)

    s_roi = roi.Roi(s_img, sx, sy, size_x=image_size[0], size_y=image_size[1])
    d_roi = roi.Roi(d_img, dx, dy, size_x=image_size[0], size_y=image_size[1])

    s_image = s_roi.clip()
    d_template = d_roi.clip()

    if s_image.shape != d_template.shape:

        s_size = s_image.shape
        d_size = d_template.shape
        updated_size_x = int(min(s_size[1], d_size[1]))
        updated_size_y = int(min(s_size[0], d_size[0]))

        # Have to subtract 1 from even entries or else the round up that
        # occurs when the size is split over the midpoint causes the
        # size to be too large by 1.
        if updated_size_x % 2 == 0:
            updated_size_x -= 1
        if updated_size_y % 2 == 0:
            updated_size_y -= 1

        # Since the image is smaller than the requested size, set the size to
        # the current maximum image size and reduce from there on potential
        # future iterations.
        size = check_image_size((updated_size_x, updated_size_y))
        s_roi = roi.Roi(s_img, sx, sy,
                        size_x=size[0], size_y=size[1])
        d_roi = roi.Roi(d_img, dx, dy,
                        size_x=size[0], size_y=size[1])
        s_image = s_roi.clip()
        d_template = d_roi.clip()

        if (s_image is None) or (d_template is None):
            return None, None, None

    (shift_y, shift_x), error, diffphase = register_translation(s_image, d_template, **kwargs)
    dx = d_roi.x - shift_x
    dy = d_roi.y - shift_y

    return dx, dy, (error, diffphase)

def subpixel_template(sx, sy, dx, dy,
                      s_img, d_img,
                      image_size=(251, 251),
                      template_size=(51,51),
                      func=pattern_match,
                      **kwargs):
    """
    Uses a pattern-matcher on subsets of two images determined from the passed-in keypoints and optional sizes to
    compute an x and y offset from the search keypoint to the template keypoint and an associated strength.

    Parameters
    ----------
    sx : Numeric
         Source X coordinate

    sy : Numeric
         Source y coordinate

    dx : Numeric
         The desintation x coordinate

    dy : Numeric
         The destination y coordinate

    s_img : GeoDataset
            The source image GeoDataset

    d_img : GeoDataset
            The destination image GeoDataset

    image_size : tuple
                 (xsize, ysize) of the image that is searched within (this should be larger
                 than the template size)

    template_size : tuple
                    (xsize, ysize) of the template to iterate over the image in order
                    to identify the area(s) of highest correlation.

    Returns
    -------
    x_shift : float
               Shift in the x-dimension

    y_shift : float
               Shift in the y-dimension

    strength : float
               Strength of the correspondence in the range [-1, 1]

    See Also
    --------
    autocnet.matcher.naive_template.pattern_match : for the kwargs that can be passed to the matcher
    autocnet.matcher.naive_template.pattern_match_autoreg : for the jwargs that can be passed to the autoreg style matcher
    """

    image_size = check_image_size(image_size)
    template_size = check_image_size(template_size)

    s_roi = roi.Roi(s_img, sx, sy, size_x=image_size[0], size_y=image_size[1])
    d_roi = roi.Roi(d_img, dx, dy, size_x=template_size[0], size_y=template_size[1])

    s_image = s_roi.clip()
    d_template = d_roi.clip()

    if (s_image is None) or (d_template is None):
        return None, None, None, None

    shift_x, shift_y, metrics, corrmap = func(d_template, s_image, **kwargs)

    dx = d_roi.x - shift_x
    dy = d_roi.y - shift_y

    return dx, dy, metrics, corrmap

def subpixel_ciratefi(sx, sy, dx, dy, s_img, d_img, search_size=251, template_size=51, **kwargs):
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
    t_roi = roi.Roi(d_img, dx, dy,
                              size_x=template_size, size_y=template_size)
    s_roi = roi.Roi(s_img, sx, sy,
                                size_x=search_size, size_y=search_size)
    template = t_roi.clip()
    search = s_roi.clip()

    if template is None or search is None:
        return None, None, None

    x_offset, y_offset, strength = ciratefi.ciratefi(template, search, **kwargs)
    dx += (x_offset + t_roi.axr)
    dy += (y_offset + t_roi.ayr)
    return dx, dy, strength

def iterative_phase(sx, sy, dx, dy, s_img, d_img, size=(51, 51), reduction=11, convergence_threshold=1.0, max_dist=50, **kwargs):
    """
    Iteratively apply a subpixel phase matcher to source (s_img) and destination (d_img)
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
    size : tuple
           Size of the template in the form (x,y)
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
        shifted_dx, shifted_dy, metrics = subpixel_phase(sx, sy, dx, dy, s_img, d_img, image_size=size, **kwargs)

        # Compute the amount of move the matcher introduced
        delta_dx = abs(shifted_dx - dx)
        delta_dy = abs(shifted_dy - dy)
        dx = shifted_dx
        dy = shifted_dy

        # Break if the solution has converged
        size = (size[0] - reduction, size[1] - reduction)
        dist = np.linalg.norm([dsample-dx, dline-dy])

        if min(size) < 1:
            return None, None, (None, None)
        if delta_dx <= convergence_threshold and\
           delta_dy<= convergence_threshold and\
           abs(dist) <= max_dist:
           break
    return dx, dy, metrics


def geom_match(base_cube, input_cube, bcenter_x, bcenter_y, size_x=60, size_y=60,
               template_kwargs={"image_size":(59,59), "template_size":(31,31)},
               phase_kwargs=None, verbose=True):

    if not isinstance(input_cube, GeoDataset):
        raise Exception("input cube must be a geodataset obj")

    if not isinstance(base_cube, GeoDataset):
        raise Exception("match cube must be a geodataset obj")

    base_startx = int(bcenter_x - size_x)
    base_starty = int(bcenter_y - size_y)
    base_stopx = int(bcenter_x + size_x)
    base_stopy = int(bcenter_y + size_y)

    image_size = input_cube.raster_size
    match_size = base_cube.raster_size

    # for now, require the entire window resides inside both cubes.
    if base_stopx > match_size[0]:
        raise Exception(f"Window: {base_stopx} > {match_size[0]}, center: {bcenter_x},{bcenter_y}")
    if base_startx < 0:
        raise Exception(f"Window: {base_startx} < 0, center: {bcenter_x},{bcenter_y}")
    if base_stopy > match_size[1]:
        raise Exception(f"Window: {base_stopy} > {match_size[1]}, center: {bcenter_x},{bcenter_y} ")
    if base_starty < 0:
        raise Exception(f"Window: {base_starty} < 0, center: {bcenter_x},{bcenter_y}")

    # specifically not putting this in a try except, because this should never fail,
    # want to throw error if there is one
    mlat, mlon = spatial.isis.image_to_ground(base_cube.file_name, bcenter_x, bcenter_y)
    center_x, center_y = spatial.isis.ground_to_image(input_cube.file_name, mlon, mlat)[::-1]

    match_points = [(base_startx,base_starty),
                    (base_startx,base_stopy),
                    (base_stopx,base_stopy),
                    (base_stopx,base_starty)]

    cube_points = []
    for x,y in match_points:
        try:
            lat, lon = spatial.isis.image_to_ground(base_cube.file_name, x, y)
            cube_points.append(spatial.isis.ground_to_image(input_cube.file_name, lon, lat)[::-1])
        except ProcessError as e:
            if 'Requested position does not project in camera model' in e.stderr:
                print(f'Skip geom_match; Region of interest corner located at ({lon}, {lat}) does not project to image {input_cube.base_name}')
                return None, None, None, None, None

    base_gcps = np.array([*match_points])
    base_gcps[:,0] -= base_startx
    base_gcps[:,1] -= base_starty

    dst_gcps = np.array([*cube_points])
    start_x = dst_gcps[:,0].min()
    start_y = dst_gcps[:,1].min()
    stop_x = dst_gcps[:,0].max()
    stop_y = dst_gcps[:,1].max()
    dst_gcps[:,0] -= start_x
    dst_gcps[:,1] -= start_y

    affine = tf.estimate_transform('affine', np.array([*base_gcps]), np.array([*dst_gcps]))

    # read_array not getting correct type by default
    isis2np_types = {
            "UnsignedByte" : "uint8",
            "SignedWord" : "int16",
            "Real" : "float64"
    }

    base_pixels = list(map(int, [match_points[0][0], match_points[0][1], size_x*2, size_y*2]))
    base_type = isis2np_types[pvl.load(base_cube.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    base_arr = base_cube.read_array(pixels=base_pixels, dtype=base_type)

    dst_pixels = list(map(int, [start_x, start_y, stop_x-start_x, stop_y-start_y]))
    dst_type = isis2np_types[pvl.load(input_cube.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    dst_arr = input_cube.read_array(pixels=dst_pixels, dtype=dst_type)

    dst_arr = tf.warp(dst_arr, affine)
    dst_arr = dst_arr[:size_y*2, :size_x*2]

    if verbose:
      fig, axs = plt.subplots(1, 2)
      axs[0].set_title("Base")
      axs[0].imshow(bytescale(base_arr), cmap="Greys_r")
      axs[1].set_title("Projected Image")
      axs[1].imshow(bytescale(dst_arr), cmap="Greys_r")
      plt.show()


    # Run through one step of template matching then one step of phase matching
    # These parameters seem to work best, should pass as kwargs later
    restemplate = subpixel_template(size_x, size_y, size_x, size_y, bytescale(base_arr), bytescale(dst_arr), **template_kwargs)

    if phase_kwargs:
        resphase = subpixel_phase(size_x, size_y, restemplate[0], restemplate[1], base_arr, dst_arr, **phase_kwargs)
        _,_,maxcorr, temp_corrmap = restemplate
        x,y,(perror, pdiff) = resphase
        if x is None or y is None:
            return None, None, None, None, None
        temp_dist = np.linalg.norm([size_x-restemplate[0], size_y-restemplate[1]])
        phase_dist = np.linalg.norm([restemplate[0]-resphase[0], restemplate[1]-resphase[1]])
        dist = (temp_dist, phase_dist)
        metric = (restemplate[2], perror, pdiff)
    else:
        x,y,maxcorr,temp_corrmap = restemplate
        if x is None or y is None:
            return None, None, None, None, None
        metric = maxcorr
        dist = np.linalg.norm([size_x/2-x, size_y/2-y])

    sample, line = affine([x,y])[0]
    sample += start_x
    line += start_y

    if verbose:
      fig, axs = plt.subplots(1, 3)
      fig.set_size_inches((30,30))
      darr = roi.Roi(input_cube.read_array(dtype=dst_type), sample, line, 100, 100).clip()
      axs[1].imshow(darr, cmap="Greys_r")
      axs[1].scatter(x=[darr.shape[1]/2], y=[darr.shape[0]/2], s=10, c="red")
      axs[1].set_title("Original Registered Image")

      axs[0].imshow(base_arr, cmap="Greys_r")
      axs[0].scatter(x=[base_arr.shape[1]/2], y=[base_arr.shape[0]/2], s=10, c="red")
      axs[0].set_title("Base")

      pcm = axs[2].imshow(temp_corrmap**2, interpolation=None, cmap="coolwarm")
      plt.show()

    # dist = np.linalg.norm([center_x-sample, center_y-line])
    return sample, line, dist, metric, temp_corrmap


def subpixel_register_measure(measureid,
                              iterative_phase_kwargs={},
                              subpixel_template_kwargs={},
                              cost_func=lambda x,y: 1/x**2 * y,
                              threshold=0.005,
                              ncg=None,
                              **kwargs):



    if isinstance(measureid, Measures):
        measureid = measureid.id

    result = {'measureid':measureid,
              'status':''}

    with ncg.session_scope() as session:
        # Setup the measure that is going to be matched
        destination = session.query(Measures).filter(Measures.id == measureid).one()
        destinationid = destination.imageid
        res = session.query(Images).filter(Images.id == destinationid).one()
        destination_node = NetworkNode(node_id=destinationid, image_path=res.path)

        # Get the point id and set up the reference measure
        pointid = destination.pointid
        source = session.query(Measures).filter(Measures.pointid==pointid).order_by(Measures.id).first()
        source.weight = 1

        sourceid = source.imageid
        res = session.query(Images).filter(Images.id == sourceid).one()
        source_node = NetworkNode(node_id=sourceid, image_path=res.path)

        new_template_x, new_template_y, template_metric, _ = subpixel_template(source.sample,
                                                                source.line,
                                                                destination.sample,
                                                                destination.line,
                                                                source_node.geodata,
                                                                destination_node.geodata,
                                                                **subpixel_template_kwargs)
        if new_template_x == None:
            destination.ignore = True # Unable to template match
            result['status'] = 'Unable to template match.'
            return result

        new_phase_x, new_phase_y, phase_metrics = iterative_phase(source.sample,
                                                                    source.line,
                                                                    new_template_x,
                                                                    new_template_y,
                                                                    source_node.geodata,
                                                                    destination_node.geodata,
                                                                    **iterative_phase_kwargs)
        if new_phase_x == None:
            destination.ignore = True # Unable to phase match
            result['status'] = 'Unable to phase match.'
            return result

        dist = np.linalg.norm([new_phase_x-new_template_x, new_phase_y-new_template_y])
        cost = cost_func(dist, template_metric)

        if cost <= threshold:
            destination.ignore = True # Threshold criteria not met
            result['status'] = 'Cost metric not met.'
            return result

        # Update the measure
        if new_phase_x:
            destination.sample = new_phase_x
            destination.line = new_phase_y
            destination.weight = cost

        # In case this is a second run, set the ignore to False if this
        # measures passed. Also, set the source measure back to ignore=False
        destination.ignore = False
        source.ignore = False
        result['status'] = 'Success.'

    return result


def subpixel_register_point(pointid, iterative_phase_kwargs={},
                            subpixel_template_kwargs={},
                            cost_func=lambda x,y: 1/x**2 * y, threshold=0.005,
                            ncg=None,
                            **kwargs):

    """
    Given some point, subpixel register all of the measures in the point to the
    first measure.

    Parameters
    ----------
    ncg : obj
          the network candidate graph that the point is associated with; used for
          the DB session that is able to access the point.

    pointid : int or obj
              The identifier of the point in the DB or a Points object

    iterative_phase_kwargs : dict
                             Any keyword arguments passed to the phase matcher

    subpixel_template_kwargs : dict
                               Ay keyword arguments passed to the template matcher

    cost : func
           A generic cost function accepting two arguments (x,y), where x is the
           distance that a point has shifted from the original, sensor identified
           intersection, and y is the correlation coefficient coming out of the
           template matcher.

    threshold : numeric
                measures with a cost <= the threshold are marked as ignore=True in
                the database.
    """
    Session = ncg.Session
    if not Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    if isinstance(pointid, Points):
        pointid = pointid.id

    with ncg.session_scope() as session:
        measures = session.query(Measures).filter(Measures.pointid == pointid).order_by(Measures.id).all()
        source = measures[0]

        source.template_metric = 1
        source.template_shift = 0
        source.phase_error = 0
        source.phase_diff = 0
        source.phase_shift = 0

        sourceid = source.imageid
        res = session.query(Images).filter(Images.id == sourceid).one()
        source_node = NetworkNode(node_id=sourceid, image_path=res.path)
        source_node.parent = ncg

        print(f'Attempting to subpixel register {len(measures)} measures for point {pointid}')

        resultlog = []
        for measure in measures[1:]:
            currentlog = {'measureid':measure.id,
                        'status':''}
            cost = None
            destinationid = measure.imageid

            res = session.query(Images).filter(Images.id == destinationid).one()
            destination_node = NetworkNode(node_id=destinationid, image_path=res.path)
            destination_node.parent = ncg

            new_x, new_y, dist, metric,  _ = geom_match(source_node.geodata, destination_node.geodata,
                                                        source.sample, source.line,
                                                        template_kwargs=subpixel_template_kwargs,
                                                        phase_kwargs=iterative_phase_kwargs, size_x=100, size_y=100)

            if new_x == None or new_y == None:
                measure.ignore = True # Unable to phase match
                currentlog['status'] = 'Failed to geom match.'
                resultlog.append(currentlog)
                continue
            # cost = cost_func(dist, template_metric)
            cost = 1

            if cost <= threshold:
                measure.ignore = True # Threshold criteria not met
                currentlog['status'] = f'Cost failed. Distance shifted: {dist}. Metric: {template_metric}.'
                resultlog.append(currentlog)
                continue

            if iterative_phase_kwargs:
                print('subpixel_register_point -> PHASE MEASURE WRITE OUT')
                measure.template_metric = metric[0]
                measure.template_shift = dist[0]
                measure.phase_error = metric[1]
                measure.phase_diff = metric[2]
                measure.phase_shift = dist[1]
            else:
                print('subpixel_register_point -> NO PHASE MEASURE WRITE OUT')
                measure.template_metric = metric
                measure.template_shift = dist

            # Update the measure
            measure.sample = new_x
            measure.line = new_y
            measure.weight = cost
            measure.choosername = 'subpixel_register_point'

            # In case this is a second run, set the ignore to False if this
            # measures passed. Also, set the source measure back to ignore=False
            measure.ignore = False
            source.ignore = False
            currentlog['status'] = f'Success.'
            resultlog.append(currentlog)

    return resultlog

def subpixel_register_points(iterative_phase_kwargs={'size': 251},
                             subpixel_template_kwargs={'image_size':(251,251)},
                             cost_kwargs={},
                             threshold=0.005,
                             Session=None):
    """
    Serial subpixel registration of all of the points in a given DB table.

    Parameters
    ----------
    Session : obj
              A SQLAlchemy Session factory.

    pointid : int
              The identifier of the point in the DB

    iterative_phase_kwargs : dict
                             Any keyword arguments passed to the phase matcher

    subpixel_template_kwargs : dict
                               Ay keyword arguments passed to the template matcher

    cost : func
           A generic cost function accepting two arguments (x,y), where x is the
           distance that a point has shifted from the original, sensor identified
           intersection, and y is the correlation coefficient coming out of the
           template matcher.

    threshold : numeric
                measures with a cost <= the threshold are marked as ignore=True in
                the database.
    """
    if not Session:
        raise BrokenPipeError('This func requires a database session.')
    session = Session()
    pointids = [point.id for point in session.query(Points)]
    session.close()
    for pointid in pointids:
        subpixel_register_point(pointid,
                                iterative_phase_kwargs=iterative_phase_kwargs,
                                subpixel_template_kwargs=subpixel_template_kwargs,
                                **cost_kwargs)

