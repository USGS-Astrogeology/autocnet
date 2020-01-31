import json
from math import modf, floor
import numpy as np

from skimage.feature import register_translation
from redis import StrictRedis
from plurmy import Slurm

from autocnet import Session, config
from autocnet.matcher import naive_template
from autocnet.matcher import ciratefi
from autocnet.io.db.model import Measures, Points, Images, JsonEncoder
from autocnet.graph.node import NetworkNode

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
    x = imagesize[0] / 2
    y = imagesize[1] / 2
    if x % 2 == 0:
        x += 1
    if y % 2 == 0:
        y += 1
    return x,y

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

    Returnsslurm-2235260_89.out.2235260_89.out
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

def subpixel_template(sx, sy, dx, dy, s_img, d_img, image_size=(251, 251), template_size=(51,51), **kwargs):
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
    """

    image_size = check_image_size(image_size)
    template_size = check_image_size(template_size)

    s_image, _, _ = clip_roi(s_img, sx, sy, size_x=image_size[0], size_y=image_size[1])
    d_template, dxr, dyr = clip_roi(d_img, dx, dy, size_x=template_size[0], size_y=template_size[1])

    if (s_image is None) or (d_template is None):
        return None, None, None

    shift_x, shift_y, metrics = naive_template.pattern_match(d_template, s_image, **kwargs)

    dx = (dx - shift_x + dxr)
    dy = (dy - shift_y + dyr)

    return dx, dy, metrics

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
    template, _, _ = clip_roi(d_img, dx, dy,
                              size_x=template_size, size_y=template_size)
    search, dxr, dyr = clip_roi(s_img, sx, sy,
                                size_x=search_size, size_y=search_size)
    if template is None or search is None:
        return None, None, None

    x_offset, y_offset, strength = ciratefi.ciratefi(template, search, **kwargs)
    dx += (x_offset + dxr)
    dy += (y_offset + dyr)
    return dx, dy, strength

def iterative_phase(sx, sy, dx, dy, s_img, d_img, size=251, reduction=11, convergence_threshold=1.0, max_dist=50, **kwargs):
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
    size : int, tuple
           One half of the total size of the template, so a 251 default results in a 502 pixel search space.
           If an int, the template is square. If a tuple, in the form (x,y), is passed an
           irregularly shaped template can be used.
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
    if isinstance(size, int):
        size = (size, size)
    while True:
        s_template, _, _ = clip_roi(s_img, sx, sy,
                                   size_x=size[0], size_y=size[1])
        d_search, dxr, dyr = clip_roi(d_img, dx, dy,
                                 size_x=size[0], size_y=size[1])

        if (s_template is None) or (d_search is None):
            return None, None, None
        if s_template.shape != d_search.shape:
            s_size = s_template.shape
            d_size = d_search.shape
            updated_size_x = int(min(s_size[1], d_size[1]))  # Why is this /2?
            updated_size_y = int(min(s_size[0], d_size[0]))
            # Since the image is smaller than the requested size, set the size to
            # the current maximum image size and reduce from there on potential
            # future iterations.
            size = (updated_size_x, updated_size_y)
            s_template, _, _ = clip_roi(s_template, sx, sy,
                                 size_x=size[0], size_y=size[1])
            d_search, dxr, dyr = clip_roi(d_search, dx, dy,
                                size_x=size[0], size_y=size[1])
            if (s_template is None) or (d_search is None):
                return None, None, None

        # Apply the phase matcher
        try:
            shift_x, shift_y, metrics = subpixel_phase(s_template, d_search, **kwargs)
        except:
            return None, None, None
        # Apply the shift to d_search and compute the new correspondence location
        dx += shift_x  # The implementation already applies the dxr, dyr shifts
        dy += shift_y 

        # Break if the solution has converged
        size = (size[0] - reduction, size[1] - reduction)

        dist = np.linalg.norm([dsample-dx, dline-dy])
        if min(size) < 1:
            return None, None, None
        if abs(shift_x) <= convergence_threshold and\
           abs(shift_y) <= convergence_threshold and\
           abs(dist) <= max_dist:
            break
    return dx, dy, metrics

def subpixel_register_point(pointid, iterative_phase_kwargs={}, subpixel_template_kwargs={},
                            cost_func=lambda x,y: 1/x**2 * y, threshold=0.005):

    """
    Given some point, subpixel register all of the measures in the point to the
    first measure.

    Parameters
    ----------
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
    session = Session()
    measures = session.query(Measures).filter(Measures.pointid == pointid).order_by(Measures.id).all()
    source = measures[0]

    sourceid = source.imageid
    res = session.query(Images).filter(Images.id == sourceid).one()
    source_node = NetworkNode(node_id=sourceid, image_path=res.path)

    for measure in measures[1:]:
        cost = None
        destinationid = measure.imageid

        res = session.query(Images).filter(Images.id == destinationid).one()
        destination_node = NetworkNode(node_id=destinationid, image_path=res.path)

        new_phase_x, new_phase_y, phase_metrics = iterative_phase(source.sample,
                                                                source.line,
                                                                measure.sample,
                                                                measure.line,
                                                                source_node.geodata,
                                                                destination_node.geodata,
                                                                **iterative_phase_kwargs)
        if new_phase_x == None:
            measure.ignore = True # Unable to phase match
            continue

        new_template_x, new_template_y, template_metric = subpixel_template(source.sample,
                                                                source.line,
                                                                new_phase_x,
                                                                new_phase_y,
                                                                source_node.geodata,
                                                                destination_node.geodata,
                                                                **subpixel_template_kwargs)
        if new_template_x == None:
            measure.ignore = True # Unable to template match
            continue

        dist = np.linalg.norm([new_phase_x-new_template_x, new_phase_y-new_template_y])
        cost = cost_func(dist, template_metric)

        if cost <= threshold:
            measure.ignore = True # Threshold criteria not met
            continue

        # Update the measure
        if new_template_x:
            measure.sample = new_template_x
            measure.line = new_template_y
            measure.weight = cost

        # In case this is a second run, set the ignore to False if this
        # measures passed. Also, set the source measure back to ignore=False
        measure.ignore = False
        source.ignore = False

    session.commit()
    session.close()

def subpixel_register_points(iterative_phase_kwargs={'size': 251},
                             subpixel_template_kwargs={'image_size':(251,251)},
                             cost_kwargs={},
                             threshold=0.005):
    """
    Serial subpixel registration of all of the points in a given DB table.

    Parameters
    ----------
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
    session = Session()
    pointids = [point.id for point in session.query(Points)]
    session.close()
    for pointid in pointids:
        subpixel_register_point(pointid,
                                iterative_phase_kwargs=iterative_phase_kwargs,
                                subpixel_template_kwargs=subpixel_template_kwargs,
                                **cost_kwargs)

def cluster_subpixel_register_points(iterative_phase_kwargs={'size': 251},
                                     subpixel_template_kwargs={'image_size':(251,251)},
                                     cost_kwargs={},
                                     threshold=0.005,
                                     filters={},
                                     walltime='00:10:00',
                                     chunksize=1000,
                                     exclude=None):
    """
    Distributed subpixel registration of all of the points in a given DB table.


    Parameters
    ----------
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
    filters : dict
              with keys equal to attributes of the Points mapping and values
              equal to some criteria.
    exclude : str
              string containing the name(s) of any slurm nodes to exclude when
              completing a cluster job. (e.g.: 'gpu1' or 'gpu1,neb12')
    """
    # Setup the redis queue
    rqueue = StrictRedis(host=config['redis']['host'],
                        port=config['redis']['port'],
                        db=0)

    # Push the job messages onto the queue
    queuename = config['redis']['processing_queue']

    session = Session()
    query = session.query(Points)
    for attr, value in filters.items():
        query = query.filter(getattr(Points, attr)==value)
    res = query.all()
    for i, point in enumerate(res):
        msg = {'id' : point.id,
               'iterative_phase_kwargs' : iterative_phase_kwargs,
               'subpixel_template_kwargs' : subpixel_template_kwargs,
               'threshold':threshold,
               'cost_kwargs': cost_kwargs,
               'walltime' : walltime}
        rqueue.rpush(queuename, json.dumps(msg, cls=JsonEncoder))
    session.close()

    job_counter = i + 1

    # Submit the jobs
    submitter = Slurm('acn_subpixel',
                 job_name='subpixel_register_points',
                 mem_per_cpu=config['cluster']['processing_memory'],
                 time=walltime,
                 partition=config['cluster']['queue'],
                 output=config['cluster']['cluster_log_dir']+f'/autocnet.subpixel_register-%j')
    submitter.submit(array='1-{}'.format(job_counter), chunksize=chunksize, exclude=exclude)
    return job_counter
