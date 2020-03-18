from skimage import transform as tf
from shapely.geometry import MultiPoint
from plio.io.io_gdal import GeoDataset
import numpy as np
import matplotlib.pyplot as plt

import ctypes
import enum
import glob
import json
import os
import os.path
import socket
from ctypes.util import find_library

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from scipy.misc import imresize
from sqlalchemy import (Boolean, Column, Float, ForeignKey, Integer,
                        LargeBinary, String, UniqueConstraint, create_engine,
                        event, orm, pool)
from sqlalchemy.ext.declarative import declarative_base

import geopandas as gpd
import plio
import pvl
import pyproj
import pysis
import cv2

from gdal import ogr

import geoalchemy2
from geoalchemy2 import Geometry, WKTElement
from geoalchemy2.shape import to_shape
from geoalchemy2 import functions

from knoten import csm

from plio.io.io_controlnetwork import from_isis, to_isis
from plio.io.io_gdal import GeoDataset

from pysis.exceptions import ProcessError
from pysis.isis import campt

from shapely import wkt
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry import Point

from redis import StrictRedis

from plurmy import Slurm

from autocnet import config, dem, engine, Session
from autocnet.io.db.model import Images, Points, Measures, JsonEncoder
from autocnet.graph.network import NetworkCandidateGraph
from autocnet.matcher.subpixel import iterative_phase, subpixel_template, clip_roi
from autocnet.cg.cg import distribute_points_in_geom
from autocnet.io.db.connection import new_connection
from autocnet.spatial import isis
from autocnet.utils.utils import bytescale
from autocnet.matcher.cpu_extractor import extract_most_interesting
from autocnet import spatial
from autocnet.transformation.spatial import reproject

import warnings




def generate_ground_points(ground_mosaic, nspts_func=lambda x: int(round(x,1)*1), ewpts_func=lambda x: int(round(x,1)*4)):
    """


    Parameters
    ----------
    ground_db_config : dict
                       In the form: {'username':'somename',
                                     'password':'somepassword',
                                     'host':'somehost',
                                     'pgbouncer_port':6543,
                                     'name':'somename'}
    nspts_func       : func
                       describes distribution of points along the north-south
                       edge of an overlap.

    ewpts_func       : func
                       describes distribution of points along the east-west
                       edge of an overlap.
    """

    if isinstance(ground_mosaic, str):
        ground_mosaic = GeoDataset(ground_mosaic)

    warnings.warn('This function is not well tested. No tests currently exists \
    in the test suite for this version of the function.')

    session = Session()
    fp_poly = wkt.loads(session.query(functions.ST_AsText(functions.ST_Union(Images.geom))).one()[0])
    session.close()

    fp_poly_bounds = list(fp_poly.bounds)

    # just hard code queries to the mars database as it exists for now

    coords = distribute_points_in_geom(fp_poly, nspts_func=nspts_func, ewpts_func=ewpts_func, method="new")
    coords = np.asarray(coords)

    records = []
    coord_list = []
    lines = []
    samples = []

    # throw out points not intersecting the ground reference images
    for i, coord in enumerate(coords):
        # res = ground_session.execute(formated_sql)
        p = Point(*coord)

        linessamples = isis.point_info(ground_mosaic.file_name, p.x, p.y, 'ground')
        sample = linessamples.get('Sample')
        line = linessamples.get('Line')
        size = 200
        image, _, _ = clip_roi(ground_mosaic, sample, line, size_x=size, size_y=size, dtype="uint64")
        interesting = extract_most_interesting(bytescale(image),  extractor_parameters={'nfeatures':30})

        # kps are in the image space with upper left origin, so convert to
        # center origin and then convert back into full image space
        newsample = sample + (interesting.x - size)
        newline = line + (interesting.y - size)

        newpoint = isis.point_info(ground_mosaic.file_name, newsample, newline, 'image')
        p = Point(newpoint.get('PositiveEast360Longitude'),
                  newpoint.get('PlanetocentricLatitude'))

        coord_list.append(p)
        lines.append(newline)
        samples.append(newsample)


    # start building the cnet
    ground_cnet = pd.DataFrame()
    ground_cnet["path"] = [ground_mosaic.file_name]*len(coord_list)
    ground_cnet["pointid"] = list(range(len(coord_list)))
    ground_cnet["point"] = coord_list
    ground_cnet['line'] = lines
    ground_cnet['sample'] = samples

    ground_cnet = gpd.GeoDataFrame(ground_cnet, geometry='point')
    return ground_cnet, fp_poly, coord_list


def propagate_point(lon, lat, pointid, paths, lines, samples, verbose=False):
    """

    """
    images = gpd.GeoDataFrame.from_postgis(f"select * from images where ST_Intersects(geom, ST_SetSRID(ST_Point({lon}, {lat}), {config['spatial']['latitudinal_srid']}))", engine, geom_col="geom")

    image_measures = pd.DataFrame(zip(paths, lines, samples), columns=["path", "line", "sample"])
    measure = image_measures.iloc[0]

    p = Point(lon, lat)
    new_measures = []

    # lazily iterate for now
    for i,image in images.iterrows():
        dest_image = GeoDataset(image["path"])

        # list of matching results in the format:
        # [measure_index, x_offset, y_offset, offset_magnitude]
        match_results = []
        for k,m in image_measures.iterrows():
            base_image = GeoDataset(m["path"])

            sx, sy = m["sample"], m["line"]

            try:
                x,y, dist, metrics, corrmap = geom_match(base_image, dest_image, sx, sy, verbose=verbose)
            except Exception as e:
                raise Exception(e)
                match_results.append(e)
                continue

            match_results.append([k, x, y,
                                     metrics, dist, corrmap, m["path"], image["path"]])

        # get best offsets, if possible we need better metric for what a
        # good match looks like
        match_results = np.asarray([res for res in match_results if isinstance(res, list)])
        if match_results.shape[0] == 0:
            # no matches
            continue

        best_results = match_results[np.argwhere(match_results[:,3] == match_results[:,3].max())][0][0]

        # apply offsets
        sample = best_results[1]
        line = best_results[2]

        if verbose:
          print("Full results: ", best_results)
          print("Winning CORR: ", best_results[3], "Themis Pixel shift: ", best_results[4])
          print("Themis Image: ", best_results[6], "CTX image:", best_results[7])
          print("Themis S,L: ", f"{sx},{sy}", "CTX S,L: ", f"{sample},{line}")

        # hardcoded for now
        if best_results[3] == None or best_results[3] < 0.7:
            continue

        if dem is None:
            height = 0
        else:
            px, py = dem.latlon_to_pixel(lat, lon)
            height = dem.read_array(1, [px, py, 1, 1])[0][0]

        semi_major = config['spatial']['semimajor_rad']
        semi_minor = config['spatial']['semiminor_rad']
        print("HEIGHT: ", height)
        # The CSM conversion makes the LLA/ECEF conversion explicit
        x, y, z = reproject([lon, lat, height],
                         semi_major, semi_minor,
                         'latlon', 'geocent')
        print(x,y,z)

        new_measures.append({
                'pointid' : pointid,
                'imageid' : image['id'],
                'serial' : image['serial'],
                'line' : line,
                'sample' : sample,
                'point_latlon' : p,
                'point_ground' : Point(x*1000, y*1000, z*1000)
        })

    return new_measures

def cluster_propagate_control_network(base_cnet, walltime='00:20:00', chunksize=1000, exclude=None):
    warnings.warn('This function is not well tested. No tests currently exists \
    in the test suite for this version of the function.')

    # Setup the redis queue
    rqueue = StrictRedis(host=config['redis']['host'],
                         port=config['redis']['port'],
                         db=0)

    # Push the job messages onto the queue
    queuename = config['redis']['processing_queue']

    groups = base_cnet.groupby('pointid').groups
    for cpoint, indices in groups.items():
        measures = base_cnet.loc[indices]
        measure = measures.iloc[0]

        p = measure.point

        # get image in the destination that overlap
        lon, lat = measures["point"].iloc[0].xy
        msg = {'lon' : lon[0],
               'lat' : lat[0],
               'pointid' : cpoint,
               'paths' : measures['path'].tolist(),
               'lines' : measures['line'].tolist(),
               'samples' : measures['sample'].tolist(),
               'walltime' : walltime}
        rqueue.rpush(queuename, json.dumps(msg, cls=JsonEncoder))

    # Submit the jobs
    submitter = Slurm('acn_propagate',
                 job_name='cross_instrument_matcher',
                 mem_per_cpu=config['cluster']['processing_memory'],
                 time=walltime,
                 partition=config['cluster']['queue'],
                 output=config['cluster']['cluster_log_dir']+'/autocnet.cim-%j')
    job_counter = len(groups.items())
    submitter.submit(array='1-{}'.format(job_counter))
    return job_counter

def propagate_control_network(base_cnet, verbose=False):
    """

    """
    warnings.warn('This function is not well tested. No tests currently exists \
    in the test suite for this version of the function.')

    groups = base_cnet.groupby('pointid').groups

    # append CNET info into structured Python list
    constrained_net = []

    # easily parrallelized on the cpoint level, dummy serial for now
    for cpoint, indices in groups.items():
        measures = base_cnet.loc[indices]
        measure = measures.iloc[0]

        p = measure.point

        # get image in the destination that overlap
        lon, lat = measures["point"].iloc[0].xy
        gp_measures = propagate_point(lon[0], lat[0], cpoint, measures["path"], measures["line"], measures["sample"], verbose=verbose)
        constrained_net.extend(gp_measures)

    ground = gpd.GeoDataFrame.from_dict(constrained_net).set_geometry('point_latlon')
    groundpoints = ground.groupby('pointid').groups

    points = []

    # upload new points
    for p,indices in groundpoints.items():
        point = ground.loc[indices].iloc[0]
        p = Points()
        p.pointtype = 3
        p.apriori = point['point_ground']
        p.adjusted = point['point_ground']

        for i in indices:
            m = ground.loc[i]
            p.measures.append(Measures(line=float(m['line']),
                                       sample = float(m['sample']),
                                       aprioriline = float(m['line']),
                                       apriorisample = float(m['sample']),
                                       imageid = int(m['imageid']),
                                       serial = m['serial'],
                                       measuretype=3))
        points.append(p)

    session = Session()
    session.add_all(points)
    session.commit()

    return ground


def geom_match(base_cube, input_cube, bcenter_x, bcenter_y, size_x=60, size_y=60,
               template_kwargs={"image_size":(59,59), "template_size":(31,31)},
               phase_kwargs=None, verbose=False):

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

    mlat, mlon = spatial.isis.image_to_ground(base_cube.file_name, bcenter_x, bcenter_y)
    center_x, center_y = spatial.isis.ground_to_image(base_cube.file_name, mlon, mlat)

    match_points = [(base_startx,base_starty),
                    (base_startx,base_stopy),
                    (base_stopx,base_stopy),
                    (base_stopx,base_starty)]

    cube_points = []
    for x,y in match_points:
#         try:
        lat, lon = spatial.isis.image_to_ground(base_cube.file_name, x, y)
        cube_points.append(spatial.isis.ground_to_image(input_cube.file_name, lon, lat)[::-1])
#         except Exception as e:
#             if verbose:
#                 print("Match Failed with: ", e)
#             return None, None, None, None, None
    for x,y in cube_points:
        if x < 0 or y < 0:
            print("projected off cube")
            return None, None, None, None, None

    print(bcenter_x, bcenter_y, size_x, size_y)
    print(input_cube.file_name)
    print(match_points)
    print(cube_points)

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
    print(base_gcps)
    print(dst_gcps)
    affine = tf.estimate_transform('affine', np.array([*base_gcps]), np.array([*dst_gcps]))

    base_pixels = list(map(int, [match_points[0][0], match_points[0][1], size_x*2, size_y*2]))
    base_arr = base_cube.read_array(pixels=base_pixels, dtype="uint64")

    dst_pixels = list(map(int, [start_x, start_y, stop_x-start_x, stop_y-start_y]))
    dst_arr = input_cube.read_array(pixels=dst_pixels, dtype="float64")

    dst_arr = tf.warp(dst_arr, affine)
    dst_arr = dst_arr[:size_y*2, :size_x*2]
    print(dst_arr.shape)

    if verbose:
      fig, axs = plt.subplots(1, 2)
      axs[0].set_title("Base")
      axs[0].imshow(bytescale(base_arr), cmap="Greys_r")
      axs[1].set_title("Projected Image")
      axs[1].imshow(bytescale(dst_arr), cmap="Greys_r")
      plt.show()

    print(dst_arr.min(), dst_arr.max())

    # Run through one step of template matching then one step of phase matching
    # These parameters seem to work best, should pass as kwargs later
    restemplate = subpixel_template(size_x, size_y, size_x, size_y, bytescale(base_arr), bytescale(dst_arr), **template_kwargs)

    if phase_kwargs:
        resphase = iterative_phase(size_x, size_y, restemplate[0], restemplate[1], base_arr, dst_arr, **phase_kwargs)
        _,_,maxcoor, corrmap = restemplate
        x,y,_ = rephase
    else:
        x,y,maxcorr,corrmap = restemplate

    print(x, y)
    if x is None or y is None:
        return None, None, None, None, None

    sample, line = affine([x,y])[0]
    print(sample, line)
    sample += start_x
    line += start_y

    print(sample, line)

    if verbose:
      fig, axs = plt.subplots(1, 3)
      fig.set_size_inches((30,30))
      darr,_,_ = clip_roi(input_cube.read_array(), sample, line, 960, 960)
      axs[1].imshow(darr, cmap="Greys_r")
      axs[1].scatter(x=[darr.shape[1]/2], y=[darr.shape[0]/2], s=10, c="red")
      axs[1].set_title("Original Registered Image")

      axs[0].imshow(base_arr, cmap="Greys_r")
      axs[0].scatter(x=[base_arr.shape[1]/2], y=[base_arr.shape[0]/2], s=10, c="red")
      axs[0].set_title("Base")

      pcm = axs[2].imshow(corrmap**2, interpolation=None, cmap="coolwarm")
      plt.show()

    dist = np.linalg.norm([center_x-x, center_y-y])
    return sample, line, dist, maxcorr, corrmap


