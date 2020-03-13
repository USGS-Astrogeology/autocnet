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

from shapely import wkt
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry import Point

from redis import StrictRedis

from plurmy import Slurm

from autocnet import config, engine, Session
from autocnet.io.db.model import Images, Points, Measures, JsonEncoder
from autocnet.graph.network import NetworkCandidateGraph
from autocnet.matcher.subpixel import iterative_phase, subpixel_template, clip_roi
from autocnet.cg.cg import distribute_points_in_geom
from autocnet.io.db.connection import new_connection
from autocnet.spatial import isis
from autocnet.utils.utils import bytescale
from autocnet.matcher.cpu_extractor import extract_most_interesting
from autocnet import spatial

import warnings



def geom_match(input_cube, base_cube, bcenter_x, bcenter_y, size_x=60, size_y=60, template_kwargs={"func": cv2.TM_CCOEFF_NORMED, "image_size":(60,60), "template_size":(31,31)}, phase_kwargs={"size":10, "reduction":1, "max_dist":2, "convergence_threshold":.5}, verbose=False):
    """
    Find some feature from base_cube denoted by a center line/sample and window into the input cube.

    100% untested for 100% Jank

    1. Reproject center to input_cube
    2. Compute an affine transformation to project input_cube onto base_cube
    3. Clip ROI from center and size_x, size_y from both cubes
    4. Apply subpixel template and sub pixel phase matcher to find base_cube's center feature on input_cube
    5. Apply inverse affine to aquire new adjusted point in input_cube and return sample, line

    Parameters
    ----------
    input_cube : GeoDataset
                 GeoDataset object for destination cube

    base_cube : GeoDataset
                GeoDataset object for source cube

    bcenter_x : Double
                Center sample for feature in base_cube

    bcenter_y : Double
                Center line for feature in base_cube

    size_x : Double
             Size in the x direction for ROI window

    size_y : Double
             Size in the y direction for ROI window

    Returns
    -------
    sample : Double
             Sample of feature detected in input_cube

    line : Double
           Line of feature detected in input_cube

    dist : Double
           Distance feature moved in projected space

    maxcorr : Double
              Correlation score at detected feature retuned by template matcher

    corrmap : np.Array
              MxN Array of correlation scores returned by template matcher
    """
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
        try:
            lat, lon = spatial.isis.image_to_ground(base_cube.file_name, x, y)
            cube_points.append(spatial.isis.ground_to_image(input_cube.file_name, lon, lat)[::-1])
        except Exception as e:
            if verbose:
                print("Match Failed with: ", e)
            return None, None, None, None, None

    affine = tf.estimate_transform('affine', np.array([*match_points]), np.array([*cube_points]))

    startx, starty, stopx, stopy = MultiPoint(cube_points).bounds

    # Do entire cube for now, should be optimized to only warp ROI
    dst_arr = tf.warp(input_cube.read_array(), affine)
    dst_arr[dst_arr==0] = np.nan
    dst_arr = dst_arr[int(match_points[0][1]):int(match_points[2][1]),int(match_points[0][0]):int(match_points[2][0])]

    pixels = list(map(int, [match_points[0][0], match_points[0][1], size_x*2, size_y*2]))
    base_arr = base_cube.read_array(pixels=pixels)

    if verbose:
      print("drawing things")
      fig, axs = plt.subplots(1, 2)
      axs[0].set_title("Base")
      axs[0].imshow(base_arr, cmap="Greys_r")
      axs[1].set_title("Projected Image")
      axs[1].imshow(dst_arr, cmap="Greys_r")
      plt.show()

    # Run through one step of template matching then one step of phase matching
    # These parameters seem to work best, should pass as kwargs later
    restemplate = subpixel_template(size_x, size_y, size_x, size_y, bytescale(base_arr), bytescale(dst_arr), **template_kwargs)
    resphase = iterative_phase(size_x, size_y, restemplate[0], restemplate[1], base_arr, dst_arr, **phase_kwargs)

    _,_,maxcorr,corrmap = restemplate
    x, y, _ = resphase
    if x is None or y is None:
        return None, None, None, None, None

    sample, line = affine([int(match_points[0][0])+x,int(match_points[0][1])+y])[0]

    if verbose:
      fig, axs = plt.subplots(1, 3)
      fig.set_size_inches((30,30))
      darr,_,_ = clip_roi(input_cube.read_array(), sample, line, 800, 800)
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


def generate_ground_points(ground_db_config, nspts_func=lambda x: int(round(x,1)*1), ewpts_func=lambda x: int(round(x,1)*4)):
    """
    Provided a config file which points to a database containing ground image path and geom information,
    generates ground points on these images within the range of a source database's images. For example,
    if creating a CTX mosaic which is grounded using themis data, the config files would look like:
        
        CTX_config -> located in config/[config_name].yml
        database:

            type: 'postgresql'
            username: 'jay'
            password: 'abcde'
            host: '130.118.160.193'
            port: 8085
            pgbouncer_port: 8083
            name: 'somename'
            timeout: 500

        Themis_config -> passed in to this function as ground_db_config
        
        ground_db_config = {'username':'jay',
                            'password':'abcde',
                            'host':'autocnet.wr.usgs.gov',
                            'pgbouncer_port':5432,
                            'name':'mars'}

    THIS FUNCTION IS CURRENTLY HARD CODED FOR themisdayir TABLE QUERY

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
    warnings.warn('This function is not well tested. No tests currently exists \
    in the test suite for this version of the function.')

    Ground_Session, ground_engine = new_connection(ground_db_config)
    ground_session = Ground_Session()

    session = Session()
    fp_poly = wkt.loads(session.query(functions.ST_AsText(functions.ST_Union(Images.geom))).one()[0])
    session.close()

    fp_poly_bounds = list(fp_poly.bounds)

    # just hard code queries to the mars database as it exists for now

    ground_image_query = f'select * from themisdayir where ST_INTERSECTS(geom, ST_MakeEnvelope({fp_poly_bounds[0]}, {fp_poly_bounds[1]}, {fp_poly_bounds[2]}, {fp_poly_bounds[3]}, {config["spatial"]["latitudinal_srid"]}))'
    themis_images = gpd.GeoDataFrame.from_postgis(ground_image_query,
                                                  ground_engine, geom_col="geom")

    coords = distribute_points_in_geom(fp_poly, nspts_func=nspts_func, ewpts_func=ewpts_func, method="new")
    coords = np.asarray(coords)

    records = []
    coord_list = []

    # throw out points not intersecting the ground reference images
    for i, coord in enumerate(coords):
        # res = ground_session.execute(formated_sql)
        p = Point(*coord)
        res = themis_images[themis_images.intersects(p)]
        adjusted = False

        for image_path in res["path"]:
            try:
                arr = GeoDataset(image_path)
                linessamples = isis.point_info(image_path, p.x, p.y, 'ground')
                sample = linessamples["GroundPoint"].get('Sample')
                line = linessamples["GroundPoint"].get('Line')
                size = 100
                image, _, _ = clip_roi(arr, sample, line, size_x=size, size_y=size)
                interesting = extract_most_interesting(image,  extractor_parameters={'nfeatures':30})

                # kps are in the image space with upper left origin, so convert to
                # center origin and then convert back into full image space
                newsample = sample + (interesting.x - size)
                newline = line + (interesting.y - size)

                newpoint = isis.point_info(image_path, newsample, newline, 'image')
                p = Point(newpoint["GroundPoint"].get('PositiveEast360Longitude').value,
                          newpoint["GroundPoint"].get('PlanetocentricLatitude').value)

                res = themis_images[themis_images.intersects(p)]
                adjusted = True
                break
            except Exception as e:
                continue
        if not adjusted:
            raise("This is some garbage")

        for k, record in res.iterrows():
            record["pointid"] = i
            records.append(record)
            coord_list.append(p)

    ground_session.close()

    # start building the cnet
    ground_cnet = pd.DataFrame.from_records(records)
    ground_cnet["point"] = coord_list
    ground_cnet['line'] = None
    ground_cnet['sample'] = None
    ground_cnet['resolution'] = None

    # generate lines and samples from ground points
    groups = ground_cnet.groupby('path')
    # group by images so campt can do multiple at a time
    for group_id, group in groups:
        lons = [p.x for p in group['point']]
        lats = [p.y for p in group['point']]

        point_list = isis.point_info(group_id, lons, lats, 'ground')
        lines = []
        samples = []
        resolutions = []
        for i, res in enumerate(point_list):
            geom = Point(res[1].get("PositiveEast360Longitude").value, res[1].get("PlanetocentricLatitude").value)
            if res[1].get('Error') is not None and not fp_poly.intersects(geom):
                lines.append(None)
                samples.append(None)
                resolutions.append(None)
            else:
                lines.append(res[1].get('Line'))
                samples.append(res[1].get('Sample'))
                resolutions.append(res[1].get('LineResolution').value)
        index = group.index.__array__()
        ground_cnet.loc[index, 'line'] = lines
        ground_cnet.loc[index, 'sample'] = samples
        ground_cnet.loc[index, 'resolution'] = resolutions

    ground_cnet = gpd.GeoDataFrame(ground_cnet, geometry='point')
    return ground_cnet, fp_poly, coord_list


def propagate_point(lon, lat, pointid, paths, lines, samples, resolutions, verbose=False):
    """

    """
    images = gpd.GeoDataFrame.from_postgis(f"select * from images where ST_Intersects(geom, ST_SetSRID(ST_Point({lon}, {lat}), {config['spatial']['latitudinal_srid']}))", engine, geom_col="geom")

    image_measures = pd.DataFrame(zip(paths, lines, samples, resolutions), columns=["path", "line", "sample", "resolution"])
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
                x,y, dist, metrics, corrmap = geom_match(dest_image, base_image, sx, sy, verbose=verbose)
            except Exception as e:
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
          print("Full results: ", match_results)
          print("Winning CORR: ", match_results[3], "Themis Pixel shift: ", match_results[4])
          print("Themis Image: ", match_results[6], "CTX image:", match_results[7])
          print("Themis S,L: ", f"{sx},{sy}", "CTX S,L: ", f"{sample},{line}")

        # hardcoded for now
        if best_results[3] < 0.7:
            continue

        pointpvl = isis.point_info(paths[0], x=lon, y=lat, point_type="ground")

        try:
            groundx, groundy, groundz = pointpvl["GroundPoint"]["BodyFixedCoordinate"].value
        except:
            groundx, groundy, groundz = pointpvl["GroundPoint"]["BodyFixedCoordinate"]
        groundx, groundy, groundz = groundx*1000, groundy*1000, groundz*1000

        new_measures.append({
                'pointid' : pointid,
                'imageid' : image['id'],
                'serial' : image['serial'],
                'line' : line,
                'sample' : sample,
                'point_latlon' : p,
                'point_ground' : Point(groundx, groundy, groundz)
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
               'resolutions' : measures['resolution'].tolist(),
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
        gp_measures = propagate_point(lon[0], lat[0], cpoint, measures["path"], measures["line"], measures["sample"], measures["resolution"], verbose=verbose)
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


