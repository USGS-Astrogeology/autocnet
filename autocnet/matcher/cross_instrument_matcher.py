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

import pandas as pd
import scipy
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

from shapely import wkt
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry import Point

from plurmy import Slurm

from autocnet.matcher.subpixel import check_match_func
from autocnet.io.db.model import Images, Points, Measures, JsonEncoder
from autocnet.cg.cg import distribute_points_in_geom, xy_in_polygon
from autocnet.io.db.connection import new_connection
from autocnet.spatial import isis
from autocnet.transformation.spatial import reproject, oc2og
from autocnet.matcher.cpu_extractor import extract_most_interesting
from autocnet.transformation import roi
from autocnet.matcher.subpixel import geom_match_simple
from autocnet.utils.utils import bytescale

import warnings

def generate_ground_points(Session, ground_mosaic, nspts_func=lambda x: int(round(x,1)*1), ewpts_func=lambda x: int(round(x,1)*4), size=(100,100)):
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

    size             : tuple of int
                       (size_x, size_y) maximum distances on either access point
                       can move when attempting to find an interesting feature.
    """

    if isinstance(ground_mosaic, str):
        ground_mosaic = GeoDataset(ground_mosaic)

    warnings.warn('This function is not well tested. No tests currently exist \
    in the test suite for this version of the function.')

    session = Session()
    fp_poly = wkt.loads(session.query(functions.ST_AsText(functions.ST_Union(Images.geom))).one()[0])
    session.close()

    coords = distribute_points_in_geom(fp_poly, nspts_func=nspts_func, ewpts_func=ewpts_func, method="new", Session=Session)
    coords = np.asarray(coords)

    old_coord_list = []
    coord_list = []
    lines = []
    samples = []
    newlines = []
    newsamples = []

    # throw out points not intersecting the ground reference images
    print('points to lay down: ', len(coords))
    for i, coord in enumerate(coords):
        # res = ground_session.execute(formated_sql)
        p = Point(*coord)
        print(f'point {i}'),


        linessamples = isis.point_info(ground_mosaic.file_name, p.x, p.y, 'ground')
        if linessamples is None:
            print('unable to find point in ground image')
            continue
        line = linessamples[0].get('Line')
        sample = linessamples[0].get('Sample')

        oldpoint = isis.point_info(ground_mosaic.file_name, sample, line, 'image')
        op = Point(oldpoint[0].get('PositiveEast360Longitude'),
                   oldpoint[0].get('PlanetocentricLatitude'))


        image = roi.Roi(ground_mosaic, sample, line, size_x=size[0], size_y=size[1])
        image_roi = image.clip(dtype="uint64")

        interesting = extract_most_interesting(bytescale(image_roi),  extractor_parameters={'nfeatures':30})

        # kps are in the image space with upper left origin, so convert to
        # center origin and then convert back into full image space
        left_x, _, top_y, _ = image.image_extent
        newsample = left_x + interesting.x
        newline = top_y + interesting.y

        newpoint = isis.point_info(ground_mosaic.file_name, newsample, newline, 'image')
        p = Point(newpoint[0].get('PositiveEast360Longitude'),
                  newpoint[0].get('PlanetocentricLatitude'))

        if not (xy_in_polygon(p.x, p.y, fp_poly)):
                print('Interesting point not in mosaic area, ignore')
                continue

        old_coord_list.append(op)
        lines.append(line)
        samples.append(sample)
        coord_list.append(p)
        newlines.append(newline)
        newsamples.append(newsample)


    # start building the cnet
    ground_cnet = pd.DataFrame()
    ground_cnet["path"] = [ground_mosaic.file_name]*len(coord_list)
    ground_cnet["pointid"] = list(range(len(coord_list)))
    ground_cnet["original point"] = old_coord_list
    ground_cnet["point"] = coord_list
    ground_cnet['original_line'] = lines
    ground_cnet['original_sample'] = samples
    ground_cnet['line'] = newlines
    ground_cnet['sample'] = newsamples
    ground_cnet = gpd.GeoDataFrame(ground_cnet, geometry='point')
    return ground_cnet, fp_poly, coord_list


def propagate_point(Session,
                    config,
                    dem,
                    lon,
                    lat,
                    pointid,
                    paths,
                    lines,
                    samples,
                    size_x=40,
                    size_y=40,
                    match_func="classic",
                    match_kwargs={'image_size': (39, 39), 'template_size': (21, 21)},
                    verbose=False,
                    cost=lambda x, y: y == np.max(x)):
    """
    Conditionally propagate a point into a stack of images. The point and all corresponding measures
    are matched against database network (to which you are propagating), best result(s) is(are) kept.

    Parameters
    ----------
    Session   : sqlalchemy.sessionmaker
                session maker associated with the database you want to propagate to

    config    : dict
                configuation file associated with database you want to propagate to
                In the form: {'username':'somename',
                              'password':'somepassword',
                              'host':'somehost',
                              'pgbouncer_port':6543,
                              'name':'somename'}

    dem       : plio.io.io_gdal.GeoDataset
                digital elevation model of target body

    lon       : np.float
                longitude of point you want to project

    lat       : np.float
                planetocentric latitude of point you want to project

    pointid   : int
                clerical input used to trace point from generate_ground_points output

    paths     : list of str
                absolute paths pointing to the image(s) from which you want to try porpagating the point

    lines     : list of np.float
                apriori line(s) corresponding to point projected in 'paths' image(s)

    samples   : list of np.float
                apriori sample(s) corresponding to point projected in 'paths' image(s)

    size_x    : int
                half width of GeoDataset that is cut from full image and affinely transfromed in geom_match;
                must be larger than 1/2 template_kwargs 'image_size'

    size_y    : int
                half height of GeoDataset that is cut from full image and affinely transfromed in geom_match;
                must be larger than 1/2 template_kwargs 'image_size'

    template_kwargs : dict
                      kwargs passed through to control matcher.subpixel_template()

    verbose   : boolean
                If True, this will print out the results of each propagation, including prints of the
                matcher areas and their correlation map.

    cost      : anonymous function
                determines to which image(s) the point should be propagated. x corresponds to a list
                of all match correlation metrics, while y corresponds to each indiviudal element
                of the x array.
                Example:
                cost = lambda x,y: y == np.max(x) will get you one result corresponding to the image that
                has the maximum correlation with the source image
                cost = lambda x,y: y > 0.6 will propagate the point to all images whose correlation
                result is greater than 0.6


    Returns
    -------
    new_measures : pd.DataFrame
                   Dataframe containing pointid, imageid, image serial number, line, sample, and ground location (both latlon
                   and cartesian) of successfully propagated points

    """

    match_func = check_match_func(match_func)

    session = Session()
    engine = session.get_bind()
    string = f"select * from images where ST_Intersects(geom, ST_SetSRID(ST_Point({lon}, {lat}), {config['spatial']['latitudinal_srid']}))"
    images = pd.read_sql(string, engine)
    session.close()

    image_measures = pd.DataFrame(zip(paths, lines, samples), columns=["path", "line", "sample"])
    measure = image_measures.iloc[0]

    p = Point(lon, lat)
    new_measures = []

    # list of matching results in the format:
    # [measure_index, x_offset, y_offset, offset_magnitude]
    match_results = []
    # lazily iterate for now
    for k,m in image_measures.iterrows():
        base_image = GeoDataset(m["path"])

        sx, sy = m["sample"], m["line"]

        for i,image in images.iterrows():
            # When grounding to THEMIS the df has a PATH to the QUAD
            dest_image = GeoDataset(image["path"])

            if os.path.basename(m['path']) == os.path.basename(image['path']):
                continue

            try:
                print(f'prop point: base_image: {base_image}')
                print(f'prop point: dest_image: {dest_image}')
                print(f'prop point: (sx, sy): ({sx}, {sy})')
                x,y, dist, metrics, corrmap = geom_match_simple(base_image, dest_image, sx, sy, 16, 16, \
                        match_func = match_func, \
                        match_kwargs=match_kwargs, \
                        verbose=verbose)
            except Exception as e:
                raise Exception(e)
                match_results.append(e)
                continue

            match_results.append([k, x, y,
                                 metrics, dist, corrmap, m["path"], image["path"],
                                 image['id'], image['serial']])

    # get best offsets
    match_results = np.asarray([res for res in match_results if isinstance(res, list) and all(r is not None for r in res)])
    if match_results.shape[0] == 0:
        # no matches
        return new_measures

    # column index 3 is the metric returned by the geom matcher
    best_results = np.asarray([match for match in match_results if cost(match_results[:,3], match[3])])
    if best_results.shape[0] == 0:
        # no matches satisfying cost
        return new_measures

    if verbose:
        print("match_results final length: ", len(match_results))
        print("best_results length: ", len(best_results))
        print("Full results: ", best_results)
        print("Winning CORRs: ", best_results[:,3], "Themis Pixel shifts: ", best_results[:,4])
        print("Themis Images: ", best_results[:,6], "CTX images:", best_results[:,7])
        print("Themis Sample: ", sx, "CTX Samples: ", best_results[:,1])
        print("Themis Line: ", sy, "CTX Lines: ", best_results[:,2])
        print('\n')

    # if the single best results metric (returned by geom_matcher) is None
    if len(best_results[:,3])==1 and best_results[:,3][0] is None:
        return new_measures

    px, py = dem.latlon_to_pixel(lat, lon)
    height = dem.read_array(1, [px, py, 1, 1])[0][0]

    semi_major = config['spatial']['semimajor_rad']
    semi_minor = config['spatial']['semiminor_rad']
    # The CSM conversion makes the LLA/ECEF conversion explicit
    # reprojection takes ographic lat
    lon_og, lat_og = oc2og(lon, lat, semi_major, semi_minor)
    x, y, z = reproject([lon_og, lat_og, height],
                         semi_major, semi_minor,
                         'latlon', 'geocent')

    for row in best_results:
        sample = row[1]
        line = row[2]

        new_measures.append({
            'pointid' : pointid,
            'imageid' : row[8],
            'serial' : row[9],
            'path': row[7],
            'line' : line,
            'sample' : sample,
            'point' : p,
            'point_ecef' : Point(x, y, z)
            })

    return new_measures

def propagate_control_network(Session,
        config,
        dem,
        base_cnet,
        size_x=40,
        size_y=40,
        match_func="classic",
        match_kwargs={'image_size': (39,39), 'template_size': (21,21)},
        verbose=False,
        cost=lambda x,y: y == np.max(x)):
    """
    Loops over a base control network's measure information (line, sample, image path) and uses image matching
    algorithms (autocnet.matcher.subpixel.geom_match) to find the corresponding line(s)/sample(s) in database images.


    Parameters
    ----------
    Session   : sqlalchemy.sessionmaker
                session maker associated with the database containing the images you want to propagate to

    config    : dict
                configuation file associated with database containing the images you want to propagate to
                In the form: {'username':'somename',
                              'password':'somepassword',
                              'host':'somehost',
                              'pgbouncer_port':6543,
                              'name':'somename'}

    dem       : plio.io.io_gdal.GeoDataset
                Digital elevation model of target body

    base_cnet : pd.DataFrame
                Dataframe representing the points you want to propagate. Must contain 'line', 'sample' location of
                the measure and the 'path' to the corresponding image

    verbose   : boolean
                Increase the level of print outs/plots recieved during propagation

    cost      : anonymous function
                determines to which image(s) the point should be propagated. x corresponds to a list
                of all match correlation metrics, while y corresponds to each indiviudal element
                of the x array.
                Example:
                cost = lambda x,y: y == np.max(x) will get you one result corresponding to the image that
                has the maximum correlation with the source image
                cost = lambda x,y: y > 0.6 will propegate the point to all images whose correlation
                result is greater than 0.6


    Returns
    -------
    ground   : pd.DataFrame
               Dataframe containing pointid, imageid, image serial number, line, sample, and ground location (both latlon
               and cartesian) of successfully propagated points

    """
    warnings.warn('This function is not well tested. No tests currently exist \
    in the test suite for this version of the function.')

    match_func = check_match_func(match_func)

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
        gp_measures = propagate_point(Session,
                                      config,
                                      dem,
                                      lon[0],
                                      lat[0],
                                      cpoint,
                                      measures["path"],
                                      measures["line"],
                                      measures["sample"],
                                      size_x,
                                      size_y,
                                      match_func,
                                      match_kwargs,
                                      verbose=verbose,
                                      cost=cost)

        # do not append if propagate_point is unable to find result
        if len(gp_measures) == 0:
            continue
        constrained_net.extend(gp_measures)

    ground = gpd.GeoDataFrame.from_dict(constrained_net).set_geometry('point')
    groundpoints = ground.groupby('pointid').groups

    points = []

    # conditionally upload a new point to DB or updated existing point with new measures
    lat_srid = config['spatial']['latitudinal_srid']
    session = Session()
    for p,indices in groundpoints.items():
        point = ground.loc[indices].iloc[0]
        # if point already exists in DB
        lon = point.point.x
        lat = point.point.y
        res = session.query(Points).filter(functions.ST_Intersects(Points.geom, functions.ST_Buffer(functions.ST_SetSRID(functions.ST_Point(lon,lat), lat_srid), 10e-10))).all()

        if len(res) > 1:
            warnings.warn(f"There is more than one point at lon: {lon}, lat: {lat}")

        elif len(res) == 1:
            for i in indices:
                row = ground.loc[i]
                pid = res[0].id
                meas = session.query(Measures.serial).filter(Measures.pointid == pid).all()
                serialnumbers = [m[0] for m in meas]

                if row['serial'] in serialnumbers:
                    continue

                points.append(Measures(pointid = pid,
                                       line=float(row['line']),
                                       sample = float(row['sample']),
                                       aprioriline = float(row['line']),
                                       apriorisample = float(row['sample']),
                                       imageid = int(row['imageid']),
                                       serial = row['serial'],
                                       measuretype=3))
        else:
            p = Points()
            p.pointtype = 3
            p.apriori = point['point_ecef']
            p.adjusted = point['point_ecef']
            for i in indices:
                row = ground.loc[i]
                p.measures.append(Measures(line=float(row['line']),
                                           sample = float(row['sample']),
                                           aprioriline = float(row['line']),
                                           apriorisample = float(row['sample']),
                                           imageid = int(row['imageid']),
                                           serial = row['serial'],
                                           measuretype=3))
            points.append(p)

    session.add_all(points)
    session.commit()
    session.close()

    return ground


