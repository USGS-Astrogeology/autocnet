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

from autocnet.io.db.model import Images, Points, Measures, JsonEncoder
from autocnet.cg.cg import distribute_points_in_geom, xy_in_polygon
from autocnet.io.db.connection import new_connection
from autocnet.spatial import isis
from autocnet.transformation.spatial import reproject
from autocnet.matcher.cpu_extractor import extract_most_interesting
from autocnet.transformation import roi
from autocnet.matcher.subpixel import geom_match
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

    warnings.warn('This function is not well tested. No tests currently exists \
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
                    template_kwargs={'image_size': (39, 39), 'template_size': (21, 21)},
                    verbose=False):
    """

    """
    session = Session()
    engine = session.get_bind()
    string = f"select * from images where ST_Intersects(geom, ST_SetSRID(ST_Point({lon}, {lat}), {config['spatial']['latitudinal_srid']}))"
    images = pd.read_sql(string, engine)
    session.close()

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
                x,y, dist, metrics, corrmap = geom_match(base_image, dest_image, sx, sy, \
                        size_x=size_x, size_y=size_y, \
                        template_kwargs=template_kwargs, \
                        verbose=verbose)
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
          print('\n')

        # hardcoded for now
        if best_results[3] == None or best_results[3] < 0.7:
            continue

        px, py = dem.latlon_to_pixel(lat, lon)
        height = dem.read_array(1, [px, py, 1, 1])[0][0]

        semi_major = config['spatial']['semimajor_rad']
        semi_minor = config['spatial']['semiminor_rad']
        # The CSM conversion makes the LLA/ECEF conversion explicit
        x, y, z = reproject([lon, lat, height],
                         semi_major, semi_minor,
                         'latlon', 'geocent')

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

def propagate_control_network(Session, config, dem, base_cnet,
        size_x=40, size_y=40,
        template_kwargs={'image_size': (39,39), 'template_size': (21,21)}, verbose=False):
    """
    Parameters
    ----------
    Session   : sqlalchemy session maker
                session maker associated with the database you want to propagate to

    config    : dict
                configuation file associated with database you want to propagate to
                In the form: {'username':'somename',
                              'password':'somepassword',
                              'host':'somehost',
                              'pgbouncer_port':6543,
                              'name':'somename'}

    dem       : plio.io.io_gdal.GeoDataset
                Digital elevation model of target body

    base_cnet : pd.DataFrame
                Dataframe representing the points you want to propagate. Must contain line, sample, path.

    verbose   : boolean
                Increase the level of print outs/plots recieved during propagation


    Output
    ------
    ground   : pd.DataFrame
               Dataframe containing successfully propagated points

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
                                      template_kwargs,
                                      verbose=verbose)

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


