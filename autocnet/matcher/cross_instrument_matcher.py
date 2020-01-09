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

from autocnet import config, engine, Session
from autocnet.io.db.model import Images, Points, Measures
from autocnet.graph.network import NetworkCandidateGraph
from autocnet.matcher.subpixel import iterative_phase
from autocnet.cg.cg import distribute_points_in_geom
from autocnet.io.db.connection import new_connection
from autocnet.spatial import isis

import warnings

ctypes.CDLL(find_library('usgscsm'))

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

    ground_image_query = f'select * from themisdayir where ST_INTERSECTS(geom, ST_MakeEnvelope({image_fp_bounds[0]}, {image_fp_bounds[1]}, {image_fp_bounds[2]}, {image_fp_bounds[3]}, {config["spatial"]["latitudinal_srid"]}))'
    themis_images = gpd.GeoDataFrame.from_postgis(ground_image_query,
                                                  ground_engine, geom_col="geom")
    spatial_index = themis_images.sindex

    coords = distribute_points_in_geom(fp_poly, nspts_func=nspts_func, ewpts_func=ewpts_func)
    coords = np.asarray(coords)

    records = []
    coord_list = []

    # throw out points not intersecting the ground reference images
    for i, coord in enumerate(coords):
        # res = ground_session.execute(formated_sql)
        p = Point(*coord)
        res = themis_images[themis_images.intersects(p)]

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
            if res[1].get('Error') is not None:
                print('Bad intersection: ', res[1].get("Error"))
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
    return ground_cnet, ground_poly, coords


def propagate_control_network(base_cnet):
    """

    """
    warnings.warn('This function is not well tested. No tests currently exists \
    in the test suite for this version of the function.')
    dest_images = gpd.GeoDataFrame.from_postgis("select * from images", engine, geom_col="geom")
    spatial_index = dest_images.sindex
    groups = base_cnet.groupby('pointid').groups

    # append to list if images, mostly used for working with the network in python
    # after this step, is this uncecceary outside of debugging? Maybe actually should return
    # more info of where everything was sourced in the original DataFrames?
    images = []

    # append CNET info into structured Python list
    constrained_net = []
    dbpoints = []
    dbmeasures = []
    i = 0
    ngroups = len(groups)
    # easily parrallelized on the cpoint level, dummy serial for now
    for cpoint, indices in groups.items():
        i+=1
        print(f"{i}/{ngroups}")
        measures = base_cnet.loc[indices]
        measure = measures.iloc[0]

        p = measure.point

        # get image in the destination that overlap
        matches = dest_images[dest_images.intersects(p)]

        # lazily iterate for now
        for i,row in matches.iterrows():
            res = isis.point_info(row["path"], p.x, p.y, point_type="ground", allow_outside=False)
            dest_line, dest_sample = res["GroundPoint"]["Line"], res["GroundPoint"]["Sample"]

            try:
                dest_resolution = res["GroundPoint"]["LineResolution"].value
            except:
                warnings.warn(f'Failed to generate ground point info on image {row["path"]} at lat={p.y} lon={p.x}')
                continue

            dest_data = GeoDataset(row["path"])
            dest_arr = dest_data.read_array()

            # dynamically set scale based on point resolution
            dest_to_base_scale = dest_resolution/measure["resolution"]

            scaled_dest_line = (dest_arr.shape[0]-dest_line)*dest_to_base_scale
            scaled_dest_sample = dest_sample*dest_to_base_scale

            dest_arr = imresize(dest_arr, dest_to_base_scale)[::-1]

            # list of matching results in the format:
            # [measure_index, x_offset, y_offset, offset_magnitude]
            match_results = []
            for k,m in measures.iterrows():
                base_arr = GeoDataset(m["path"]).read_array()

                sx, sy = m["sample"], m["line"]
                dx, dy = scaled_dest_sample, scaled_dest_line
                try:
                    # not sure what the best parameters are here
                    ret = iterative_phase(sx, sy, dx, dy, base_arr, dest_arr, size=10, reduction=1, max_dist=2, convergence_threshold=1)
                except Exception as ex:
                    match_results.append(ex)
                    continue

                if ret is not None and None not in ret:
                    x,y,metrics = ret
                else:
                    match_results.append("Failed to Converge")
                    continue

                dist = np.linalg.norm([x-dx, -1*(y-dy)])
                match_results.append([k, x-dx, -1*(y-dy), dist])

            # get best offsets, if possible we need better metric for what a
            # good match looks like
            match_results = np.asarray([res for res in match_results if isinstance(res, list)])
            if match_results.shape[0] == 0:
                # no matches
                print("No Mathces")
                continue

            match_results = match_results[np.argwhere(match_results[:,3] == match_results[:,3].min())][0][0]

            if match_results[3] > 2:
                # best match diverged too much
                continue

            measure = measures.loc[match_results[0]]

            # apply offsets
            sample = (match_results[1]/dest_to_base_scale) + dest_sample
            line = (match_results[2]/dest_to_base_scale) + dest_line

            pointpvl = isis.point_info(row["path"], sample, line, point_type="image")
            groundx, groundy, groundz = pointpvl["GroundPoint"]["BodyFixedCoordinate"].value
            groundx, groundy, groundz = groundx*1000, groundy*1000, groundz*1000

            images.append(row["path"])
            constrained_net.append({
                    'pointid' : cpoint,
                    'imageid' : row['id'],
                    'serial' : row.serial,
                    'line' : line,
                    'sample' : sample,
                    'point_latlon' : p,
                    'point_ground' : Point(groundx, groundy, groundz)
                })

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
