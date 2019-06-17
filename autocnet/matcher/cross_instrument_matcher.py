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
from skimage.transform import resize
from sqlalchemy import (Boolean, Column, Float, ForeignKey, Integer,
                        LargeBinary, String, UniqueConstraint, create_engine,
                        event, orm, pool)
from sqlalchemy.ext.declarative import declarative_base

import geoalchemy2
import geopandas as gpd
import plio
import pvl
import pyproj
import pysis
from autocnet import config
from autocnet.graph.network import NetworkCandidateGraph
from autocnet.matcher.subpixel import iterative_phase
from autocnet import engine
from gdal import ogr
from geoalchemy2 import Geometry, WKTElement
from geoalchemy2.shape import to_shape
from knoten import csm
from plio.io.io_controlnetwork import from_isis, to_isis
from plio.io.io_gdal import GeoDataset
from pysis.exceptions import ProcessError
from pysis.isis import campt
from shapely import wkt
from shapely.geometry.multipolygon import MultiPolygon

from autocnet import engine

ctypes.CDLL(find_library('usgscsm'))


def themis_ground_to_ctx_matcher(cnet):
    """
    Hardcoded to be named exactly what it does, if we get meaningful results should
    be simple to generalize to a cross instrument CNET matcher type thing.
    """

    # Get neccessary tables from DB,
    # later this should probably be done through SQL comamnds and joins rather than in DataFrames

    # Get ground images from database, in this case THEMIS
    # hardcoded for now until we can finalize our databases
    db_uri = '{}://{}:{}@{}:{}/{}'.format('postgresql',
                                          'jay',
                                          'abcde',
                                          'smalls',
                                          '8083',
                                          'mars')
    themis_engine = create_engine(db_uri, poolclass=pool.NullPool,
                           isolation_level="AUTOCOMMIT")

    themis_images = gpd.GeoDataFrame.from_postgis("select * from themis_ir", themis_engine, geom_col="footprint_latlon")

    # useful for masking out paths, sometimes data is split between shared directories and
    # scratch causing errors when working anywhere besides the custer
    themis_images["valid_path"] = [os.path.isfile(p) for p in themis_images["path"]]
    themis_images = themis_images[themis_images["valid_path"]]

    # We need to join the CNET dataframe on serial numbers
    themis_images["serial"] = [plio.io.isis_serial_number.generate_serial_number(d) if d else None for d in themis_images["path"]]

    ctx_images = gpd.GeoDataFrame.from_postgis("select * from images", engine, geom_col="footprint_latlon")

    themis_cnet = from_isis(cnet)
    # we need image path information for each measure
    themis_cnet = themis_cnet.merge(themis_images, how='left', left_on="serialnumber", right_on="serial")

    # this doesn't need to be serial
    lats = []
    lons = []
    resolutions = []

    # Get lats, lons and pixel resolution through campt
    for i,r in themis_cnet.iterrows():
        try:
            res = pvl.loads(campt(from_=r["path"],
                                  line=r["line"], sample=r["sample"],
                                  type="image"))
        except ProcessError as e:
            # should probably do something here...
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)

        lats.append(res["GroundPoint"]["PlanetocentricLatitude"].value)
        lons.append(res["GroundPoint"]["PositiveEast360Longitude"].value)

        # We can assume line/sample resolution to be equal?
        resolutions.append(res["GroundPoint"]["LineResolution"].value)

    themis_cnet["resolution"] = resolutions
    themis_cnet = gpd.GeoDataFrame(themis_cnet, geometry=gpd.points_from_xy(lons, lats))

    spatial_index = ctx_images.sindex

    # We are going to iterate on points
    groups = themis_cnet.groupby("id_x").groups


    # append to list if images, mostly used for working with the network in python
    # after this step, is this uncecceary outside of debugging? Maybe actually should return
    # more info of where everything was sourced in the original DataFrames?
    images = []

    # append CNET info into structured Python list
    ctx_constrained_net = []

    # easily parrallelized on the cpoint level, dummy serial for now
    for cpoint, indices in groups.items():
        measures = themis_cnet.loc[indices]
        measure = measures.iloc[0]
        p = measure.geometry
        possible_matches_index = list(spatial_index.intersection(p.bounds))
        possible_matches = ctx_images.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(p)]

        if precise_matches.empty:
            continue

        # lazily iterate for now
        for i,row in precise_matches.iterrows():
            try:
                res = pvl.loads(campt(from_=row["path"],
                                      longitude=p.x,latitude=p.y,
                                      type="ground", allowoutside=False))
            except ProcessError as e:
                continue

            ctx_line, ctx_sample = res["GroundPoint"]["Line"], res["GroundPoint"]["Sample"]
            ctx_resolution = res["GroundPoint"]["LineResolution"].value

            ctx_data = GeoDataset(row["path"])
            ctx_arr = ctx_data.read_array()

            # dynamically set scale based on point resolution
            ctx_to_themis_scale = ctx_resolution/measure["resolution"]

            scaled_ctx_line = (ctx_arr.shape[0]-ctx_line)*ctx_to_themis_scale
            scaled_ctx_sample = ctx_sample*ctx_to_themis_scale

            ctx_arr = resize(ctx_arr, ctx_to_themis_scale)[::-1]

            # list of matching results in the format:
            # [measure_index, x_offset, y_offset, offset_magnitude]
            match_results = []
            for k,m in measures.iterrows():
                themis_arr = GeoDataset(m["path"]).read_array()

                sx, sy = m["sample"], m["line"]
                dx, dy = scaled_ctx_sample, scaled_ctx_line
                try:
                    # not sure what the best parameters are here
                    ret = iterative_phase(sx, sy, dx, dy, themis_arr, ctx_arr, size=20, reduction=1, max_dist=2, convergence_threshold=2)
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
                continue
            match_results = match_results[np.argwhere(match_results[:,3] == match_results[:,3].min())][0][0]

            if match_results[3] > 2:
                # best match diverged too much
                continue

            measure = measures.loc[match_results[0]]

            # apply offsets
            sample = (match_results[1]/ctx_to_themis_scale) + ctx_sample
            line = (match_results[2]/ctx_to_themis_scale) + ctx_line

            images.append(row["path"])
            ctx_constrained_net.append([cpoint,                # point id
                                       4,                      # point type
                                       "autocnet",             # choosername
                                       measure["datetime"].iloc[1],    # datetime
                                       False,                  # EditLock
                                       False,                  # ignore
                                       False,                  # jigsawRejected
                                       6,                      # reference Index
                                       0,                      # aprioriSurfPointSource, 3 for reference?
                                       measure["path"],        # aprioriSurfPointSourceFile
                                       measure["aprioriRadiusSource"], # aprioriRadiusSource
                                       measure["aprioriSurfPointSourceFile"], # aprioriRadiusSourceFile
                                       True,                   # latitudeConstrained
                                       True,                   # longitudeConstrained
                                       True,                   # radiusConstrained
                                       measure["aprioriX"],    # aprioriX
                                       measure["aprioriY"],    # aprioriY
                                       measure["aprioriZ"],    # aprioriZ
                                       measure["aprioriCovar"], # aprioriCovar
                                       measure["adjustedX"],   # adjustedX
                                       measure["adjustedY"],   # adjustedY
                                       measure["adjustedZ"],   # adjustedZ
                                       plio.io.isis_serial_number.generate_serial_number(row["path"]), #serial number
                                       0,                      # diameter
                                       sample,                 # sample
                                       line,                   # line
                                       0,                      # sample residual
                                       0,                      # line residual
                                       ctx_sample,             # apriorisample
                                       ctx_line,               # aprioriline
                                       0,                      # sample sigma
                                       0                       # line sigma
                                       ])


    # These should be defined somewhere in Autocnet/plio, if so it should be imported
    columns = ['point_id', 'type', 'chooserName', 'datetime', 'editLock', 'ignore',
           'jigsawRejected', 'referenceIndex', 'AprioriSource',
           'aprioriSurfPointSourceFile', 'RadiusSource',
           'aprioriRadiusSourceFile', 'latitudeConstrained',
           'longitudeConstrained', 'radiusConstrained', 'aprioriX', 'aprioriY',
           'aprioriZ', 'aprioriCovar', 'adjustedX', 'adjustedY', 'adjustedZ',
            'serialnumber', 'diameter', 'x', 'y',
           'sampleResidual', 'lineResidual', 'apriorisample', 'aprioriline',
           'samplesigma', 'linesigma']


    new_cnet = pd.DataFrame(data=ctx_constrained_net, columns=columns)
    return new_cnet, images
