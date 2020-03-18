import warnings
import json

from redis import StrictRedis
import numpy as np
import pyproj
import shapely
import sqlalchemy
from plio.io.io_gdal import GeoDataset

from autocnet import config, dem, Session
from autocnet.cg import cg as compgeom
from autocnet.io.db.model import Images, Measures, Overlay, Points, JsonEncoder
from autocnet.spatial import isis
from autocnet.matcher.cpu_extractor import extract_most_interesting
from autocnet.transformation.spatial import reproject
from autocnet.transformation import roi

from plurmy import Slurm
import csmapi


# SQL query to decompose pairwise overlaps
compute_overlaps_sql = """
WITH intersectiongeom AS
(SELECT geom AS geom FROM ST_Dump((
   SELECT ST_Polygonize(the_geom) AS the_geom FROM (
     SELECT ST_Union(the_geom) AS the_geom FROM (
     SELECT ST_ExteriorRing((ST_DUMP(geom)).geom) AS the_geom
       FROM images WHERE images.geom IS NOT NULL) AS lines
  ) AS noded_lines))),
iid AS (
 SELECT images.id, intersectiongeom.geom AS geom
    FROM images, intersectiongeom
    WHERE images.geom is NOT NULL AND
    ST_INTERSECTS(intersectiongeom.geom, images.geom) AND
    ST_AREA(ST_INTERSECTION(intersectiongeom.geom, images.geom)) > 0.000001
)
INSERT INTO overlay(intersections, geom) SELECT row.intersections, row.geom FROM
(SELECT iid.geom, array_agg(iid.id) AS intersections
  FROM iid GROUP BY iid.geom) AS row WHERE array_length(intersections, 1) > 1;
"""

def place_points_in_overlaps(nodes, size_threshold=0.0007,
                             distribute_points_kwargs={}, cam_type='csm'):
    """
    Place points in all of the overlap geometries by back-projecing using
    sensor models.

    Parameters
    ----------
    nodes : dict-link
            A dict like object with a shared key with the intersection
            field of the database Overlay table and a cg node object
            as the value. This could be a NetworkCandidateGraph or some
            other dict-like object.

    size_threshold : float
                     overlaps with area <= this threshold are ignored
    """
    points = []
    for o in Overlay.overlapping_larger_than(size_threshold):
        overlaps = o.intersections
        if overlaps == None:
            continue

        overlapnodes = [nodes[id]["data"] for id in overlaps]
        points.extend(place_points_in_overlap(overlapnodes, o.geom, cam_type=cam_type,
                                              distribute_points_kwargs=distribute_points_kwargs))
    Points.bulkadd(points)

def cluster_place_points_in_overlaps(size_threshold=0.0007,
                                     distribute_points_kwargs={},
                                     walltime='00:10:00',
                                     chunksize=1000,
                                     exclude=None,
                                     cam_type="csm",
                                     query_string='SELECT overlay.id FROM overlay LEFT JOIN points ON ST_INTERSECTS(overlay.geom, points.geom) WHERE points.id IS NULL AND ST_AREA(overlay.geom) >= {};'):
    """
    Place points in all of the overlap geometries by back-projecing using
    sensor models. This method uses the cluster to process all of the overlaps
    in parallel. See place_points_in_overlap and acn_overlaps.

    Parameters
    ----------
    size_threshold : float
        overlaps with area <= this threshold are ignored

    walltime : str
        Cluster job wall time as a string HH:MM:SS

    cam_type : str
               options: {"csm", "isis"}
               Pick what kind of camera model implementation to use

    query : str

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
    past = 0
    session = Session()
    ids = [i[0] for i in session.execute(query_string.format(size_threshold))]
    session.close()
    for i, id in enumerate(ids):
        msg = {'id' : id,
               'distribute_points_kwargs' : distribute_points_kwargs,
               'walltime' : walltime,
               'cam_type': cam_type}
        rqueue.rpush(queuename, json.dumps(msg, cls=JsonEncoder))
    # Submit the jobs
    submitter = Slurm('acn_overlaps',
                 job_name='place_points',
                 mem_per_cpu=config['cluster']['processing_memory'],
                 time=walltime,
                 partition=config['cluster']['queue'],
                 output=config['cluster']['cluster_log_dir']+'/autocnet.place_points-%j')
    job_counter = i+1
    submitter.submit(array='1-{}%24'.format(job_counter), chunksize=chunksize, exclude=exclude)
    return job_counter

def place_points_in_overlap(nodes, geom, cam_type="csm",
                            size=71,
                            distribute_points_kwargs={}):
    """
    Place points into an overlap geometry by back-projecing using sensor models.
    The DEM specified in the config file will be used to calculate point elevations.

    Parameters
    ----------
    nodes : list of Nodes
        The Nodes or Networknodes of all the images that intersect the overlap

    geom : geometry
        The geometry of the overlap region

    cam_type : str
               options: {"csm", "isis"}
               Pick what kind of camera model implementation to use

    size : int
           The size of the window used to extractor features to find an
           interesting feature to which the point is shifted.

    Returns
    -------
    points : list of Points
        The list of points seeded in the overlap
    """
    avail_cams = {"isis", "csm"}
    cam_type = cam_type.lower()
    if cam_type not in cam_type:
        raise Exception(f'{cam_type} is not one of valid camera: {avail_cams}')

    points = []
    semi_major = config['spatial']['semimajor_rad']
    semi_minor = config['spatial']['semiminor_rad']

    valid = compgeom.distribute_points_in_geom(geom, **distribute_points_kwargs)
    if not valid:
        warnings.warn('Failed to distribute points in overlap')
        return []

    for v in valid:
        lon = v[0]
        lat = v[1]

        # Calculate the height, the distance (in meters) above or
        # below the aeroid (meters above or below the BCBF spheroid).
        if dem is None:
            height = 0
        else:
            px, py = dem.latlon_to_pixel(lat, lon)
            height = dem.read_array(1, [px, py, 1, 1])[0][0]

        # Need to get the first node and then convert from lat/lon to image space
        node = nodes[0]
        if cam_type == "isis":
            line, sample = isis.ground_to_image(node["image_path"], lon ,lat)
        if cam_type == "csm":
            # The CSM conversion makes the LLA/ECEF conversion explicit
            x, y, z = reproject([lon, lat, height],
                                 semi_major, semi_minor,
                                 'latlon', 'geocent')
            gnd = csmapi.EcefCoord(x, y, z)
            image_coord = node.camera.groundToImage(gnd)
            sample, line = image_coord.samp, image_coord.line

        # Extract ORB features in a sub-image around the desired point
        image_roi = roi.Roi(node.geodata, sample, line, size_x=size, size_y=size)
        image = image_roi.clip()
        try:
            interesting = extract_most_interesting(image)
        except:
            warnings.warn('Could not find an interesting feature around point')
            continue

        # kps are in the image space with upper left origin and the roi
        # could be the requested size or smaller if near an image boundary.
        # So use the roi upper left_x and top_y for the actual origin.
        left_x, _, top_y, _ = image_roi.image_extent
        newsample = left_x + interesting.x
        newline = top_y + interesting.y

        # Get the updated lat/lon from the feature in the node
        if cam_type == "isis":
            p = isis.point_info(node["image_path"], newsample, newline, point_type="image")
            x, y, z = p["GroundPoint"]["BodyFixedCoordinate"].value
            if p["GroundPoint"]["BodyFixedCoordinate"].units.lower() == "km":
                x = x * 1000
                y = y * 1000
                z = z * 1000
        elif cam_type == "csm":
            image_coord = csmapi.ImageCoord(newline, newsample)
            pcoord = node.camera.imageToGround(image_coord)
            # Get the BCEF coordinate from the lon, lat
            updated_lon, updated_lat, _ = reproject([pcoord.x, pcoord.y, pcoord.z],
                                                    semi_major, semi_minor, 'geocent', 'latlon')

            # Get the new DEM height
            if dem is None:
                updated_height = 0
            else:
                px, py = dem.latlon_to_pixel(updated_lat, updated_lon)
                updated_height = dem.read_array(1, [px, py, 1, 1])[0][0]


            # Get the BCEF coordinate from the lon, lat
            x, y, z = reproject([updated_lon, updated_lat, updated_height],
                                semi_major, semi_minor, 'latlon', 'geocent')

        # If the updated point is outside of the overlap, then revert back to the
        # original point and hope the matcher can handle it when sub-pixel registering
        updated_lon, updated_lat, updated_height = reproject([x, y, z], semi_major, semi_minor,
                                                             'geocent', 'latlon')
        if not geom.contains(shapely.geometry.Point(updated_lon, updated_lat)):
            x, y, z = reproject([lon, lat, height],
                                semi_major, semi_minor, 'latlon', 'geocent')

        point_geom = shapely.geometry.Point(x, y, z)
        point = Points(apriori=point_geom,
                       adjusted=point_geom,
                       pointtype=2, # Would be 3 or 4 for ground
                       cam_type=cam_type)

        gnd = csmapi.EcefCoord(x, y, z)
        for node in nodes:
            if cam_type == "csm":
                image_coord = node.camera.groundToImage(gnd)
                sample, line = image_coord.samp, image_coord.line
            if cam_type == "isis":
                line, sample = isis.ground_to_image(node["image_path"], updated_lon, updated_lat)

            point.measures.append(Measures(sample=sample,
                                           line=line,
                                           apriorisample=sample,
                                           aprioriline=line,
                                           imageid=node['node_id'],
                                           serial=node.isis_serial,
                                           measuretype=3))

        if len(point.measures) >= 2:
            points.append(point)
    return points
