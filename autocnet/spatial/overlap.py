import warnings
import json

from redis import StrictRedis
import pyproj
import shapely
import sqlalchemy
from plio.io.io_gdal import GeoDataset

from autocnet import config, dem
from autocnet.cg import cg as compgeom
from autocnet.io.db.model import Images, Measures, Overlay, Points
from autocnet.matcher.subpixel import iterative_phase
from autocnet.spatial import isis

from plurmy import Slurm
import csmapi


# SQL query to decompose pairwise overlaps
compute_overlaps_sql = """
WITH intersectiongeom AS
(SELECT geom AS geom FROM ST_Dump((
   SELECT ST_Polygonize(the_geom) AS the_geom FROM (
     SELECT ST_Union(the_geom) AS the_geom FROM (
	   SELECT ST_ExteriorRing((ST_DUMP(footprint_latlon)).geom) AS the_geom
	     FROM images WHERE images.footprint_latlon IS NOT NULL) AS lines
	) AS noded_lines))),
iid AS (
 SELECT images.id, intersectiongeom.geom AS geom
		FROM images, intersectiongeom
		WHERE images.footprint_latlon is NOT NULL AND
		ST_INTERSECTS(intersectiongeom.geom, images.footprint_latlon) AND
		ST_AREA(ST_INTERSECTION(intersectiongeom.geom, images.footprint_latlon)) > 0.000001
)
INSERT INTO overlay(intersections, geom) SELECT row.intersections, row.geom FROM
(SELECT iid.geom, array_agg(iid.id) AS intersections
  FROM iid GROUP BY iid.geom) AS row WHERE array_length(intersections, 1) > 1;
"""

def place_points_in_overlaps(nodes, size_threshold=0.0007,
                             iterative_phase_kwargs={'size':71},
                             distribute_points_kwargs={}):
    """
    Place points in all of the overlap geometries by back-projecing using
    sensor models.

    The DEM specified in the config file will be used to calculate point elevations.

    Parameters
    ----------
    nodes : dict-link
            A dict like object with a shared key with the intersection
            field of the database Overlay table and a cg node object
            as the value. This could be a NetworkCandidateGraph or some
            other dict-like object.

    size_threshold : float
                     overlaps with area <= this threshold are ignored

    iterative_phase_kwargs : dict
        Dictionary of keyword arguments for the iterative phase matcher function
    """
    points = []
    for o in Overlay.overlapping_larger_than(size_threshold):
        overlaps = o.intersections
        if overlaps == None:
            continue
        overlapnodes = [nodes[id]["data"] for id in overlaps]
        points.extend(place_points_in_overlap(overlapnodes, o.geom, dem=dem,
                                              iterative_phase_kwargs=iterative_phase_kwargs,
                                              distribute_points_kwargs=distribute_points_kwargs))

    Points.bulkadd(points)

def cluster_place_points_in_overlaps(size_threshold=0.0007,
                                     iterative_phase_kwargs={'size':71},
                                     distribute_points_kwargs={},
                                     walltime='00:10:00', cam_type="csm"):
    """
    Place points in all of the overlap geometries by back-projecing using
    sensor models. This method uses the cluster to process all of the overlaps
    in parallel. See place_points_in_overlap and acn_overlaps.

    The DEM specified in the config file will be used to calculate point elevations.

    Parameters
    ----------
    size_threshold : float
        overlaps with area <= this threshold are ignored

    iterative_phase_kwargs : dict
        Dictionary of keyword arguments for the iterative phase matcher function

    walltime : str
        Cluster job wall time as a string HH:MM:SS

    cam_type : str
               options: {"csm", "isis"}
               Pick what kind of camera model implementation to use
    """
    # Get all of the overlaps over the size threshold
    overlaps = Overlay.overlapping_larger_than(size_threshold)

    # Setup the redis queue
    rqueue = StrictRedis(host=config['redis']['host'],
                         port=config['redis']['port'],
                         db=0)

    # Push the job messages onto the queue
    queuename = config['redis']['processing_queue']
    for overlap in overlaps:
        msg = {'id' : overlap.id,
               'iterative_phase_kwargs' : iterative_phase_kwargs,
               'distribute_points_kwargs' : distribute_points_kwargs,
               'walltime' : walltime,
               'cam_type': cam_type}
        rqueue.rpush(queuename, json.dumps(msg))
    job_counter = len([*overlaps]) + 1

    # Submit the jobs
    submitter = Slurm('acn_overlaps',
                 mem_per_cpu=config['cluster']['processing_memory'],
                 time=walltime,
                 partition=config['cluster']['queue'],
                 output=config['cluster']['cluster_log_dir']+'/autocnet.place_points-%j')
    submitter.submit(array='1-{}'.format(job_counter))
    return job_counter

def place_points_in_overlap(nodes, geom, dem=dem, cam_type="csm",
                            iterative_phase_kwargs={'size':71},
                            distribute_points_kwargs={}):
    """
    Place points into an overlap geometry by back-projecing using sensor models.

    Parameters
    ----------
    nodes : list of Nodes
        The Nodes or Networknodes of all the images that intersect the overlap

    geom : geometry
        The geometry of the overlap region

    dem : GeoDataset
         The DEM used to compute point elevations. An elevation of 0 is used
         if no DEM is passed in.

    iterative_phase_kwargs : dict
        Dictionary of keyword arguments for the iterative phase matcher function

    cam_type : str
               options: {"csm", "isis"}
               Pick what kind of camera model implementation to use

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
    ecef = pyproj.Proj(proj='geocent', a=semi_major, b=semi_minor)
    lla = pyproj.Proj(proj='latlon', a=semi_major, b=semi_minor)

    valid = compgeom.distribute_points_in_geom(geom, **distribute_points_kwargs)
    if not valid:
        warnings.warn('Failed to distribute points in overlap')
        return []

    # Grab the source image. This is just the node with the lowest ID, nothing smart.
    source = nodes[0]
    nodes.remove(source)
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
        # Get the BCEF coordinate from the lon, lat
        x, y, z = pyproj.transform(lla, ecef, lon, lat, height)
        geom = shapely.geometry.Point(x, y, z)
        point = Points(apriori=geom,
                       adjusted=geom,
                       pointtype=2) # Would be 3 or 4 for ground

        if cam_type == "csm":
            gnd = csmapi.EcefCoord(x, y, z)
            sic = source.camera.groundToImage(gnd)
            ssample, sline = sic.samp, sic.line
        if cam_type == "isis":
            sline, ssample = isis.ground_to_image(source["image_path"], lat ,lon)

        point.measures.append(Measures(sample=ssample,
                                       line=sline,
                                       imageid=source['node_id'],
                                       serial=source.isis_serial,
                                       measuretype=3))


        for i, dest in enumerate(nodes):
            if cam_type == "csm":
                dic = dest.camera.groundToImage(gnd)
                dline, dsample = dic.line, dic.samp
            if cam_type == "isis":
                dline, dsample = isis.ground_to_image(dest["image_path"], lat, lon)

            dx, dy, _ = iterative_phase(ssample, sline, dsample, dline,
                                        source.geodata, dest.geodata,
                                        **iterative_phase_kwargs)
            if dx is not None or dy is not None:
                point.measures.append(Measures(sample=dx,
                                               line=dy,
                                               imageid=dest['node_id'],
                                               serial=dest.isis_serial,
                                               measuretype=3))
        if len(point.measures) >= 2:
            points.append(point)
    return points
