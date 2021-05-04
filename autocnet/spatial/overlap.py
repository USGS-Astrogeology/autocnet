import warnings
import json

from redis import StrictRedis
import numpy as np
import pyproj
import shapely
import sqlalchemy
from plio.io.io_gdal import GeoDataset
from pysis.exceptions import ProcessError

from autocnet.cg import cg as compgeom
from autocnet.graph.node import NetworkNode
from autocnet.io.db.model import Images, Measures, Overlay, Points, JsonEncoder
from autocnet.spatial import isis
from autocnet.matcher.cpu_extractor import extract_most_interesting
from autocnet.transformation.spatial import reproject, og2oc, oc2og
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

def place_points_in_overlaps(size_threshold=0.0007,
                             distribute_points_kwargs={},
                             cam_type='csm',
                             point_type=2,
                             ncg=None):
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

    Session : obj
              The session object from the NetworkCandidateGraph

    size_threshold : float
                     overlaps with area <= this threshold are ignored
    cam_type : str
               Either 'csm' (default) or 'isis'. The type of sensor model to use.

    point_type : int
                 Either 2 (free;default) or 3 (constrained). Point type 3 should be used for
                 ground.
    """
    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    for overlap in Overlay.overlapping_larger_than(size_threshold, Session):
        if overlap.intersections == None:
            continue
        place_points_in_overlap(overlap,
                                cam_type=cam_type,
                                distribute_points_kwargs=distribute_points_kwargs,
                                point_type=point_type,
                                ncg=ncg)

def place_points_in_overlap(overlap,
                            identifier="autocnet",
                            cam_type="csm",
                            size=71,
                            distribute_points_kwargs={},
                            point_type=2,
                            ncg=None,
                            use_cache=False,
                            **kwargs):
    """
    Place points into an overlap geometry by back-projecing using sensor models.
    The DEM specified in the config file will be used to calculate point elevations.

    Parameters
    ----------
    overlap : obj
              An autocnet.io.db.model Overlay model instance.

    identifier: str
                The tag used to distiguish points laid down by this function.

    cam_type : str
               options: {"csm", "isis"}
               Pick what kind of camera model implementation to use.

    size : int
           The amount of pixel around a points initial location to search for an
           interesting feature to which to shift the point.

    distribute_points_kwargs: dict
                              kwargs to pass to autocnet.cg.cg.distribute_points_in_geom

    point_type: int
                The type of point being placed. Default is pointtype=2, corresponding to
                free points.

    ncg: obj
         An autocnet.graph.network NetworkCandidateGraph instance representing the network
         to apply this function to

    use_cache : bool
                If False (default) this func opens a database session and writes points
                and measures directly to the respective tables. If True, this method writes 
                messages to the point_insert (defined in ncg.config) redis queue for 
                asynchronous (higher performance) inserts.

    Returns
    -------
    points : list of Points
        The list of points seeded in the overlap

    See Also
    --------
    autocnet.io.db.model.Overlay: for associated properties of the Overlay object

    autocnet.cg.cg.distribute_points_in_geom: for the possible arguments to pass through using
    disribute_points_kwargs.

    autocnet.model.io.db.PointType: for the point type options.

    autocnet.graph.network.NetworkCandidateGraph: for associated properties and functionalities of the
    NetworkCandidateGraph class
    """
    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    # Determine what sensor type to use
    avail_cams = {"isis", "csm"}
    cam_type = cam_type.lower()
    if cam_type not in cam_type:
        raise Exception(f'{cam_type} is not one of valid camera: {avail_cams}')

    points = []
    semi_major = ncg.config['spatial']['semimajor_rad']
    semi_minor = ncg.config['spatial']['semiminor_rad']

    # Determine the point distribution in the overlap geom
    geom = overlap.geom
    valid = compgeom.distribute_points_in_geom(geom, **distribute_points_kwargs, **kwargs)
    if not valid.any():
        warnings.warn('Failed to distribute points in overlap')
        return []

    print(f'Have {len(valid)} potential points to place.')

    # Setup the node objects that are covered by the geom
    nodes = []
    with ncg.session_scope() as session:
        for id in overlap.intersections:
            res = session.query(Images).filter(Images.id == id).one()
            nn = NetworkNode(node_id=id, image_path=res.path)
            nn.parent = ncg
            nodes.append(nn)
    
    print(f'Attempting to place measures in {len(nodes)} images.')
    for v in valid:
        lon = v[0]
        lat = v[1]

        # Calculate the height, the distance (in meters) above or
        # below the aeroid (meters above or below the BCBF spheroid).
        px, py = ncg.dem.latlon_to_pixel(lat, lon)
        height = ncg.dem.read_array(1, [px, py, 1, 1])[0][0]

        # Need to get the first node and then convert from lat/lon to image space
        for reference_index, node in enumerate(nodes):  
            # reference_index is the index into the list of measures for the image that is not shifted and is set at the 
            # reference against which all other images are registered.
            if cam_type == "isis":
                try:
                    line, sample = isis.ground_to_image(node["image_path"], lon, lat)
                except ProcessError as e:
                    if 'Requested position does not project in camera model' in e.stderr:
                        print(f'point ({lon}, {lat}) does not project to reference image {node["image_path"]}')
                        continue
            if cam_type == "csm":
                lon_og, lat_og = oc2og(lon, lat, semi_major, semi_minor)
                x, y, z = reproject([lon_og, lat_og, height],
                                    semi_major, semi_minor,
                                    'latlon', 'geocent')
                # The CSM conversion makes the LLA/ECEF conversion explicit
                gnd = csmapi.EcefCoord(x, y, z)
                image_coord = node.camera.groundToImage(gnd)
                sample, line = image_coord.samp, image_coord.line

            # Extract ORB features in a sub-image around the desired point
            image_roi = roi.Roi(node.geodata, sample, line, size_x=size, size_y=size)
            if image_roi.variance == 0:
                warnings.warn(f'Failed to find interesting features in image {node.image_name}.')
                continue
            image = image_roi.clip()

            # Extract the most interesting feature in the search window
            interesting = extract_most_interesting(image)
            if interesting is not None:
                # We have found an interesting feature and have identified the reference point.
                break
 
        if interesting is None:
            warnings.warn('Unable to find an interesting point, falling back to the a priori pointing')
            newsample = sample
            newline = line
        else:
            # kps are in the image space with upper left origin and the roi
            # could be the requested size or smaller if near an image boundary.
            # So use the roi upper left_x and top_y for the actual origin.
            left_x, _, top_y, _ = image_roi.image_extent
            newsample = left_x + interesting.x
            newline = top_y + interesting.y

        # Get the updated lat/lon from the feature in the node
        if cam_type == "isis":
            try:
                p = isis.point_info(node["image_path"], newsample, newline, point_type="image")
            except ProcessError as e:
                if 'Requested position does not project in camera model' in e.stderr:
                    print(node["image_path"])
                    print(f'interesting point ({newsample}, {newline}) does not project back to ground')
                    continue
            try:
                x, y, z = p["BodyFixedCoordinate"].value
            except:
                x, y, z = ["BodyFixedCoordinate"]

            if getattr(p["BodyFixedCoordinate"], "units", "None").lower() == "km":
                x = x * 1000
                y = y * 1000
                z = z * 1000
        elif cam_type == "csm":
            image_coord = csmapi.ImageCoord(newline, newsample)
            pcoord = node.camera.imageToGround(image_coord)
            # Get the BCEF coordinate from the lon, lat
            updated_lon_og, updated_lat_og, _ = reproject([pcoord.x, pcoord.y, pcoord.z],
                                                           semi_major, semi_minor, 'geocent', 'latlon')
            updated_lon, updated_lat = og2oc(updated_lon_og, updated_lat_og, semi_major, semi_minor)

            px, py = ncg.dem.latlon_to_pixel(updated_lat, updated_lon)
            updated_height = ncg.dem.read_array(1, [px, py, 1, 1])[0][0]


            # Get the BCEF coordinate from the lon, lat
            x, y, z = reproject([updated_lon_og, updated_lat_og, updated_height],
                                semi_major, semi_minor, 'latlon', 'geocent')

        # If the updated point is outside of the overlap, then revert back to the
        # original point and hope the matcher can handle it when sub-pixel registering
        updated_lon_og, updated_lat_og, updated_height = reproject([x, y, z], semi_major, semi_minor,
                                                             'geocent', 'latlon')
        updated_lon, updated_lat = og2oc(updated_lon_og, updated_lat_og, semi_major, semi_minor)

        if not geom.contains(shapely.geometry.Point(updated_lon, updated_lat)):
            lon_og, lat_og = oc2og(lon, lat, semi_major, semi_minor)
            x, y, z = reproject([lon_og, lat_og, height],
                                semi_major, semi_minor, 'latlon', 'geocent')
            updated_lon_og, updated_lat_og, updated_height = reproject([x, y, z], semi_major, semi_minor,
                                                                 'geocent', 'latlon')
            updated_lon, updated_lat = og2oc(updated_lon_og, updated_lat_og, semi_major, semi_minor)

        point_geom = shapely.geometry.Point(x, y, z)
        point = Points(identifier=identifier,
                       overlapid=overlap.id,
                       apriori=point_geom,
                       adjusted=point_geom,
                       pointtype=point_type, # Would be 3 or 4 for ground
                       cam_type=cam_type,
                       reference_index=reference_index)

        # Compute ground point to back project into measurtes
        gnd = csmapi.EcefCoord(x, y, z)

        for node in nodes:
            if cam_type == "csm":
                image_coord = node.camera.groundToImage(gnd)
                sample, line = image_coord.samp, image_coord.line
            if cam_type == "isis":
                try:
                    line, sample = isis.ground_to_image(node["image_path"], updated_lon, updated_lat)
                except ProcessError as e:
                    if 'Requested position does not project in camera model' in e.stderr:
                        print(f'interesting point ({updated_lon},{updated_lat}) does not project to image {node["image_path"]}')
                        continue

            point.measures.append(Measures(sample=sample,
                                           line=line,
                                           apriorisample=sample,
                                           aprioriline=line,
                                           imageid=node['node_id'],
                                           serial=node.isis_serial,
                                           measuretype=3,
                                           choosername='place_points_in_overlap'))

        if len(point.measures) >= 2:
            points.append(point)
    print(f'Able to place {len(points)} points.')

    # Insert the points into the database asynchronously (via redis) or synchronously via the ncg
    if use_cache:
        # Push
        ncg.redis_queue.rpush(ncg.point_insert_queue, *[json.dumps(point.to_dict(_hide=[]), cls=JsonEncoder) for point in points])
        ncg.redis_queue.incr(ncg.point_insert_counter, amount=len(points))
    else:
        Points.bulkadd(points, ncg.Session)
    return points

def place_points_in_image(image,
                          identifier="autocnet",
                          cam_type="csm",
                          size=71,
                          distribute_points_kwargs={},
                          ncg=None,
                          **kwargs):
    """
    Place points into an image and then attempt to place measures
    into all overlapping images. This function is funcitonally identical
    to place_point_in_overlap except it works on individual images.

    Parameters
    ----------
    image : obj
            An autocnet Images model object

    identifier: str
                The tag used to distiguish points laid down by this function.

    cam_type : str
               options: {"csm", "isis"}
               Pick what kind of camera model implementation to use

    size : int
           The size of the window used to extractor features to find an
           interesting feature to which the point is shifted.

    distribute_points_kwargs: dict
                              kwargs to pass to autocnet.cg.cg.distribute_points_in_geom

    ncg: obj
         An autocnet.graph.network NetworkCandidateGraph instance representing the network
         to apply this function to

    See Also
    --------
    autocnet.cg.cg.distribute_points_in_geom: for the possible arguments to pass through using
    disribute_points_kwargs.

    autocnet.graph.network.NetworkCandidateGraph: for associated properties and functionalities of the
    NetworkCandidateGraph class
    """
    # Arg checking
    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    # Determine what sensor type to use
    avail_cams = {"isis", "csm"}
    cam_type = cam_type.lower()
    if cam_type not in cam_type:
        raise Exception(f'{cam_type} is not one of valid camera: {avail_cams}')

    points = []
    semi_major = ncg.config['spatial']['semimajor_rad']
    semi_minor = ncg.config['spatial']['semiminor_rad']

    # Logic
    geom = image.geom
    # Put down a grid of points over the image; the density is intentionally high
    valid = compgeom.distribute_points_in_geom(geom, **distribute_points_kwargs)
    print(f'Have {len(valid)} potential points to place.')
    for v in valid:
        lon = v[0]
        lat = v[1]
        point_geometry = f'SRID=949900;POINT({v[0]} {v[1]})'

        # Calculate the height, the distance (in meters) above or
        # below the aeroid (meters above or below the BCBF spheroid).
        px, py = ncg.dem.latlon_to_pixel(lat, lon)
        height = ncg.dem.read_array(1, [px, py, 1, 1])[0][0]

        with ncg.session_scope() as session:
            intersecting_images = session.query(Images.id, Images.path).filter(Images.geom.ST_Intersects(point_geometry)).all()
            node_res= [i for i in intersecting_images]
            nodes = []

            for nid, image_path  in node_res:
                # Setup the node objects that are covered by the geom
                nn = NetworkNode(node_id=nid, image_path=image_path)
                nn.parent = ncg
                nodes.append(nn)

        # Need to get the first node and then convert from lat/lon to image space
        node = nodes[0]
        if cam_type == "isis":
            try:
                line, sample = isis.ground_to_image(node["image_path"], lon, lat)
            except ProcessError as e:
                if 'Requested position does not project in camera model' in e.stderr:
                    print(f'point ({lon}, {lat}) does not project to reference image {node["image_path"]}')
                    continue
        if cam_type == "csm":
            lon_og, lat_og = oc2og(lon, lat, semi_major, semi_minor)
            x, y, z = reproject([lon_og, lat_og, height],
                                semi_major, semi_minor,
                                'latlon', 'geocent')
            # The CSM conversion makes the LLA/ECEF conversion explicit
            gnd = csmapi.EcefCoord(x, y, z)
            image_coord = node.camera.groundToImage(gnd)
            sample, line = image_coord.samp, image_coord.line

        # Extract ORB features in a sub-image around the desired point
        image_roi = roi.Roi(node.geodata, sample, line, size_x=size, size_y=size)
        image = image_roi.clip()
        try:
            interesting = extract_most_interesting(image)
        except:
            continue

        # kps are in the image space with upper left origin and the roi
        # could be the requested size or smaller if near an image boundary.
        # So use the roi upper left_x and top_y for the actual origin.
        left_x, _, top_y, _ = image_roi.image_extent
        newsample = left_x + interesting.x
        newline = top_y + interesting.y

        # Get the updated lat/lon from the feature in the node
        if cam_type == "isis":
            try:
                p = isis.point_info(node["image_path"], newsample, newline, point_type="image")
            except ProcessError as e:
                if 'Requested position does not project in camera model' in e.stderr:
                    print(node["image_path"])
                    print(f'interesting point ({newsample}, {newline}) does not project back to ground')
                    continue
            try:
                x, y, z = p["BodyFixedCoordinate"].value
            except:
                x, y, z = ["BodyFixedCoordinate"]

            if getattr(p["BodyFixedCoordinate"], "units", "None").lower() == "km":
                x = x * 1000
                y = y * 1000
                z = z * 1000
        elif cam_type == "csm":
            image_coord = csmapi.ImageCoord(newline, newsample)
            pcoord = node.camera.imageToGround(image_coord)
            # Get the BCEF coordinate from the lon, lat
            updated_lon_og, updated_lat_og, _ = reproject([pcoord.x, pcoord.y, pcoord.z],
                                                           semi_major, semi_minor, 'geocent', 'latlon')
            updated_lon, updated_lat = og2oc(updated_lon_og, updated_lat_og, semi_major, semi_minor)

            px, py = ncg.dem.latlon_to_pixel(updated_lat, updated_lon)
            updated_height = ncg.dem.read_array(1, [px, py, 1, 1])[0][0]


            # Get the BCEF coordinate from the lon, lat
            x, y, z = reproject([updated_lon_og, updated_lat_og, updated_height],
                                semi_major, semi_minor, 'latlon', 'geocent')

        # If the updated point is outside of the overlap, then revert back to the
        # original point and hope the matcher can handle it when sub-pixel registering
        updated_lon_og, updated_lat_og, updated_height = reproject([x, y, z], semi_major, semi_minor,
                                                             'geocent', 'latlon')
        updated_lon, updated_lat = og2oc(updated_lon_og, updated_lat_og, semi_major, semi_minor)

        if not geom.contains(shapely.geometry.Point(updated_lon, updated_lat)):
            lon_og, lat_og = oc2og(lon, lat, semi_major, semi_minor)
            x, y, z = reproject([lon_og, lat_og, height],
                                semi_major, semi_minor, 'latlon', 'geocent')
            updated_lon_og, updated_lat_og, updated_height = reproject([x, y, z], semi_major, semi_minor,
                                                                 'geocent', 'latlon')
            updated_lon, updated_lat = og2oc(updated_lon_og, updated_lat_og, semi_major, semi_minor)

        point_geom = shapely.geometry.Point(x, y, z)

        # Insert a spatial query to find which overlap this is in.
        with ncg.session_scope() as session:
            oid = session.query(Overlay.id).filter(Overlay.geom.ST_Contains(point_geometry)).one()[0]

        point = Points(identifier=identifier,
                       overlapid=oid,
                       apriori=point_geom,
                       adjusted=point_geom,
                       pointtype=2, # Would be 3 or 4 for ground
                       cam_type=cam_type)

        for node in nodes:
            insert = True
            if cam_type == "csm":
                image_coord = node.camera.groundToImage(gnd)
                sample, line = image_coord.samp, image_coord.line
            if cam_type == "isis":
                try:
                    line, sample = isis.ground_to_image(node["image_path"], updated_lon, updated_lat)
                except ProcessError as e:
                    if 'Requested position does not project in camera model' in e.stderr:
                        print(f'interesting point ({lon},{lat}) does not project to image {node["image_path"]}')
                        insert = False

            point.measures.append(Measures(sample=sample,
                                           line=line,
                                           apriorisample=sample,
                                           aprioriline=line,
                                           imageid=node['node_id'],
                                           serial=node.isis_serial,
                                           measuretype=3,
                                           choosername='place_points_in_image'))

        if len(point.measures) >= 2:
            points.append(point)
    print(f'Able to place {len(points)} points.')
    Points.bulkadd(points, ncg.Session)

def add_measures_to_point(pointid, cam_type='isis', ncg=None, Session=None):
    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')
    
    if isinstance(pointid, Points):
        pointid = pointid.id

    
    with ncg.session_scope() as session:
        point = session.query(Points).filter(Points.id == pointid).one()
        point_lon = point.geom.x
        point_lat = point.geom.y

        reference_index = point.reference_index
        reference_measure = point.measures[reference_index]
        reference_image_id = reference_measure.imageid

        images = session.query(Images).filter(Images.geom.ST_Intersects(point._geom)).all()
        print(f'Placing measures into {len(images)-1} images.')
        for image in images:
            if image.id == reference_image_id:
                continue  # This is the reference image, so pass on adding a new measure
            
            if cam_type == "isis":
                try:
                    line, sample = isis.ground_to_image(image.path, point_lon, point_lat)
                except ProcessError as e:
                    if 'Requested position does not project in camera model' in e.stderr:
                        print(f'interesting point ({point_lon},{point_lat}) does not project to image {image.name}')

            point.measures.append(Measures(sample=sample,
                                           line=line,
                                           apriorisample=sample,
                                           aprioriline=line,
                                           imageid=image.id,
                                           serial=image.serial,
                                           measuretype=3,
                                           choosername='add_measures_to_point')) 
            i = 0
            for m in point.measures:
                if m.measuretype == 2 or m.measuretype == 3:
                    i += 1
            if i >= 2:
                point.ignore = False      
    return
