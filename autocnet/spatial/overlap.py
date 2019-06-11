import warnings
from autocnet import config
from autocnet.cg import cg as compgeom
from autocnet.io.db.model import Images, Measures, Overlay, Points
from autocnet.matcher.subpixel import iterative_phase
from autocnet import Session, engine

import csmapi
import numpy as np
import pyproj
import shapely
import sqlalchemy


def place_points_in_overlaps(cg, size_threshold=0.0007, reference=None, height=0,
                             iterative_phase_kwargs={'size':71}):
    """
    Given a geometry, place points into the geometry by back-projecing using
    a sensor model.compgeom

    TODO: This shoucompgeomn once that package is stable.

    Parameters
    ----------
    cg : CandiateGraph object
         that is used to access sensor information

    size_threshold : float
                     overlaps with area <= this threshold are ignored

    reference : int
                the i.d. of a reference node to use when placing points. If not
                speficied, this is the node with the lowest id

    height : numeric
             The distance (in meters) above or below the aeroid (meters above or
             below the BCBF spheroid).
    """
    if not Session:
        warnings.warn('This function requires a database connection configured via an autocnet config file.')
        return

    points = []
    session = Session()
    srid = config['spatial']['srid']
    semi_major = config['spatial']['semimajor_rad']
    semi_minor = config['spatial']['semiminor_rad']
    ecef = pyproj.Proj(proj='geocent', a=semi_major, b=semi_minor)
    lla = pyproj.Proj(proj='latlon', a=semi_major, b=semi_minor)

    # TODO: This should be a passable query where we can subset.
    for o in session.query(Overlay).\
             filter(sqlalchemy.func.ST_Area(Overlay.geom) >= size_threshold):

        valid = compgeom.distribute_points_in_geom(o.geom)
        if not valid:
            continue

        overlaps = o.intersections

        if overlaps == None:
            continue

        if reference is None:
            source = overlaps[0]
        else:
            source = reference
        overlaps.remove(source)
        source = cg.node[source]['data']
        source_camera = source.camera
        for v in valid:
            point = Points(geom=shapely.geometry.Point(*v),
                           pointtype=2) # Would be 3 or 4 for ground

            # Get the BCEF coordinate from the lon, lat
            x, y, z = pyproj.transform(lla, ecef, v[0], v[1], height)  # -3000 working well in elysium, need aeroid
            gnd = csmapi.EcefCoord(x, y, z)

            # Grab the source image. This is just the node with the lowest ID, nothing smart.
            sic = source_camera.groundToImage(gnd)
            point.measures.append(Measures(sample=sic.samp,
                                           line=sic.line,
                                           imageid=source['node_id'],
                                           serial=source.isis_serial,
                                           measuretype=3))


            for i, d in enumerate(overlaps):
                destination = cg.node[d]['data']
                destination_camera = destination.camera
                dic = destination_camera.groundToImage(gnd)
                dx, dy, metrics = iterative_phase(sic.samp, sic.line, dic.samp, dic.line,
                                                  source.geodata, destination.geodata,
                                                  **iterative_phase_kwargs)
                if dx is not None or dy is not None:
                    point.measures.append(Measures(sample=dx,
                                                   line=dy,
                                                   imageid=destination['node_id'],
                                                   serial=destination.isis_serial,
                                                   measuretype=3))
            if len(point.measures) >= 2:
                points.append(point)
    session.add_all(points)
    session.commit()

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