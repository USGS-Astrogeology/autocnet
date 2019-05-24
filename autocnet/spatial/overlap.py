import warnings
from autocnet import config
from autocnet.cg import cg as compgeom
from autocnet.io.db.model import Images, Measures, MeasureType, Overlay, Points, PointType
from autocnet.matcher.subpixel import iterative_phase
from autocnet import Session, engine

import csmapi
import numpy as np
import pyproj
import geoalchemy2
import shapely
import sqlalchemy

def compute_overlaps(sql='(SELECT * FROM images) AS images'):
    """
    For the candidate graph, compute the overlapping polygons that
    comprise the entire candidate graph / footprint map. Each overlap
    includes an 'overlaps' attribute/column that includes a list of the
    footprint polygons that have contributed to given overlap.

    """
    query = f"""
SELECT ST_AsEWKB(geom) AS geom FROM ST_Dump((
    SELECT ST_Polygonize(the_geom) AS the_geom FROM (
        SELECT ST_Union(the_geom) AS the_geom FROM (
            SELECT ST_ExteriorRing((ST_DUMP(footprint_latlon)).geom) AS the_geom
            FROM {sql}) AS lines
    ) AS noded_lines
)
)"""
    if not Session:
        warnings.warn('This function requires a database connection configured via an autocnet config file.')
        return 

    session = Session()
    oquery = session.query(Overlay)
    iquery = session.query(Images)
    
    srid = config['spatial']['srid']
    
    rows = []
    for q in engine.execute(query).fetchall():
        overlaps = []
        b = bytes(q['geom'])
        qgeom = shapely.wkb.loads(b)
        res = iquery.filter(Images.footprint_latlon.ST_Intersects(geoalchemy2.shape.from_shape(qgeom,
                                                                                               srid=srid)))
        for i in res:
            fgeom = geoalchemy2.shape.to_shape(i.footprint_latlon)
            area = qgeom.intersection(fgeom).area
            if area < 1e-6:
                continue
            overlaps.append(i.id)
        o = Overlay(geom=f'srid={srid};{qgeom.wkt}', overlaps=overlaps)
        res = oquery.filter(Overlay.overlaps == o.overlaps).first()
        if res is None:
            rows.append(o)

    session.bulk_save_objects(rows)
    session.commit()

    # If an overlap has only 1 entry, it is a sliver and we want to remove it.
    res = oquery.filter(sqlalchemy.func.array_length(Overlay.overlaps, 1) <= 1)
    res.delete(synchronize_session=False)
    session.commit()
    session.close()

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
    for o in session.query(Overlay.id, Overlay.geom, Overlay.overlaps).\
             filter(sqlalchemy.func.ST_Area(Overlay.geom) >= size_threshold):

        valid = compgeom.distribute_points_in_geom(geoalchemy2.shape.to_shape(o.geom))
        if not valid:
            continue
        overlaps = o.overlaps
    
        if reference is None:
            source = overlaps[0]
        else:
            source = reference
        overlaps.remove(source)
        source = cg.node[source]['data']
        source_camera = source.camera
        for v in valid:
            geom = f'srid={srid};Point({v[0]} {v[1]})'
            point = Points(geom=geom,
                           pointtype=PointType(2)) # Would be 3 or 4 for ground

            # Get the BCEF coordinate from the lon, lat
            x, y, z = pyproj.transform(lla, ecef, v[0], v[1], height)  # -3000 working well in elysium, need aeroid
            gnd = csmapi.EcefCoord(x, y, z)

            # Grab the source image. This is just the node with the lowest ID, nothing smart.
            sic = source_camera.groundToImage(gnd)
            point.measures.append(Measures(sample=sic.samp, 
                                           line=sic.line,
                                           imageid=source['node_id'],
                                           serial=source.isis_serial,
                                           measuretype=MeasureType(3)))


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
                                                   measuretype=MeasureType(3)))
            if len(point.measures) >= 2:
                points.append(point)
    session.add_all(points)
    session.commit()

