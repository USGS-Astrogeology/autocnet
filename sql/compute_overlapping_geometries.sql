INSERT INTO overlay(geom)
(
	SELECT ST_AsEWKB(geom) AS geom FROM ST_Dump((
        SELECT ST_Polygonize(the_geom) AS the_geom FROM (
            SELECT ST_Union(the_geom) AS the_geom FROM (
                SELECT ST_ExteriorRing((ST_DUMP(footprint_latlon)).geom) AS the_geom
                FROM images WHERE images.footprint_latlon IS NOT NULL) AS lines
        ) AS noded_lines
    ))
);

UPDATE overlay
SET intersections = imgs.iid
FROM (
		SELECT overlay.id, array_agg(images.id) as iid
		FROM overlay, images
		WHERE images.footprint_latlon is NOT NULL AND
		ST_INTERSECTS(overlay.geom, images.footprint_latlon) AND
		ST_AREA(ST_INTERSECTION(overlay.geom, images.footprint_latlon)) > 0.000001
		GROUP BY overlay.id
	) AS imgs
WHERE imgs.id = overlay.id;
