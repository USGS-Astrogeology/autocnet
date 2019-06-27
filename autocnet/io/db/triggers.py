from sqlalchemy.schema import DDL

from autocnet import config

valid_geom_function = DDL("""
CREATE OR REPLACE FUNCTION validate_geom()
  RETURNS trigger AS
$BODY$
  BEGIN
      NEW.footprint_latlon = ST_MAKEVALID(NEW.footprint_latlon);
      RETURN NEW;
    EXCEPTION WHEN OTHERS THEN
      NEW.active = false;
      RETURN NEW;
END;
$BODY$

LANGUAGE plpgsql VOLATILE -- Says the function is implemented in the plpgsql language; VOLATILE says the function has side effects.
COST 100; -- Estimated execution cost of the function.
""")

valid_geom_trigger = DDL("""
CREATE TRIGGER image_inserted
  BEFORE INSERT OR UPDATE
  ON images
  FOR EACH ROW
EXECUTE PROCEDURE validate_geom();
""")

valid_point_function = DDL("""
CREATE OR REPLACE FUNCTION validate_points()
  RETURNS trigger AS
$BODY$
BEGIN
 IF (SELECT COUNT(*)
	 FROM MEASURES
	 WHERE pointid = NEW.pointid AND active = True) < 2
 THEN
   UPDATE points
     SET active = False
	 WHERE points.id = NEW.pointid;
 ELSE
   UPDATE points
   SET active = True
   WHERE points.id = NEW.pointid;
 END IF;

 RETURN NEW;
END;
$BODY$

LANGUAGE plpgsql VOLATILE -- Says the function is implemented in the plpgsql language; VOLATILE says the function has side effects.
COST 100; -- Estimated execution cost of the function.
""")

valid_point_trigger = DDL("""
CREATE TRIGGER active_measure_changes
  AFTER UPDATE
  ON measures
  FOR EACH ROW
EXECUTE PROCEDURE validate_points();
""")

latitudinal_srid = config['spatial']['latitudinal_srid']

update_point_function = DDL("""
CREATE OR REPLACE FUNCTION update_points()
  RETURNS trigger AS
$BODY$
BEGIN
    NEW.geom = ST_Force_2D(ST_Transform(NEW.adjusted, {}));
    RETURN NEW;
  EXCEPTION WHEN OTHERS THEN
    NEW.geom = Null;
    RETURN NEW;
END;
$BODY$

LANGUAGE plpgsql VOLATILE -- Says the function is implemented in the plpgsql language; VOLATILE says the function has side effects.
COST 100; -- Estimated execution cost of the function.
""".format(latitudinal_srid))

update_point_trigger = DDL("""
CREATE TRIGGER point_inserted
  BEFORE INSERT OR UPDATE
  ON points
  FOR EACH ROW
EXECUTE PROCEDURE update_points();
""")
