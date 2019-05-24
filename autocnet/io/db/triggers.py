from sqlalchemy.schema import DDL

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