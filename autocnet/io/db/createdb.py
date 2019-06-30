from sqlalchemy.ext.declarative import declarative_base
from autocnet.io.db.model import (Points, Measures, Images, Costs,
                                Edges, Overlay, Keypoints, Cameras,
                                Matches)
from autocnet.io.db.triggers import valid_point_function, valid_point_trigger, update_point_function, update_point_trigger, valid_geom_function, valid_geom_trigger
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy import create_engine, pool, orm, event
from sqlalchemy.orm import sessionmaker


Base = declarative_base()

def createdb(Session, engine):
    if not isinstance(Session, sessionmaker):
        return
    # Create the database
    if not database_exists(engine.url):
        create_database(engine.url, template='template_postgis')  # This is a hardcode to the local template

        # Trigger that watches for points that should be active/inactive
        # based on the point count.
        event.listen(Base.metadata, 'before_create', valid_point_function)
        event.listen(Measures.__table__, 'after_create', valid_point_trigger)
        event.listen(Base.metadata, 'before_create', update_point_function)
        event.listen(Images.__table__, 'after_create', update_point_trigger)
        event.listen(Base.metadata, 'before_create', valid_geom_function)
        event.listen(Images.__table__, 'after_create', valid_geom_trigger)

    Base.metadata.bind = engine
    # If the table does not exist, this will create it. This is used in case a
    # user has manually dropped a table so that the project is not wrecked.
    Base.metadata.create_all(tables=[Overlay.__table__,
                                    Edges.__table__, Costs.__table__, Matches.__table__,
                                    Cameras.__table__, Points.__table__,
                                    Measures.__table__, Images.__table__,
                                    Keypoints.__table__])