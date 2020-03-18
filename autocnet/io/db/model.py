import enum
import json

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (Column, String, Integer, Float, \
                        ForeignKey, Boolean, LargeBinary, \
                        UniqueConstraint, event)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import relationship, backref
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.types import TypeDecorator
from sqlalchemy.ext.hybrid import hybrid_property

from geoalchemy2 import Geometry
from geoalchemy2.shape import from_shape, to_shape

import osgeo
import shapely
from shapely.geometry import Point
from autocnet import engine, Session, config
from autocnet.transformation.spatial import reproject
from autocnet.utils.serializers import JsonEncoder

Base = declarative_base()

# Default to mars if no config is set
spatial = config.get('spatial', {'latitudinal_srid': 949900, 'rectangular_srid': 949980})
latitudinal_srid = spatial['latitudinal_srid']
rectangular_srid = spatial['rectangular_srid']

class BaseMixin(object):
    @classmethod
    def create(cls, session, **kw):
        obj = cls(**kw)
        session.add(obj)
        session.commit()
        return obj

    @staticmethod
    def bulkadd(iterable):
        session = Session()
        session.add_all(iterable)
        session.commit()
        session.close()

class IntEnum(TypeDecorator):
    """
    Mapper for enum type to sqlalchemy and back again
    """
    impl = Integer
    def __init__(self, enumtype, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enumtype = enumtype

    def process_bind_param(self, value, dialect):
        if hasattr(value, 'value'):
            value = value.value
        return value

    def process_result_value(self, value, dialect):
        return self._enumtype(value)

class ArrayType(TypeDecorator):
    """
    Sqlite does not support arrays. Therefore, use a custom type decorator.

    See http://docs.sqlalchemy.org/en/latest/core/types.html#sqlalchemy.types.TypeDecorator
    """
    impl = String

    def process_bind_param(self, value, dialect):
        return json.dumps(value, cls=JsonEncoder)

    def process_result_value(self, value, dialect):
        return json.loads(value)

    def copy(self):
        return ArrayType(self.impl.length)

class Json(TypeDecorator):
    """
    Sqlite does not have native JSON support. Therefore, use a custom type decorator.

    See http://docs.sqlalchemy.org/en/latest/core/types.html#sqlalchemy.types.TypeDecorator
    """
    impl = String

    @property
    def python_type(self):
        return object

    def process_bind_param(self, value, dialect):
        return json.dumps(value, cls=JsonEncoder)

    def process_literal_param(self, value, dialect):
        return value

    def process_result_value(self, value, dialect):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None


class Keypoints(BaseMixin, Base):
    __tablename__ = 'keypoints'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"))
    convex_hull_image = Column(Geometry('POLYGON'))
    convex_hull_latlon = Column(Geometry('POLYGON', srid=latitudinal_srid))
    path = Column(String)
    nkeypoints = Column(Integer)

    def __repr__(self):
        try:
            chll = to_shape(self.convex_hull_latlon).__geo_interface__
        except:
            chll = None
        return json.dumps({'id':self.id,
                           'image_id':self.image_id,
                           'convex_hull':self.convex_hull_image,
                           'convex_hull_latlon':chll,
                           'path':self.path,
                           'nkeypoints':self.nkeypoints})

class Edges(BaseMixin, Base):
    __tablename__ = 'edges'
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(Integer)
    destination = Column(Integer)
    ring = Column(ArrayType())
    fundamental = Column(ArrayType())
    ignore = Column(Boolean, default=False)
    masks = Column(Json())

class Costs(BaseMixin, Base):
    __tablename__ = 'costs'
    match_id = Column(Integer, ForeignKey("matches.id", ondelete="CASCADE"), primary_key=True)
    _cost = Column(JSONB)

class Matches(BaseMixin, Base):
    __tablename__ = 'matches'
    id = Column(Integer, primary_key=True, autoincrement=True)
    point_id = Column(Integer)
    source_measure_id = Column(Integer)
    destin_measure_id = Column(Integer)
    source = Column(Integer, nullable=False)
    source_idx = Column(Integer, nullable=False)
    destination = Column(Integer, nullable=False)
    destination_idx = Column(Integer, nullable=False)
    lat = Column(Float)
    lon = Column(Float)
    _geom = Column("geom", Geometry('POINT', dimension=2, srid=latitudinal_srid, spatial_index=True))
    source_x = Column(Float)
    source_y = Column(Float)
    destination_x = Column(Float)
    destination_y = Column(Float)
    shift_x = Column(Float)
    shift_y = Column(Float)
    original_destination_x = Column(Float)
    original_destination_y = Column(Float)

    @hybrid_property
    def geom(self):
        try:
            return to_shape(self._geom)
        except:
            return self._geom

    @geom.setter
    def geom(self, geom):
        if geom:  # Supports instances where geom is explicitly set to None.
            self._geom = from_shape(geom, srid=latitudinal_srid)

class Cameras(BaseMixin, Base):
    __tablename__ = 'cameras'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), unique=True)
    camera = Column(Json())
    camtype = Column(String)

class Images(BaseMixin, Base):
    __tablename__ = 'images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    path = Column(String)
    serial = Column(String, unique=True)
    ignore = Column(Boolean, default=False)
    _geom = Column("geom", Geometry('MultiPolygon', srid=latitudinal_srid, dimension=2, spatial_index=True))
    footprint_bodyfixed = Column(Geometry('MULTIPOLYGON', dimension=2))
    cam_type = Column(String)
    #footprint_bodyfixed = Column(Geometry('POLYGON',dimension=3))

    # Relationships
    keypoints = relationship(Keypoints, passive_deletes='all', backref="images", uselist=False)
    cameras = relationship(Cameras, passive_deletes='all', backref='images', uselist=False)
    measures = relationship("Measures")

    def __repr__(self):
        try:
            footprint = to_shape(self.geom).__geo_interface__
        except:
            footprint = None
        return json.dumps({'id':self.id,
                'name':self.name,
                'path':self.path,
                'geom':footprint,
                'footprint_bodyfixed':self.footprint_bodyfixed})

    @hybrid_property
    def geom(self):
        try:
            return to_shape(self._geom)
        except:
            return self._geom

    @geom.setter
    def geom(self, newgeom):
        if isinstance(newgeom, osgeo.ogr.Geometry):
            # If an OGR geom, convert to shapely
            newgeom = shapely.wkt.loads(newgeom.ExportToWkt())
        if newgeom is None:
            self._geom = None
        else:
            self._geom = from_shape(newgeom, srid=latitudinal_srid)

class Overlay(BaseMixin, Base):
    __tablename__ = 'overlay'
    id = Column(Integer, primary_key=True, autoincrement=True)
    intersections = Column(ARRAY(Integer))
    #geom = Column(Geometry(geometry_type='POLYGON', management=True))  # sqlite
    _geom = Column("geom", Geometry('POLYGON', srid=latitudinal_srid, dimension=2, spatial_index=True))  # postgresql

    @hybrid_property
    def geom(self):
        try:
            return to_shape(self._geom)
        except:
            return self._geom
    @geom.setter
    def geom(self, geom):
        self._geom = from_shape(geom, srid=latitudinal_srid)

    @classmethod
    def overlapping_larger_than(cls, size_threshold):
        """
        Query the Overlay table for an iterable of responses where the objects
        in the iterable have an area greater than a given size.

        Parameters
        ----------
        size_threshold : Number
                        area >= this arg are returned
        """
        session = Session()
        res = session.query(cls).\
                filter(sqlalchemy.func.ST_Area(cls.geom) >= size_threshold).\
                filter(sqlalchemy.func.array_length(cls.intersections, 1) > 1)
        session.close()
        return res

class PointType(enum.IntEnum):
    """
    Enum to enforce point type for ISIS control networks
    """
    free = 2
    constrained = 3
    fixed = 4

class Points(BaseMixin, Base):
    __tablename__ = 'points'
    id = Column(Integer, primary_key=True, autoincrement=True)
    _pointtype = Column("pointType", IntEnum(PointType), nullable=False)  # 2, 3, 4 - Could be an enum in the future, map str to int in a decorator
    identifier = Column(String, unique=True)
    _geom = Column("geom", Geometry('POINT', srid=latitudinal_srid, dimension=2, spatial_index=True))
    cam_type = Column(String)
    ignore = Column("pointIgnore", Boolean, default=False)
    _apriori = Column("apriori", Geometry('POINTZ', srid=rectangular_srid, dimension=3, spatial_index=False))
    _adjusted = Column("adjusted", Geometry('POINTZ', srid=rectangular_srid, dimension=3, spatial_index=False))
    measures = relationship('Measures')

    @hybrid_property
    def geom(self):
        try:
            return to_shape(self._geom)
        except:
            return self._geom

    @geom.setter
    def geom(self, geom):
        raise TypeError("The geom column for Points cannot be set." \
                        " Set the adjusted column to update the geom.")

    @hybrid_property
    def apriori(self):
        try:
            return to_shape(self._apriori)
        except:
            return self._apriori

    @apriori.setter
    def apriori(self, apriori):
        if apriori:
            self._apriori = from_shape(apriori, srid=rectangular_srid)
        else:
            self._apriori = apriori

    @hybrid_property
    def adjusted(self):
        try:
            return to_shape(self._adjusted)
        except:
            return self._adjusted

    @adjusted.setter
    def adjusted(self, adjusted):
        if adjusted:
            self._adjusted = from_shape(adjusted, srid=rectangular_srid)
            lat, lon, _ = reproject([adjusted.x, adjusted.y, adjusted.z],
                                    spatial['semimajor_rad'], spatial['semiminor_rad'],
                                    'geocent', 'latlon')
            self._geom = from_shape(Point(lat, lon), latitudinal_srid)
        else:
            self._adjusted = adjusted
            self._geom = None

    @hybrid_property
    def pointtype(self):
        return self._pointtype

    @pointtype.setter
    def pointtype(self, v):
        if isinstance(v, int):
            v = PointType(v)
        self._pointtype = v

class MeasureType(enum.IntEnum):
    """
    Enum to enforce measure type for ISIS control networks
    """
    candidate = 0
    manual = 1
    pixelregistered = 2
    subpixelregistered = 3

class Measures(BaseMixin, Base):
    __tablename__ = 'measures'
    id = Column(Integer,primary_key=True, autoincrement=True)
    pointid = Column(Integer, ForeignKey('points.id'), nullable=False)
    imageid = Column(Integer, ForeignKey('images.id'))
    serial = Column("serialnumber", String, nullable=False)
    _measuretype = Column("measureType", IntEnum(MeasureType), nullable=False)  # [0,3]  # Enum as above
    sample = Column(Float, nullable=False)
    line = Column(Float, nullable=False)
    sampler = Column(Float)  # Sample Residual
    liner = Column(Float)  # Line Residual
    ignore = Column("measureIgnore", Boolean, default=False)
    jigreject = Column("measureJigsawRejected", Boolean, default=False)  # jigsaw rejected
    aprioriline = Column(Float)
    apriorisample = Column(Float)
    samplesigma = Column(Float)
    linesigma = Column(Float)
    weight = Column(Float, default=None)
    rms = Column(Float)

    @hybrid_property
    def measuretype(self):
        return self._measuretype

    @measuretype.setter
    def measuretype(self, v):
        if isinstance(v, int):
            v = MeasureType(v)
        self._measuretype = v

if isinstance(Session, sqlalchemy.orm.sessionmaker):
    from autocnet.io.db.triggers import valid_point_function, valid_point_trigger, valid_geom_function, valid_geom_trigger, ignore_image_function, ignore_image_trigger

    # Create the database
    if not database_exists(engine.url):
        create_database(engine.url, template='template_postgis')  # This is a hardcode to the local template

    # Trigger that watches for points that should be active/inactive
    # based on the point count.
    if not engine.dialect.has_table(engine, "points"):
        event.listen(Base.metadata, 'before_create', valid_point_function)
        event.listen(Measures.__table__, 'after_create', valid_point_trigger)
        event.listen(Base.metadata, 'before_create', valid_geom_function)
        event.listen(Images.__table__, 'after_create', valid_geom_trigger)
        event.listen(Base.metadata, 'before_create', ignore_image_function)
        event.listen(Images.__table__, 'after_create', ignore_image_trigger)

    Base.metadata.bind = engine
    # If the table does not exist, this will create it. This is used in case a
    # user has manually dropped a table so that the project is not wrecked.
    Base.metadata.create_all(tables=[Overlay.__table__,
                                     Edges.__table__, Costs.__table__, Matches.__table__,
                                     Cameras.__table__, Points.__table__,
                                     Measures.__table__, Images.__table__,
                                     Keypoints.__table__])
