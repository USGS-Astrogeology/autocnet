import datetime
import enum
import json

import numpy as np

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
from autocnet import engine, Session, config

Base = declarative_base()

# Default to mars if no config is set
spatial = config.get('spatial', {'srid': 949900})
srid = spatial['srid']

class BaseMixin(object):
    @classmethod
    def create(cls, session, **kw):
        obj = cls(**kw)
        session.add(obj)
        session.commit()
        return obj

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, datetime.datetime):
            return obj.__str__()
        if isinstance(obj, bytes):
            return obj.decode("utf-8")
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj,  shapely.geometry.base.BaseGeometry):
            return obj.wkt
        return json.JSONEncoder.default(self, obj)

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
    convex_hull_latlon = Column(Geometry('POLYGON', srid=srid))
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
    active = Column(Boolean)
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
    _geom = Column("geom", Geometry('POINT', dimension=2, srid=srid, spatial_index=True))
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
            self._geom = from_shape(geom, srid=srid)

class Cameras(BaseMixin, Base):
    __tablename__ = 'cameras'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), unique=True)
    camera = Column(Json())

class Images(BaseMixin, Base):
    __tablename__ = 'images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    path = Column(String)
    serial = Column(String, unique=True)
    active = Column(Boolean, default=True)
    _footprint_latlon = Column("footprint_latlon", Geometry('MultiPolygon', srid=srid, dimension=2, spatial_index=True))
    footprint_bodyfixed = Column(Geometry('MULTIPOLYGON', dimension=2))
    #footprint_bodyfixed = Column(Geometry('POLYGON',dimension=3))

    # Relationships
    keypoints = relationship(Keypoints, passive_deletes='all', backref="images", uselist=False)
    cameras = relationship(Cameras, passive_deletes='all', backref='images', uselist=False)
    measures = relationship("Measures")

    def __repr__(self):
        try:
            footprint = to_shape(self.footprint_latlon).__geo_interface__
        except:
            footprint = None
        return json.dumps({'id':self.id,
                'name':self.name,
                'path':self.path,
                'footprint_latlon':footprint,
                'footprint_bodyfixed':self.footprint_bodyfixed})

    @hybrid_property
    def footprint_latlon(self):
        try:
            return to_shape(self._footprint_latlon)
        except:
            return self._footprint_latlon

    @footprint_latlon.setter
    def footprint_latlon(self, geom):
        if isinstance(geom, osgeo.ogr.Geometry):
            # If an OGR geom, convert to shapely
            geom = shapely.wkt.loads(geom.ExportToWkt())
        self._footprint_latlon = from_shape(geom, srid=srid)

class Overlay(BaseMixin, Base):
    __tablename__ = 'overlay'
    id = Column(Integer, primary_key=True, autoincrement=True)
    intersections = Column(ARRAY(Integer))
    #geom = Column(Geometry(geometry_type='POLYGON', management=True))  # sqlite
    _geom = Column("geom", Geometry('POLYGON', srid=srid, dimension=2, spatial_index=True))  # postgresql

    @hybrid_property
    def geom(self):
        try:
            return to_shape(self._geom)
        except:
            return self._geom
    @geom.setter
    def geom(self, geom):
        self._geom = from_shape(geom, srid=srid)

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
    _pointtype = Column("pointtype", IntEnum(PointType), nullable=False)  # 2, 3, 4 - Could be an enum in the future, map str to int in a decorator
    identifier = Column(String, unique=True)
    _geom = Column("geom", Geometry('POINT', srid=srid, dimension=2, spatial_index=True))
    active = Column(Boolean, default=True)
    apriorix = Column(Float)
    aprioriy = Column(Float)
    aprioriz = Column(Float)
    adjustedx = Column(Float)
    adjustedy = Column(Float)
    adjustedz = Column(Float)
    measures = relationship('Measures')
    rms = Column(Float)

    @hybrid_property
    def geom(self):
        try:
            return to_shape(self._geom)
        except:
            return self._geom

    @geom.setter
    def geom(self, geom):
        if geom:
            self._geom = from_shape(geom, srid=srid)

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
    serial = Column(String, nullable=False)
    _measuretype = Column("measuretype", IntEnum(MeasureType), nullable=False)  # [0,3]  # Enum as above
    sample = Column(Float, nullable=False)
    line = Column(Float, nullable=False)
    sampler = Column(Float)  # Sample Residual
    liner = Column(Float)  # Line Residual
    active = Column(Boolean, default=True)
    jigreject = Column(Boolean, default=False)  # jigsaw rejected
    aprioriline = Column(Float)
    apriorisample = Column(Float)
    samplesigma = Column(Float)
    linesigma = Column(Float)
    rms = Column(Float)

    @hybrid_property
    def measuretype(self):
        return self._measuretype

    @measuretype.setter
    def measuretype(self, v):
        if isinstance(v, int):
            v = MeasureType(v)
        self._measuretype = v

if Session:
    from autocnet.io.db.triggers import valid_point_function, valid_point_trigger, valid_geom_function, valid_geom_trigger
    # Create the database
    if not database_exists(engine.url):
        create_database(engine.url, template='template_postgis')  # This is a hardcode to the local template

        # Trigger that watches for points that should be active/inactive
        # based on the point count.
        event.listen(Base.metadata, 'before_create', valid_point_function)
        event.listen(Measures.__table__, 'after_create', valid_point_trigger)
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
