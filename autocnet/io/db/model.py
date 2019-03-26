import json

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, Float, ForeignKey, Boolean, LargeBinary, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import relationship, backref
from sqlalchemy.types import TypeDecorator
from geoalchemy2 import Geometry
from geoalchemy2.shape import to_shape


Base = declarative_base()

attr_dict = {'__tablename__':None,
             '__table_args__': {'useexisting':True},
             'id':Column(Integer, primary_key=True, autoincrement=True),
             'name':Column(String),
             'path':Column(String),
             'footprint':Column(Geometry('POLYGON')),
             'keypoint_path':Column(String),
             'nkeypoints':Column(Integer),
             'kp_min_x':Column(Float),
             'kp_max_x':Column(Float),
             'kp_min_y':Column(Float),
             'kp_max_y':Column(Float)}

def create_table_cls(name, clsname):
    attrs = attr_dict
    attrs['__tablename__'] = name
    return type(clsname, (Base,), attrs)

Base = declarative_base()


class ArrayType(TypeDecorator):
    """ 
    Sqlite does not support arrays. Therefore, use a custom type decorator.

    See http://docs.sqlalchemy.org/en/latest/core/types.html#sqlalchemy.types.TypeDecorator
    """
    impl = String

    def process_bind_param(self, value, dialect):
        return json.dumps(value)

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
        return json.dumps(value)

    def process_literal_param(self, value, dialect):
        return value

    def process_result_value(self, value, dialect):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None

class Keypoints(Base):
    __tablename__ = 'keypoints'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"))
    convex_hull_image = Column(Geometry('POLYGON'))
    convex_hull_latlon = Column(Geometry('POLYGON', srid=949900))
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

class Edges(Base):
    __tablename__ = 'edges'
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(Integer)
    destination = Column(Integer)
    ring = Column(ArrayType())
    fundamental = Column(ArrayType())
    active = Column(Boolean)
    masks = Column(Json())

class Costs(Base):
    __tablename__ = 'costs'
    match_id = Column(Integer, ForeignKey("matches.id", ondelete="CASCADE"), primary_key=True)
    _cost = Column(JSONB)

class Matches(Base):
    __tablename__ = 'matches'
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(Integer, nullable=False)
    source_idx = Column(Integer, nullable=False)
    destination = Column(Integer, nullable=False)
    destination_idx = Column(Integer, nullable=False)
    lat = Column(Float)
    lon = Column(Float)
    geom = Column(Geometry('POINTZ', dimension=3, srid=949900, spatial_index=True))
    source_x = Column(Float)
    source_y = Column(Float)
    destination_x = Column(Float)
    destination_y = Column(Float)
    shift_x = Column(Float)
    shift_y = Column(Float)
    original_destination_x = Column(Float)
    original_destination_y = Column(Float)


class Cameras(Base):
    __tablename__ = 'cameras'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), unique=True)
    camera = Column(Json())



class Images(Base):
    __tablename__ = 'images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True)
    path = Column(String, unique=True)
    active = Column(Boolean)
    footprint_latlon = Column(Geometry('MULTIPOLYGONZ', srid=949900, dimension=3, spatial_index=True))
    footprint_bodyfixed = Column(Geometry('MULTIPOLYGONZ', dimension=3))
    #footprint_bodyfixed = Column(Geometry('POLYGON',dimension=3))

    # Relationships
    keypoints = relationship(Keypoints, passive_deletes='all', backref="images", uselist=False)
    cameras = relationship(Cameras, passive_deletes='all', backref='images', uselist=False)

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

class Network(Base):
    __tablename__ = 'network'
    id = Column(Integer, primary_key=True, autoincrement=True)
    #TODO: Document that images on delete will CASCADE into all other tables
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"))  # Links back to the source image
    match_id = Column(Integer)
    point_id = Column(String) # User defined point identifier
    keypoint_id = Column(Integer)  # id to link to a keypoint in the correct file
    x = Column(Float)
    y = Column(Float)
    geom = Column(Geometry('POINTZ', dimension=3, srid=949900, spatial_index=True))
    
class Overlay(Base):
    __tablename__ = 'overlay'
    id = Column(Integer, primary_key=True, autoincrement=True)
    overlaps = Column(ArrayType())
    geom = Column(Geometry(geometry_type='POLYGON', management=True))  # sqlite
    #geom = Column(Geometry('POLYGONZ', srid=949900, dimension=3, spatial_index=True))  # postgresql
