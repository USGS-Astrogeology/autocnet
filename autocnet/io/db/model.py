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
from sqlalchemy.orm.attributes import QueryableAttribute

from geoalchemy2 import Geometry
from geoalchemy2.shape import from_shape, to_shape

import osgeo
import shapely
from shapely.geometry import Point
from autocnet.transformation.spatial import reproject, og2oc
from autocnet.utils.serializers import JsonEncoder

Base = declarative_base()

class BaseMixin(object):
    def to_dict(self, show=None, _hide=[], _path=None):     
        """Return a dictionary representation of this model."""

        show = show or []

        hidden = self._hidden_fields if hasattr(self, "_hidden_fields") else []
        default = self._default_fields if hasattr(self, "_default_fields") else []
        
        default.extend(['id', 'modified_at', 'created_at'])

        
        if not _path:
            _path = self.__tablename__.lower()

            def prepend_path(item):
                item = item.lower()
                if item.split(".", 1)[0] == _path:
                    return item
                if len(item) == 0:
                    return item
                if item[0] != ".":
                    item = ".%s" % item
                item = "%s%s" % (_path, item)
                return item

            _hide[:] = [prepend_path(x) for x in _hide]
            show[:] = [prepend_path(x) for x in show]

        columns = self.__table__.columns.keys()
        relationships = self.__mapper__.relationships.keys()
        properties = dir(self)

        ret_data = {}

        for key in columns:
            if key.startswith("_"):
                continue
            check = "%s.%s" % (_path, key)
            if check in _hide or key in hidden:
                continue
            if check in show or key in default:
                ret_data[key] = getattr(self, key)
            
        for key in relationships:
            if key.startswith("_"):
                continue
            check = "%s.%s" % (_path, key)
            if check in _hide or key in hidden:
                continue
            if check in show or key in default:
                _hide.append(check)
                is_list = self.__mapper__.relationships[key].uselist
                if is_list:
                    try:
                        # Obj is in a detached state and can't be loaded to
                        # serialize.
                        items = getattr(self, key)
                    except:
                        continue
                    if self.__mapper__.relationships[key].query_class is not None:
                        if hasattr(items, "all"):
                            items = items.all()
                    ret_data[key] = []
                    for item in items:
                        ret_data[key].append(
                            item.to_dict(
                                show=list(show),
                                _hide=list(_hide),
                                _path=("%s.%s" % (_path, key.lower())),
                            )
                        )
                else:
                    if (
                        self.__mapper__.relationships[key].query_class is not None
                        or self.__mapper__.relationships[key].instrument_class
                        is not None
                    ):
                        item = getattr(self, key)
                        if item is not None:
                            ret_data[key] = item.to_dict(
                                show=list(show),
                                _hide=list(_hide),
                                _path=("%s.%s" % (_path, key.lower())),
                            )
                        else:
                            ret_data[key] = None
                    else:
                        ret_data[key] = getattr(self, key)

        for key in list(set(properties) - set(columns) - set(relationships)):
            if key.startswith("_"):
                continue
            if not hasattr(self.__class__, key):
                continue
            attr = getattr(self.__class__, key)
            if not (isinstance(attr, property) or isinstance(attr, QueryableAttribute)):
                continue
            check = "%s.%s" % (_path, key)
            if check in _hide or key in hidden:
                continue
            if check in show or key in default:
                val = getattr(self, key)
                if hasattr(val, "to_dict"):
                    ret_data[key] = val.to_dict(
                        show=list(show),
                        _hide=list(_hide),
                        _path=('%s.%s' % (path, key.lower())),
                    )
                else:
                    try:
                        ret_data[key] = json.loads(json.dumps(val))
                    except:
                        pass

        return ret_data

    @classmethod
    def create(cls, session, **kw):
        obj = cls(**kw)
        session.add(obj)
        session.commit()
        return obj

    @staticmethod
    def bulkadd(iterable, Session):
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
    latitudinal_srid = -1
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
    weights = Column(Json())

class Costs(BaseMixin, Base):
    __tablename__ = 'costs'
    match_id = Column(Integer, ForeignKey("matches.id", ondelete="CASCADE"), primary_key=True)
    _cost = Column(JSONB)

class Matches(BaseMixin, Base):
    __tablename__ = 'matches'
    latitudinal_srid = -1
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
    source_apriori_x = Column(Float)
    source_apriori_y = Column(Float)
    destination_x = Column(Float)
    destination_y = Column(Float)
    destination_apriori_x = Column(Float)
    destination_apriori_y = Column(Float)
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
            self._geom = from_shape(geom, srid=self.latitudinal_srid)

class Cameras(BaseMixin, Base):
    __tablename__ = 'cameras'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), unique=True)
    camera = Column(Json())
    camtype = Column(String)

class Images(BaseMixin, Base):
    __tablename__ = 'images'
    latitudinal_srid = -1
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
        return json.dumps({'id':self.id,
                'name':self.name,
                'path':self.path,
                'geom':self.geom.wkt})

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
            self._geom = from_shape(newgeom, srid=self.latitudinal_srid)

class Overlay(BaseMixin, Base):
    __tablename__ = 'overlay'
    latitudinal_srid = -1
    id = Column(Integer, primary_key=True, autoincrement=True)
    intersections = Column(ARRAY(Integer))
    #geom = Column(Geometry(geometry_type='POLYGON', management=True))  # sqlite
    _geom = Column("geom", Geometry('POLYGON', srid=latitudinal_srid, dimension=2, spatial_index=True))  # postgresql
    points = relationship('Points',
                          primaryjoin='func.ST_Contains(foreign(Overlay.geom), Points.geom).as_comparison(1,2)',
                          backref=backref('overlay', uselist=False),
                          sync_backref=False,
                          viewonly=True,
                          uselist=True)


    @hybrid_property
    def geom(self):
        try:
            return to_shape(self._geom)
        except:
            return self._geom
    @geom.setter
    def geom(self, geom):
        self._geom = from_shape(geom, srid=self.latitudinal_srid)

    @classmethod
    def overlapping_larger_than(cls, size_threshold, Session):
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

class Points(Base, BaseMixin):
    __tablename__ = 'points'
    latitudinal_srid = -1
    rectangular_srid = -1
    semimajor_rad = 1
    semiminor_rad = 1

    id = Column(Integer, primary_key=True, autoincrement=True)
    _pointtype = Column("pointType", IntEnum(PointType), nullable=False)  # 2, 3, 4 - Could be an enum in the future, map str to int in a decorator
    identifier = Column(String, unique=True)
    overlapid = Column(Integer, ForeignKey('overlay.id'))
    _geom = Column("geom", Geometry('POINT', srid=latitudinal_srid, dimension=2, spatial_index=True))
    cam_type = Column(String)
    ignore = Column("pointIgnore", Boolean, default=False)
    _apriori = Column("apriori", Geometry('POINTZ', srid=rectangular_srid, dimension=3, spatial_index=False))
    _adjusted = Column("adjusted", 
                       Geometry('POINTZ', 
                                srid=rectangular_srid, 
                                dimension=3, 
                                spatial_index=False))
    measures = relationship('Measures', 
                            order_by="asc(Measures.id)", 
                            backref=backref('point', lazy='joined'))
    reference_index = Column("referenceIndex", Integer, default=0)

    _default_fields = [
        "pointtype",
        "identifier",
        "overlapid",
        "adjusted",
        "cam_type",
        "ignore",
        "measures"
    ]

    def __repr__(self):
        try:
            return 'Point: ' + str(self.to_dict())
        except:
            return 'Unable to serialize for string output'

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
            self._apriori = from_shape(apriori, srid=self.rectangular_srid)
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
            self._adjusted = from_shape(adjusted, srid=self.rectangular_srid)
            lon_og, lat_og, _ = reproject([adjusted.x, adjusted.y, adjusted.z],
                                    self.semimajor_rad, self.semiminor_rad,
                                    'geocent', 'latlon')
            lon, lat = og2oc(lon_og, lat_og, self.semimajor_rad, self.semiminor_rad)
            self._geom = from_shape(Point(lon, lat), srid=self.latitudinal_srid)
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

    #def subpixel_register(self, Session, pointid, **kwargs):
    #    subpixel.subpixel_register_point(args=(Session, pointid), **kwargs)


class CandidateGroundPoints(BaseMixin, Base):
    __tablename__ = 'candidategroundpoints'
    latitudinal_srid = -1

    id = Column(Integer,primary_key=True, autoincrement=True)
    path = Column(String, nullable=False)
    choosername = Column("ChooserName", String)
    apriorisample = Column(Float)
    aprioriline = Column(Float)
    sample = Column(Float, nullable=False)
    line = Column(Float, nullable=False)
    _geom = Column("geom", Geometry('POINT', srid=latitudinal_srid, dimension=2, spatial_index=True))
    ignore = Column(Boolean, default=False)

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
            self._geom = from_shape(newgeom, srid=self.latitudinal_srid)


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
    pointid = Column(Integer, ForeignKey('points.id'), nullable=False, index=True)
    imageid = Column(Integer, ForeignKey('images.id'), index=True)
    serial = Column("serialnumber", String, nullable=False)
    _measuretype = Column("measureType", IntEnum(MeasureType), nullable=False)  # [0,3]  # Enum as above
    ignore = Column("measureIgnore", Boolean, default=False)
    sample = Column(Float, nullable=False)
    line = Column(Float, nullable=False)
    template_metric = Column("templateMetric", Float)
    template_shift = Column("templateShift", Float)
    phase_error = Column("phaseError", Float)
    phase_diff = Column("phaseDiff", Float)
    phase_shift = Column("phaseShift", Float)
    choosername = Column("ChooserName", String)
    apriorisample = Column(Float)
    aprioriline = Column(Float)
    sampler = Column(Float)  # Sample Residual
    liner = Column(Float)  # Line Residual
    residual = Column(Float)
    jigreject = Column("measureJigsawRejected", Boolean, default=False)  # jigsaw rejected
    samplesigma = Column(Float)
    linesigma = Column(Float)
    weight = Column(Float, default=None)
    rms = Column(Float)
    point = relationship("Points", back_populates="measures")

    
    _default_fields = ['id', 'pointid', 'imageid', 'serial', 'measuretype', 'ignore',
                       'line', 'sample', 'template_metric', 'template_shift', 'phase_error',
                       'phase_diff', 'phase_shift', 'choosername', 'apriorisample', 'aprioriline',
                       'sampler', 'liner', 'residual', 'jigreject', 'samplesigma', 'linesigma',
                       'weight', 'rms']
    

    @hybrid_property
    def measuretype(self):
        return self._measuretype

    @measuretype.setter
    def measuretype(self, v):
        if isinstance(v, int):
            v = MeasureType(v)
        self._measuretype = v

def try_db_creation(engine, config):
    from autocnet.io.db.triggers import valid_point_function, valid_point_trigger, valid_geom_function, valid_geom_trigger, ignore_image_function, ignore_image_trigger

    # Create the database
    if not database_exists(engine.url):
        create_database(engine.url, template='template_postgis')  # This is a hardcode to the local template

    # Trigger that watches for points that should be active/inactive
    # based on the point count.
    if not sqlalchemy.inspect(engine).has_table("points"):
        event.listen(Base.metadata, 'before_create', valid_point_function)
        event.listen(Measures.__table__, 'after_create', valid_point_trigger)
        event.listen(Base.metadata, 'before_create', valid_geom_function)
        event.listen(Images.__table__, 'after_create', valid_geom_trigger)
        event.listen(Base.metadata, 'before_create', ignore_image_function)
        event.listen(Images.__table__, 'after_create', ignore_image_trigger)

    Base.metadata.bind = engine

    # Set the class attributes for the SRIDs
    spatial = config['spatial']
    latitudinal_srid = spatial['latitudinal_srid']
    rectangular_srid = spatial['rectangular_srid']

    Points.rectangular_srid = rectangular_srid
    Points.semimajor_rad = spatial['semimajor_rad']
    Points.semiminor_rad = spatial['semiminor_rad']
    for cls in [Points, Overlay, Images, Keypoints, Matches, CandidateGroundPoints]:
        setattr(cls, 'latitudinal_srid', latitudinal_srid)

    # If the table does not exist, this will create it. This is used in case a
    # user has manually dropped a table so that the project is not wrecked.
    Base.metadata.create_all(tables=[Overlay.__table__,
                                     Edges.__table__, Costs.__table__, Matches.__table__,
                                     Cameras.__table__, Points.__table__,
                                     Measures.__table__, Images.__table__,
                                     Keypoints.__table__, CandidateGroundPoints.__table__])
