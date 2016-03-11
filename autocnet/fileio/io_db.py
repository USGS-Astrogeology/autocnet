from sqlalchemy import Column, Integer, String, create_engine, orm
from sqlalchemy.ext import declarative

from autocnet.fileio.sqlalchemy_json.alchemy import NestedJsonObject

Base = declarative.declarative_base()


def setup_db_session(db):
    """
    Add a database session object to the root namespace

    Parameters
    ----------
    db : str
         Database name

    Returns
    -------
     : object
       A SQLAlchemy session object
    """
    engine = create_engine('sqlite:///{}'.format(db))
    Base.metadata.bind = engine
    Base.metadata.create_all()
    return orm.sessionmaker(bind=engine)()


class Translations(Base):  # pragma: no cover
    """
    Table mapping for the ISIS Translation file table
    """
    __tablename__ = 'isis_translations'
    id = Column(Integer, primary_key=True)
    mission = Column(String)
    instrument = Column(String)
    translation = Column(NestedJsonObject)

    def __init__(self, mission, instrument, translation):
        self.mission = mission
        self.instrument = instrument
        self.translation = translation


class StringToMission(Base):  # pragma: no cover
    """
    Table mapping for the ISIS mission name cleaner table
    """
    __tablename__ = 'isis_mission_to_standard'
    id = Column(Integer, primary_key=True)
    key = Column(String)
    value = Column(String)

    def __init__(self, key, value):
        self.key = key
        self.value = value
