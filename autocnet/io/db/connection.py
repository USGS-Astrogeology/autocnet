from contextlib import contextmanager

import sqlalchemy
from sqlalchemy import create_engine, pool, orm

import os
import socket
import warnings
import yaml

from autocnet import config

try:
    db_uri = '{}://{}:{}@{}:{}/{}'.format(config['database']['type'],
                                            config['database']['username'],
                                            config['database']['password'],
                                            config['database']['host'],
                                            config['database']['pgbouncer_port'],
                                            config['database']['name'])
    hostname = socket.gethostname()
    engine = create_engine(db_uri, poolclass=pool.NullPool,
                    connect_args={"application_name":"AutoCNet_{}".format(hostname)},
                    isolation_level="AUTOCOMMIT")                   
    Session = orm.session.sessionmaker(bind=engine)

except: 
    def sessionwarn():
        raise RuntimeError('This call requires a database connection.')
    
    Session = sessionwarn
    engine = sessionwarn


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

