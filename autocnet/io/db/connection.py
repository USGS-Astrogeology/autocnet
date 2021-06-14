import socket

import sqlalchemy
from sqlalchemy import create_engine, pool, orm
from sqlalchemy.orm import create_session, scoped_session, sessionmaker

import os
import socket
import warnings
import yaml


class Parent:
    def __init__(self, config):
        Session, _ = new_connection(config)
        self.session = Session()
        self.session.begin()

def new_connection(dbconfig):
    """
    Using the user supplied config create a NullPool database connection.

    Returns
    -------
    Session : object
              An SQLAlchemy session object

    engine : object
             An SQLAlchemy engine object
    """
    db_uri = 'postgresql://{}:{}@{}:{}/{}'.format(dbconfig['username'],
                                                  dbconfig['password'],
                                                  dbconfig['host'],
                                                  dbconfig['pgbouncer_port'],
                                                  dbconfig['name'])
    hostname = socket.gethostname()
    engine = sqlalchemy.create_engine(db_uri,
                poolclass=sqlalchemy.pool.NullPool,
                connect_args={"application_name":f"AutoCNet_{hostname}"},
                isolation_level="AUTOCOMMIT")
    Session = orm.sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return Session, engine
