from contextlib import contextmanager

import sqlalchemy
from sqlalchemy import create_engine, pool, orm
from sqlalchemy.orm import create_session, scoped_session, sessionmaker

import os
import socket
import warnings
import yaml


class Parent:
    def __init__(self, config):
        self.session, _ = new_connection(config)
        self.session.begin()

def new_connection(config):
    """
    Using the user supplied config create a NullPool database connection.

    Returns
    -------
    Session : object
              An SQLAlchemy session object

    engine : object
             An SQLAlchemy engine object
    """
    db = config['database']
    db_uri = 'postgresql://{}:{}@{}:{}/{}'.format(db['username'],
                                                  db['password'],
                                                  db['host'],
                                                  db['pgbouncer_port'],
                                                  db['name'])    
    engine = sqlalchemy.create_engine(db_uri,
                                      poolclass=sqlalchemy.pool.NullPool)
    Session = sqlalchemy.orm.sessionmaker(bind=engine, autocommit=True)
    return Session(), engine
