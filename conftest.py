import os
from unittest.mock import Mock, MagicMock

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from autocnet.control import control
from autocnet.graph.network import CandidateGraph, NetworkCandidateGraph
from autocnet.graph import edge, node
from autocnet.graph.node import Node
from autocnet.io.db import model


from plio.io.io_gdal import GeoDataset

@pytest.fixture(scope='session')
def candidategraph(node_a, node_b, node_c):
    # TODO: Getting this fixture from the global conf is causing deepycopy
    # to fail.  Why?
    cg = CandidateGraph()

    # Create a candidategraph object - we instantiate a real CandidateGraph to
    # have access of networkx functionality we do not want to test and then
    # mock all autocnet functionality to control test behavior.
    edges = [(0,1,{'data':edge.Edge(0,1)}),
             (0,2,{'data':edge.Edge(0,2)}),
             (1,2,{'data':edge.Edge(1,2)})]

    cg.add_edges_from(edges)



    match_indices = [([0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7]),
                     ([0,1,2,3,4,5,8,9], [0,1,2,3,4,5,8,9]),
                     ([0,1,2,3,4,5,8,9], [0,1,2,3,4,5,6,7])]

    matches = []
    for i, e in enumerate(edges):
        c = match_indices[i]
        source_image = np.repeat(e[0], 8)
        destin_image = np.repeat(e[1], 8)
        coords = np.zeros(8)
        data = np.vstack((source_image, c[0], destin_image, c[1],
                          coords, coords, coords, coords)).T
        matches_df = pd.DataFrame(data, columns=['source_image', 'source_idx', 'destination_image', 'destination_idx',
                                                 'source_x', 'source_y', 'destination_x', 'destination_y'])
        matches.append(matches_df)

    # Mock in autocnet methods
    cg.get_matches = MagicMock(return_value=matches)

    # Mock in the node objects onto the candidate graph
    cg.nodes[0]['data'] = node_a
    cg.nodes[1]['data'] = node_b
    cg.nodes[2]['data'] = node_c

    return cg

@pytest.fixture
def default_configuration():
    config = {'cluster': {'maxfailures': 3,
              'queue': '',
              'cluster_log_dir': '',
              'cluster_submission': '',
              'tmp_scratch_dir': '',
              'extractor_memory': 8192,
              'processing_memory': 8192},
              'database': {'type': 'postgresql',
                  'username': 'postgres',
                  'password': 'NotTheDefault',
                  'host': 'localhost',
                  'port': 5432,
                  'pgbouncer_port': 5432,
                  'name': 'travis_ci_test',
                  'timeout': 500},
              'pfeffernusse': {'url': ''},
              'redis': {'basename': 'basename',
                  'host': 'host',
                  'port': '1111',
                  'completed_queue': 'basename:done',
                  'processing_queue': 'basename:proc',
                  'working_queue': 'basename:working'},
              'spatial': {'target': 'MARS',
                  'latitudinal_srid': 4326,
                  'rectangular_srid': 4978,
                  'semimajor_rad': 3396190,
                  'semiminor_rad': 3376200,
                  'proj4_str': '+proj:longlat +a:3396190 +b:3376200 +no_defs',
                  'dem': None}}
    if os.environ.get('TRAVIS', False):
        config['database']['password'] = ''
    return config

@pytest.fixture()
def ncg(default_configuration):
    ncg = NetworkCandidateGraph()
    ncg.config_from_dict(default_configuration)
    return ncg

@pytest.fixture(scope='session')
def node_a(geodata_a):
    na = Node(node_id=0, image_path='/foo/bar/', image_name='pretty.png')
    na._geodata = geodata_a
    return na

@ pytest.fixture(scope='session')
def node_b(geodata_b):
    nb = Node(node_id=1, image_path='/foo/bar/', image_name='ugly.tif')
    nb._geodata = geodata_b
    return nb

@ pytest.fixture(scope='session')
def node_c(geodata_c):
    nc = Node(node_id=2, image_path='/foo/bar/', image_name='duckling.jpg')
    nc._geodata = geodata_c
    return nc

#TODO: Can these be a single parameterized fixture - so much boilerplate!
@pytest.fixture(scope='session')
def geodata_a():
    a = Mock(spec=GeoDataset)
    a.pixel_to_latlon = MagicMock(side_effect=lambda x, y: (x, y))
    return a

@pytest.fixture(scope='session')
def geodata_b():
    b = Mock(spec=GeoDataset)
    b.pixel_to_latlon = MagicMock(side_effect=lambda x, y: (x, y))
    return b

@pytest.fixture(scope='session')
def geodata_c():
    c = Mock(spec=Node)
    c = Mock(spec=GeoDataset)
    c.pixel_to_latlon = MagicMock(side_effect=lambda x, y: (x, y))
    return c

@pytest.fixture(scope='session')
def controlnetwork():
    df = pd.DataFrame([[0, 0.0, 0.0, (0.0, 1.0), 0, 0.0, 0.0, 0, 0, np.inf, True],
                       [0, 1.0, 0.0, (0.0, 1.0), 0, 0.0, 0.0, 0, 0, np.inf, True],
                       [1, 0.0, 1.0, (0.0, 1.0), 1, 0.0, 0.0, 0, 0, np.inf, True],
                       [1, 1.0, 1.0, (0.0, 1.0), 1, 0.0, 0.0, 0, 0, np.inf, True],
                       [2, 0.0, 2.0, (0.0, 1.0), 2, 0.0, 0.0, 0, 0, np.inf, True],
                       [2, 1.0, 2.0, (0.0, 1.0), 2, 0.0, 0.0, 0, 0, np.inf, True],
                       [3, 0.0, 3.0, (0.0, 1.0), 3, 0.0, 0.0, 0, 0, np.inf, True],
                       [3, 1.0, 3.0, (0.0, 1.0), 3, 0.0, 0.0, 0, 0, np.inf, True],
                       [4, 0.0, 4.0, (0.0, 1.0), 4, 0.0, 0.0, 0, 0, np.inf, True],
                       [4, 1.0, 4.0, (0.0, 1.0), 4, 0.0, 0.0, 0, 0, np.inf, True],
                       [5, 0.0, 5.0, (0.0, 1.0), 5, 0.0, 0.0, 0, 0, np.inf, True],
                       [5, 1.0, 5.0, (0.0, 1.0), 5, 0.0, 0.0, 0, 0, np.inf, True],
                       [6, 0.0, 6.0, (0.0, 1.0), 6, 0.0, 0.0, 0, 0, np.inf, True],
                       [6, 1.0, 6.0, (0.0, 1.0), 6, 0.0, 0.0, 0, 0, np.inf, True],
                       [7, 0.0, 7.0, (0.0, 1.0), 7, 0.0, 0.0, 0, 0, np.inf, True],
                       [7, 1.0, 7.0, (0.0, 1.0), 7, 0.0, 0.0, 0, 0, np.inf, True],
                       [0, 2.0, 0.0, (0.0, 2.0), 0, 0.0, 0.0, 0, 0, np.inf, True],
                       [1, 2.0, 1.0, (0.0, 2.0), 1, 0.0, 0.0, 0, 0, np.inf, True],
                       [2, 2.0, 2.0, (0.0, 2.0), 2, 0.0, 0.0, 0, 0, np.inf, True],
                       [3, 2.0, 3.0, (0.0, 2.0), 3, 0.0, 0.0, 0, 0, np.inf, True],
                       [4, 2.0, 4.0, (0.0, 2.0), 4, 0.0, 0.0, 0, 0, np.inf, True],
                       [5, 2.0, 5.0, (0.0, 2.0), 5, 0.0, 0.0, 0, 0, np.inf, True],
                       [8, 0.0, 8.0, (0.0, 2.0), 6, 0.0, 0.0, 0, 0, np.inf, True],
                       [8, 2.0, 8.0, (0.0, 2.0), 6, 0.0, 0.0, 0, 0, np.inf, True],
                       [9, 0.0, 9.0, (0.0, 2.0), 7, 0.0, 0.0, 0, 0, np.inf, True],
                       [9, 2.0, 9.0, (0.0, 2.0), 7, 0.0, 0.0, 0, 0, np.inf, True],
                       [10, 1.0, 8.0, (1.0, 2.0), 6, 0.0, 0.0, 0, 0, np.inf, True],
                       [10, 2.0, 6.0, (1.0, 2.0), 6, 0.0, 0.0, 0, 0, np.inf, True],
                       [11, 1.0, 9.0, (1.0, 2.0), 7, 0.0, 0.0, 0, 0, np.inf, True],
                       [11, 2.0, 7.0, (1.0, 2.0), 7, 0.0, 0.0, 0, 0, np.inf, True]],
                       columns=['point_id', 'image_index', 'keypoint_index',
                                'edge', 'match_idx', 'x', 'y','x_off', 'y_off',
                                'corr', 'valid'])

    df.index.name = 'measure_id'

    #Fix types
    df['point_id'] = df['point_id'].astype(object)
    df['match_idx'] = df['match_idx'].astype(object)

    return df

@pytest.fixture
def tables(ncg):
    engine = ncg.Session().get_bind()
    return engine.table_names()

@pytest.fixture
def session(tables, request, ncg):

    session = ncg.Session()

    def cleanup():
        session.rollback()  # Necessary because some tests intentionally fail
        for t in reversed(tables):
            # Skip the srid table
            if t != 'spatial_ref_sys':
                session.execute(f'TRUNCATE TABLE {t} CASCADE')
            # Reset the autoincrementing
            if t in ['Images', 'Cameras', 'Matches', 'Measures']:
                session.execute(f'ALTER SEQUENCE {t}_id_seq RESTART WITH 1')
        session.commit()
        # Ensure that this is the only connection to the DB
        num_con = session.execute('SELECT sum(numbackends) FROM pg_stat_database;').scalar()
        assert num_con == 1
        session.close()

    request.addfinalizer(cleanup)

    return session

@pytest.fixture
def db_controlnetwork(session):

    # Create the images
    i1 = {'id':0, 'serial':'foo'}
    i2 = {'id':1, 'serial':'bar'}
    for i in [i1, i2]:
        model.Images.create(session, **i)

    for i, j in enumerate([0,2,4]):
        ptype = 2
        if j == 4:
            ptype=3  # Ground
        model.Points.create(session,
                            id=i,
                            _pointtype=ptype,
                            measures=[model.Measures(id=k+j,
                                                    imageid=k,
                                                    serial='None',
                                                    _measuretype=3,
                                                    sample=k,
                                                    line=k,
                                                    aprioriline=k,
                                                    apriorisample=k) for k in range(2)])
    session.close()

"""@pytest.fixture(scope='session')
def bad_controlnetwork(controlnetwork_data):
    cn = control.ControlNetwork()
    cn.data = controlnetwork_data
    # Since the data is being patched in, fix the measure counter
    cn._measure_id = len(cn.data) + 1
    # Add a duplicate measure in image 0 to point 0
    cn.add_measure((0,11), (0,1), 2, [1,1], point_id=0)
    return cn"""
