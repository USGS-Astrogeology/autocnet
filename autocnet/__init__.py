import os
import warnings
import yaml
import socket

from sqlalchemy import create_engine, pool, orm
from sqlalchemy.event import listen

from pkg_resources import get_distribution, DistributionNotFound

try:
    _dist = get_distribution('autocnet')
    # Normalize case for Windows systems
    dist_loc = os.path.normcase(_dist.location)
    here = os.path.normcase(__file__)
    if not here.startswith(os.path.join(dist_loc, 'autocnet')):
        # not installed, but there is another version that *is*
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = 'Please install this project with setup.py'
else:
    __version__ = _dist.version

#Load the config file and setup a global DB session factory
try:
    with open(os.environ['autocnet_config'], 'r') as f:
        config = yaml.safe_load(f)
except:
    warnings.warn('No autocnet_config environment variable set. Defaulting to an en empty configuration.')
    config = {}

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
    Session = None
    engine = None

import autocnet.examples
import autocnet.camera
import autocnet.cg
import autocnet.control
import autocnet.graph
import autocnet.matcher
import autocnet.transformation
import autocnet.utils
import autocnet.spatial

# Patch the candidate graph into the root namespace
from autocnet.graph.network import CandidateGraph

def get_data(filename):
    packagdir = autocnet.__path__[0]
    dirname = os.path.join(os.path.dirname(packagdir), 'data')
    fullname = os.path.join(dirname, filename)
    return fullname

def cuda(enable=False, gpu=0):
    # Classes/Methods that can vary if GPU is available
    from autocnet.graph.node import Node
    from autocnet.graph.edge import Edge
    if enable:
        try:
            import cudasift as cs
            cs.PyInitCuda(gpu)

            # Here is where the GPU methods get patched into the class
            from autocnet.matcher.cuda_extractor import extract_features
            Node._extract_features = staticmethod(extract_features)

            from autocnet.matcher.cuda_matcher import match
            Edge._match = staticmethod(match)

            from autocnet.matcher.cuda_decompose import decompose_and_match
            Edge.decompose_and_match = decompose_and_match

            from autocnet.matcher.cuda_outlier_detector import distance_ratio
            Edge._ratio_check = staticmethod(distance_ratio)

        except Exception:
            warnings.warn('Failed to enable Cuda')
        return

    # Here is where the CPU methods get patched into the class
    from autocnet.matcher.cpu_extractor import extract_features
    Node._extract_features = staticmethod(extract_features)

    from autocnet.matcher.cpu_matcher import match
    Edge._match = staticmethod(match)

    from autocnet.matcher.cpu_decompose import decompose_and_match
    Edge.decompose_and_match = decompose_and_match

    from autocnet.matcher.cpu_outlier_detector import  distance_ratio
    Edge._ratio_check = staticmethod(distance_ratio)

cuda()
