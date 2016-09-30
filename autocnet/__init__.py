import os
import autocnet

__version__ = "0.1.0"

def get_data(filename):
    packagdir = autocnet.__path__[0]
    dirname = os.path.join(os.path.dirname(packagdir), 'data')
    fullname = os.path.join(dirname, filename)
    return fullname

import autocnet.examples
import autocnet.camera
import autocnet.cg
import autocnet.control
import autocnet.graph
import autocnet.matcher
import autocnet.transformation
import autocnet.utils
import autocnet.utils