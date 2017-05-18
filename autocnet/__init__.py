import os
import warnings

import autocnet

import autocnet.examples
import autocnet.camera
import autocnet.cg
import autocnet.control
import autocnet.graph
import autocnet.matcher
import autocnet.transformation
import autocnet.utils
import autocnet.utils

__version__ = "0.1.0"

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
            Edge.match = match

            from autocnet.matcher.cuda_decompose import decompose_and_match
            Edge.decompose_and_match = decompose_and_match
        except Exception:
            warning.warn('Failed to enable Cuda')
        return

    # Here is where the CPU methods get patched into the class
    from autocnet.matcher.feature_extractor import extract_features
    Node._extract_features = staticmethod(extract_features)

    from autocnet.matcher.feature_matcher import match
    Edge.match = match

    from autocnet.matcher.cpu_decompose import decompose_and_match
    Edge.decompose_and_match = decompose_and_match

cuda()
