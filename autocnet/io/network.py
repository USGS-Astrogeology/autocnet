from io import BytesIO
import json
import os
import warnings
from zipfile import ZipFile

from networkx.readwrite import json_graph
import numpy as np
import pandas as pd

import autocnet


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        """If input object is an ndarray it will be converted into a dict
        holding dtype, shape and the data, base64 encoded.
        """
        if isinstance(obj, np.ndarray):
            return dict(__ndarray__= obj.tolist(),
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

def save(network, projectname):
    """
    Save an AutoCNet candiate graph to disk in a compressed file.  The
    graph adjacency structure is stored as human readable JSON and all
    potentially large numpy arrays are stored as compressed binary. The
    project archive is a standard .zip file that can have any ending,
    e.g., <projectname>.project, <projectname>.zip, <projectname>.myname.

    TODO: This func. writes a intermediary .npz to disk when saving.  Can
    we write the .npz to memory?

    Parameters
    ----------
    network : object
              The AutoCNet Candidate Graph object

    projectname : str
                  The PATH to the output file.
    """
    # Convert the graph into json format
    js = json_graph.node_link_data(network)

    with ZipFile(projectname, 'w') as pzip:
        js_str = json.dumps(js, cls=NumpyEncoder, sort_keys=True, indent=4)
        pzip.writestr('graph.json', js_str)

        # Write the array node_attributes to hdf
        for n, data in network.nodes_iter(data=True):
            grp = data['node_id']
            np.savez('{}.npz'.format(data['node_id']),
                     descriptors=data.descriptors,
                     keypoints=data._keypoints,
                     keypoints_idx=data._keypoints.index,
                     keypoints_columns=data._keypoints.columns)
            pzip.write('{}.npz'.format(data['node_id']))
            os.remove('{}.npz'.format(data['node_id']))

        # Write the array edge attributes to hdf
        for s, d, data in network.edges_iter(data=True):
            if s > d:
                s, d = d, s
            grp = str((s,d))
            np.savez('{}_{}.npz'.format(s, d),
                     matches=data.matches,
                     matches_idx=data.matches.index,
                     matches_columns=data.matches.columns,
                     _masks=data._masks,
                     _masks_idx=data._masks.index,
                     _masks_columns=data._masks.columns)
            pzip.write('{}_{}.npz'.format(s, d))
            os.remove('{}_{}.npz'.format(s, d))

def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = np.asarray(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

def load(projectname):

    with ZipFile(projectname, 'r') as pzip:
        # Read the graph object
        with pzip.open('graph.json', 'r') as g:
            data = json.loads(g.read().decode(),object_hook=json_numpy_obj_hook)

        cg = autocnet.graph.network.CandidateGraph()
        Edge = autocnet.graph.edge.Edge
        Node = autocnet.graph.node.Node
        # Reload the graph attributes
        cg.graph = data['graph']
        # Handle nodes
        for d in data['nodes']:
            n = Node(image_name=d['image_name'], image_path=d['image_path'], node_id=d['id'])
            n['hash'] = d['hash']
            try:
                # Load the byte stream for the nested npz file into memory and then unpack
                n.load_features(BytesIO(pzip.read('{}.npz'.format(d['id']))))
            except:
                pass  # The node does not have features to load.
            cg.add_node(d['node_id'])
            cg.node[d['node_id']] = n
        for e in data['links']:
            cg.add_edge(e['source'], e['target'])
            edge = Edge()
            edge.source = cg.node[e['source']]
            edge.destination = cg.node[e['target']]
            edge['fundamental_matrix'] = e['fundamental_matrix']
            edge['weights'] = e['weights']
            try:
                nzf = np.load(BytesIO(pzip.read('{}_{}.npz'.format(e['source'], e['target']))))
                edge._masks = pd.DataFrame(nzf['_masks'], index=nzf['_masks_idx'], columns=nzf['_masks_columns'])
                edge.matches = pd.DataFrame(nzf['matches'], index=nzf['matches_idx'], columns=nzf['matches_columns'])
            except:
                pass
            # Add a mock edge
            cg.edge[e['source']][e['target']] = edge

    return cg
