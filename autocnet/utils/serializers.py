from base64 import encodebytes, decodebytes
import datetime
import json

import dill
import numpy as np
import shapely
from shapely import wkt  # Not available in shapely.wkt

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, datetime.datetime):
            return obj.__str__()
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj,  shapely.geometry.base.BaseGeometry):
            return obj.wkt
        if callable(obj):
            return encodebytes(dill.dumps(obj)).decode()
        return json.JSONEncoder.default(self, obj)

def object_hook(dct):
    for k, v in dct.items():
        if isinstance(v, str):
            try:
                # Decodes serialized functions
                decoded = decodebytes(v.encode())
                v = dill.loads(decoded)
                dct[k] = v
                continue
            except: pass
            
            # Decodes WKT points
            if 'POINT' in v:
                try:
                    v = wkt.loads(v)
                    dct[k] = v
                    continue   
                except: pass
        # All other obj should be readable
        dct[k] = v
    return dct