from base64 import encodebytes, decodebytes
import datetime
import json

import dill
import numpy as np
import shapely

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
                decoded = decodebytes(v.encode())
                v = dill.loads(decoded)
            except: pass
            dct[k] = v
    return dct
        