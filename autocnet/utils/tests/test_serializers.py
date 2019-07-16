import pytest
import numpy as np

from datetime import datetime
from inspect import signature
import json

from shapely.geometry import Point

from autocnet.utils.serializers import JsonEncoder, object_hook

@pytest.mark.parametrize("data, serialized", [
    ({'foo':np.arange(5)}, {"foo": [0, 1, 2, 3, 4]}),
    ({'foo':np.int64(1)}, {"foo": 1}),
    ({'foo':b'bar'}, {"foo": "bar"}),
    ({'foo':set(['a', 'b', 'c'])}, {"foo": ["a", "b", "c"]}),
    ({'foo':Point(0,0)}, {"foo": 'POINT (0 0)'}),
    ({'foo':datetime(1982, 9, 8)}, {"foo": '1982-09-08 00:00:00'})
])
def test_json_encoder(data, serialized):
    res = json.dumps(data, cls=JsonEncoder)
    res = json.loads(res, object_hook=object_hook)
    if isinstance(res['foo'], list):
        res['foo'] = sorted(res['foo'])
    assert res == serialized

@pytest.mark.parametrize("data", [
    ({'func':lambda x:True}),
    ({'func':lambda x:True, 'other':1}),
    ({'func':lambda x:True, 'other':1,
       'nested':{'inside':'foo'}}),
    ({'func':lambda x,y:x+y})
])
def test_json_roundtrip(data):
    as_str = json.dumps(data, cls=JsonEncoder)
    as_dict = json.loads(as_str, object_hook=object_hook)
    for k, v in data.items():
        if callable(v):
            sig = signature(v)
            nparams = len(sig.parameters)
            args = [True] * nparams
            assert as_dict[k](*args) == v(*args)
            continue
        assert as_dict[k] == v