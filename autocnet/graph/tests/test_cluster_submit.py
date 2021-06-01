import json
from unittest.mock import patch

import fakeredis
import numpy as np
import pytest

from autocnet.utils.serializers import JsonEncoder, object_hook
from autocnet.graph import cluster_submit
from autocnet.graph.node import NetworkNode
from autocnet.graph.edge import NetworkEdge
from autocnet.io.db.model import Points


@pytest.fixture
def args():
    arg_dict = {'working_queue':'working',
                'processing_queue':'processing'}
    return arg_dict
            
@pytest.fixture
def queue():
    return fakeredis.FakeStrictRedis()

@pytest.fixture
def simple_message():
    return json.dumps({"job":"do some work",
                       "success":False}, cls=JsonEncoder
    )

@pytest.fixture
def complex_message():
    return json.dumps({'job':'do some complex work',
                      'arr':np.ones(5),
                      'func':lambda x:x}, cls=JsonEncoder)

def test_manage_simple_messages(args, queue, simple_message, mocker, capfd):
    queue.rpush(args['processing_queue'], simple_message)

    response_msg = {'success':True, 'results':'Things were good.'}
    mocker.patch('autocnet.graph.cluster_submit.process', return_value=response_msg)
    
    cluster_submit.manage_messages(args, queue)
    
    # Check that logging to stdout is working
    out, err = capfd.readouterr()
    assert out == str(response_msg) + '\n' 

    # Check that the messages are finalizing
    assert queue.llen(args['working_queue']) == 0

def test_manage_complex_messages(args, queue, complex_message, mocker, capfd):
    queue.rpush(args['processing_queue'], complex_message)

    response_msg = {'success':True, 'results':'Things were good.'}
    mocker.patch('autocnet.graph.cluster_submit.process', return_value=response_msg)
    
    cluster_submit.manage_messages(args, queue)
    
    # Check that logging to stdout is working
    out, err = capfd.readouterr()
    assert out == str(response_msg) + '\n' 

    # Check that the messages are finalizing
    assert queue.llen(args['working_queue']) == 0

def test_transfer_message_to_work_queue(args, queue, simple_message):
    queue.rpush(args['processing_queue'], simple_message)
    cluster_submit.transfer_message_to_work_queue(queue, args['processing_queue'], args['working_queue'])
    msg = queue.lpop(args['working_queue'])
    assert msg.decode() == simple_message

def test_finalize_message_from_work_queue(args, queue, simple_message):
    remove_key = simple_message
    queue.rpush(args['working_queue'], simple_message)
    cluster_submit.finalize_message_from_work_queue(queue, args['working_queue'], remove_key)
    assert queue.llen(args['working_queue']) == 0
    
def test_no_msg(args, queue):
    with pytest.warns(UserWarning, match='Expected to process a cluster job, but the message queue is empty.'):
        cluster_submit.manage_messages(args, queue)


# Classes and funcs for testing job submission.
class Foo():
    def test(self, *args, **kwargs):
        return True

def _do_nothing(*args, **kwargs): 
    return True

def _generate_obj(msg, ncg):
    return Foo()

@pytest.mark.parametrize("along, func, msg_additions", [
                            ('edge', _do_nothing, {'id':(0,1), 'image_path':('/foo.img', '/foo2.img')}),  # Case: callable func
                            ('node', _do_nothing, {'id':0, 'image_path':'/foo.img'}),   # Case: callable func
                            ('edge', 'test', {'id':(0,1), 'image_path':('/foo.img', '/foo2.img')}),  # Case:method
                            ('node', 'test', {'id':0, 'image_path':'/foo.img'}),   # Case: method
                            ('edge', 'graph.tests.test_cluster_submit._do_nothing', {'id':(0,1), 'image_path':('/foo.img', '/foo2.img')}),  # Case: imported func
                            ('node', 'graph.tests.test_cluster_submit._do_nothing', {'id':0, 'image_path':'/foo.img'}),   # Case: imported func
                        ])
def test_process_obj(along, func, msg_additions, mocker):
    msg = {'along':along,
          'config':{},
          'func':func,
          'args':[],
          'kwargs':{}}
    msg ={**msg, **msg_additions}
    mocker.patch('autocnet.graph.cluster_submit._instantiate_obj', side_effect=_generate_obj)
    mocker.patch('autocnet.graph.network.NetworkCandidateGraph.Session', return_value=True)
    mocker.patch('autocnet.graph.network.NetworkCandidateGraph.config_from_dict')

    msg = cluster_submit.process(msg)
    
    # Message result should be the same as 
    assert msg['results'] == True
    
    cluster_submit._instantiate_obj.assert_called_once()

@pytest.mark.parametrize("along, func, msg_additions", [
    ('points', _do_nothing, {}),
    ('measures', _do_nothing, {}),
    ('overlaps', _do_nothing, {}),
    ('images', _do_nothing, {})
])
def test_process_row(along, func, msg_additions, mocker):
    msg = {'along':along,
        'config':{},
        'func':func,
        'args':[],
        'kwargs':{}}
    msg ={**msg, **msg_additions}
    mocker.patch('autocnet.graph.cluster_submit._instantiate_row', side_effect=_generate_obj)
    mocker.patch('autocnet.graph.network.NetworkCandidateGraph.Session', return_value=True)
    mocker.patch('autocnet.graph.network.NetworkCandidateGraph.config_from_dict')
    msg = cluster_submit.process(msg)
    
    # Message result should be the same as 
    assert msg['results'] == True
    
    cluster_submit._instantiate_row.assert_called_once()

@pytest.mark.parametrize("along, func, msg_additions",[
                        ([1,2,3,4,5], _do_nothing, {})
                        ])
def test_process_generic(along, func, msg_additions, mocker):
    msg = {'along':along,
        'config':{},
        'func':func,
        'args':[],
        'kwargs':{}}
    msg ={**msg, **msg_additions}
    mocker.patch('autocnet.graph.cluster_submit._instantiate_row', side_effect=_generate_obj)
    mocker.patch('autocnet.graph.cluster_submit._instantiate_obj', side_effect=_generate_obj)
    mocker.patch('autocnet.graph.network.NetworkCandidateGraph.Session', return_value=True)
    mocker.patch('autocnet.graph.network.NetworkCandidateGraph.config_from_dict')
    
    assert not cluster_submit._instantiate_row.called
    assert not cluster_submit._instantiate_obj.called

    msg = cluster_submit.process(msg)

    # Message result should be the same as 
    assert msg['results'] == True

@pytest.mark.parametrize("msg, expected", [
                            ({'along':'node','id':0, 'image_path':'/foo.img'}, NetworkNode),
                            ({'along':'edge','id':(0,1), 'image_path':('/foo.img', '/foo2.img')}, NetworkEdge)
                            ])
def test_instantiate_obj(msg, expected):
    obj = cluster_submit._instantiate_obj(msg, None)
    assert isinstance(obj, expected)

@pytest.mark.parametrize("msg, expected", [
                            ({'along':'points','id':0}, Points),
                            ])

def test_instantiate_row(msg, expected, mocker):
    mock_ncg = mocker.MagicMock()
    # Mock the db query to return a row of the requested type
    mock_ncg.apply_iterable_options.return_value = {'points':Points}
    mock_ncg.session_scope.return_value.__enter__.return_value.query.return_value.filter.return_value.one.return_value = expected()

    obj = cluster_submit._instantiate_row(msg, mock_ncg)
    assert isinstance(obj, expected)