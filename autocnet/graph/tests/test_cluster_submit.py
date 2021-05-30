import json

import fakeredis
import numpy as np
import pytest

from autocnet.utils.serializers import JsonEncoder, object_hook
from autocnet.graph import cluster_submit

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
                            ('edge', 'test', {'id':(0,1), 'image_path':('/foo.img', '/foo2.img')}),  # Case: callable func
                            ('node', 'test', {'id':0, 'image_path':'/foo.img'}),   # Case: callable func
                            ('edge', 'graph.tests.test_cluster_submit._do_nothing', {'id':(0,1), 'image_path':('/foo.img', '/foo2.img')}),  # Case: callable func
                            ('node', 'graph.tests.test_cluster_submit._do_nothing', {'id':0, 'image_path':'/foo.img'}),   # Case: callable func
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
