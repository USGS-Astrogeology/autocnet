#!/usr/bin/env python

import argparse
import copy
import os
import json
import sys
import warnings

from redis import StrictRedis

from autocnet.graph.network import NetworkCandidateGraph
from autocnet.graph.node import NetworkNode
from autocnet.graph.edge import NetworkEdge
from autocnet.io.db.model import Points, Measures, Overlay
from autocnet.utils.utils import import_func
from autocnet.utils.serializers import JsonEncoder, object_hook


def parse_args():  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--host', help='The host URL for the redis queue to to pull messages from.')
    parser.add_argument('-p', '--port', help='The port for used by redis.')
    parser.add_argument('processing_queue', help='The name of the processing queue to draw messages from.')
    parser.add_argument('working_queue', help='The name of the queue to push messages to while they process.')

    return parser.parse_args()

def _instantiate_obj(msg, ncg):
    """
    Instantiate either a NetworkNode or a NetworkEdge that is the 
    target of processing.

    """
    along = msg['along']
    id = msg['id']
    image_path = msg['image_path']
    if along == 'node':
        obj = NetworkNode(node_id=id, image_path=image_path)
    elif along == 'edge':
        obj = NetworkEdge()
        obj.source = NetworkNode(node_id=id[0], image_path=image_path[0])
        obj.destination = NetworkNode(node_id=id[1], image_path=image_path[1])
    obj.parent = ncg
    return obj

def _instantiate_row(msg, ncg):
    """
    Instantiate some db.io.model row object that is the target
    of processing.
    """
    # Get the dict mapping iterable keyword types to the objects
    objdict = ncg.apply_iterable_options
    rowid = msg['id']
    obj = objdict[msg['along']]
    with ncg.session_scope() as session:
        res = session.query(obj).filter(getattr(obj, 'id')==msg['id']).one()
        session.expunge_all() # Disconnect the object from the session
    return res

def process(msg):
    """
    Given a message, instantiate the necessary processing objects and 
    apply some generic function or method.

    Parameters
    ----------
    msg : dict
          The message that parametrizes the job.
    """
    ncg = NetworkCandidateGraph()
    ncg.config_from_dict(msg['config'])
    if msg['along'] in ['node', 'edge']:
        obj = _instantiate_obj(msg, ncg)
    elif msg['along'] in ['points', 'measures', 'overlaps', 'images']:
        obj = _instantiate_row(msg, ncg)
    else:
        obj = msg['along']

    # Grab the function and apply. This assumes that the func is going to
    # have a True/False return value. Basically, all processing needs to
    # occur inside of the func, nothing belongs in here.
    #
    # All args/kwargs are passed through the RedisQueue, and then right on to the func.
    func = msg['func']
    if callable(func):  # The function is a de-serialzied function
        msg['args'] = (obj, *msg['args'])
        msg['kwargs']['ncg'] = ncg
    elif hasattr(obj, msg['func']):  # The function is a method on the object
        func = getattr(obj, msg['func'])
    else:  # The func is a function from a library to be imported
        func = import_func(msg['func'])
        # Get the object begin processed prepended into the args.
        msg['args'] = (obj, *msg['args'])
        # For now, pass all the potential config items through
        # most funcs will simply discard the unnecessary ones.
        msg['kwargs']['ncg'] = ncg
        msg['kwargs']['Session'] = ncg.Session

    # Now run the function.
    res = func(*msg['args'], **msg['kwargs'])
    
    # Update the message with the True/False
    msg['results'] = res
    # Update the message with the correct callback function

    return msg

def transfer_message_to_work_queue(queue, queue_from, queue_to):
    """
    Atomic pop/push from one redis list to another

    Parameters
    ----------
    queue : object
            PyRedis queue
    
    queue_from : str
                 The name of the queue to pop a message from
    
    queue_to : str
               The name of the queue to push a message to

    Returns
    -------
      : str
        The message from the queue
    """
    return queue.rpoplpush(queue_from, queue_to)

def finalize_message_from_work_queue(queue, queue_name, remove_key):
    """
    Remove a message from a queue

    Parameters
    ----------
    queue : object
            PyRedis queue

    queue_name : str
                 The name of the queue to remove an object from

    remove_key : obj
                 The message to remove from the list
    """
    # The operation completed. Remove this message from the working queue.  
    queue.lrem(queue_name, 0, remove_key)

def manage_messages(args, queue):
    """
    This function manages pulling a message from a redis list, atomically pushing 
    the message to another redis list, launching a generic processing job, 
    and finalizing the message by removing it from the intermediary redis list.

    This function is an easily testable main for the cluster_submit CLI.

    Parameters
    ----------
    args : dict
           A dictionary with queue names that are parsed from the CLI

    queue : obj
            A py-Redis queue object

    """
    # Pop the message from the left queue and push to the right queue; atomic operation
    msg = transfer_message_to_work_queue(queue, 
                                         args['processing_queue'],
                                         args['working_queue'])

    if msg is None:
        warnings.warn('Expected to process a cluster job, but the message queue is empty.')
        return

    # The key to remove from the working queue is the message. Essentially, find this element
    # in the list where the element is the JSON representation of the message. Maybe swap to a hash?
    remove_key = msg

    # Apply the algorithm
    response = process(msg)
    # Should go to a logger someday!
    print(response)

    finalize_message_from_work_queue(queue, args['working_queue'], remove_key)

def main():  # pragma: no cover
    args = vars(parse_args())
    
    # Get the message
    queue = StrictRedis(host=args['host'], port=args['port'], db=0)
    manage_messages(args, queue)
