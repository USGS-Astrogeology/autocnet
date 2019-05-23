import json
import time

import numpy as np

from plurmy import slurm_walltime_to_seconds

def pop_computetime_push(queue, inqueue, outqueue):
    """
    Pop a message from a 'todo' queue, compute the maximum possible walltime,
    push the updated message to 'processing queue', and return the
    original message.' 

    Parameters
    ----------
    queue : object
            A Redis queue object

    inqueue : str
              The key for the redis store to pop from

    outqueue : str
               The key for the redis store to push to
    Returns
    -------
    msg : dict
          The message from the processing queue.
    """
    # Load the message out of the processing queue and add a max processing time key
    msg = json.loads(queue.rpop(inqueue))
    msg['max_time'] = time.time() + slurm_walltime_to_seconds(msg['walltime'])

    # Push the message to the processing queue with the updated max_time
    queue.rpush(outqueue, json.dumps(msg))

    return msg

def finalize(response, remove_key, queue, outqueue, removequeue):
    """
    Given a successful processing run, finalize the job in both processing queues

    Parameters
    ----------
    response : dict
               The reponse to the callback function
    
    remove_key : dict
                 The key to remove from the removequeue

    queue : obj
            A Redis queue object

    outqueue : str
               The name of the redis queue to push the reponse to

    removequeue : str
                  The name of the queue to remove a reponse
    """
    for k, v in response.items():
        if isinstance(v, np.ndarray):
            response[k] = v.tolist()
    queue.rpush(outqueue, json.dumps(response))

    # Now that work is done, clean out the 'working queue'
    queue.lrem(removequeue, 0, json.dumps(remove_key))