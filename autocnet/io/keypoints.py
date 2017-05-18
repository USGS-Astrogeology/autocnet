import os

import numpy as np
import pandas as pd
from plio.io import io_hdf

from autocnet.utils import utils

def from_hdf(in_path, key=None):
    """
    For a given node, load the keypoints and descriptors from a hdf5 file.

    Parameters
    ----------
    in_path : str
              handle to the file

    key : str
          An optional path into the HDF5.  For example key='image_name', will
          search /image_name/descriptors for the descriptors.

    Returns
    -------
    keypoints : DataFrame
                A pandas dataframe of keypoints.

    descriptors : ndarray
                  A numpy array of descriptors
    """
    if isinstance(in_path, str):
        hdf = io_hdf.HDFDataset(in_path, mode='r')
    else:
        hdf = in_path

    if key:
        outd = '{}/descriptors'.format(key)
        outk = '{}/keypoints'.format(key)
    else:
        outd = '/descriptors'
        outk = '/keypoints'

    descriptors = hdf[outd][:]
    raw_kps = hdf[outk][:]
    index = raw_kps['index']
    clean_kps = utils.remove_field_name(raw_kps, 'index')
    columns = clean_kps.dtype.names

    allkps = pd.DataFrame(data=clean_kps, columns=columns, index=index)

    if isinstance(in_path, str):
        hdf = None

    return allkps, descriptors


def to_hdf(keypoints, descriptors, out_path, key=None):
    """
    Save keypoints and descriptors to HDF at a given out_path at either
    the root or at some arbitrary path given by a key.

    Parameters
    ----------
    keypoints : DataFrame
                Pandas dataframe of keypoints

    descriptors : ndarray
                  of feature descriptors

    out_path : str
               to the HDF5 file

    key : str
          path within the HDF5 file.  If given, the keypoints and descriptors
          are save at <key>/keypoints and <key>/descriptors respectively.
    """
    # If the out_path is a string, access the HDF5 file
    if isinstance(out_path, str):
        if os.path.exists(out_path):
            mode = 'a'
        else:
            mode = 'w'
        hdf = io_hdf.HDFDataset(out_path, mode=mode)
    else:
        hdf = out_path

    #try:
    if key:
        outd = '{}/descriptors'.format(key)
        outk = '{}/keypoints'.format(key)
    else:
        outd = '/descriptors'
        outk = '/keypoints'
    hdf.create_dataset(outd,
                       data=descriptors,
                       compression=io_hdf.DEFAULT_COMPRESSION,
                       compression_opts=io_hdf.DEFAULT_COMPRESSION_VALUE)
    hdf.create_dataset(outk,
                       data=hdf.df_to_sarray(keypoints.reset_index()),
                       compression=io_hdf.DEFAULT_COMPRESSION,
                       compression_opts=io_hdf.DEFAULT_COMPRESSION_VALUE)
    #except:
        #warnings.warn('Descriptors for the node {} are already stored'.format(self['image_name']))

    # If the out_path is a string, assume this method is being called as a singleton
    # and close the hdf file gracefully.  If an object, let the instantiator of the
    # object close the file
    if isinstance(out_path, str):
        hdf = None

def from_npy(in_path):
    """
    Load keypoints and descriptors from a .npz file.

    Parameters
    ----------
    in_path : str
              PATH to the npz file

    Returns
    -------
    keypoints : DataFrame
                of keypoints

    descriptors : ndarray
                  of feature descriptors
    """
    nzf = np.load(in_path)
    descriptors = nzf['descriptors']
    keypoints = pd.DataFrame(nzf['keypoints'], index=nzf['keypoints_idx'], columns=nzf['keypoints_columns'])

    return keypoints, descriptors

def to_npy(keypoints, descriptors, out_path):
    """
    Save keypoints and descriptors to a .npz file at some out_path

    Parameters
    ----------
    keypoints : DataFrame
                of keypoints

    descriptors : ndarray
                  of feature descriptors

    out_path : str
               PATH and filename to save the features
    """
    np.savez(out_path, descriptors=descriptors,
             keypoints=keypoints,
             keypoints_idx=keypoints.index,
             keypoints_columns=keypoints.columns)
