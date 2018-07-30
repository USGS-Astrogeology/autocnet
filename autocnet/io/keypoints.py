import os

import numpy as np
import pandas as pd
from plio.io import io_hdf

from autocnet.utils import utils

def from_hdf(in_path, index=None, keypoints=True, descriptors=True):
    """
    For a given node, load the keypoints and descriptors from a hdf5 file. The
    keypoints and descriptors kwargs support returning only keypoints or descriptors.
    The index kwarg supports returning a subset of the data.

    Parameters
    ----------
    in_path : str
              handle to the file

    key : str
          An optional path into the HDF5.  For example key='image_name', will
          search /image_name/descriptors for the descriptors.

    index : iterable
            an h5py accepted indexer to pull only a subset of the keypoints
            off disk. Default is None to pull all keypoints.

    keypoints : bool
                if True (default) return the keypoints

    descriptors : bool
                  if True (default) return the descriptors

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

    outd = '/descriptors'
    outk = '/keypoints'

    if index is not None:
        index=np.asarray(index)

        # The indices into HDF have to be sorted lists. When indices get passed in
        # they are frequently ordered, so this pulls the data using the sorted
        # index and then reorders the data.
        i = np.argsort(index)
        ii = np.argsort(i)
        # Is is important to use sorted() so that an in-place sort is NOT used.
        if descriptors:
            desc = hdf[outd][index[i].tolist()]
            desc = desc[ii]
        if keypoints:
            raw_kps = hdf[outk][index[i].tolist()]
            raw_kps = raw_kps[ii]
    else:
        # Unlike numpy hdf does not handle NoneType as a proxy for `:`
        if descriptors:
            desc = hdf[outd][:]
        if keypoints:
            raw_kps = hdf[outk][:]
    
    if keypoints:
        index = raw_kps['index']
        clean_kps = utils.remove_field_name(raw_kps, 'index')
        columns = clean_kps.dtype.names

        allkps = pd.DataFrame(data=clean_kps, columns=columns, index=index)

    if isinstance(in_path, str):
        hdf = None

    if keypoints and descriptors:
        return allkps, desc
    elif keypoints:
        return allkps
    else:
        return desc


def to_hdf(out_path, keypoints=None, descriptors=None, key=None):
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
        hdf = io_hdf.HDFDataset(out_path, mode='a')
    else:
        hdf = out_path

    grps = list(hdf.keys())

    outd = '/descriptors'
    outk = '/keypoints'
    if descriptors is not None:
        # Strip the leading slash
        if outd[1:] in grps:
            del hdf[outd] # Prep to replace

        hdf.create_dataset(outd,
                        data=descriptors,
                        compression=io_hdf.DEFAULT_COMPRESSION,
                        compression_opts=io_hdf.DEFAULT_COMPRESSION_VALUE)

    if keypoints is not None:
        if outk[1:] in grps:
            del hdf[outk]

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
        del hdf

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
