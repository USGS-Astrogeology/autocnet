import numpy as np
import pandas as pd
import pytest

from plio.io import io_hdf

from .. import keypoints

@pytest.fixture
def kd(scope='module'):
    kps = pd.DataFrame(np.random.random((128,3)), columns=['a', 'b', 'c'])
    desc = np.random.random((128,128))

    return kps, desc

def test_read_write_npy(tmpdir, kd):
    kps, desc = kd
    path = tmpdir.join('out.npz')
    keypoints.to_npy(kps, desc, path.strpath)
    reloaded_kps, reloaded_desc = keypoints.from_npy(path.strpath)

    assert reloaded_kps.equals(kps)
    np.testing.assert_array_equal(reloaded_desc, desc)

def test_read_write_hdf(tmpdir, kd):
    kps, desc = kd
    path = tmpdir.join('out.h5')
    keypoints.to_hdf(kps, desc, path.strpath)
    reloaded_kps, reloaded_desc = keypoints.from_hdf(path.strpath)

    assert reloaded_kps.equals(kps)
    np.testing.assert_array_equal(reloaded_desc, desc)

def test_read_write_hdf_with_live_file(tmpdir, kd):
    kps, desc = kd
    path = tmpdir.join('live.h5')
    hf = io_hdf.HDFDataset(path.strpath, mode='w')
    keypoints.to_hdf(kps, desc, hf)
    reloaded_kps, reloaded_desc = keypoints.from_hdf(hf)

    assert reloaded_kps.equals(kps)
    np.testing.assert_array_equal(reloaded_desc, desc)
